#!/usr/bin/env python3
"""
Stitch 12 sub-image masks (PNG dirs) into full-image mask + JSON + overlay.
- Reads per-instance PNG masks from IMG_20230819_203159_{r}_{c}.png/ dirs
- Computes polygon contours from each mask
- Converts local (patch) coords to global (full-image) coords
- Merges into one LabelMe-style JSON
- Saves: stitched mask, merged JSON, overlay on original image
"""
import cv2, numpy as np, json
from pathlib import Path

BASE   = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/pinjie')
ORIG   = BASE / 'IMG_20230819_203159.jpg'
OUT_MASK    = BASE / 'IMG_20230819_203159_full_mask.png'
OUT_JSON    = BASE / 'IMG_20230819_203159_full.json'
OUT_OVERLAY = BASE / 'IMG_20230819_203159_full_overlay.jpg'

PATCH = 512; NR = 4; NC = 3

def starts(total, n, p):
    if n == 1: return [0]
    if total <= p: return [0]*n
    s = (total-p)/(n-1)
    return [int(round(i*s)) for i in range(n)]

# Load original image
orig = cv2.imread(str(ORIG))
if orig is None: raise FileNotFoundError(ORIG)
FULL_H, FULL_W = orig.shape[:2]
HALF_W, HALF_H = FULL_W//2, FULL_H//2
print(f'Original: {FULL_W}x{FULL_H}, half: {HALF_W}x{HALF_H}')

# The sub-image JPGs are already at half size (512x512 patches of the half image)
rs = starts(HALF_H, NR, PATCH)
cs = starts(HALF_W, NC, PATCH)
grid = [(x0,y0,r,c) for r,y0 in enumerate(rs) for c,x0 in enumerate(cs)]
print(f'Grid: rows={rs}, cols={cs}')

# ── Stitch instance mask ──────────────────────────────────────────────────────
imask    = np.zeros((HALF_H, HALF_W), dtype=np.int32)
area_map = np.zeros((HALF_H, HALF_W), dtype=np.int32)
gid = 1
all_shapes = []

for x0,y0,r,c in grid:
    roi_h = min(HALF_H, y0+PATCH)-y0
    roi_w = min(HALF_W, x0+PATCH)-x0
    patch_dir = BASE / f'IMG_20230819_203159_{r}_{c}.png'
    if not patch_dir.is_dir():
        print(f'  MISSING dir: {patch_dir.name}')
        continue
    pngs = sorted(patch_dir.glob('*.png'))
    print(f'  Patch ({r},{c}) offset=({x0},{y0}) roi={roi_w}x{roi_h}: {len(pngs)} masks', flush=True)

    for png_path in pngs:
        m = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        m_roi = m[:roi_h, :roi_w]
        fg = m_roi > 127
        if not np.any(fg): continue
        new_area = int(np.sum(fg))

        # Place in global mask (larger area wins on overlap)
        reg  = imask[y0:y0+roi_h, x0:x0+roi_w]
        areg = area_map[y0:y0+roi_h, x0:x0+roi_w]
        empty  = fg & (reg == 0)
        bigger = fg & (reg > 0) & (new_area > areg)
        reg[empty | bigger]  = gid
        areg[empty | bigger] = new_area
        imask[y0:y0+roi_h, x0:x0+roi_w]    = reg
        area_map[y0:y0+roi_h, x0:x0+roi_w] = areg

        # Extract polygon contour in patch-local coords
        mask_bin = (m_roi > 127).astype(np.uint8) * 255
        # Clip to roi
        mask_bin_roi = np.zeros_like(mask_bin)
        mask_bin_roi[:roi_h, :roi_w] = mask_bin[:roi_h, :roi_w]
        cnts, _ = cv2.findContours(mask_bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: gid += 1; continue
        # Use largest contour
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 4: gid += 1; continue
        # Simplify and convert to global coords
        eps = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        # Global coords (half-image space)
        pts_global = [[float(pt[0][0]+x0), float(pt[0][1]+y0)] for pt in approx]
        if len(pts_global) < 3: gid += 1; continue

        # bbox in global half-image coords
        xs = [p[0] for p in pts_global]; ys = [p[1] for p in pts_global]
        bbox = [float(min(xs)), float(min(ys)),
                float(max(xs)-min(xs)), float(max(ys)-min(ys))]
        cx = float(np.mean(xs)); cy = float(np.mean(ys))

        all_shapes.append({
            'label': 'lettuce',
            'points': pts_global,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {},
            'instance_id': gid,
            'bbox': bbox,
            'centroid': [cx, cy],
            'area': new_area,
        })
        gid += 1

print(f'Total instances: {len(all_shapes)}')

# ── Save instance mask ────────────────────────────────────────────────────────
imask16 = np.clip(imask, 0, 65535).astype(np.uint16)
cv2.imwrite(str(OUT_MASK), imask16)
print(f'Instance mask saved (uint16): {HALF_W}x{HALF_H}')

# ── Save merged JSON (LabelMe style, half-image coords) ──────────────────────
coco_json = {
    'version': '5.0.1',
    'flags': {},
    'shapes': all_shapes,
    'imagePath': ORIG.name,
    'imageData': None,
    'imageHeight': HALF_H,
    'imageWidth':  HALF_W,
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(coco_json, f, ensure_ascii=False, indent=2)
print(f'Full JSON saved: {len(all_shapes)} shapes')

# ── Save overlay on half-size original ───────────────────────────────────────
img_half = cv2.resize(orig, (HALF_W, HALF_H), interpolation=cv2.INTER_AREA)
vis = img_half.copy()
for i, shp in enumerate(all_shapes):
    hue = int((i * 137.508) % 180)
    color = cv2.cvtColor(np.uint8([[[hue,200,200]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
    pts = np.array(shp['points'], dtype=np.int32)
    ov = vis.copy(); cv2.fillPoly(ov, [pts], color)
    vis = cv2.addWeighted(ov, 0.3, vis, 0.7, 0)
    cv2.polylines(vis, [pts], True, color, 1)
cv2.imwrite(str(OUT_OVERLAY), vis)
print(f'Overlay saved: {HALF_W}x{HALF_H}')
print('Done.')
