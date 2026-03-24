#!/usr/bin/env python3
"""Stitch 12 image patches + per-instance masks into full image."""
import cv2, numpy as np, json
from pathlib import Path
import numpy as np
base = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/pinjie')
hw, hh =np.int32(1699/2) , np.int32(3398/2)
PATCH = 512; NR = 4; NC = 3

def starts(total, n, p):
    if n == 1: return [0]
    if total <= p: return [0]*n
    s = (total-p)/(n-1)
    return [int(round(i*s)) for i in range(n)]

rs = starts(hh, NR, PATCH)
cs = starts(hw, NC, PATCH)
grid = [(x0, y0, r, c) for r,y0 in enumerate(rs) for c,x0 in enumerate(cs)]

# ── 1. Stitch image patches ───────────────────────────────────────────────────
canvas = np.zeros((hh, hw, 3), dtype=np.uint8)
for x0,y0,r,c in grid:
    p = cv2.imread(str(base/f'IMG_20230819_203159_{r}_{c}.jpg'))
    if p is None: print(f'MISSING patch {r}_{c}'); continue
    rh = min(hh, y0+PATCH)-y0; rw = min(hw, x0+PATCH)-x0
    canvas[y0:y0+rh, x0:x0+rw] = p[:rh, :rw]
cv2.imwrite(str(base/'IMG_20230819_203159_stitched.jpg'), canvas)
print(f'Stitched image saved: {hw}x{hh}')

# ── 2. Stitch instance masks ──────────────────────────────────────────────────
# Strategy: assign unique global_id per instance, place in mask
# Overlaps: keep the instance with larger original area (count in patch)
imask   = np.zeros((hh, hw), dtype=np.int32)
area_map= np.zeros((hh, hw), dtype=np.int32)  # area of instance currently in that pixel
total_inst = 0
gid_counter = 1
instance_info = {}  # global_id -> info dict

for x0,y0,r,c in grid:
    rh = min(hh, y0+PATCH)-y0; rw = min(hw, x0+PATCH)-x0
    patch_dir = base/f'IMG_20230819_203159_{r}_{c}.png'
    if not patch_dir.exists(): print(f'MISSING dir {r}_{c}'); continue
    pngs = sorted(patch_dir.glob('*.png'))
    print(f'  Patch ({r},{c}): {len(pngs)} masks', flush=True)
    for png in pngs:
        m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        m_roi = m[:rh, :rw]
        fg = m_roi > 127
        if not np.any(fg): continue
        new_area = int(np.sum(fg))
        gid = gid_counter; gid_counter += 1
        total_inst += 1
        # Region in global mask
        reg  = imask[y0:y0+rh, x0:x0+rw]
        areg = area_map[y0:y0+rh, x0:x0+rw]
        # Where empty: place directly
        empty = fg & (reg == 0)
        reg[empty]  = gid
        areg[empty] = new_area
        # Where occupied and new is larger: overwrite
        bigger = fg & (reg > 0) & (new_area > areg)
        reg[bigger]  = gid
        areg[bigger] = new_area
        imask[y0:y0+rh, x0:x0+rw]    = reg
        area_map[y0:y0+rh, x0:x0+rw] = areg
        instance_info[gid] = {'instance_id': gid, 'area': new_area, 'score': 1.0}

print(f'Total masks: {total_inst}, unique instances: {len(instance_info)}')

# Compute final bbox + centroid from stitched mask
print('Computing final bboxes...', flush=True)
for gid in instance_info:
    ys,xs = np.where(imask == gid)
    if len(xs) == 0: continue
    instance_info[gid].update({
        'bbox': [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())],
        'centroid': [float(np.mean(xs)), float(np.mean(ys))],
        'area': int(len(xs)),
    })

# Save uint16 instance mask
imask16 = np.clip(imask, 0, 65535).astype(np.uint16)
cv2.imwrite(str(base/'IMG_20230819_203159_instance_mask.png'), imask16)
print('Instance mask (uint16) saved')

# Colored visualization overlaid on stitched image
vis = canvas.copy()
for gid in instance_info:
    hue = int((gid * 137.508) % 180)
    color = cv2.cvtColor(np.uint8([[[hue,200,200]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
    mask_px = imask == gid
    ov = vis.copy(); ov[mask_px] = color
    vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)
cv2.imwrite(str(base/'IMG_20230819_203159_instance_vis.jpg'), vis)
print('Instance visualization saved')

# Save instance info JSON
with open(str(base/'IMG_20230819_203159_instances.json'), 'w') as f:
    json.dump(list(instance_info.values()), f, indent=2)
print(f'Instance info JSON saved: {len(instance_info)} instances')
print('Done.')
