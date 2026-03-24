#!/usr/bin/env python3
"""
Step 1 QA: Stitch patch vis_annotations back into full image and overlay
original annotation polygons on the half-size image for comparison.

Outputs to: mydata/step1_output/stitch_compare/
  <stem>_stitched.jpg   - patches stitched back (with green polygon overlays)
  <stem>_original.jpg   - half-size original image with original polygons
  <stem>_compare.jpg    - side-by-side comparison
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ORG_DIR  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_img_label')
OUT_DIR  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_output')
VIS_DIR  = OUT_DIR / 'vis_annotations'
STITCH_DIR = OUT_DIR / 'stitch_compare'
STITCH_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 512
N_ROWS = 4
N_COLS = 3


def compute_crop_grid(img_w, img_h, n_rows=N_ROWS, n_cols=N_COLS, patch=PATCH_SIZE):
    def starts(total, n, p):
        if n == 1: return [0]
        if total <= p: return [0]*n
        stride = (total - p) / (n - 1)
        return [int(round(i*stride)) for i in range(n)]
    row_starts = starts(img_h, n_rows, patch)
    col_starts = starts(img_w, n_cols, patch)
    grid = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = col_starts[c]; y0 = row_starts[r]
            grid.append((x0, y0, x0+patch, y0+patch, r, c))
    return grid


def draw_polygons(img, poly_list, color=(0,255,0), alpha=0.3):
    """Draw filled+outlined polygons with alpha blend."""
    vis = img.copy()
    overlay = img.copy()
    for pts in poly_list:
        arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [arr], color)
    vis = cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0)
    for pts in poly_list:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(vis, [arr], True, color, 1)
    return vis


def process_stem(stem):
    jpg_path = ORG_DIR / f'{stem}.jpg'
    json_path = ORG_DIR / f'{stem}.json'
    if not jpg_path.exists():
        print(f'  [skip] {stem}: no image')
        return

    # ── Load and half-size original image ──────────────────────────────────
    img_bgr = cv2.imread(str(jpg_path))
    orig_h, orig_w = img_bgr.shape[:2]
    half_w, half_h = orig_w // 2, orig_h // 2
    img_half = cv2.resize(img_bgr, (half_w, half_h), interpolation=cv2.INTER_AREA)

    # ── Draw original annotation polygons on half image ────────────────────
    orig_polys = []
    if json_path.exists():
        with open(json_path) as f:
            ldata = json.load(f)
        ann_w = ldata.get('imageWidth', half_w)
        ann_h = ldata.get('imageHeight', half_h)
        sx = half_w / ann_w
        sy = half_h / ann_h
        for shp in ldata.get('shapes', []):
            if shp.get('shape_type', 'polygon') != 'polygon': continue
            pts = shp.get('points', [])
            if len(pts) < 3: continue
            scaled = [(float(x)*sx, float(y)*sy) for x,y in pts]
            orig_polys.append(scaled)

    orig_vis = draw_polygons(img_half, orig_polys)

    # ── Stitch patches back into full canvas ────────────────────────────────
    canvas = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    grid = compute_crop_grid(half_w, half_h)
    for x0, y0, x1, y1, r, c in grid:
        patch_name = f'{stem}_r{r}_c{c}.jpg'
        patch_path = VIS_DIR / patch_name
        if not patch_path.exists():
            continue
        patch = cv2.imread(str(patch_path))
        if patch is None:
            continue
        # Actual region in canvas (may be smaller than patch due to padding)
        actual_h = min(half_h, y1) - y0
        actual_w = min(half_w, x1) - x0
        # Use max-blend to handle overlapping regions (keep best annotation)
        roi = canvas[y0:y0+actual_h, x0:x0+actual_w]
        patch_crop = patch[:actual_h, :actual_w]
        # Where canvas is black (0,0,0) use patch; otherwise max-blend
        mask_black = (roi[:,:,0] == 0) & (roi[:,:,1] == 0) & (roi[:,:,2] == 0)
        roi[mask_black] = patch_crop[mask_black]
        # For overlapping areas blend
        blend = cv2.addWeighted(roi, 0.5, patch_crop, 0.5, 0)
        roi[~mask_black] = blend[~mask_black]
        canvas[y0:y0+actual_h, x0:x0+actual_w] = roi

    # ── Side-by-side comparison ─────────────────────────────────────────────
    # Resize both to same height for comparison
    h_orig = orig_vis.shape[0]
    h_stitch = canvas.shape[0]
    target_h = max(h_orig, h_stitch)

    def pad_to_height(img, h):
        dh = h - img.shape[0]
        if dh > 0:
            return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(50,50,50))
        return img

    left  = pad_to_height(orig_vis, target_h)
    right = pad_to_height(canvas,   target_h)

    # Add labels
    def add_label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0,0), (img.shape[1], 30), (30,30,30), -1)
        cv2.putText(out, text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,100), 2)
        return out

    left  = add_label(left,  f'Original half-size + annotations ({len(orig_polys)} polys)')
    right = add_label(right, f'Stitched patches ({stem})')
    compare = np.concatenate([left, right], axis=1)

    # Save
    cv2.imwrite(str(STITCH_DIR / f'{stem}_original.jpg'), orig_vis)
    cv2.imwrite(str(STITCH_DIR / f'{stem}_stitched.jpg'), canvas)
    cv2.imwrite(str(STITCH_DIR / f'{stem}_compare.jpg'), compare)
    print(f'  {stem}: {len(orig_polys)} original polys, grid={len(grid)} patches')


def main():
    stems = sorted(set(p.stem for p in ORG_DIR.glob('*.jpg')))
    print(f'Processing {len(stems)} images...')
    for stem in stems:
        process_stem(stem)
    print(f'\nDone. Compare images saved to: {STITCH_DIR}')


if __name__ == '__main__':
    main()
