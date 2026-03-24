#!/usr/bin/env python3
"""
矫正图形 Step1 check:
- Images in 矫正图形/ are 3072x4096 (actual)
- JSONs describe annotations in 1752x3504 space (geometrically corrected tray)
- Task: find tray 4 corners in actual image, compute perspective transform,
  map annotation polygons back to actual image space
- Test multiple candidate transforms, pick best by IoU with green foreground
- Save: overlay on orig, overlay on half-size, transformed image, corrected JSON, log
"""
import json, cv2, numpy as np, os, shutil
from pathlib import Path
from itertools import product

SRC_JZX  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/矫正图形')
OUT_DIR  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_check_out')
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEMS = ['IMG_20240310_112137', 'IMG_20240311_103331', 'IMG_20240312_101729']


def extract_green_mask(img_bgr):
    """Extract green plant foreground mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Green hue range
    m1 = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # Also include darker greens
    m2 = cv2.inRange(hsv, (25, 30, 30), (95, 255, 200))
    mask = cv2.bitwise_or(m1, m2)
    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def polys_to_mask(polys, h, w):
    """Render polygon list to binary mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    for pts in polys:
        arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(m, [arr], 255)
    return m


def score_transform(polys_transformed, img_bgr):
    """Score how well transformed polygons match green foreground."""
    h, w = img_bgr.shape[:2]
    ann_mask = polys_to_mask(polys_transformed, h, w)
    green = extract_green_mask(img_bgr)
    inter = cv2.bitwise_and(ann_mask, green)
    union = cv2.bitwise_or(ann_mask, green)
    iou = (np.sum(inter) + 1) / (np.sum(union) + 1)
    dice = (2*np.sum(inter) + 1) / (np.sum(ann_mask) + np.sum(green) + 1)
    # foreground coverage: what fraction of green is covered by annotations
    cov = (np.sum(inter) + 1) / (np.sum(green) + 1)
    return 0.4*iou + 0.3*dice + 0.3*cov


def find_tray_corners_auto(img_bgr, json_w, json_h):
    """
    Attempt to find 4 tray corners in the actual image.
    Returns list of candidate src_pts (4 points in actual image) that map
    the json rect [0,0,json_w,json_h] to those corners.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Try to find the tray as the largest rectangular contour
    candidates = []
    
    # Method 1: Edge-based
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.1*h*w:
            pts = approx.reshape(4,2).astype(np.float32)
            candidates.append(('contour4', pts))
            break
    
    # Method 2: Green mask bounding rect
    green = extract_green_mask(img_bgr)
    cnts2, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts2:
        largest = max(cnts2, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)
        candidates.append(('green_minrect', box))
        x,y,bw,bh = cv2.boundingRect(largest)
        box2 = np.array([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]], dtype=np.float32)
        candidates.append(('green_boundingrect', box2))
    
    # Method 3: Simple scale (direct and proportional)
    # Direct: json coords directly on actual image (no transform)
    candidates.append(('direct_scale_1', None))  # identity
    # Proportional scale from json space to actual space
    sx = w / json_w; sy = h / json_h
    candidates.append(('prop_scale', (sx, sy, 0, 0)))
    # Centered proportional scale
    off_x = (w - json_w*min(sx,sy))/2
    off_y = (h - json_h*min(sx,sy))/2
    s = min(sx,sy)
    candidates.append(('centered_scale', (s, s, off_x, off_y)))
    
    return candidates


def order_pts(pts):
    """Order: TL, TR, BR, BL."""
    pts = pts[np.argsort(pts[:,1])]  # sort by y
    top = pts[:2][np.argsort(pts[:2,0])]  # top: sort by x
    bot = pts[2:][np.argsort(pts[2:,0])]  # bottom: sort by x
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def apply_transform(shapes, transform_type, params, json_w, json_h, actual_w, actual_h):
    """Apply a candidate transform to annotation polygons."""
    result = []
    for shp in shapes:
        pts = np.array(shp['points'], dtype=np.float32)
        if transform_type == 'direct_scale_1':
            # Use JSON coords directly on actual image
            new_pts = pts.copy()
        elif transform_type == 'prop_scale':
            sx, sy, ox, oy = params
            new_pts = pts * np.array([sx, sy]) + np.array([ox, oy])
        elif transform_type == 'centered_scale':
            sx, sy, ox, oy = params
            new_pts = pts * np.array([sx, sy]) + np.array([ox, oy])
        elif transform_type in ('contour4', 'green_minrect', 'green_boundingrect'):
            # Perspective transform: json rect -> tray corners in actual image
            src_pts = order_pts(params)  # 4 corners in actual image
            dst_pts = np.array([[0,0],[json_w,0],[json_w,json_h],[0,json_h]],
                               dtype=np.float32)
            # M maps from json space to actual space
            M = cv2.getPerspectiveTransform(dst_pts, src_pts)
            ones = np.ones((len(pts),1), dtype=np.float32)
            pts_h = np.hstack([pts, ones])
            transformed = (M @ pts_h.T).T
            new_pts = (transformed[:,:2] / transformed[:,2:3])
        else:
            new_pts = pts.copy()
        # Clip to image bounds
        new_pts[:,0] = np.clip(new_pts[:,0], 0, actual_w-1)
        new_pts[:,1] = np.clip(new_pts[:,1], 0, actual_h-1)
        result.append(new_pts.tolist())
    return result


def draw_polys_colored(img, poly_list):
    vis = img.copy()
    for i, pts in enumerate(poly_list):
        hue = int((i*137.508)%180)
        color = cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
        arr = np.array(pts, dtype=np.int32)
        ov = vis.copy(); cv2.fillPoly(ov,[arr],color)
        vis = cv2.addWeighted(ov,0.3,vis,0.7,0)
        cv2.polylines(vis,[arr],True,color,2)
    return vis


def process_stem(stem, log_lines):
    img_path = SRC_JZX / f'{stem}.jpg'
    json_path = SRC_JZX / f'{stem}.json'
    if not img_path.exists():
        log_lines.append(f'{stem}: IMAGE NOT FOUND - skip')
        return

    img_bgr = cv2.imread(str(img_path))
    ah, aw = img_bgr.shape[:2]
    with open(json_path) as f: d = json.load(f)
    json_w = d['imageWidth']; json_h = d['imageHeight']
    shapes = d['shapes']

    log_lines.append(f'\n=== {stem} ===')
    log_lines.append(f'  actual: {aw}x{ah}, json: {json_w}x{json_h}, shapes: {len(shapes)}')

    # Get all candidate transforms
    candidates = find_tray_corners_auto(img_bgr, json_w, json_h)

    best_score = -1
    best_name = None
    best_polys = None
    best_params = None

    for (tname, params) in candidates:
        try:
            polys_t = apply_transform(shapes, tname, params, json_w, json_h, aw, ah)
            sc = score_transform(polys_t, img_bgr)
            log_lines.append(f'  transform={tname} score={sc:.4f}')
            if sc > best_score:
                best_score = sc; best_name = tname
                best_polys = polys_t; best_params = params
        except Exception as e:
            log_lines.append(f'  transform={tname} FAILED: {e}')

    log_lines.append(f'  BEST: {best_name} score={best_score:.4f}')

    # Save green foreground
    green = extract_green_mask(img_bgr)
    cv2.imwrite(str(OUT_DIR/f'{stem}_green.jpg'), green)

    # Save overlay on original
    overlay_orig = draw_polys_colored(img_bgr, best_polys)
    cv2.putText(overlay_orig, f'Transform: {best_name} score={best_score:.3f}',
                (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
    cv2.imwrite(str(OUT_DIR/f'{stem}_overlay_orig.jpg'), overlay_orig)

    # Save half-size overlay
    hw = aw//2; hh = ah//2
    img_half = cv2.resize(img_bgr, (hw, hh), interpolation=cv2.INTER_AREA)
    polys_half = [[[x*0.5, y*0.5] for x,y in pts] for pts in best_polys]
    overlay_half = draw_polys_colored(img_half, polys_half)
    cv2.imwrite(str(OUT_DIR/f'{stem}_overlay_half.jpg'), overlay_half)

    # Build perspective-corrected (transformed) image if perspective transform was used
    if best_name in ('contour4','green_minrect','green_boundingrect'):
        src_ordered = order_pts(best_params)
        dst_pts = np.array([[0,0],[json_w,0],[json_w,json_h],[0,json_h]], dtype=np.float32)
        M_inv = cv2.getPerspectiveTransform(src_ordered, dst_pts)
        transformed = cv2.warpPerspective(img_bgr, M_inv, (json_w, json_h))
        cv2.imwrite(str(OUT_DIR/f'{stem}_transformed.jpg'), transformed)
        log_lines.append(f'  Saved perspective-corrected image: {json_w}x{json_h}')
    else:
        # For scale transforms, just save resized to json space
        if best_name == 'direct_scale_1':
            # Crop/pad actual image to json size for comparison
            tmp = img_bgr[:json_h, :json_w] if ah>=json_h and aw>=json_w else \
                  cv2.resize(img_bgr,(json_w,json_h))
        else:
            tmp = cv2.resize(img_bgr, (json_w, json_h))
        cv2.imwrite(str(OUT_DIR/f'{stem}_transformed.jpg'), tmp)

    # Save corrected JSON with updated imageWidth/imageHeight
    d_out = dict(d)
    d_out['imageWidth'] = aw
    d_out['imageHeight'] = ah
    new_shapes = []
    for shp, new_pts in zip(shapes, best_polys):
        s = dict(shp)
        s['points'] = [[float(x), float(y)] for x,y in new_pts]
        new_shapes.append(s)
    d_out['shapes'] = new_shapes
    out_json = OUT_DIR / f'{stem}_corrected.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(d_out, f, ensure_ascii=False, indent=2)
    log_lines.append(f'  Saved corrected JSON: imageSize={aw}x{ah} shapes={len(new_shapes)}')

    # Instance-level matching stats
    ann_mask = polys_to_mask(best_polys, ah, aw)
    green_full = extract_green_mask(img_bgr)
    inter = cv2.bitwise_and(ann_mask, green_full)
    iou = np.sum(inter)/(np.sum(cv2.bitwise_or(ann_mask,green_full))+1)
    log_lines.append(f'  Global IoU(ann,green)={iou:.3f}')
    # Per-instance centroid check
    matched = 0
    for pts in best_polys:
        arr = np.array(pts, dtype=np.int32)
        cx = int(np.mean(arr[:,0])); cy = int(np.mean(arr[:,1]))
        if 0<=cy<ah and 0<=cx<aw and green_full[cy,cx]>0:
            matched+=1
    log_lines.append(f'  Instance centroid in green: {matched}/{len(best_polys)}')
    if matched/max(len(best_polys),1) < 0.3:
        log_lines.append(f'  WARNING: low centroid match - transform may be unreliable')
    else:
        log_lines.append(f'  STATUS: OK')


def main():
    log_lines = ['Step1 矫正图形 Transform Check Log', '='*60]
    for stem in STEMS:
        process_stem(stem, log_lines)
    log_lines.append('\n' + '='*60 + '\nDone.')
    log_text = '\n'.join(log_lines)
    print(log_text)
    (OUT_DIR/'step1_check.log').write_text(log_text+'\n')
    print(f'\nOutput saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
