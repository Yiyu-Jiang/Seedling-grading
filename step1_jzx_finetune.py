#!/usr/bin/env python3
"""
Fast fine-tune: align annotations to green foreground using
direct bbox-to-bbox transform (no exhaustive grid search).
Then do a small local refinement (+/- 50px translation only).
"""
import json, cv2, numpy as np
from pathlib import Path

SRC_JZX  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/矫正图形')
CHECK_DIR = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_check_out')
STEMS = ['IMG_20240310_112137','IMG_20240311_103331','IMG_20240312_101729']


def get_green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv,(25,25,25),(95,255,255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    m = cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)
    m = cv2.morphologyEx(m,cv2.MORPH_OPEN,k)
    return m


def transform_polys(shapes, sx, sy, orig_cx, orig_cy, dx, dy):
    result = []
    for shp in shapes:
        pts = np.array(shp['points'],dtype=np.float32)
        pts[:,0] = (pts[:,0]-orig_cx)*sx + orig_cx + dx
        pts[:,1] = (pts[:,1]-orig_cy)*sy + orig_cy + dy
        result.append(pts.tolist())
    return result


def score_polys(polys, green, h, w):
    ann = np.zeros((h,w),np.uint8)
    for pts in polys:
        arr = np.array(pts,dtype=np.int32)
        arr[:,0] = np.clip(arr[:,0],0,w-1)
        arr[:,1] = np.clip(arr[:,1],0,h-1)
        cv2.fillPoly(ann,[arr],255)
    inter = cv2.bitwise_and(ann,green)
    iou = (np.sum(inter)+1)/(np.sum(cv2.bitwise_or(ann,green))+1)
    hit = 0
    for pts in polys:
        cy = int(np.mean([p[1] for p in pts]))
        cx = int(np.mean([p[0] for p in pts]))
        if 0<=cy<h and 0<=cx<w and green[cy,cx]>0:
            hit+=1
    cov = hit/max(len(polys),1)
    return 0.6*iou+0.4*cov, iou, cov


def draw_polys(img, polys):
    vis=img.copy()
    for i,pts in enumerate(polys):
        hue=int((i*137.508)%180)
        color=cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
        arr=np.array(pts,dtype=np.int32)
        arr[:,0]=np.clip(arr[:,0],0,vis.shape[1]-1)
        arr[:,1]=np.clip(arr[:,1],0,vis.shape[0]-1)
        ov=vis.copy(); cv2.fillPoly(ov,[arr],color)
        vis=cv2.addWeighted(ov,0.3,vis,0.7,0)
        cv2.polylines(vis,[arr],True,color,2)
    return vis


def optimize_stem(stem):
    img = cv2.imread(str(SRC_JZX/f'{stem}.jpg'))
    ah,aw = img.shape[:2]
    with open(CHECK_DIR/f'{stem}_corrected.json') as f: d=json.load(f)
    shapes = d['shapes']
    green = get_green_mask(img)

    # Annotation centroid and bbox
    all_pts = np.array([p for s in shapes for p in s['points']],dtype=np.float32)
    orig_cx = float(np.mean(all_pts[:,0])); orig_cy = float(np.mean(all_pts[:,1]))
    a_x0=all_pts[:,0].min(); a_y0=all_pts[:,1].min()
    a_x1=all_pts[:,0].max(); a_y1=all_pts[:,1].max()
    a_w=a_x1-a_x0; a_h=a_y1-a_y0
    a_cx=(a_x0+a_x1)/2; a_cy=(a_y0+a_y1)/2

    # Green bbox
    gy,gx_arr = np.where(green>0)
    if len(gx_arr)==0:
        print(f'  {stem}: no green, using whole image')
        g_x0,g_y0,g_x1,g_y1=0,0,aw,ah
    else:
        g_x0=int(gx_arr.min()); g_y0=int(gy.min())
        g_x1=int(gx_arr.max()); g_y1=int(gy.max())
    g_w=g_x1-g_x0; g_h=g_y1-g_y0
    g_cx=(g_x0+g_x1)/2; g_cy=(g_y0+g_y1)/2

    print(f'  ann bbox: ({a_x0:.0f},{a_y0:.0f})-({a_x1:.0f},{a_y1:.0f}) center=({a_cx:.0f},{a_cy:.0f})')
    print(f'  green bbox: ({g_x0},{g_y0})-({g_x1},{g_y1}) center=({g_cx:.0f},{g_cy:.0f})')

    # Candidate transforms (all bbox-to-bbox or simple scale+shift)
    candidates = []

    # 1. Direct bbox match: scale ann bbox to match green bbox exactly
    if a_w>0 and a_h>0:
        sx1 = g_w/a_w; sy1 = g_h/a_h
        dx1 = g_cx - a_cx*sx1 + (orig_cx*sx1-orig_cx)  # correct for scale-around-centroid
        # Actually: new_x = (x-orig_cx)*sx + orig_cx + dx
        # We want: a_x0 maps to g_x0 => (a_x0-orig_cx)*sx+orig_cx+dx = g_x0
        # Solve: dx = g_x0 - (a_x0-orig_cx)*sx - orig_cx
        dx1 = g_x0 - (a_x0-orig_cx)*sx1 - orig_cx
        dy1 = g_y0 - (a_y0-orig_cy)*sy1 - orig_cy
        candidates.append(('bbox_exact',sx1,sy1,dx1,dy1))

    # 2. Center alignment only (no scale)
    dx2 = g_cx - a_cx; dy2 = g_cy - a_cy
    candidates.append(('center_only',1.0,1.0,dx2,dy2))

    # 3. Uniform scale to fit green height, center align
    if a_h>0:
        su = g_h/a_h
        dx3 = g_cx - orig_cx*su + (orig_cx*su-orig_cx)
        dx3 = g_x0 - (a_x0-orig_cx)*su - orig_cx
        dy3 = g_y0 - (a_y0-orig_cy)*su - orig_cy
        candidates.append(('uniform_h',su,su,dx3,dy3))

    # 4. Prop scale from json to actual
    sx4=aw/d['imageWidth']; sy4=ah/d['imageHeight'] if d['imageHeight']>0 else 1
    # ann was already in actual coords, try just proportional
    candidates.append(('prop',sx4,sy4,0.0,0.0))

    # 5. Keep current (baseline)
    candidates.append(('current',1.0,1.0,0.0,0.0))

    best_sc=-1; best_name=None; best_polys=None; best_tf=None
    base_polys=[s['points'] for s in shapes]
    for (name,sx,sy,dx,dy) in candidates:
        polys_t=transform_polys(shapes,sx,sy,orig_cx,orig_cy,dx,dy)
        sc,iou,cov=score_polys(polys_t,green,ah,aw)
        print(f'    {name}: score={sc:.4f} iou={iou:.4f} cov={cov:.4f} sx={sx:.3f} sy={sy:.3f} dx={dx:.1f} dy={dy:.1f}')
        if sc>best_sc: best_sc=sc; best_name=name; best_polys=polys_t; best_tf=(sx,sy,dx,dy)

    # Local translation refinement (+/-150px, step 25)
    sx,sy,dx,dy=best_tf
    print(f'  Refining {best_name} with local translation search...')
    for ddx in np.arange(-150,151,25):
        for ddy in np.arange(-150,151,25):
            polys_t=transform_polys(shapes,sx,sy,orig_cx,orig_cy,dx+ddx,dy+ddy)
            sc,iou,cov=score_polys(polys_t,green,ah,aw)
            if sc>best_sc:
                best_sc=sc; best_polys=polys_t; best_tf=(sx,sy,dx+ddx,dy+ddy)
    sx,sy,dx,dy=best_tf
    _,best_iou,best_cov=score_polys(best_polys,green,ah,aw)
    print(f'  FINAL: score={best_sc:.4f} iou={best_iou:.4f} cov={best_cov:.4f} sx={sx:.3f} sy={sy:.3f} dx={dx:.1f} dy={dy:.1f}')

    # Save JSON
    d_out=dict(d); d_out['imageWidth']=aw; d_out['imageHeight']=ah
    new_shapes=[]
    for shp,pts in zip(shapes,best_polys):
        s=dict(shp); s['points']=[[float(x),float(y)] for x,y in pts]
        new_shapes.append(s)
    d_out['shapes']=new_shapes
    with open(CHECK_DIR/f'{stem}_corrected.json','w',encoding='utf-8') as f:
        json.dump(d_out,f,ensure_ascii=False,indent=2)

    # Save overlays
    overlay=draw_polys(img,best_polys)
    cv2.putText(overlay,f'iou={best_iou:.3f} cov={best_cov:.3f} {best_name} sx={sx:.2f} sy={sy:.2f}',
                (20,80),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
    cv2.imwrite(str(CHECK_DIR/f'{stem}_overlay_orig.jpg'),overlay)
    img_h=cv2.resize(img,(aw//2,ah//2),interpolation=cv2.INTER_AREA)
    polys_h=[[[x*0.5,y*0.5] for x,y in pts] for pts in best_polys]
    cv2.imwrite(str(CHECK_DIR/f'{stem}_overlay_half.jpg'),draw_polys(img_h,polys_h))
    print(f'  Saved overlays and corrected JSON')


if __name__=='__main__':
    for stem in STEMS:
        print(f'\n=== {stem} ===')
        optimize_stem(stem)
    print('\nAll done.')
