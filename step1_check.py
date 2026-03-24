#!/usr/bin/env python3
"""
Step1 check: org_data + org_img_label
Tests multiple global transforms, picks best by IoU+coverage vs green foreground.
"""
import json, cv2, numpy as np
from pathlib import Path

DIRS = [
    Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_data'),
    # Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_img_label'),
]
OUT = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_check_out')
OUT.mkdir(parents=True, exist_ok=True)
FAIL_THRESH = 0.25


def get_green(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    m=cv2.inRange(hsv,(25,25,20),(100,255,255))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)
    return cv2.morphologyEx(m,cv2.MORPH_OPEN,k)


def score_polys(polys,green,h,w):
    ann=np.zeros((h,w),np.uint8)
    for pts in polys:
        arr=np.array(pts,dtype=np.int32)
        arr[:,0]=np.clip(arr[:,0],0,w-1); arr[:,1]=np.clip(arr[:,1],0,h-1)
        cv2.fillPoly(ann,[arr],255)
    inter=cv2.bitwise_and(ann,green); union=cv2.bitwise_or(ann,green)
    iou=(np.sum(inter)+1)/(np.sum(union)+1)
    dice=(2*np.sum(inter)+1)/(np.sum(ann)+np.sum(green)+1)
    hit=sum(1 for pts in polys
            if green[int(np.clip(np.mean([p[1] for p in pts]),0,h-1)),
                       int(np.clip(np.mean([p[0] for p in pts]),0,w-1))]>0)
    cov=hit/max(len(polys),1)
    return 0.3*iou+0.2*dice+0.4*cov, iou, cov


def tf_polys(polys,sx,sy,dx,dy):
    return [[[x*sx+dx,y*sy+dy] for x,y in pts] for pts in polys]


def get_candidates(shapes,jw,jh,aw,ah):
    raw=[s['points'] for s in shapes]
    all_pts=np.array([p for pts in raw for p in pts],np.float32)
    ax0,ay0=all_pts[:,0].min(),all_pts[:,1].min()
    ax1,ay1=all_pts[:,0].max(),all_pts[:,1].max()
    aw_=ax1-ax0; ah_=ay1-ay0
    cands=[]
    # identity
    cands.append(('identity',1,1,0,0))
    # prop scale json->actual from origin
    if jw and jh and (jw!=aw or jh!=ah):
        cands.append(('prop_orig',aw/jw,ah/jh,0,0))
        # prop scale keeping bbox
        cands.append(('prop_bbox',aw/jw,ah/jh,ax0*(1-aw/jw),ay0*(1-ah/jh)))
    # fit bbox to full image
    if aw_>0 and ah_>0:
        sx=aw/aw_; sy=ah/ah_
        cands.append(('fit_full',sx,sy,-ax0*sx,-ay0*sy))
        su=min(sx,sy)
        cands.append(('fit_uniform',su,su,-ax0*su+(aw-aw_*su)/2,-ay0*su+(ah-ah_*su)/2))
    # hflip
    cands.append(('hflip',-1,1,aw,0))
    # vflip
    cands.append(('vflip',1,-1,0,ah))
    return raw, cands


def draw_polys(img,polys):
    vis=img.copy()
    for i,pts in enumerate(polys):
        hue=int((i*137.508)%180)
        c=cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
        arr=np.array(pts,dtype=np.int32)
        arr[:,0]=np.clip(arr[:,0],0,vis.shape[1]-1)
        arr[:,1]=np.clip(arr[:,1],0,vis.shape[0]-1)
        ov=vis.copy(); cv2.fillPoly(ov,[arr],c)
        vis=cv2.addWeighted(ov,0.3,vis,0.7,0)
        cv2.polylines(vis,[arr],True,c,2)
    return vis


def process(img_path,json_path,log):
    stem=img_path.stem
    img=cv2.imread(str(img_path))
    if img is None: log.append(f'{stem}: UNREADABLE'); return
    ah,aw=img.shape[:2]
    with open(json_path) as f: d=json.load(f)
    jw=d.get('imageWidth',aw); jh=d.get('imageHeight',ah)
    shapes=d.get('shapes',[])
    if not shapes: log.append(f'{stem}: NO SHAPES'); return
    green=get_green(img)
    log.append(f'\n{stem}: actual={aw}x{ah} json={jw}x{jh} shapes={len(shapes)}')
    raw,cands=get_candidates(shapes,jw,jh,aw,ah)
    best_sc=-1; best_name=''; best_polys=raw; best_iou=0; best_cov=0
    for name,sx,sy,dx,dy in cands:
        try:
            pt=tf_polys(raw,sx,sy,dx,dy)
            sc,iou,cov=score_polys(pt,green,ah,aw)
            if sc>best_sc: best_sc=sc;best_name=name;best_polys=pt;best_iou=iou;best_cov=cov
        except: pass
    # local translation refinement
    for ddx in np.arange(-150,151,30):
        for ddy in np.arange(-150,151,30):
            pt=[[[x+ddx,y+ddy]for x,y in pts]for pts in best_polys]
            sc,iou,cov=score_polys(pt,green,ah,aw)
            if sc>best_sc: best_sc=sc;best_polys=pt;best_iou=iou;best_cov=cov
    status='OK' if best_cov>=FAIL_THRESH else 'FAILED'
    log.append(f'  transform={best_name} iou={best_iou:.3f} cov={best_cov:.3f} [{status}]')
    # save corrected json
    d_out=dict(d); d_out['imageWidth']=aw; d_out['imageHeight']=ah
    d_out['shapes']=[{**s,'points':[[float(x),float(y)]for x,y in pts]}
                     for s,pts in zip(shapes,best_polys)]
    with open(OUT/f'{stem}_corrected.json','w',encoding='utf-8') as f:
        json.dump(d_out,f,ensure_ascii=False,indent=2)
    # save overlays
    ov=draw_polys(img,best_polys)
    cv2.putText(ov,f'{status} iou={best_iou:.2f} cov={best_cov:.2f}',
                (20,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),3)
    cv2.imwrite(str(OUT/f'{stem}_overlay_orig.jpg'),ov)
    ih=cv2.resize(img,(aw//2,ah//2),interpolation=cv2.INTER_AREA)
    ph=[[[x*.5,y*.5]for x,y in pts]for pts in best_polys]
    cv2.imwrite(str(OUT/f'{stem}_overlay_half.jpg'),draw_polys(ih,ph))
    # save half-size corrected json
    d_half=dict(d_out); d_half['imageWidth']=aw//2; d_half['imageHeight']=ah//2
    d_half['shapes']=[{**s,'points':[[x*.5,y*.5]for x,y in pts]}
                      for s,pts in zip(d_out['shapes'],best_polys)]
    with open(OUT/f'{stem}_corrected_half.json','w',encoding='utf-8') as f:
        json.dump(d_half,f,ensure_ascii=False,indent=2)
    # save green mask
    cv2.imwrite(str(OUT/f'{stem}_green.jpg'),green)
    print(f'  {stem}: {status} iou={best_iou:.3f} cov={best_cov:.3f}')


def main():
    log=['Step1 Check Log','='*65]
    pairs=[]
    seen=set()
    for d in DIRS:
        for jp in sorted(d.glob('*.jpg')):
            jp2=jp.with_suffix('.json')
            if jp2.exists() and jp.stem not in seen:
                pairs.append((jp,jp2)); seen.add(jp.stem)
    print(f'Processing {len(pairs)} image+JSON pairs...')
    ok=fail=0
    for jp,jp2 in pairs:
        process(jp,jp2,log)
        if log and 'FAILED' in log[-1]: fail+=1
        else: ok+=1
    log+=['','='*65,f'Total: {len(pairs)} processed, OK={ok} FAILED={fail}']
    lp=OUT/'step1_check.log'; lp.write_text('\n'.join(log)+'\n')
    print(f'\nDone. Log: {lp}')


if __name__=='__main__':
    main()
