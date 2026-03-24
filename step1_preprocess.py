#!/usr/bin/env python3
"""
Step 1: 主体归属 + 重叠区域保留
- Actual image size is coordinate basis
- Primary patch = largest visible area; overlap patches keep if ratio >= OVERLAP_RATIO
"""
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path
import cv2, numpy as np

PATCH_SIZE=512; N_ROWS=4; N_COLS=3
MIN_AREA_RATIO=0.10; MIN_BBOX_SIZE=6; MIN_AREA_PX=10; OVERLAP_RATIO=0.05

def _poly_area(pts):
    if len(pts)<3: return 0.0
    s=0.0
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]+pts[:1]): s+=x1*y2-x2*y1
    return abs(s)*0.5

def _poly_bbox(pts):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return [float(min(xs)),float(min(ys)),float(max(xs)-min(xs)),float(max(ys)-min(ys))]

def _clip_sh(pts,x0,y0,x1,y1):
    def inside(p,e):
        if e=='left': return p[0]>=x0
        if e=='right': return p[0]<=x1
        if e=='top': return p[1]>=y0
        return p[1]<=y1
    def isect(p1,p2,e):
        xa,ya=p1; xb,yb=p2
        if e in('left','right'):
            bx=x0 if e=='left' else x1; t=(bx-xa)/(xb-xa+1e-12)
            return(bx,ya+t*(yb-ya))
        by=y0 if e=='top' else y1; t=(by-ya)/(yb-ya+1e-12)
        return(xa+t*(xb-xa),by)
    out=list(pts)
    for e in('left','right','top','bottom'):
        if not out: break
        inp=out; out=[]
        for i in range(len(inp)):
            c=inp[i]; p=inp[i-1]
            if inside(c,e):
                if not inside(p,e): out.append(isect(p,c,e))
                out.append(c)
            elif inside(p,e): out.append(isect(p,c,e))
    return out

def _icolor(idx):
    hue=int((idx*137.508)%180)
    return cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

def _draw(img,polys):
    vis=img.copy()
    for i,pts in enumerate(polys):
        color=_icolor(i); arr=np.array(pts,dtype=np.int32)
        ov=vis.copy(); cv2.fillPoly(ov,[arr],color)
        vis=cv2.addWeighted(ov,0.35,vis,0.65,0)
        cv2.polylines(vis,[arr],True,color,1)
    return vis

def crop_grid(w,h,nr=N_ROWS,nc=N_COLS,p=PATCH_SIZE):
    def st(total,n,p):
        if n==1: return[0]
        if total<=p: return[0]*n
        s=(total-p)/(n-1); return[int(round(i*s))for i in range(n)]
    return[(x0,y0,x0+p,y0+p,r,c)
           for r,y0 in enumerate(st(h,nr,p))for c,x0 in enumerate(st(w,nc,p))]


def process_one_image(img_path,label_path,out_img_dir,out_vis_dir,stem):
    img_bgr=cv2.imread(str(img_path))
    if img_bgr is None: print(f'[WARN] {img_path}'); return[],[],0,0,0
    ah,aw=img_bgr.shape[:2]
    hw=aw//2; hh=ah//2
    img_half=cv2.resize(img_bgr,(hw,hh),interpolation=cv2.INTER_AREA)
    shapes=[]
    if label_path and label_path.exists():
        with label_path.open('r',encoding='utf-8') as f: ld=json.load(f)
        jw=ld.get('imageWidth',aw); jh=ld.get('imageHeight',ah)
        cx=hw/jw; cy=hh/jh
        for shp in ld.get('shapes',[]):
            if shp.get('shape_type','polygon')!='polygon': continue
            pts=shp.get('points',[])
            if len(pts)<3: continue
            shapes.append({'label':shp.get('label','lettuce'),
                           'points':[(float(x)*cx,float(y)*cy)for x,y in pts]})
    n_orig=len(shapes)
    orig_areas=[_poly_area(s['points'])for s in shapes]
    grid=crop_grid(hw,hh)
    pclips=[]
    for(x0,y0,x1,y1,r,c)in grid:
        rw=min(hw,x1)-x0; rh=min(hh,y1)-y0
        clips={}
        for si,shp in enumerate(shapes):
            local=[(px-x0,py-y0)for px,py in shp['points']]
            cl=_clip_sh(local,0,0,float(rw),float(rh))
            if len(cl)<3: clips[si]=None; continue
            ca=_poly_area(cl)
            if ca<MIN_AREA_PX: clips[si]=None; continue
            clips[si]=(cl,ca,ca/(orig_areas[si]+1e-6),_poly_bbox(cl))
        pclips.append(clips)
    primary=[-1]*n_orig
    for si in range(n_orig):
        best=-1; bar=-1
        for pid in range(len(grid)):
            info=pclips[pid].get(si)
            if info and info[2]>bar: bar=info[2]; best=pid
        primary[si]=best
    keep=[set()for _ in range(len(grid))]
    n_filt=0
    for si in range(n_orig):
        pp=primary[si]; kept=False
        for pid in range(len(grid)):
            info=pclips[pid].get(si)
            if info is None: continue
            _,ca,ar,bb=info
            if bb[2]<MIN_BBOX_SIZE or bb[3]<MIN_BBOX_SIZE: continue
            if pid==pp and ar>=MIN_AREA_RATIO: keep[pid].add(si); kept=True
            elif pid!=pp and ar>=OVERLAP_RATIO: keep[pid].add(si); kept=True
        if not kept: n_filt+=1
    pimgs=[]; pannos=[]
    n_assigned=sum(len(k)for k in keep)
    for pid,(x0,y0,x1,y1,r,c)in enumerate(grid):
        rw=min(hw,x1)-x0; rh=min(hh,y1)-y0
        roi=img_half[y0:y0+rh,x0:x0+rw]
        pb=PATCH_SIZE-rh; pr=PATCH_SIZE-rw
        crop=(cv2.copyMakeBorder(roi,0,pb,0,pr,cv2.BORDER_CONSTANT,value=(0,0,0))
              if(pb>0 or pr>0)else roi.copy())
        pname=f'{stem}_r{r}_c{c}.jpg'
        cv2.imwrite(str(out_img_dir/pname),crop)
        pimgs.append({'file_name':pname,'width':PATCH_SIZE,'height':PATCH_SIZE})
        ppolys=[]
        for si in sorted(keep[pid]):
            info=pclips[pid].get(si)
            if info is None: continue
            cl,ca,ar,bb=info
            pannos.append({'patch_file_name':pname,'category_name':shapes[si]['label'],
                           'segmentation':[[coord for pt in cl for coord in pt]],
                           'area':float(ca),'bbox':bb,'iscrowd':0})
            ppolys.append(cl)
        cv2.imwrite(str(out_vis_dir/pname),_draw(crop,ppolys))
    return pimgs,pannos,n_orig,n_assigned,n_filt


def stitch_compare(stem,img_path,label_path,dvis,dst,hw,hh):
    grid=crop_grid(hw,hh)
    canvas=np.zeros((hh,hw,3),dtype=np.uint8)
    for(x0,y0,x1,y1,r,c)in grid:
        p=cv2.imread(str(dvis/f'{stem}_r{r}_c{c}.jpg'))
        if p is None: continue
        rh=min(hh,y1)-y0; rw=min(hw,x1)-x0
        reg=canvas[y0:y0+rh,x0:x0+rw]; src=p[:rh,:rw]
        blk=(reg[:,:,0]==0)&(reg[:,:,1]==0)&(reg[:,:,2]==0)
        reg[blk]=src[blk]; reg[~blk]=cv2.addWeighted(reg,0.5,src,0.5,0)[~blk]
        canvas[y0:y0+rh,x0:x0+rw]=reg
    ib=cv2.imread(str(img_path)); ah,aw=ib.shape[:2]
    orig=cv2.resize(ib,(hw,hh),interpolation=cv2.INTER_AREA)
    polys=[]
    if label_path and label_path.exists():
        with open(label_path)as f: ld=json.load(f)
        jw=ld.get('imageWidth',aw); jh=ld.get('imageHeight',ah)
        cx=hw/jw; cy=hh/jh
        for shp in ld.get('shapes',[]):
            if shp.get('shape_type','polygon')!='polygon': continue
            pts=shp.get('points',[])
            if len(pts)<3: continue
            polys.append([(float(x)*cx,float(y)*cy)for x,y in pts])
    ov=_draw(orig,polys)
    def lbl(img,txt):
        o=img.copy(); cv2.rectangle(o,(0,0),(img.shape[1],32),(20,20,20),-1)
        cv2.putText(o,txt,(5,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,80),2); return o
    cmp=np.concatenate([lbl(ov,f'Original({len(polys)})'),lbl(canvas,'Stitched')],axis=1)
    cv2.imwrite(str(dst/f'{stem}_original.jpg'),ov)
    cv2.imwrite(str(dst/f'{stem}_stitched.jpg'),canvas)
    cv2.imwrite(str(dst/f'{stem}_compare.jpg'),cmp)


def build_coco(aimgs,aannos,cats):
    c2id={n:i+1 for i,n in enumerate(cats)}; f2id={}; ci=[]
    for i,img in enumerate(aimgs,1):
        f2id[img['file_name']]=i
        ci.append({'id':i,'file_name':img['file_name'],'width':img['width'],'height':img['height']})
    ca=[]; aid=1
    for a in aannos:
        iid=f2id.get(a['patch_file_name'])
        if iid is None: continue
        cat=a['category_name']
        if cat not in c2id: c2id[cat]=max(c2id.values(),default=0)+1
        ca.append({'id':aid,'image_id':iid,'category_id':c2id[cat],
                   'segmentation':a['segmentation'],'area':a['area'],
                   'bbox':a['bbox'],'iscrowd':a['iscrowd']}); aid+=1
    return{'images':ci,'annotations':ca,
           'categories':[{'id':v,'name':k,'supercategory':'object'}
                         for k,v in sorted(c2id.items(),key=lambda x:x[1])]}


def split_coco(coco,tr,vr,seed):
    imgs=coco['images']; annos=coco['annotations']; cats=coco['categories']
    rng=random.Random(seed); ids=[i['id']for i in imgs]; rng.shuffle(ids)
    nv=max(1,int(round(len(ids)*vr))); nt=max(1,int(round(len(ids)*(1-tr-vr))))
    na=len(ids)-nv-nt
    sp={'train':set(ids[:na]),'val':set(ids[na:na+nv]),'test':set(ids[na+nv:])}
    def sub(keep):
        o2n={}; si=[]
        for i,img in enumerate(imgs,1):
            if img['id']in keep: o2n[img['id']]=i; si.append({**img,'id':i})
        sa=[]
        for j,a in enumerate(annos,1):
            if a['image_id']in keep: sa.append({**a,'id':j,'image_id':o2n[a['image_id']]})
        return{'images':si,'annotations':sa,'categories':cats}
    return sub(sp['train']),sub(sp['val']),sub(sp['test'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src',type=Path,
        default=Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_img_label'))
    ap.add_argument('--out',type=Path,
        default=Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_output'))
    ap.add_argument('--train-ratio',type=float,default=0.70)
    ap.add_argument('--val-ratio',type=float,default=0.15)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    src,out=args.src,args.out
    if not src.is_dir(): raise FileNotFoundError(src)
    dimg=out/'images'; dcoco=out/'coco'; dvis=out/'vis_annotations'; dst=out/'stitch_compare'
    if args.overwrite and out.exists(): shutil.rmtree(out)
    for d in(dimg,dcoco,dvis,dst): d.mkdir(parents=True,exist_ok=True)
    jpgs=sorted(src.glob('*.jpg'))
    print(f'Found {len(jpgs)} images')
    all_imgs,all_annos,all_cats=[],[],set()
    log=['Step1 Consistency+Fragment Log','='*60]
    to=ta=tf=0
    for jp in jpgs:
        lp=jp.with_suffix('.json')
        if not lp.exists(): lp=None
        stem=jp.stem
        print(f'  {stem}...',end=' ',flush=True)
        pi,pa,no,na,nf=process_one_image(jp,lp,dimg,dvis,stem)
        all_imgs.extend(pi); all_annos.extend(pa)
        for a in pa: all_cats.add(a['category_name'])
        ib=cv2.imread(str(jp)); ah,aw=ib.shape[:2]
        stitch_compare(stem,jp,lp,dvis,dst,aw//2,ah//2)
        print(f'orig={no} assigned={na} filtered={nf}')
        log.append(f'{stem}: orig={no} assigned={na} filtered={nf} [{"OK"if nf==0 else f"FILTERED={nf}"}]')
        to+=no; ta+=na; tf+=nf
    cats=sorted(all_cats)or['lettuce']
    log+=['='*60,f'TOTAL orig={to} assigned={ta} filtered={tf}','='*60]
    (out/'consistency_check.log').write_text('\n'.join(log)+'\n')
    ca=build_coco(all_imgs,all_annos,cats)
    with(dcoco/'instances_all.json').open('w')as f: json.dump(ca,f)
    print('Saved instances_all.json')
    tr,vl,te=split_coco(ca,args.train_ratio,args.val_ratio,args.seed)
    for nm,sub in[('train',tr),('val',vl),('test',te)]:
        with(dcoco/f'instances_{nm}.json').open('w')as f: json.dump(sub,f)
        print(f'Saved instances_{nm}.json images={len(sub["images"])} annos={len(sub["annotations"])}')
    print(f'\nStep 1 complete.\n  images:{dimg}\n  coco:{dcoco}\n  vis:{dvis}\n  stitch:{dst}')


if __name__=='__main__':
    main()
