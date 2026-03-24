#!/usr/bin/env python3
"""
Step 4: step2_jiaozheng corrected images + JSON -> half-size -> 12 patches -> COCO
"""
import json, cv2, numpy as np, random
from pathlib import Path

SRC = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_jiaozheng')
OUT = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step4_output')
PATCH=512; NR=4; NC=3
MIN_AREA_RATIO=0.10; MIN_BBOX=6; MIN_AREA_PX=10; OVERLAP_RATIO=0.05

for d in (OUT/'images',OUT/'coco',OUT/'vis_annotations'):
    d.mkdir(parents=True,exist_ok=True)

def starts(total,n,p):
    if n==1: return[0]
    if total<=p: return[0]*n
    s=(total-p)/(n-1); return[int(round(i*s))for i in range(n)]

def poly_area(pts):
    if len(pts)<3: return 0.0
    s=0.0
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]+pts[:1]): s+=x1*y2-x2*y1
    return abs(s)*0.5

def poly_bbox(pts):
    xs=[p[0]for p in pts]; ys=[p[1]for p in pts]
    return[float(min(xs)),float(min(ys)),float(max(xs)-min(xs)),float(max(ys)-min(ys))]

def clip_sh(pts,x0,y0,x1,y1):
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

def icolor(idx):
    hue=int((idx*137.508)%180)
    return cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

def draw_inst(img,polys):
    vis=img.copy()
    for i,pts in enumerate(polys):
        color=icolor(i); arr=np.array(pts,dtype=np.int32)
        ov=vis.copy(); cv2.fillPoly(ov,[arr],color)
        vis=cv2.addWeighted(ov,0.35,vis,0.65,0)
        cv2.polylines(vis,[arr],True,color,1)
    return vis

def crop_grid(w,h):
    return[(x0,y0,x0+PATCH,y0+PATCH,r,c)
           for r,y0 in enumerate(starts(h,NR,PATCH))
           for c,x0 in enumerate(starts(w,NC,PATCH))]

def process(img_path,json_path,stem,all_imgs,all_annos,all_cats,log):
    img=cv2.imread(str(img_path))
    if img is None: log.append(f'{stem}: UNREADABLE'); return
    ah,aw=img.shape[:2]
    hw=aw//2; hh=ah//2
    img_half=cv2.resize(img,(hw,hh),interpolation=cv2.INTER_AREA)
    with open(json_path)as f: d=json.load(f)
    jw=d.get('imageWidth',aw); jh=d.get('imageHeight',ah)
    cx_=hw/jw; cy_=hh/jh
    shapes=[]
    for shp in d.get('shapes',[]):
        if shp.get('shape_type','polygon')!='polygon': continue
        pts=shp.get('points',[])
        if len(pts)<3: continue
        shapes.append({'label':shp.get('label','lettuce'),
                       'points':[(float(x)*cx_,float(y)*cy_)for x,y in pts]})
    n_orig=len(shapes)
    orig_areas=[poly_area(s['points'])for s in shapes]
    grid=crop_grid(hw,hh)
    pclips=[]
    for(x0,y0,x1,y1,r,c)in grid:
        rw=min(hw,x1)-x0; rh=min(hh,y1)-y0
        clips={}
        for si,shp in enumerate(shapes):
            local=[(px-x0,py-y0)for px,py in shp['points']]
            cl=clip_sh(local,0,0,float(rw),float(rh))
            if len(cl)<3: clips[si]=None; continue
            ca=poly_area(cl)
            if ca<MIN_AREA_PX: clips[si]=None; continue
            clips[si]=(cl,ca,ca/(orig_areas[si]+1e-6),poly_bbox(cl))
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
            if bb[2]<MIN_BBOX or bb[3]<MIN_BBOX: continue
            if pid==pp and ar>=MIN_AREA_RATIO: keep[pid].add(si); kept=True
            elif pid!=pp and ar>=OVERLAP_RATIO: keep[pid].add(si); kept=True
        if not kept: n_filt+=1
    for pid,(x0,y0,x1,y1,r,c)in enumerate(grid):
        rw=min(hw,x1)-x0; rh=min(hh,y1)-y0
        roi=img_half[y0:y0+rh,x0:x0+rw]
        pb=PATCH-rh; pr=PATCH-rw
        crop=(cv2.copyMakeBorder(roi,0,pb,0,pr,cv2.BORDER_CONSTANT,value=(0,0,0))
              if(pb>0 or pr>0)else roi.copy())
        pname=f'{stem}_r{r}_c{c}.jpg'
        cv2.imwrite(str(OUT/'images'/pname),crop)
        all_imgs.append({'file_name':pname,'width':PATCH,'height':PATCH})
        ppolys=[]
        for si in sorted(keep[pid]):
            info=pclips[pid].get(si)
            if info is None: continue
            cl,ca,ar,bb=info
            cat=shapes[si]['label']; all_cats.add(cat)
            all_annos.append({'patch_file_name':pname,'category_name':cat,
                              'segmentation':[[coord for pt in cl for coord in pt]],
                              'area':float(ca),'bbox':bb,'iscrowd':0})
            ppolys.append(cl)
        cv2.imwrite(str(OUT/'vis_annotations'/pname),draw_inst(crop,ppolys))
    log.append(f'{stem}: orig={n_orig} filtered={n_filt}')
    print(f'  {stem}: orig={n_orig} filtered={n_filt}',flush=True)

def build_coco(aimgs,aannos,cats):
    c2id={n:i+1 for i,n in enumerate(sorted(cats))}; f2id={}; ci=[]
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

def split_coco(coco,tr=0.70,vr=0.15,seed=42):
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
    jsons=sorted(SRC.glob('*_corrected.json'))
    print(f'Found {len(jsons)} corrected pairs')
    all_imgs=[]; all_annos=[]; all_cats=set()
    log=['Step4 Log','='*60]
    for jp in jsons:
        stem=jp.stem
        img_path=jp.parent/(stem+'.jpg')
        if not img_path.exists(): log.append(f'{stem}: MISSING jpg'); continue
        process(img_path,jp,stem,all_imgs,all_annos,all_cats,log)
    cats=sorted(all_cats)or['lettuce']
    ca=build_coco(all_imgs,all_annos,cats)
    with open(OUT/'coco'/'instances_all.json','w')as f: json.dump(ca,f)
    tr,vl,te=split_coco(ca)
    for nm,sub in[('train',tr),('val',vl),('test',te)]:
        with open(OUT/'coco'/f'instances_{nm}.json','w')as f: json.dump(sub,f)
        print(f'Saved instances_{nm}.json images={len(sub["images"])} annos={len(sub["annotations"])}')
    log+=['='*60,f'Total: {len(all_imgs)} patches, {len(all_annos)} annos']
    (OUT/'step4.log').write_text('\n'.join(log)+'\n')
    print(f'\nStep4 done: {len(all_imgs)} patches, {len(all_annos)} annotations')
    print(f'Output: {OUT}')

if __name__=='__main__':
    main()
