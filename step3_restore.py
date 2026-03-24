#!/usr/bin/env python3
"""
Restore org_img_label JSONs from patch-level COCO (mydata/coco),
then rebuild step3_output COCO + overlays.
"""
import json, cv2, numpy as np, random, re
from pathlib import Path
from collections import defaultdict

ORG_LABEL = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_img_label')
ORG_DATA  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_data')
CHECK_OUT = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_check_out')
OUT       = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step3_output')
(OUT/'overlays').mkdir(parents=True, exist_ok=True)

PATCH=512; NR=4; NC=3

def starts(total,n,p):
    if n==1: return[0]
    if total<=p: return[0]*n
    s=(total-p)/(n-1); return[int(round(i*s))for i in range(n)]

def icolor(i):
    hue=int((i*137.508)%180)
    return cv2.cvtColor(np.uint8([[[hue,210,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

def poly_area(pts):
    if len(pts)<3: return 0.0
    s=0.0
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]+pts[:1]): s+=x1*y2-x2*y1
    return abs(s)*0.5

def draw_overlay(img,shapes):
    vis=img.copy()
    for i,shp in enumerate(shapes):
        pts=shp.get('points',[])
        if len(pts)<3: continue
        arr=np.array(pts,dtype=np.int32)
        color=icolor(i)
        ov=vis.copy(); cv2.fillPoly(ov,[arr],color)
        vis=cv2.addWeighted(ov,0.3,vis,0.7,0)
        cv2.polylines(vis,[arr],True,color,1)
    return vis

def remap(shapes,jw,jh,aw,ah):
    sx=aw/jw; sy=ah/jh
    for s in shapes:
        s['points']=[[float(x)*sx,float(y)*sy] for x,y in s['points']]
    return shapes

# Load patch COCO
print('Loading patch COCO...')
all_patch={'images':[],'annotations':[],'categories':[]}
for cp in [Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/coco/instances_train.json'),
           Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/coco/instances_val.json')]:
    c=json.load(open(cp))
    off_i=max((x['id'] for x in all_patch['images']),default=0)
    off_a=max((x['id'] for x in all_patch['annotations']),default=0)
    imap={}
    for img in c['images']:
        nid=img['id']+off_i; imap[img['id']]=nid
        all_patch['images'].append({**img,'id':nid})
    for ann in c['annotations']:
        all_patch['annotations'].append({**ann,'id':ann['id']+off_a,'image_id':imap[ann['image_id']]})
    all_patch['categories']=c['categories']
print(f'Patch COCO: {len(all_patch["images"])} imgs, {len(all_patch["annotations"])} annos')

# Group by stem
by_stem=defaultdict(list)
patch_img_map={img['file_name']:img for img in all_patch['images']}
patch_ann_map=defaultdict(list)
for ann in all_patch['annotations']:
    iid=ann['image_id']
    for img in all_patch['images']:
        if img['id']==iid:
            patch_ann_map[img['file_name']].append(ann); break
for fname in patch_img_map:
    m=re.match(r'(.+)_r(\d+)_c(\d+)\.jpg$',fname)
    if m: by_stem[m.group(1)].append((fname,int(m.group(2)),int(m.group(3))))
print(f'Stems in patch COCO: {len(by_stem)}')

# Restore each org_img_label image
for jp in sorted(ORG_LABEL.glob('*.jpg')):
    stem=jp.stem
    img_bgr=cv2.imread(str(jp))
    if img_bgr is None: continue
    ah,aw=img_bgr.shape[:2]
    hw=aw//2; hh=ah//2

    if stem in by_stem:
        rs=starts(hh,NR,PATCH); cs_=starts(hw,NC,PATCH)
        grid={(r,c):(x0,y0) for r,y0 in enumerate(rs) for c,x0 in enumerate(cs_)}
        shapes=[]; seen=set()
        for fname,r,c in by_stem[stem]:
            x0,y0=grid.get((r,c),(0,0))
            for ann in patch_ann_map.get(fname,[]):
                seg=ann.get('segmentation',[])
                if not seg: continue
                flat=seg[0]
                pts=[[float(flat[i]+x0)*2,float(flat[i+1]+y0)*2]
                     for i in range(0,len(flat),2)]
                cx=round(sum(p[0] for p in pts)/len(pts))
                cy=round(sum(p[1] for p in pts)/len(pts))
                if (cx,cy) in seen: continue
                seen.add((cx,cy))
                shapes.append({'label':'lettuce','points':pts,
                               'group_id':None,'shape_type':'polygon','flags':{}})
        source=f'patch_coco'
    else:
        shapes=[]; source='empty'
        for src,path in [('check_out',CHECK_OUT/f'{stem}_corrected.json'),
                          ('org_data',ORG_DATA/f'{stem}.json')]:
            if path.exists():
                d2=json.load(open(path))
                jw=d2.get('imageWidth',aw); jh=d2.get('imageHeight',ah)
                shapes=d2.get('shapes',[])
                if jw!=aw or jh!=ah: shapes=remap(shapes,jw,jh,aw,ah)
                source=src; break

    d_out={'version':'5.0.1','flags':{},'shapes':shapes,
           'imagePath':jp.name,'imageData':None,'imageHeight':ah,'imageWidth':aw}
    with open(ORG_LABEL/(stem+'.json'),'w',encoding='utf-8') as f:
        json.dump(d_out,f,ensure_ascii=False,indent=2)
    ov=draw_overlay(img_bgr,shapes)
    cv2.putText(ov,f'{source} n={len(shapes)}',(10,50),
                cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
    cv2.imwrite(str(OUT/'overlays'/f'{stem}_overlay.jpg'),ov)
    print(f'  {stem}: {source} shapes={len(shapes)}')

print('Restoration done. Building COCO...')

# Build COCO from restored JSONs
all_imgs=[]; all_annos=[]; img_id=1; ann_id=1
for jp in sorted(ORG_LABEL.glob('*.jpg')):
    stem=jp.stem
    jf=ORG_LABEL/(stem+'.json')
    img=cv2.imread(str(jp))
    if img is None or not jf.exists(): continue
    ah,aw=img.shape[:2]
    d=json.load(open(jf))
    shapes=d.get('shapes',[])
    all_imgs.append({'id':img_id,'file_name':jp.name,'width':aw,'height':ah})
    for shp in shapes:
        pts=shp.get('points',[])
        if len(pts)<3: continue
        area=poly_area(pts)
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        bbox=[float(min(xs)),float(min(ys)),float(max(xs)-min(xs)),float(max(ys)-min(ys))]
        seg=[[coord for pt in pts for coord in pt]]
        all_annos.append({'id':ann_id,'image_id':img_id,'category_id':1,
                          'segmentation':seg,'area':float(area),'bbox':bbox,'iscrowd':0})
        ann_id+=1
    img_id+=1

cats=[{'id':1,'name':'lettuce','supercategory':'plant'}]
coco_all={'images':all_imgs,'annotations':all_annos,'categories':cats}
with open(OUT/'instances_all.json','w') as f: json.dump(coco_all,f)
print(f'COCO all: {len(all_imgs)} images, {len(all_annos)} annos')

# Train/val/test split
rng=random.Random(42)
ids=[i['id'] for i in all_imgs]; rng.shuffle(ids)
nv=max(1,int(round(len(ids)*0.15))); nt=max(1,int(round(len(ids)*0.15)))
na=len(ids)-nv-nt
sp={'train':set(ids[:na]),'val':set(ids[na:na+nv]),'test':set(ids[na+nv:])}
def sub(keep):
    o2n={}; si=[]
    for i,img in enumerate(all_imgs,1):
        if img['id'] in keep: o2n[img['id']]=i; si.append({**img,'id':i})
    sa=[]
    for j,a in enumerate(all_annos,1):
        if a['image_id'] in keep: sa.append({**a,'id':j,'image_id':o2n[a['image_id']]})
    return{'images':si,'annotations':sa,'categories':cats}
for nm,keep in sp.items():
    s=sub(keep)
    with open(OUT/f'instances_{nm}.json','w') as f: json.dump(s,f)
    print(f'COCO {nm}: {len(s["images"])} imgs, {len(s["annotations"])} annos')

log=f'step3_restore: {len(all_imgs)} images, {len(all_annos)} total annotations'
(OUT/'step3.log').write_text(log+'\n')
print('Done.',log)
