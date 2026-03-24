#!/usr/bin/env python3
"""
Step3 (corrected): Convert org_img_label + org_data annotations -> X-AnyLabeling compatible JSON
+ COCO format + overlay images.

Priority:
  1. org_img_label JSON (already in actual image coords) - use directly
  2. org_data JSON (may need coord remapping if size mismatch)
  3. Empty stub if no annotation exists
"""
import json, cv2, numpy as np, random, shutil
from pathlib import Path

ORG_LABEL = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_img_label')
ORG_DATA  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/org_data')
OUT       = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step3_output')
(OUT/'overlays').mkdir(parents=True, exist_ok=True)
(OUT/'coco').mkdir(parents=True, exist_ok=True)

TRAIN_R = 0.70; VAL_R = 0.15; SEED = 42


def icolor(i):
    hue=int((i*137.508)%180)
    return cv2.cvtColor(np.uint8([[[hue,210,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()


def remap_shapes(shapes, src_w, src_h, dst_w, dst_h):
    """Proportionally remap annotation coords from src to dst space."""
    sx=dst_w/src_w; sy=dst_h/src_h
    out=[]
    for shp in shapes:
        s=dict(shp)
        s['points']=[[float(x)*sx,float(y)*sy] for x,y in shp['points']]
        out.append(s)
    return out


def load_annotation(img_path):
    """Load best available annotation for this image.
    Returns (shapes, json_dict) with coords in actual image space."""
    stem=img_path.stem
    img=cv2.imread(str(img_path))
    if img is None: return [],None
    ah,aw=img.shape[:2]

    # Priority 1: org_img_label JSON
    lp=ORG_LABEL/(stem+'.json')
    if lp.exists():
        with open(lp) as f: d=json.load(f)
        jw=d.get('imageWidth',aw); jh=d.get('imageHeight',ah)
        shapes=d.get('shapes',[])
        if jw!=aw or jh!=ah:
            shapes=remap_shapes(shapes,jw,jh,aw,ah)
        d['imageWidth']=aw; d['imageHeight']=ah; d['shapes']=shapes
        return shapes, d, 'org_img_label'

    # Priority 2: org_data JSON
    dp=ORG_DATA/(stem+'.json')
    if dp.exists():
        with open(dp) as f: d=json.load(f)
        jw=d.get('imageWidth',aw); jh=d.get('imageHeight',ah)
        shapes=d.get('shapes',[])
        if jw!=aw or jh!=ah:
            shapes=remap_shapes(shapes,jw,jh,aw,ah)
        d['imageWidth']=aw; d['imageHeight']=ah; d['shapes']=shapes
        return shapes, d, 'org_data'

    # Empty stub
    d={'version':'5.0.1','flags':{},'shapes':[],'imagePath':img_path.name,
       'imageData':None,'imageHeight':ah,'imageWidth':aw}
    return [], d, 'empty'


def draw_overlay(img, shapes):
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


def poly_area(pts):
    if len(pts)<3: return 0.0
    s=0.0
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]+pts[:1]): s+=x1*y2-x2*y1
    return abs(s)*0.5


def main():
    # Collect all images from org_img_label
    jpgs=sorted(ORG_LABEL.glob('*.jpg'))
    # Also add any from org_data not in org_img_label
    stems_label={j.stem for j in jpgs}
    for jp in sorted(ORG_DATA.glob('*.jpg')):
        if jp.stem not in stems_label and 'copy' not in jp.stem:
            jpgs.append(jp)
    jpgs=sorted(jpgs,key=lambda x:x.stem)
    print(f'Total images: {len(jpgs)}')

    log=['Step3 Corrected Auto-Label + COCO Log','='*60]
    all_coco_imgs=[]; all_coco_annos=[]; all_cats={'lettuce':1}
    img_id=1; ann_id=1

    for jp in jpgs:
        img=cv2.imread(str(jp))
        if img is None: print(f'  SKIP {jp.name}'); continue
        ah,aw=img.shape[:2]
        stem=jp.stem
        shapes,d,source=load_annotation(jp)

        log.append(f'{stem}: source={source} size={aw}x{ah} shapes={len(shapes)}')
        print(f'  {stem}: {source} shapes={len(shapes)}')

        # Save corrected JSON to org_img_label (X-AnyLabeling compatible)
        out_json=ORG_LABEL/(stem+'.json')
        if d is not None:
            d['imagePath']=jp.name
            d['imageWidth']=aw; d['imageHeight']=ah
            with open(out_json,'w',encoding='utf-8') as f:
                json.dump(d,f,ensure_ascii=False,indent=2)

        # Save overlay
        ov=draw_overlay(img,shapes)
        cv2.putText(ov,f'{source} n={len(shapes)}',(20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),3)
        cv2.imwrite(str(OUT/'overlays'/f'{stem}_overlay.jpg'),ov)

        # COCO
        all_coco_imgs.append({'id':img_id,'file_name':jp.name,
                              'width':aw,'height':ah})
        for shp in shapes:
            pts=shp.get('points',[])
            if len(pts)<3: continue
            area=poly_area(pts)
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            bbox=[float(min(xs)),float(min(ys)),
                  float(max(xs)-min(xs)),float(max(ys)-min(ys))]
            seg=[[coord for pt in pts for coord in pt]]
            cat=shp.get('label','lettuce')
            if cat not in all_cats: all_cats[cat]=len(all_cats)+1
            all_coco_annos.append({'id':ann_id,'image_id':img_id,
                'category_id':all_cats[cat],'segmentation':seg,
                'area':float(area),'bbox':bbox,'iscrowd':0})
            ann_id+=1
        img_id+=1

    cats=[{'id':v,'name':k,'supercategory':'plant'}
          for k,v in sorted(all_cats.items(),key=lambda x:x[1])]
    coco_all={'images':all_coco_imgs,'annotations':all_coco_annos,'categories':cats}

    # Split
    rng=random.Random(SEED)
    ids=[i['id'] for i in all_coco_imgs]; rng.shuffle(ids)
    nv=max(1,int(round(len(ids)*VAL_R)))
    nt=max(1,int(round(len(ids)*(1-TRAIN_R-VAL_R))))
    na=len(ids)-nv-nt
    sp={'train':set(ids[:na]),'val':set(ids[na:na+nv]),'test':set(ids[na+nv:])}

    def sub(keep):
        o2n={}; si=[]
        for i,img in enumerate(all_coco_imgs,1):
            if img['id'] in keep: o2n[img['id']]=i; si.append({**img,'id':i})
        sa=[]
        for j,a in enumerate(all_coco_annos,1):
            if a['image_id'] in keep:
                sa.append({**a,'id':j,'image_id':o2n[a['image_id']]})
        return{'images':si,'annotations':sa,'categories':cats}

    with open(OUT/'instances_all.json','w') as f: json.dump(coco_all,f)
    for nm,keep in sp.items():
        with open(OUT/f'instances_{nm}.json','w') as f: json.dump(sub(keep),f)
        s=sub(keep)
        print(f'  COCO {nm}: {len(s["images"])} imgs {len(s["annotations"])} annos')

    log+=['='*60,
          f'COCO total: {len(all_coco_imgs)} images {len(all_coco_annos)} annotations',
          f'Categories: {list(all_cats.keys())}']
    (OUT/'step3.log').write_text('\n'.join(log)+'\n')
    print(f'Done. {len(all_coco_imgs)} images, {len(all_coco_annos)} annotations')


if __name__=='__main__':
    main()
