#!/usr/bin/env python3
"""
Step 5: Online data augmentation on step4_output patches.
Types: rotate, translate, flip_h, flip_v, affine, brightness/contrast.
Each image augmented N_AUG times; all aug types covered; annotations sync-transformed.
Padding areas have NO annotations.
"""
import json, cv2, numpy as np, random
from pathlib import Path
from copy import deepcopy

SRC_IMG = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step4_output/images')
SRC_COCO= Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step4_output/coco/instances_all.json')
OUT     = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step5_augmented')
(OUT/'images').mkdir(parents=True,exist_ok=True)
(OUT/'vis').mkdir(parents=True,exist_ok=True)
PATCH=512; N_AUG=6; SEED=42
rng=random.Random(SEED); np.random.seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
def poly_area(pts):
    if len(pts)<3: return 0.0
    s=0.0
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]+pts[:1]): s+=x1*y2-x2*y1
    return abs(s)*0.5

def clip_pts(pts,w,h):
    return[[float(np.clip(x,0,w-1)),float(np.clip(y,0,h-1))]for x,y in pts]

def valid_ann(pts,min_area=10,min_side=4):
    if not pts or len(pts)<3: return False
    xs=[p[0]for p in pts]; ys=[p[1]for p in pts]
    return(poly_area(pts)>=min_area and max(xs)-min(xs)>=min_side and max(ys)-min(ys)>=min_side)

def centroid_valid(pts,valid):
    if not pts: return False
    h,w=valid.shape[:2]
    cx=int(np.clip(np.mean([p[0]for p in pts]),0,w-1))
    cy=int(np.clip(np.mean([p[1]for p in pts]),0,h-1))
    return valid[cy,cx]>0

def apply_M(pts,M,valid):
    arr=np.array(pts,np.float32)
    t=(M@np.hstack([arr,np.ones((len(arr),1),np.float32)]).T).T
    np_=[[[float(p[0]),float(p[1])]] for p in t]
    res=[[float(p[0][0]),float(p[0][1])]for p in np_]
    return res if centroid_valid(res,valid) else None

def warp(img,M,dsize):
    return cv2.warpAffine(img,M,dsize,flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def warp_valid(h,w,M,dsize):
    v=np.ones((h,w),np.uint8)*255
    return cv2.warpAffine(v,M,dsize,flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def center_crop_pad(img,valid):
    h,w=img.shape[:2]
    y0=max(0,(h-PATCH)//2); x0=max(0,(w-PATCH)//2)
    img=img[y0:y0+PATCH,x0:x0+PATCH]
    valid=valid[y0:y0+PATCH,x0:x0+PATCH]
    ph=PATCH-img.shape[0]; pw=PATCH-img.shape[1]
    if ph>0 or pw>0:
        img=cv2.copyMakeBorder(img,0,ph,0,pw,cv2.BORDER_CONSTANT,value=0)
        valid=cv2.copyMakeBorder(valid,0,ph,0,pw,cv2.BORDER_CONSTANT,value=0)
    return img,valid

# ── Augmentation functions ────────────────────────────────────────────────────
def aug_rotate(img,polys):
    angle=rng.uniform(-30,30); h,w=img.shape[:2]; cx,cy=w/2,h/2
    M=cv2.getRotationMatrix2D((cx,cy),angle,1.0)
    cos,sin=abs(M[0,0]),abs(M[0,1])
    nw=int(h*sin+w*cos); nh=int(h*cos+w*sin)
    M[0,2]+=(nw-w)/2; M[1,2]+=(nh-h)/2
    img_o=warp(img,M,(nw,nh)); valid=warp_valid(h,w,M,(nw,nh))
    img_o,valid=center_crop_pad(img_o,valid)
    # shift M for crop offset
    y0=max(0,(nh-PATCH)//2); x0=max(0,(nw-PATCH)//2)
    Mc=M.copy(); Mc[0,2]-=x0; Mc[1,2]-=y0
    return img_o,[apply_M(p,Mc,valid) for p in polys]

def aug_translate(img,polys):
    dx=rng.randint(-60,60); dy=rng.randint(-60,60)
    h,w=img.shape[:2]; M=np.float32([[1,0,dx],[0,1,dy]])
    img_o=warp(img,M,(w,h)); valid=warp_valid(h,w,M,(w,h))
    return img_o,[apply_M(p,M,valid) for p in polys]

def aug_flip_h(img,polys):
    h,w=img.shape[:2]; img_o=cv2.flip(img,1)
    valid=np.ones((h,w),np.uint8)*255
    res=[]
    for p in polys:
        fp=[[w-1-x,y]for x,y in p]
        res.append(fp if centroid_valid(fp,valid) else None)
    return img_o,res

def aug_flip_v(img,polys):
    h,w=img.shape[:2]; img_o=cv2.flip(img,0)
    valid=np.ones((h,w),np.uint8)*255
    res=[]
    for p in polys:
        fp=[[x,h-1-y]for x,y in p]
        res.append(fp if centroid_valid(fp,valid) else None)
    return img_o,res

def aug_affine(img,polys):
    h,w=img.shape[:2]; cx,cy=w/2,h/2
    sx=rng.uniform(-0.12,0.12); sy=rng.uniform(-0.12,0.12); sc=rng.uniform(0.85,1.10)
    M=np.float32([[sc,sx,cx*(1-sc)-sx*cy],[sy,sc,cy*(1-sc)-sy*cx]])
    img_o=warp(img,M,(w,h)); valid=warp_valid(h,w,M,(w,h))
    return img_o,[apply_M(p,M,valid) for p in polys]

def aug_brightness(img,polys):
    a=rng.uniform(0.6,1.4); b=rng.randint(-40,40)
    img_o=np.clip(img.astype(np.float32)*a+b,0,255).astype(np.uint8)
    return img_o,polys  # no spatial change

AUGS=[('rotate',aug_rotate),('translate',aug_translate),
      ('flip_h',aug_flip_h),('flip_v',aug_flip_v),
      ('affine',aug_affine),('brightness',aug_brightness)]

def icolor(i):
    hue=int((i*137.508)%180)
    return cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

def draw_vis(img,polys):
    vis=img.copy()
    for i,pts in enumerate(polys):
        if not pts or len(pts)<3: continue
        c=icolor(i); arr=np.array(pts,dtype=np.int32)
        ov=vis.copy(); cv2.fillPoly(ov,[arr],c)
        vis=cv2.addWeighted(ov,0.35,vis,0.65,0)
        cv2.polylines(vis,[arr],True,c,1)
    return vis

def main():
    print('Loading COCO...',flush=True)
    coco=json.load(open(SRC_COCO))
    cats=coco['categories']
    img2ann={}
    for ann in coco['annotations']:
        img2ann.setdefault(ann['image_id'],[]).append(ann)
    out_imgs=[]; out_annos=[]; img_id=1; ann_id=1
    n_total=len(coco['images'])
    print(f'Augmenting {n_total} images x {N_AUG} each...',flush=True)
    for oidx,orig_img in enumerate(coco['images']):
        fname=orig_img['file_name']
        img=cv2.imread(str(SRC_IMG/fname))
        if img is None: print(f'  SKIP {fname}'); continue
        anns=img2ann.get(orig_img['id'],[])
        polys=[]; meta=[]
        for ann in anns:
            seg=ann.get('segmentation',[])
            if not seg: continue
            flat=seg[0]
            pts=[[flat[i],flat[i+1]]for i in range(0,len(flat),2)]
            polys.append(pts); meta.append(ann['category_id'])
        for aug_i in range(N_AUG):
            # guaranteed coverage: first N_AUG slots use each aug type once
            if aug_i<len(AUGS):
                chosen=[AUGS[aug_i]]
            else:
                chosen=rng.sample(AUGS,rng.randint(2,3))
            aug_img=img.copy(); aug_polys=deepcopy(polys)
            for _,fn in chosen:
                aug_img,aug_polys=fn(aug_img,aug_polys)
            # filter annotations
            valid_p=[]; valid_m=[]
            for pts,cat_id in zip(aug_polys,meta):
                if pts is None: continue
                pts_c=clip_pts(pts,PATCH,PATCH)
                if valid_ann(pts_c): valid_p.append(pts_c); valid_m.append(cat_id)
            aug_types='_'.join(n for n,_ in chosen)
            aug_fname=f'aug_{fname.replace(".jpg","")}_{aug_i}_{aug_types}.jpg'
            cv2.imwrite(str(OUT/'images'/aug_fname),aug_img)
            out_imgs.append({'id':img_id,'file_name':aug_fname,
                             'width':PATCH,'height':PATCH,
                             'aug_type':aug_types,'source':fname})
            for pts_c,cat_id in zip(valid_p,valid_m):
                seg=[[coord for pt in pts_c for coord in pt]]
                xs=[p[0]for p in pts_c]; ys=[p[1]for p in pts_c]
                bbox=[float(min(xs)),float(min(ys)),
                      float(max(xs)-min(xs)),float(max(ys)-min(ys))]
                out_annos.append({'id':ann_id,'image_id':img_id,'category_id':cat_id,
                                  'segmentation':seg,'area':float(poly_area(pts_c)),
                                  'bbox':bbox,'iscrowd':0})
                ann_id+=1
            # save vis for first 3 augs of first 10 source images
            if aug_i<3 and oidx<10:
                vis=draw_vis(aug_img,valid_p)
                cv2.imwrite(str(OUT/'vis'/aug_fname),vis)
            img_id+=1
        if (oidx+1)%30==0: print(f'  {oidx+1}/{n_total} done',flush=True)
    # Save COCO JSON
    coco_out={'images':out_imgs,'annotations':out_annos,'categories':cats}
    with open(OUT/'instances_aug.json','w')as f: json.dump(coco_out,f)
    # Save split
    import random as rnd
    rnd2=rnd.Random(SEED); ids=[i['id']for i in out_imgs]; rnd2.shuffle(ids)
    nv=max(1,int(round(len(ids)*0.15))); nt=max(1,int(round(len(ids)*0.15)))
    na=len(ids)-nv-nt
    id2img={i['id']:i for i in out_imgs}
    img2ano={}
    for a in out_annos: img2ano.setdefault(a['image_id'],[]).append(a)
    def sub(keep):
        o2n={}; si=[]
        for j,iid in enumerate(sorted(keep),1):
            o2n[iid]=j; si.append({**id2img[iid],'id':j})
        sa=[]
        for k,a in enumerate(sum([img2ano.get(iid,[])for iid in keep],[]),1):
            sa.append({**a,'id':k,'image_id':o2n[a['image_id']]})
        return{'images':si,'annotations':sa,'categories':cats}
    sp={'train':set(ids[:na]),'val':set(ids[na:na+nv]),'test':set(ids[na+nv:])}
    for nm,keep in sp.items():
        s=sub(keep)
        with open(OUT/f'instances_{nm}.json','w')as f: json.dump(s,f)
        print(f'Saved instances_{nm}: {len(s["images"])} imgs {len(s["annotations"])} annos')
    print(f'\nStep5 done: {len(out_imgs)} aug images, {len(out_annos)} annotations')
    print(f'Output: {OUT}')

if __name__=='__main__':
    main()
