#!/usr/bin/env python3
"""
Step 2: Online data augmentation on step1_output patches.

Augmentation types applied as random subset per image:
  1. Random rotation with zero-padding
  2. Random translation
  3. Random horizontal/vertical flip
  4. Random affine transform
  5. Brightness/contrast perturbation
  6. Local occlusion (random black rectangle)

All polygon annotations transform in sync with the image.

Outputs: mydata/step2_output/
  images/   - augmented patch .jpg files
  coco/     - instances_all/train/val/test.json
  samples/  - visualised samples for manual inspection
"""
from __future__ import annotations
import argparse, json, math, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

PATCH   = 512
SAMPLE_N = 30

# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

def _transform_pts(pts: List[Tuple[float,float]], M: np.ndarray) -> List[Tuple[float,float]]:
    """Apply 2x3 affine matrix M to a list of (x,y) points."""
    if not pts:
        return pts
    arr = np.array(pts, dtype=np.float32)          # (N,2)
    ones = np.ones((len(arr), 1), dtype=np.float32)
    aug  = np.hstack([arr, ones])                   # (N,3)
    out  = (M @ aug.T).T                            # (N,2)
    return [(float(x), float(y)) for x, y in out]


def _clip_polygon(pts, x0, y0, x1, y1):
    """Sutherland-Hodgman clip to [x0,x1]x[y0,y1]."""
    def inside(p, e):
        if e == 'left':   return p[0] >= x0
        if e == 'right':  return p[0] <= x1
        if e == 'top':    return p[1] >= y0
        return p[1] <= y1
    def intersect(p1, p2, e):
        x1_, y1_ = p1; x2_, y2_ = p2
        if e in ('left', 'right'):
            bx = x0 if e == 'left' else x1
            t  = (bx - x1_) / (x2_ - x1_ + 1e-12)
            return (bx, y1_ + t*(y2_ - y1_))
        by = y0 if e == 'top' else y1
        t  = (by - y1_) / (y2_ - y1_ + 1e-12)
        return (x1_ + t*(x2_ - x1_), by)
    output = list(pts)
    for e in ('left', 'right', 'top', 'bottom'):
        if not output: break
        inp = output; output = []
        for i in range(len(inp)):
            curr = inp[i]; prev = inp[i-1]
            if inside(curr, e):
                if not inside(prev, e): output.append(intersect(prev, curr, e))
                output.append(curr)
            elif inside(prev, e):
                output.append(intersect(prev, curr, e))
    return output


def _polygon_area(pts):
    if len(pts) < 3: return 0.0
    s = 0.0
    for (x1,y1),(x2,y2) in zip(pts, pts[1:]+pts[:1]):
        s += x1*y2 - x2*y1
    return abs(s)*0.5


def _polygon_bbox(pts):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return [float(min(xs)), float(min(ys)),
            float(max(xs)-min(xs)), float(max(ys)-min(ys))]


# ---------------------------------------------------------------------------
# Individual augmentation functions
# Each returns (aug_image, M) where M is the 2x3 affine applied to coords
# ---------------------------------------------------------------------------

def aug_rotate(img: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(-30, 30)
    cx, cy = PATCH/2, PATCH/2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0).astype(np.float64)
    out = cv2.warpAffine(img, M, (PATCH, PATCH),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, M


def aug_translate(img: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    tx = rng.uniform(-64, 64)
    ty = rng.uniform(-64, 64)
    M  = np.array([[1,0,tx],[0,1,ty]], dtype=np.float64)
    out = cv2.warpAffine(img, M, (PATCH, PATCH),
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, M


def aug_flip(img: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    mode = rng.choice([0, 1, -1])   # 0=vert, 1=horiz, -1=both
    out  = cv2.flip(img, mode)
    if mode == 1:    # horizontal: x -> W-1-x
        M = np.array([[-1,0,PATCH-1],[0,1,0]], dtype=np.float64)
    elif mode == 0:  # vertical:   y -> H-1-y
        M = np.array([[1,0,0],[0,-1,PATCH-1]], dtype=np.float64)
    else:            # both
        M = np.array([[-1,0,PATCH-1],[0,-1,PATCH-1]], dtype=np.float64)
    return out, M


def aug_affine(img: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Slight shear + scale affine."""
    sx  = rng.uniform(0.85, 1.15)
    sy  = rng.uniform(0.85, 1.15)
    shx = rng.uniform(-0.1, 0.1)
    shy = rng.uniform(-0.1, 0.1)
    cx, cy = PATCH/2, PATCH/2
    # centre, apply, un-centre
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
    A  = np.array([[sx, shx, 0],[shy, sy, 0],[0,0,1]])
    T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
    full = (T2 @ A @ T1).astype(np.float64)
    M    = full[:2, :]   # 2x3
    out  = cv2.warpAffine(img, M, (PATCH, PATCH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, M


def aug_brightness_contrast(img: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    alpha = rng.uniform(0.6, 1.4)   # contrast
    beta  = rng.uniform(-40, 40)    # brightness
    out   = np.clip(img.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
    M     = np.eye(2, 3, dtype=np.float64)   # identity (pixels unchanged)
    return out, M


def aug_occlusion(img: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Place 1-3 random black rectangles on the image."""
    out = img.copy()
    n   = rng.randint(1, 3)
    for _ in range(n):
        w = rng.randint(20, 100)
        h = rng.randint(20, 100)
        x = rng.randint(0, PATCH - w)
        y = rng.randint(0, PATCH - h)
        out[y:y+h, x:x+w] = 0
    M = np.eye(2, 3, dtype=np.float64)
    return out, M


AUG_FUNCS = [
    aug_rotate,
    aug_translate,
    aug_flip,
    aug_affine,
    aug_brightness_contrast,
    # aug_occlusion excluded per step 2 spec
]


# ---------------------------------------------------------------------------
# Compose multiple augmentations, transform annotations
# ---------------------------------------------------------------------------

def apply_augmentations(img: np.ndarray,
                        segs: List[List[float]],
                        rng: random.Random
                        ) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Randomly select >=1 augmentation(s), apply sequentially.
    segs: list of flat [x1,y1,x2,y2,...] segmentation polygons.
    Returns augmented image and updated segmentations.
    """
    # Ensure every aug type is eventually represented; pick random subset >=1
    n_pick = rng.randint(1, len(AUG_FUNCS))
    chosen = rng.sample(AUG_FUNCS, n_pick)

    cur_img = img.copy()
    # Convert segs to list-of-point-lists
    poly_list = []
    for seg in segs:
        coords = seg[0] if isinstance(seg[0], list) else seg
        pts = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
        poly_list.append(pts)

    for fn in chosen:
        cur_img, M = fn(cur_img, rng)
        # always apply coord transform; identity M leaves coords unchanged
        poly_list = [_transform_pts(pts, M) for pts in poly_list]
        # For occlusion / brightness: M is identity, coords unchanged

    # Clip all polygons back to [0, PATCH]
    new_segs = []
    for pts in poly_list:
        clipped = _clip_polygon(pts, 0, 0, PATCH, PATCH)
        if len(clipped) >= 3 and _polygon_area(clipped) >= 1.0:
            flat = [coord for pt in clipped for coord in pt]
            new_segs.append([flat])
        else:
            new_segs.append(None)  # annotation lost after augmentation

    return cur_img, new_segs


# ---------------------------------------------------------------------------
# Process one COCO split
# ---------------------------------------------------------------------------

def augment_split(coco: Dict,
                  src_img_dir: Path,
                  out_img_dir: Path,
                  rng: random.Random,
                  split_name: str) -> Dict:
    """
    For every image in the split, produce one augmented version.
    Original images are ALSO copied through unchanged so the dataset doubles.
    Returns new COCO dict.
    """
    id_to_anns: Dict[int, list] = {}
    for ann in coco['annotations']:
        id_to_anns.setdefault(ann['image_id'], []).append(ann)

    new_images = []
    new_annos  = []
    new_img_id = 1
    new_ann_id = 1

    total = len(coco['images'])
    for idx, img_rec in enumerate(coco['images']):
        src_path = src_img_dir / img_rec['file_name']
        if not src_path.exists():
            continue

        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            continue

        anns = id_to_anns.get(img_rec['id'], [])
        segs = [a['segmentation'] for a in anns]

        # --- copy original ---
        orig_name = img_rec['file_name']
        cv2.imwrite(str(out_img_dir / orig_name), img_bgr)
        new_images.append({'id': new_img_id, 'file_name': orig_name,
                           'width': PATCH, 'height': PATCH})
        for ann in anns:
            new_annos.append({**ann, 'id': new_ann_id, 'image_id': new_img_id})
            new_ann_id += 1
        orig_img_id = new_img_id
        new_img_id += 1

        # --- 4 augmented versions per image ---
        stem = Path(orig_name).stem
        for aug_idx in range(4):
            aug_img, new_segs = apply_augmentations(img_bgr, segs, rng)
            aug_name = f'{stem}_aug{aug_idx:02d}.jpg'
            cv2.imwrite(str(out_img_dir / aug_name), aug_img)
            new_images.append({'id': new_img_id, 'file_name': aug_name,
                               'width': PATCH, 'height': PATCH})
            for ann, new_seg in zip(anns, new_segs):
                if new_seg is None:
                    continue
                flat = new_seg[0]
                pts  = [(flat[i], flat[i+1]) for i in range(0, len(flat)-1, 2)]
                new_annos.append({
                    'id':          new_ann_id,
                    'image_id':    new_img_id,
                    'category_id': ann['category_id'],
                    'segmentation': new_seg,
                    'area':        float(_polygon_area(pts)),
                    'bbox':        _polygon_bbox(pts),
                    'iscrowd':     0,
                })
                new_ann_id += 1
            new_img_id += 1

        if (idx+1) % 50 == 0:
            print(f'    [{split_name}] {idx+1}/{total} done', flush=True)

    return {'images': new_images, 'annotations': new_annos,
            'categories': coco['categories']}


# ---------------------------------------------------------------------------
# Draw annotations on image for sample visualisation
# ---------------------------------------------------------------------------

def draw_annotations(img: np.ndarray, anns: list) -> np.ndarray:
    out = img.copy()
    for ann in anns:
        seg = ann['segmentation']
        coords = seg[0] if isinstance(seg[0], list) else seg
        pts = np.array([[coords[i], coords[i+1]]
                        for i in range(0, len(coords)-1, 2)],
                       dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description='Step 2 augmentation')
    ap.add_argument('--src-imgs', type=Path,
        default=Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_output/images'))
    ap.add_argument('--src-coco', type=Path,
        default=Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step1_output/coco'))
    ap.add_argument('--out', type=Path,
        default=Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    if args.overwrite and args.out.exists():
        shutil.rmtree(args.out)

    out_img_dir  = args.out / 'images'
    out_coco_dir = args.out / 'coco'
    out_samp_dir = args.out / 'samples'
    for d in (out_img_dir, out_coco_dir, out_samp_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    splits = ['train', 'val', 'test']
    all_images, all_annos, cats = [], [], []
    img_id_offset = 0
    ann_id_offset = 0

    for split in splits:
        src_json = args.src_coco / f'instances_{split}.json'
        if not src_json.exists():
            print(f'[SKIP] {src_json} not found')
            continue
        with src_json.open('r', encoding='utf-8') as f:
            coco = json.load(f)
        cats = coco['categories']
        print(f'[{split}] {len(coco["images"])} images -> augmenting...')
        new_coco = augment_split(coco, args.src_imgs, out_img_dir, rng, split)

        # Save per-split JSON
        out_path = out_coco_dir / f'instances_{split}.json'
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(new_coco, f, ensure_ascii=False)
        print(f'  Saved {out_path.name}: images={len(new_coco["images"])}, annos={len(new_coco["annotations"])}')

        # Accumulate for all.json (re-id to avoid clashes)
        for img in new_coco['images']:
            all_images.append({**img, 'id': img['id'] + img_id_offset})
        for ann in new_coco['annotations']:
            all_annos.append({**ann,
                              'id':       ann['id'] + ann_id_offset,
                              'image_id': ann['image_id'] + img_id_offset})
        img_id_offset += len(new_coco['images'])
        ann_id_offset += len(new_coco['annotations'])

    # Save all.json
    coco_all = {'images': all_images, 'annotations': all_annos, 'categories': cats}
    with (out_coco_dir / 'instances_all.json').open('w', encoding='utf-8') as f:
        json.dump(coco_all, f, ensure_ascii=False)
    print(f'Saved instances_all.json: images={len(all_images)}, annos={len(all_annos)}')

    # Save representative samples
    print(f'Saving {SAMPLE_N} annotated samples to {out_samp_dir} ...')
    id_to_anns: Dict[int, list] = {}
    for ann in all_annos:
        id_to_anns.setdefault(ann['image_id'], []).append(ann)
    sample_imgs = random.Random(args.seed).sample(all_images, min(SAMPLE_N, len(all_images)))
    for rec in sample_imgs:
        img_path = out_img_dir / rec['file_name']
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        anns = id_to_anns.get(rec['id'], [])
        vis  = draw_annotations(img, anns)
        cv2.imwrite(str(out_samp_dir / rec['file_name']), vis)

    print('')
    print('Step 2 complete.')
    print(f'  Augmented images : {out_img_dir}')
    print(f'  COCO jsons       : {out_coco_dir}')
    print(f'  Visual samples   : {out_samp_dir}')


if __name__ == '__main__':
    main()
