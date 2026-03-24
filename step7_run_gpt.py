#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import re
import argparse
import numpy as np
from PIL import Image
import cv2

# =========================
# 路径配置
# =========================
TILE_MASK_DIR = "/ssd/home/jiangyiyu/unit3code/mydata/output/test_blendmask_r50/mask"
TILE_JSON_DIR = "/ssd/home/jiangyiyu/unit3code/mydata/output/test_blendmask_r50/json"
ORG_DIRS = [
    "/ssd/home/jiangyiyu/unit3code/mydata/org_img",
    "/ssd/home/jiangyiyu/unit3code/mydata/org_img_label",
]
OUT_DIR = "/ssd/home/jiangyiyu/unit3code/mydata/output/step7_stitch"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 参数配置
# =========================
TILE_SIZE = 512
GRID_COLS = 3
GRID_ROWS = 4

# 步骤1：先缩小到 1/2 再裁剪
RAW_SCALE = 0.5
UPSCALE = 1.0 / RAW_SCALE  # 2.0

MIN_AREA = 20

# bbox 级 NMS 阈值（抑制重复框）
NMS_IOU_THR = 0.35

# 像素级写回约束：写回后保留面积太小则丢弃
MIN_KEEP_AREA = 20
MIN_KEEP_RATIO = 0.15  # 写回后至少保留原面积的多少

# 文件名格式：basename_rN_cN
TILE_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)$")

# 一个实例一种颜色
rng = np.random.default_rng(42)
COLOR_TABLE = rng.integers(50, 255, size=(65536, 3), dtype=np.uint8)
COLOR_TABLE[0] = [0, 0, 0]

# bbox绘制参数
BOX_THICKNESS = 1
TEXT_SCALE = 0.35
TEXT_THICKNESS = 1


# =========================
# 基础工具函数
# =========================
def compute_tile_origins(img_w, img_h, tile=TILE_SIZE, cols=GRID_COLS, rows=GRID_ROWS):
    """
    与步骤1一致：
    在半尺寸整图坐标系中，计算每个 tile 的左上角坐标。
    """
    def o1d(length, n, t):
        if length <= t:
            return [0] * n
        step = (length - t) / (n - 1) if n > 1 else 0
        return [round(i * step) for i in range(n)]

    xs = o1d(img_w, cols, tile)
    ys = o1d(img_h, rows, tile)

    origins = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            origins.append((x, y))
    return origins


def find_org_img(stem):
    for d in ORG_DIRS:
        for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    return None


def safe_load_json(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_size_match(name, image_bgr, mask):
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            f"[{name}] size mismatch: image={image_bgr.shape[:2]}, mask={mask.shape[:2]}"
        )


def bbox_iou_xywh(box1, box2):
    """
    box: [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_overlay(image_bgr, inst_mask):
    """
    一个实例一种颜色，叠加到底图。
    要求 image 与 mask 尺寸完全一致。
    """
    if image_bgr.shape[:2] != inst_mask.shape[:2]:
        raise RuntimeError(
            f"Overlay size mismatch: image={image_bgr.shape[:2]}, mask={inst_mask.shape[:2]}"
        )

    overlay = image_bgr.copy().astype(np.float32)
    ids = np.unique(inst_mask)
    ids = ids[ids > 0]

    for iid in ids:
        color = COLOR_TABLE[int(iid) % len(COLOR_TABLE)].astype(np.float32)
        region = (inst_mask == iid)
        overlay[region] = overlay[region] * 0.4 + color[[2, 1, 0]] * 0.6

    result = overlay.astype(np.uint8)

    # 画轮廓
    for iid in ids:
        region = (inst_mask == iid).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = COLOR_TABLE[int(iid) % len(COLOR_TABLE)]
        cv2.drawContours(result, contours, -1, (int(color[2]), int(color[1]), int(color[0])), 1)

    return result


def draw_bboxes(image_bgr, instances, bbox_key="bbox_half", show_id=True, show_score=False):
    """
    在图像上绘制预测框。
    bbox_key:
      - 半尺寸图使用 "bbox_half"
      - 全尺寸图使用 "bbox"
    """
    out = image_bgr.copy()

    for ins in instances:
        bbox = ins.get(bbox_key, None)
        if bbox is None or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))
        if w <= 0 or h <= 0:
            continue

        iid = int(ins.get("instance_id", 0))
        color = COLOR_TABLE[iid % len(COLOR_TABLE)]
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))

        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color_bgr, BOX_THICKNESS)

        text_parts = []
        if show_id:
            text_parts.append(f"id:{iid}")
        if show_score and ins.get("score", None) is not None:
            try:
                text_parts.append(f"{float(ins['score']):.2f}")
            except Exception:
                pass

        if text_parts:
            text = " ".join(text_parts)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
            tx1 = x
            ty1 = max(0, y - th - 4)
            tx2 = min(out.shape[1] - 1, x + tw + 4)
            ty2 = max(th + 4, y)

            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), color_bgr, thickness=-1)
            cv2.putText(
                out, text,
                (tx1 + 2, ty2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                (255, 255, 255),
                TEXT_THICKNESS,
                lineType=cv2.LINE_AA
            )

    return out


def build_overlay_with_bboxes(image_bgr, inst_mask, instances, bbox_key="bbox_half",
                              show_id=True, show_score=False):
    over = build_overlay(image_bgr, inst_mask)
    over = draw_bboxes(over, instances, bbox_key=bbox_key, show_id=show_id, show_score=show_score)
    return over


# =========================
# 实例特征与候选提取
# =========================
def compute_boundary_continuity(mask_crop):
    """
    简单轮廓连续性评分：面积 / 周长
    """
    m = mask_crop.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0

    area = float(m.sum())
    peri = 0.0
    for c in contours:
        peri += cv2.arcLength(c, True)
    if peri <= 1e-6:
        return 0.0
    return area / peri


def extract_candidate_instances(stem, tile_masks, tile_jsons, origins, canvas_h, canvas_w):
    """
    从所有 tile 中提取候选实例。
    每个候选实例仅保存 bbox 内的局部二值 mask crop，减少内存占用。
    """
    candidates = []
    tmp_instance_id = 0

    for ti, (x0, y0) in enumerate(origins):
        row = ti // GRID_COLS
        col = ti % GRID_COLS
        tile_name = f"{stem}_r{row}_c{col}"

        if tile_name not in tile_masks:
            continue

        tile_mask = tile_masks[tile_name]
        if tile_mask.ndim == 3:
            tile_mask = tile_mask[:, :, 0]

        tile_json = tile_jsons.get(tile_name, {})
        tile_info = {}
        for x in tile_json.get("instances", []):
            if "instance_id" in x:
                try:
                    tile_info[int(x["instance_id"])] = x
                except Exception:
                    pass

        h, w = tile_mask.shape[:2]
        local_ids = sorted(int(v) for v in np.unique(tile_mask) if v > 0)

        for lid in local_ids:
            local_region = (tile_mask == lid)
            area_half = int(local_region.sum())
            if area_half < MIN_AREA:
                continue

            ys, xs = np.nonzero(local_region)
            if len(xs) == 0:
                continue

            xs_global = xs + x0
            ys_global = ys + y0

            valid = (
                (xs_global >= 0) & (xs_global < canvas_w) &
                (ys_global >= 0) & (ys_global < canvas_h)
            )
            xs_global = xs_global[valid]
            ys_global = ys_global[valid]
            if len(xs_global) == 0:
                continue

            bx = int(xs_global.min())
            by = int(ys_global.min())
            bw = int(xs_global.max()) - bx + 1
            bh = int(ys_global.max()) - by + 1

            crop_mask = np.zeros((bh, bw), dtype=bool)
            crop_mask[ys_global - by, xs_global - bx] = True

            cx = float(xs_global.mean())
            cy = float(ys_global.mean())

            local_cx = float(xs.mean())
            local_cy = float(ys.mean())
            min_margin = min(local_cx, w - 1 - local_cx, local_cy, h - 1 - local_cy)

            edge_touch_count = 0
            edge_touch_count += int(np.sum(local_region[0, :]))
            edge_touch_count += int(np.sum(local_region[-1, :]))
            edge_touch_count += int(np.sum(local_region[:, 0]))
            edge_touch_count += int(np.sum(local_region[:, -1]))

            continuity = compute_boundary_continuity(crop_mask)

            src = tile_info.get(lid, {})
            score = src.get("score", None)
            if score is None:
                score_val = -1.0
            else:
                try:
                    score_val = float(score)
                except Exception:
                    score_val = -1.0

            completeness_score = (
                1.0 * area_half
                + 8.0 * min_margin
                + 200.0 * continuity
                - 2.0 * edge_touch_count
            )

            tmp_instance_id += 1
            candidates.append({
                "instance_id": tmp_instance_id,   # 临时ID，后续会重新编号
                "source_tile": tile_name,
                "tile_index": ti,
                "tile_row": row,
                "tile_col": col,
                "tile_local_id": lid,
                "category_name": src.get("category_name", "lettuce"),
                "score": score,
                "score_val": score_val,
                "bbox_half": [bx, by, bw, bh],
                "centroid_half": [round(cx, 2), round(cy, 2)],
                "area_half": int(crop_mask.sum()),
                "edge_touch_count": edge_touch_count,
                "min_margin_to_tile_boundary": float(min_margin),
                "boundary_continuity_score": float(continuity),
                "completeness_score": float(completeness_score),
                "mask_crop_half": crop_mask,
            })

    return candidates


# =========================
# 方案A：面积优先
# =========================
def instance_priority(ins):
    """
    方案A：
    同一区域若有多个实例重叠，优先保留面积最大的那个；
    若较小实例仍有不重叠剩余部分，可继续保留。
    优先级：
    1) area_half 大优先
    2) score 高优先
    3) completeness_score 高优先
    4) tile顺序稳定
    """
    return (
        -int(ins.get("area_half", 0)),
        -float(ins.get("score_val", -1.0)),
        -float(ins.get("completeness_score", 0.0)),
        int(ins.get("tile_row", 0)),
        int(ins.get("tile_col", 0)),
        int(ins.get("tile_local_id", 0)),
    )


# =========================
# bbox NMS
# =========================
def nms_instances(candidates, iou_thr=NMS_IOU_THR):
    """
    对候选实例做 bbox 级 NMS，减少重复实例。
    面积最大的优先保留。
    """
    if len(candidates) == 0:
        return [], {"before": 0, "after": 0, "suppressed": 0, "details": []}

    ordered = sorted(candidates, key=instance_priority)
    kept = []
    suppressed = []

    for cand in ordered:
        keep_this = True
        box_c = cand["bbox_half"]

        for k in kept:
            iou = bbox_iou_xywh(box_c, k["bbox_half"])
            if iou >= iou_thr:
                keep_this = False
                suppressed.append({
                    "suppressed_instance_id": int(cand["instance_id"]),
                    "kept_instance_id": int(k["instance_id"]),
                    "iou": round(float(iou), 6)
                })
                break

        if keep_this:
            kept.append(cand)

    debug = {
        "before": len(candidates),
        "after": len(kept),
        "suppressed": len(suppressed),
        "details": suppressed[:500]
    }
    return kept, debug


# =========================
# 像素级无重叠写回（方案A）
# =========================
def paste_instances_without_overlap(candidates, canvas_h, canvas_w,
                                    min_keep_area=MIN_KEEP_AREA,
                                    min_keep_ratio=MIN_KEEP_RATIO):
    """
    将实例按优先级从高到低写入半尺寸整图。
    方案A：
    - 面积最大的实例先占位；
    - 后来的较小实例如果仍有未被占用的区域，则保留这些剩余区域；
    - 若剩余区域过小，则丢弃。
    """
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    new_mask_half = np.zeros((canvas_h, canvas_w), dtype=np.uint32)
    kept_instances = []
    debug_details = []

    ordered = sorted(candidates, key=instance_priority)

    for ins in ordered:
        bx, by, bw, bh = ins["bbox_half"]
        crop = ins["mask_crop_half"]
        if bw <= 0 or bh <= 0:
            continue

        occ_view = occupied[by:by + bh, bx:bx + bw]
        free_crop = crop & (~occ_view)

        orig_area = int(crop.sum())
        keep_area = int(free_crop.sum())

        if orig_area <= 0:
            continue

        keep_ratio = keep_area / orig_area

        if keep_area < min_keep_area or keep_ratio < min_keep_ratio:
            debug_details.append({
                "instance_id": int(ins["instance_id"]),
                "source_tile": ins["source_tile"],
                "orig_area": orig_area,
                "keep_area": keep_area,
                "keep_ratio": round(float(keep_ratio), 6),
                "status": "discard_after_pixel_dedup"
            })
            continue

        # 重新编号
        new_id = len(kept_instances) + 1

        new_mask_half[by:by + bh, bx:bx + bw][free_crop] = new_id
        occupied[by:by + bh, bx:bx + bw] |= free_crop

        ys, xs = np.nonzero(free_crop)
        x0 = int(xs.min()) + bx
        y0 = int(ys.min()) + by
        x1 = int(xs.max()) + bx
        y1 = int(ys.max()) + by

        new_ins = dict(ins)
        new_ins["old_instance_id"] = int(ins["instance_id"])
        new_ins["instance_id"] = new_id
        new_ins["bbox_half"] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        new_ins["centroid_half"] = [round(float(xs.mean()) + bx, 2), round(float(ys.mean()) + by, 2)]
        new_ins["area_half"] = keep_area
        new_ins["mask_crop_half"] = free_crop
        kept_instances.append(new_ins)

    debug = {
        "before": len(candidates),
        "after": len(kept_instances),
        "suppressed": len(candidates) - len(kept_instances),
        "details": debug_details[:1000]
    }
    return new_mask_half, kept_instances, debug


# =========================
# 半尺寸 -> 全尺寸属性
# =========================
def convert_instances_half_to_full(instances_half):
    out = []
    for ins in instances_half:
        bx_h, by_h, bw_h, bh_h = ins["bbox_half"]
        cx_h, cy_h = ins["centroid_half"]
        area_half = ins["area_half"]

        bx = int(round(bx_h * UPSCALE))
        by = int(round(by_h * UPSCALE))
        bw = int(round(bw_h * UPSCALE))
        bh = int(round(bh_h * UPSCALE))
        cx = float(cx_h * UPSCALE)
        cy = float(cy_h * UPSCALE)
        area = int(round(area_half * (UPSCALE ** 2)))

        new_ins = dict(ins)
        new_ins["bbox"] = [bx, by, bw, bh]
        new_ins["centroid"] = [round(cx, 2), round(cy, 2)]
        new_ins["area"] = area
        new_ins.pop("mask_crop_half", None)
        out.append(new_ins)

    return out


# =========================
# 参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在输出")
    parser.add_argument("--images", nargs="*", default=None, help="仅处理指定图像，如 IMG_xxx.jpg 或 stem")
    parser.add_argument("--min-area", type=int, default=MIN_AREA, help="极小实例过滤阈值")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THR, help="bbox级NMS阈值")
    parser.add_argument("--min-keep-area", type=int, default=MIN_KEEP_AREA, help="像素级写回后最小保留面积")
    parser.add_argument("--min-keep-ratio", type=float, default=MIN_KEEP_RATIO, help="像素级写回后最小保留比例")
    parser.add_argument("--no-box-id", action="store_true", help="half/full vis上不显示实例ID")
    parser.add_argument("--show-score", action="store_true", help="half/full vis上显示score")
    return parser.parse_args()


def normalize_requested_stems(images_arg):
    if not images_arg:
        return None
    stems = set()
    for x in images_arg:
        base = os.path.basename(x)
        stem, _ = os.path.splitext(base)
        stems.add(stem)
    return stems


# =========================
# 主流程
# =========================
def main():
    args = parse_args()
    requested_stems = normalize_requested_stems(args.images)

    mask_files = sorted(glob.glob(os.path.join(TILE_MASK_DIR, "*_mask.png")))

    stems = sorted(set(
        m.group(1)
        for mf in mask_files
        for m in [TILE_RE.match(os.path.basename(mf).replace("_mask.png", ""))]
        if m
    ))

    if requested_stems is not None:
        stems = [s for s in stems if s in requested_stems]

    print(f"Found {len(stems)} stems from {len(mask_files)} tile masks")

    summary = []

    for idx, stem in enumerate(stems, 1):
        out_mask_path = os.path.join(OUT_DIR, stem + "_instance_mask.png")
        if args.skip_existing and os.path.exists(out_mask_path):
            print(f"[{idx}/{len(stems)}] {stem}: [SKIP] already done")
            summary.append({"stem": stem, "skipped": True, "mask_path": out_mask_path})
            continue

        org_path = find_org_img(stem)
        if org_path is None:
            print(f"[{idx}/{len(stems)}] {stem}: [SKIP] no original image")
            continue

        org_bgr = cv2.imread(org_path)
        if org_bgr is None:
            print(f"[{idx}/{len(stems)}] {stem}: [SKIP] failed to read original image")
            continue

        full_h, full_w = org_bgr.shape[:2]
        half_w = int(round(full_w * RAW_SCALE))
        half_h = int(round(full_h * RAW_SCALE))

        print(f"[{idx}/{len(stems)}] {stem}  full={full_w}x{full_h}  half={half_w}x{half_h}", flush=True)

        # 半尺寸底图
        half_img_bgr = cv2.resize(org_bgr, (half_w, half_h), interpolation=cv2.INTER_LINEAR)

        origins = compute_tile_origins(half_w, half_h)

        # 读取所有 tile mask/json
        tile_masks = {}
        tile_jsons = {}
        for ti, (x0, y0) in enumerate(origins):
            row = ti // GRID_COLS
            col = ti % GRID_COLS
            tile_name = f"{stem}_r{row}_c{col}"

            mask_path = os.path.join(TILE_MASK_DIR, tile_name + "_mask.png")
            json_path = os.path.join(TILE_JSON_DIR, tile_name + ".json")

            if os.path.exists(mask_path):
                tile_mask = np.array(Image.open(mask_path))
                if tile_mask.ndim == 3:
                    tile_mask = tile_mask[:, :, 0]
                tile_masks[tile_name] = tile_mask

            if os.path.exists(json_path):
                tile_jsons[tile_name] = safe_load_json(json_path)

        if len(tile_masks) == 0:
            print(f"[{idx}/{len(stems)}] {stem}: [SKIP] no tile masks found")
            continue

        # 1) 提取候选实例
        candidates = extract_candidate_instances(
            stem=stem,
            tile_masks=tile_masks,
            tile_jsons=tile_jsons,
            origins=origins,
            canvas_h=half_h,
            canvas_w=half_w
        )

        # 2) bbox级 NMS：面积最大的优先保留
        candidates_nms, nms_debug = nms_instances(
            candidates,
            iou_thr=args.nms_iou
        )

        # 3) 像素级无重叠写回：面积最大实例先占位，小实例若有剩余可保留
        global_mask_half, instances_half, raster_debug = paste_instances_without_overlap(
            candidates_nms,
            canvas_h=half_h,
            canvas_w=half_w,
            min_keep_area=args.min_keep_area,
            min_keep_ratio=args.min_keep_ratio
        )

        # 4) 半尺寸 mask 输出
        half_mask_path = os.path.join(OUT_DIR, stem + "_half_instance_mask.png")
        Image.fromarray(np.clip(global_mask_half, 0, 65535).astype(np.uint16)).save(half_mask_path)

        # 5) 半尺寸 overlay + bbox
        verify_size_match(stem + "_half", half_img_bgr, global_mask_half)
        half_vis = build_overlay_with_bboxes(
            half_img_bgr,
            global_mask_half.astype(np.uint16),
            instances_half,
            bbox_key="bbox_half",
            show_id=(not args.no_box_id),
            show_score=args.show_score
        )
        half_vis_path = os.path.join(OUT_DIR, stem + "_half_vis.jpg")
        cv2.imwrite(half_vis_path, half_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 6) 调试信息输出
        debug_path = os.path.join(OUT_DIR, stem + "_merge_debug.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({
                "candidate_instances_before_nms": len(candidates),
                "candidate_instances_after_nms": len(candidates_nms),
                "instances_after_raster_merge": len(instances_half),
                "nms_debug": nms_debug,
                "raster_debug": raster_debug,
            }, f, ensure_ascii=False, indent=2)

        # 7) 恢复到原图全尺寸
        global_mask_half_u16 = np.clip(global_mask_half, 0, 65535).astype(np.uint16)
        full_mask = cv2.resize(
            global_mask_half_u16,
            (full_w, full_h),
            interpolation=cv2.INTER_NEAREST
        )

        verify_size_match(stem + "_full", org_bgr, full_mask)

        # 8) 全尺寸实例 mask 输出
        Image.fromarray(full_mask).save(out_mask_path)

        # 9) 半尺寸实例属性转全尺寸
        instances_full = convert_instances_half_to_full(instances_half)

        # 10) 全尺寸 overlay + bbox
        vis = build_overlay_with_bboxes(
            org_bgr,
            full_mask,
            instances_full,
            bbox_key="bbox",
            show_id=(not args.no_box_id),
            show_score=args.show_score
        )
        vis_path = os.path.join(OUT_DIR, stem + "_vis.jpg")
        cv2.imwrite(vis_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 11) JSON
        json_path = os.path.join(OUT_DIR, stem + "_instances.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "imagePath": os.path.basename(org_path),
                "imageWidth": full_w,
                "imageHeight": full_h,
                "halfWidth": half_w,
                "halfHeight": half_h,
                "instances": instances_full
            }, f, ensure_ascii=False, indent=2)

        summary.append({
            "stem": stem,
            "original_image": org_path,
            "full_wh": [full_w, full_h],
            "half_wh": [half_w, half_h],
            "n_tile_masks": len(tile_masks),
            "candidate_instances_before_nms": len(candidates),
            "candidate_instances_after_nms": len(candidates_nms),
            "n_instances_half": len(instances_half),
            "n_instances_full": len(instances_full),
            "debug_path": debug_path,
            "half_mask_path": half_mask_path,
            "half_vis_path": half_vis_path,
            "mask_path": out_mask_path,
            "vis_path": vis_path,
            "json_path": json_path,
        })

        print(
            f"  tile_masks={len(tile_masks)}  cand={len(candidates)} -> nms={len(candidates_nms)} -> final={len(instances_full)}",
            flush=True
        )

    summary_path = os.path.join(OUT_DIR, "stitch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    total_instances = sum(x.get("n_instances_full", 0) for x in summary if not x.get("skipped"))
    print("\n=== Step 7 Done ===")
    print(f"  stems={len(summary)}  total_instances={total_instances}")
    print(f"  output: {OUT_DIR}")


if __name__ == "__main__":
    main()