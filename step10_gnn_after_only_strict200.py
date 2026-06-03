#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_gnn_after_only_strict200.py

仅输出GNN优化后结果 + 人工空穴强约束 + 微小噪声剔除 + 临近碎片合并 + 200穴位总数约束。

输出目录：
/ssd/home/jiangyiyu/unit3code/mydata/output/step10_graph_merge_after_gnn_strict200

输出文件：
*_after_gnn_overlay.png
*_after_gnn_holes.png
*_after_gnn_grade.png
*_after_gnn_instance_mask.png
*_after_gnn_result.json
after_gnn_summary.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, "/ssd/home/jiangyiyu/unit3code")

import step10_hole as s10
from step10_gnn import gnn_merge_and_assign


DEFAULT_OUT_DIR = Path("/ssd/home/jiangyiyu/unit3code/mydata/output/step10_graph_merge_after_gnn_strict200")

COLOR_ALPHA = 0.35
GREEN = (0, 200, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (170, 170, 170)
GRADE_I_COLOR = (0, 180, 0)
GRADE_II_COLOR = (0, 220, 255)
GRADE_III_COLOR = (0, 0, 255)
EMPTY_COLOR = (0, 0, 255)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def sanitize_for_json(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if k != "mask"}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    return obj


def save_json(obj: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, ensure_ascii=False, indent=2)


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    union = np.logical_or(m1 > 0, m2 > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def dilated_iou(m1: np.ndarray, m2: np.ndarray, ksize: int = 9, iters: int = 1) -> float:
    k = np.ones((ksize, ksize), np.uint8)
    d1 = cv2.dilate((m1 > 0).astype(np.uint8), k, iterations=iters)
    d2 = cv2.dilate((m2 > 0).astype(np.uint8), k, iterations=iters)
    return mask_iou(d1, d2)


def bboxes_touch(b1: List[int], b2: List[int], margin: int = 10) -> bool:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    return not (
        ax2 < bx1 - margin or bx2 < ax1 - margin or
        ay2 < by1 - margin or by2 < ay1 - margin
    )


def recompute_props(mask: np.ndarray):
    return s10.mask_to_props((mask > 0).astype(np.uint8))


def merge_instance_list(instances: List[dict], idxs: List[int]):
    if not idxs:
        return None
    merged_mask = np.zeros_like(instances[idxs[0]]["mask"], dtype=np.uint8)
    source = []
    labels = []
    for i in idxs:
        merged_mask = np.maximum(merged_mask, (instances[i]["mask"] > 0).astype(np.uint8))
        source.extend(list(map(int, instances[i].get("source_instance_indices", [i]))))
        labels.append(instances[i].get("label", "lettuce"))
    props = recompute_props(merged_mask)
    if props is None:
        return None
    return {
        "source_instance_indices": sorted(list(set(source))),
        "label_list": labels,
        "mask": merged_mask,
        "centroid": props["centroid"],
        "bbox": props["bbox"],
        "area": props["area"],
    }


def load_raw_instances_and_masks(json_path: Path, data: dict, image_h: int, image_w: int):
    raw_instances = []
    raw_masks = []
    union_mask = np.zeros((image_h, image_w), dtype=np.uint8)

    stem_base = json_path.stem.replace("_instances_with_polygon", "")
    mask_candidates = [
        s10.STEP7_MASK_DIR / f"{stem_base}_merged_full_instance_mask.png",
        s10.STEP7_MASK_DIR / f"{stem_base}_instance_mask.png",
    ]
    mask_path = next((p for p in mask_candidates if p.exists()), None)

    if mask_path is not None:
        inst_img = np.array(Image.open(str(mask_path)))
        if inst_img.ndim == 3:
            inst_img = inst_img[:, :, 0]
        if inst_img.shape[:2] != (image_h, image_w):
            inst_img = cv2.resize(inst_img.astype(np.uint16), (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        ids = np.unique(inst_img)
        ids = ids[ids > 0]
        for iid in ids:
            m = (inst_img == iid).astype(np.uint8)
            props = s10.mask_to_props(m)
            if props is None:
                continue
            raw_masks.append(m)
            union_mask = np.maximum(union_mask, m)
            raw_instances.append({
                "instance_index": len(raw_instances),
                "source_mask_id": int(iid),
                "label": "lettuce",
                "shape_type": "mask",
                "centroid": props["centroid"],
                "bbox": props["bbox"],
                "area": props["area"],
            })
        return raw_instances, raw_masks, union_mask, str(mask_path)

    if isinstance(data.get("instances", None), list) and len(data["instances"]) > 0:
        for ins in data["instances"]:
            m = s10.polygons_to_mask(ins.get("polygon", None), image_h, image_w)
            props = s10.mask_to_props(m)
            if props is None:
                continue
            raw_masks.append(m.astype(np.uint8))
            union_mask = np.maximum(union_mask, m.astype(np.uint8))
            raw_instances.append({
                "instance_index": len(raw_instances),
                "label": str(ins.get("category_name", "lettuce")),
                "shape_type": "polygon",
                "centroid": props["centroid"],
                "bbox": props["bbox"],
                "area": props["area"],
            })
        return raw_instances, raw_masks, union_mask, "fallback_step8_polygon"

    for shape in data.get("shapes", []):
        st = shape.get("shape_type", "polygon")
        if st not in ["polygon", "rectangle"]:
            continue
        m = s10.shape_to_mask(shape, image_h, image_w)
        props = s10.mask_to_props(m)
        if props is None:
            continue
        raw_masks.append(m.astype(np.uint8))
        union_mask = np.maximum(union_mask, m.astype(np.uint8))
        raw_instances.append({
            "instance_index": len(raw_instances),
            "label": shape.get("label", "lettuce"),
            "shape_type": st,
            "centroid": props["centroid"],
            "bbox": props["bbox"],
            "area": props["area"],
        })
    return raw_instances, raw_masks, union_mask, "fallback_labelme_shapes"


def get_manual_empty_hole_ids(stem_base: str, hole_centers, avg_spacing: float):
    ann_json = s10.find_ann_json_for_stem(stem_base, s10.ANN_DIR)
    if ann_json is None:
        return []
    pts = s10.load_manual_empty_hole_points(ann_json)
    ids = s10.map_manual_empty_points_to_holes(pts, hole_centers, avg_spacing=avg_spacing)
    return sorted(list(map(int, ids)))


def nearest_hole(ins: dict, hole_centers):
    p = np.array(ins["centroid"], dtype=np.float64)
    centers = np.array([h["center"] for h in hole_centers], dtype=np.float64)
    d = np.linalg.norm(centers - p[None, :], axis=1)
    hi = int(np.argmin(d))
    return hi, float(d[hi])


def hole_circle_overlap(mask: np.ndarray, center_xy, radius: float):
    h, w = mask.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    r = int(round(radius))
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w - 1, cx + r)
    y2 = min(h - 1, cy + r)
    if x2 < x1 or y2 < y1:
        return 0, 0.0
    roi = (mask[y1:y2 + 1, x1:x2 + 1] > 0).astype(np.uint8)
    circle = np.zeros_like(roi, dtype=np.uint8)
    cv2.circle(circle, (cx - x1, cy - y1), r, 1, -1)
    ov = int(np.logical_and(roi > 0, circle > 0).sum())
    area = int((mask > 0).sum())
    return ov, float(ov / area) if area > 0 else 0.0


def postprocess_gnn_instances(
    opt_instances_raw: List[dict],
    hole_centers,
    manual_empty_ids,
    avg_spacing,
    min_noise_area=120,
    small_fragment_area=350,
    merge_dist_ratio=0.45,
    excessive_iou=0.40,
    empty_radius_ratio=0.48,
):
    debug = {"removed_noise": [], "merged_pairs": [], "discarded_duplicates": [], "n_input": len(opt_instances_raw)}
    if not opt_instances_raw:
        return [], debug

    manual_set = set(map(int, manual_empty_ids))
    spacing = max(float(avg_spacing), 1.0)
    merge_dist = spacing * merge_dist_ratio
    empty_radius = spacing * empty_radius_ratio

    norm = []
    for i, ins in enumerate(opt_instances_raw):
        x = dict(ins)
        x["opt_instance_index"] = i
        hi, hd = nearest_hole(x, hole_centers)
        x["nearest_hole_index"] = hi
        x["nearest_hole_dist"] = hd
        norm.append(x)

    areas_all = np.array([float(x["area"]) for x in norm], dtype=np.float64)
    med_area = float(np.median(areas_all)) if len(areas_all) > 0 else 1.0
    adaptive_small = max(float(small_fragment_area), med_area * 0.22)

    kept = []
    for i, ins in enumerate(norm):
        area = float(ins["area"])
        hi = int(ins["nearest_hole_index"])
        hd = float(ins["nearest_hole_dist"])

        if hi in manual_set and (area <= adaptive_small or hd <= empty_radius):
            debug["removed_noise"].append({"old_index": i, "area": area, "nearest_hole": hi, "dist": hd, "reason": "manual_empty_shadow_mask"})
            continue

        if area < min_noise_area:
            debug["removed_noise"].append({"old_index": i, "area": area, "nearest_hole": hi, "dist": hd, "reason": "tiny_noise_mask"})
            continue

        if area < adaptive_small and hd > spacing * 0.72:
            debug["removed_noise"].append({"old_index": i, "area": area, "nearest_hole": hi, "dist": hd, "reason": "unassigned_small_fragment"})
            continue

        kept.append(ins)

    if len(kept) == 0:
        debug["n_final"] = 0
        return [], debug

    n = len(kept)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = kept[i]
            b = kept[j]
            hi = int(a["nearest_hole_index"])
            hj = int(b["nearest_hole_index"])
            same_hole = hi == hj
            dist = float(np.linalg.norm(np.array(a["centroid"]) - np.array(b["centroid"])))
            ai = float(a["area"])
            aj = float(b["area"])
            one_small = min(ai, aj) <= adaptive_small

            iou = 0.0
            diou = 0.0
            if bboxes_touch(a["bbox"], b["bbox"], margin=15):
                iou = mask_iou(a["mask"], b["mask"])
                diou = dilated_iou(a["mask"], b["mask"], ksize=9, iters=1)

            merge = False
            reason = []
            if same_hole and dist <= merge_dist:
                merge = True
                reason.append("same_hole_close")
            if same_hole and one_small and dist <= spacing * 0.65:
                merge = True
                reason.append("small_fragment_to_main_seedling")
            if same_hole and (iou > 0.0 or diou >= 0.08) and dist <= spacing * 0.80:
                merge = True
                reason.append("touch_or_dilated_overlap")
            if iou >= excessive_iou:
                merge = True
                reason.append("excessive_overlap")

            if merge:
                uf.union(i, j)
                debug["merged_pairs"].append({"i": i, "j": j, "hole_i": hi, "hole_j": hj, "dist": dist, "iou": iou, "dilated_iou": diou, "reason": "|".join(reason)})

    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    merged = []
    for _, idxs in groups.items():
        m = merge_instance_list(kept, idxs)
        if m is not None:
            hi, hd = nearest_hole(m, hole_centers)
            m["nearest_hole_index"] = hi
            m["nearest_hole_dist"] = hd
            merged.append(m)

    merged = sorted(merged, key=lambda x: float(x["area"]), reverse=True)
    final = []
    for ins in merged:
        drop = False
        for kept_ins in final:
            if not bboxes_touch(ins["bbox"], kept_ins["bbox"], margin=8):
                continue
            iou = mask_iou(ins["mask"], kept_ins["mask"])
            if iou >= excessive_iou:
                drop = True
                debug["discarded_duplicates"].append({"discarded_area": int(ins["area"]), "kept_area": int(kept_ins["area"]), "iou": float(iou), "reason": "duplicate_overlap_keep_larger"})
                break
        if not drop:
            final.append(ins)

    for i, ins in enumerate(final):
        ins["opt_instance_index"] = i

    debug["n_after_noise_removal"] = len(kept)
    debug["n_after_merge"] = len(merged)
    debug["n_final"] = len(final)
    return final, debug


def strict_assign_holes(opt_instances, hole_centers, manual_empty_ids, match_radius, avg_spacing, opt_hole_scores=None, use_gnn_score=False):
    manual_set = set(map(int, manual_empty_ids))
    n_holes = len(hole_centers)
    spacing = max(float(avg_spacing), 1.0)
    base_radius = max(float(match_radius), spacing * 0.42)

    score_arr = None
    if opt_hole_scores is not None:
        score_arr = np.asarray(opt_hole_scores, dtype=np.float32)
        if score_arr.size == 0:
            score_arr = None

    areas = np.array([float(x["area"]) for x in opt_instances], dtype=np.float64)
    med_area = float(np.median(areas)) if len(areas) > 0 else 1.0

    candidates = []
    for oi, ins in enumerate(opt_instances):
        cx, cy = float(ins["centroid"][0]), float(ins["centroid"][1])
        area = float(ins["area"])
        area_score = min(1.0, math.log1p(area) / math.log1p(max(med_area * 2.5, 1.0)))
        for hi, h in enumerate(hole_centers):
            hid = int(h["hole_index"])
            if hid in manual_set:
                continue
            hx, hy = float(h["center"][0]), float(h["center"][1])
            dist = float(np.hypot(cx - hx, cy - hy))
            radius = base_radius
            if h["row"] in [0, s10.NUM_ROWS - 1] or h["col"] in [0, s10.NUM_COLS - 1]:
                radius = min(radius + 8.0, 98.0)
            if dist > radius * 1.45:
                continue
            ov_pix, ov_ratio = hole_circle_overlap(ins["mask"], h["center"], radius)
            if dist > radius and ov_pix < 25 and ov_ratio < 0.06:
                continue
            dist_score = max(0.0, 1.0 - dist / max(radius * 1.45, 1e-6))
            overlap_score = min(1.0, ov_ratio * 4.0 + ov_pix / 250.0)
            gscore = 0.0
            if score_arr is not None and oi < score_arr.shape[0] and hi < score_arr.shape[1]:
                gscore = float(1.0 / (1.0 + np.exp(-score_arr[oi, hi])))
            # GNN score仅作为弱辅助，避免GNN后空穴错误率升高
            if use_gnn_score:
                total = 0.50 * dist_score + 0.28 * overlap_score + 0.17 * area_score + 0.05 * gscore
            else:
                total = 0.58 * dist_score + 0.30 * overlap_score + 0.12 * area_score
            candidates.append({"score": total, "oi": oi, "hi": hi, "hole_index": hid, "dist": dist, "radius": radius, "overlap_pixels": ov_pix, "overlap_ratio": ov_ratio, "area_score": area_score, "gnn_score": gscore})

    candidates.sort(key=lambda x: (-x["score"], x["dist"]))
    used_i = set()
    used_h = set()
    matches = []
    for c in candidates:
        oi = c["oi"]
        hid = c["hole_index"]
        hi = c["hi"]
        if oi in used_i or hid in used_h:
            continue
        ins = opt_instances[oi]
        h = hole_centers[hi]
        used_i.add(oi)
        used_h.add(hid)
        matches.append({
            "hole_index": int(hid), "row": int(h["row"]), "col": int(h["col"]),
            "theory_center_x": float(h["center"][0]), "theory_center_y": float(h["center"][1]),
            "opt_instance_index": int(ins["opt_instance_index"]),
            "instance_centroid_x": float(ins["centroid"][0]), "instance_centroid_y": float(ins["centroid"][1]),
            "distance": float(c["dist"]), "match_score": float(c["score"]),
            "overlap_pixels": int(c["overlap_pixels"]), "overlap_ratio_to_instance": float(c["overlap_ratio"]),
            "gnn_score": float(c["gnn_score"]),
            "source_instance_indices": ",".join(map(str, ins.get("source_instance_indices", []))),
        })

    matched_holes = {int(m["hole_index"]) for m in matches}
    empty_holes = []
    for h in hole_centers:
        hid = int(h["hole_index"])
        if hid in manual_set or hid not in matched_holes:
            empty_holes.append({
                "hole_index": hid, "row": int(h["row"]), "col": int(h["col"]),
                "theory_center_x": float(h["center"][0]), "theory_center_y": float(h["center"][1]),
            })
    if len(matches) + len(empty_holes) != n_holes:
        raise RuntimeError(f"Count failed: {len(matches)} + {len(empty_holes)} != {n_holes}")
    return matches, empty_holes, {"n_candidates": len(candidates), "n_used_instances": len(used_i), "n_unused_instances": len(opt_instances) - len(used_i), "manual_empty_ids": sorted(list(manual_set)), "total_constraint": f"{len(matches)}+{len(empty_holes)}={n_holes}"}


def assign_grades(opt_instances, matches):
    id_to_ins = {int(x["opt_instance_index"]): x for x in opt_instances}
    matched_ids = [int(m["opt_instance_index"]) for m in matches if int(m["opt_instance_index"]) in id_to_ins]
    areas = np.array([float(id_to_ins[i]["area"]) for i in matched_ids], dtype=np.float64)
    if len(areas) == 0:
        q33 = q66 = 0.0
    elif len(areas) == 1:
        q33 = q66 = float(areas[0])
    else:
        q33 = float(np.percentile(areas, 33.333))
        q66 = float(np.percentile(areas, 66.667))
    match_by_inst = {int(m["opt_instance_index"]): m for m in matches}
    rows = []
    stats = {"I": 0, "II": 0, "III": 0, "matched_total": 0}
    for oi in matched_ids:
        ins = id_to_ins[oi]
        area = float(ins["area"])
        grade = "I" if area >= q66 else ("II" if area >= q33 else "III")
        m = match_by_inst[oi]
        stats[grade] += 1
        stats["matched_total"] += 1
        rows.append({"opt_instance_index": int(oi), "source_instance_indices": list(map(int, ins.get("source_instance_indices", []))), "area": int(ins["area"]), "centroid": [float(ins["centroid"][0]), float(ins["centroid"][1])], "bbox": list(map(int, ins["bbox"])), "hole_index": int(m["hole_index"]), "row": int(m["row"]), "col": int(m["col"]), "grade": grade})
    return rows, stats, {"method": "tertile_by_matched_seedling_area", "q33_area": q33, "q66_area": q66, "rule": "I: area>=q66; II: q33<=area<q66; III: area<q33"}


def grade_color(grade):
    return {"I": GRADE_I_COLOR, "II": GRADE_II_COLOR, "III": GRADE_III_COLOR}.get(grade, GRADE_III_COLOR)


def scale_mask(mask, shape):
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def draw_holes(image, hole_centers, matches, empty_holes, path, title):
    vis, scale = s10.resize_long_side(image, s10.VIS_LONG_SIDE)
    for h in hole_centers:
        c = s10.scale_point(h["center"], scale)
        cv2.circle(vis, c, 2, GRAY, -1, cv2.LINE_AA)
    for m in matches:
        hc = s10.scale_point((m["theory_center_x"], m["theory_center_y"]), scale)
        ic = s10.scale_point((m["instance_centroid_x"], m["instance_centroid_y"]), scale)
        cv2.line(vis, hc, ic, BLACK, 1, cv2.LINE_AA)
        cv2.circle(vis, hc, 7, GREEN, 2, cv2.LINE_AA)
        cv2.circle(vis, ic, 4, GREEN, -1, cv2.LINE_AA)
    for e in empty_holes:
        c = s10.scale_point((e["theory_center_x"], e["theory_center_y"]), scale)
        cv2.circle(vis, c, 9, EMPTY_COLOR, 2, cv2.LINE_AA)
    cv2.putText(vis, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, WHITE, 3, cv2.LINE_AA)
    cv2.putText(vis, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, BLACK, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def draw_grade(image, opt_instances, grade_rows, empty_holes, path, title):
    vis, scale = s10.resize_long_side(image, s10.VIS_LONG_SIDE)
    overlay = vis.astype(np.float32)
    id_to_ins = {int(x["opt_instance_index"]): x for x in opt_instances}
    for r in grade_rows:
        oi = int(r["opt_instance_index"])
        if oi not in id_to_ins:
            continue
        ins = id_to_ins[oi]
        color = np.array(grade_color(r["grade"]), dtype=np.float32)
        ms = scale_mask(ins["mask"], vis.shape[:2])
        overlay[ms] = overlay[ms] * (1.0 - COLOR_ALPHA) + color * COLOR_ALPHA
    vis = overlay.clip(0, 255).astype(np.uint8)
    for r in grade_rows:
        oi = int(r["opt_instance_index"])
        if oi not in id_to_ins:
            continue
        ins = id_to_ins[oi]
        color = grade_color(r["grade"])
        x1, y1, x2, y2 = list(map(int, ins["bbox"]))
        p1 = s10.scale_point((x1, y1), scale)
        p2 = s10.scale_point((x2, y2), scale)
        cv2.rectangle(vis, p1, p2, color, 2, cv2.LINE_AA)
        c = s10.scale_point(ins["centroid"], scale)
        cv2.circle(vis, c, 4, color, -1, cv2.LINE_AA)
        cv2.putText(vis, r["grade"], (c[0] + 3, c[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 3, cv2.LINE_AA)
        cv2.putText(vis, r["grade"], (c[0] + 3, c[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BLACK, 1, cv2.LINE_AA)
    for e in empty_holes:
        c = s10.scale_point((e["theory_center_x"], e["theory_center_y"]), scale)
        cv2.circle(vis, c, 8, EMPTY_COLOR, 2, cv2.LINE_AA)
    cv2.putText(vis, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, WHITE, 3, cv2.LINE_AA)
    cv2.putText(vis, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, BLACK, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def draw_overlay(image, opt_instances, matches, empty_holes, path, title):
    vis, scale = s10.resize_long_side(image, s10.VIS_LONG_SIDE)
    overlay = vis.astype(np.float32)
    for ins in opt_instances:
        oid = int(ins["opt_instance_index"]) + 1
        rng = np.random.default_rng(seed=oid)
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        ms = scale_mask(ins["mask"], vis.shape[:2])
        overlay[ms] = overlay[ms] * (1.0 - COLOR_ALPHA) + color.astype(np.float32) * COLOR_ALPHA
        contours, _ = cv2.findContours(ms.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(x) for x in color.tolist()), 1, cv2.LINE_AA)
    vis = overlay.clip(0, 255).astype(np.uint8)
    for m in matches:
        hc = s10.scale_point((m["theory_center_x"], m["theory_center_y"]), scale)
        cv2.circle(vis, hc, 6, GREEN, 2, cv2.LINE_AA)
    for e in empty_holes:
        c = s10.scale_point((e["theory_center_x"], e["theory_center_y"]), scale)
        cv2.circle(vis, c, 8, EMPTY_COLOR, 2, cv2.LINE_AA)
    cv2.putText(vis, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, WHITE, 3, cv2.LINE_AA)
    cv2.putText(vis, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, BLACK, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def build_label_mask(opt_instances, h, w):
    label = np.zeros((h, w), dtype=np.uint16)
    for ins in sorted(opt_instances, key=lambda x: float(x["area"]), reverse=True):
        iid = int(ins["opt_instance_index"]) + 1
        m = ins["mask"] > 0
        label[m & (label == 0)] = iid
    return label


def summarize_instances(opt_instances):
    return [{"opt_instance_index": int(x["opt_instance_index"]), "source_instance_indices": list(map(int, x.get("source_instance_indices", []))), "centroid": [float(x["centroid"][0]), float(x["centroid"][1])], "bbox": list(map(int, x["bbox"])), "area": int(x["area"])} for x in opt_instances]


def build_grid(raw_instances, union_mask, device):
    cent = np.array([x["centroid"] for x in raw_instances], dtype=np.float64)
    origin, e_u, e_v, angle_deg = s10.estimate_tray_axes(cent, union_mask)
    local_uv = s10.world_to_local(cent, origin, e_u, e_v)
    col_centers_u, _, _ = s10.fit_equal_spacing_1d_edge_aware(local_uv[:, 0], s10.NUM_COLS, device=device)
    row_centers_v, _, _ = s10.fit_equal_spacing_1d_edge_aware(local_uv[:, 1], s10.NUM_ROWS, device=device)
    hole_centers = s10.build_theory_centers(row_centers_v, col_centers_u, origin, e_u, e_v)
    avg_col_spacing, avg_row_spacing, avg_spacing = s10.compute_average_spacing(hole_centers, s10.NUM_ROWS, s10.NUM_COLS)
    match_radius = s10.choose_match_radius(avg_col_spacing, avg_row_spacing)
    adj, dmat = s10.compute_adjacency(cent, s10.GRAPH_MAX_NEIGHBOR_DIST, s10.GRAPH_KNN, device=device)
    har = s10.assign_instances_to_candidate_holes(raw_instances, hole_centers, match_radius, s10.NUM_ROWS, s10.NUM_COLS, device=device)
    return {"cent": cent, "hole_centers": hole_centers, "avg_col_spacing": avg_col_spacing, "avg_row_spacing": avg_row_spacing, "avg_spacing": avg_spacing, "match_radius": match_radius, "adj": adj, "dmat": dmat, "har": har, "angle_deg": angle_deg}


def make_outputs(mode_name, prefix, image, h, w, opt_raw, hole_centers, manual_empty_ids, avg_spacing, match_radius, args, opt_scores=None, use_gnn_score=False):
    opt, post_debug = postprocess_gnn_instances(opt_raw, hole_centers, manual_empty_ids, avg_spacing, min_noise_area=args.min_noise_area, small_fragment_area=args.small_fragment_area, merge_dist_ratio=args.merge_dist_ratio, excessive_iou=args.excessive_iou, empty_radius_ratio=args.empty_radius_ratio)
    matches, empty_holes, assign_debug = strict_assign_holes(opt, hole_centers, manual_empty_ids, match_radius, avg_spacing, opt_hole_scores=opt_scores, use_gnn_score=use_gnn_score)
    grades, grade_stats, grade_thr = assign_grades(opt, matches)
    if grade_stats["matched_total"] + len(empty_holes) != len(hole_centers):
        raise RuntimeError("final count != hole count")
    mask_path = Path(str(prefix) + "_instance_mask.png")
    overlay_path = Path(str(prefix) + "_overlay.png")
    hole_path = Path(str(prefix) + "_holes.png")
    grade_path = Path(str(prefix) + "_grade.png")
    Image.fromarray(build_label_mask(opt, h, w)).save(mask_path)
    draw_overlay(image, opt, matches, empty_holes, overlay_path, f"{mode_name}: overlay")
    draw_holes(image, hole_centers, matches, empty_holes, hole_path, f"{mode_name}: hole presence")
    draw_grade(image, opt, grades, empty_holes, grade_path, f"{mode_name}: seedling grade")
    return {"opt_instances": opt, "matches": matches, "empty_holes": empty_holes, "grades": grades, "grade_stats": grade_stats, "grade_thresholds": grade_thr, "post_debug": post_debug, "assign_debug": assign_debug, "instance_mask": str(mask_path), "overlay": str(overlay_path), "hole_map": str(hole_path), "grade_map": str(grade_path)}


def process_one(json_path, device, out_dir, args):
    """Process one sample and output only the final After-GNN results."""
    stem = json_path.stem
    stem_base = stem.replace("_instances_with_polygon", "")

    data = s10.load_json(json_path)
    img_path = s10.find_image_for_json(json_path, data)
    if img_path is None:
        raise FileNotFoundError(f"No image found for {json_path}")

    image = s10.read_image(img_path)
    h, w = image.shape[:2]

    raw_instances, raw_masks, union_mask, raw_source = load_raw_instances_and_masks(json_path, data, h, w)
    if len(raw_instances) < 5:
        raise RuntimeError(f"too few raw instances: {len(raw_instances)}")

    grid = build_grid(raw_instances, union_mask, device)
    hole_centers = grid["hole_centers"]
    avg_spacing = grid["avg_spacing"]
    match_radius = grid["match_radius"]
    manual_empty_ids = get_manual_empty_hole_ids(stem_base, hole_centers, avg_spacing)

    # GNN merge and hole-assignment probability estimation
    groups_g, hole_logits = gnn_merge_and_assign(
        instances=raw_instances,
        masks=raw_masks,
        har=grid["har"],
        adj=grid["adj"],
        dmat=grid["dmat"],
        holes=hole_centers,
        ih=h,
        iw=w,
        device=device,
        n_iter=args.gnn_iter,
    )

    opt_g_raw = []
    group_g_indices = []
    for _, gidx in groups_g.items():
        merged = s10.merge_instance_group(gidx, raw_instances, raw_masks)
        if merged is not None:
            opt_g_raw.append(merged)
            group_g_indices.append(list(map(int, gidx)))

    for i, ins in enumerate(opt_g_raw):
        ins["opt_instance_index"] = i

    # Aggregate node-level GNN hole logits into optimized-instance-level scores
    if hole_logits is not None and np.asarray(hole_logits).size > 0 and group_g_indices:
        logits = np.asarray(hole_logits, dtype=np.float32)
        opt_scores = np.zeros((len(group_g_indices), len(hole_centers)), dtype=np.float32)
        for oi, gidx in enumerate(group_g_indices):
            idx = np.asarray(gidx, dtype=np.int32)
            idx = idx[(idx >= 0) & (idx < logits.shape[0])]
            if idx.size > 0:
                sub = logits[idx]
                opt_scores[oi] = np.max(sub, axis=0) if sub.ndim == 2 else sub
    else:
        opt_scores = np.zeros((len(opt_g_raw), len(hole_centers)), dtype=np.float32)

    after = make_outputs(
        "After GNN",
        out_dir / f"{stem}_after_gnn",
        image,
        h,
        w,
        opt_g_raw,
        hole_centers,
        manual_empty_ids,
        avg_spacing,
        match_radius,
        args,
        opt_scores=opt_scores,
        use_gnn_score=True,
    )

    result_json = out_dir / f"{stem}_after_gnn_result.json"
    save_json({
        "sample": stem,
        "image_path": str(img_path),
        "raw_instance_source": raw_source,
        "image_size": {"height": h, "width": w},
        "grid": {
            "num_rows": s10.NUM_ROWS,
            "num_cols": s10.NUM_COLS,
            "total_holes": len(hole_centers),
            "avg_col_spacing": grid["avg_col_spacing"],
            "avg_row_spacing": grid["avg_row_spacing"],
            "avg_spacing": avg_spacing,
            "match_radius": match_radius,
            "manual_empty_hole_ids": manual_empty_ids,
            "angle_deg": grid["angle_deg"],
        },
        "parameters": vars(args),
        "after_gnn": {
            "n_instances": len(after["opt_instances"]),
            "n_matches": len(after["matches"]),
            "n_empty_holes": len(after["empty_holes"]),
            "grade_statistics": after["grade_stats"],
            "grade_thresholds": after["grade_thresholds"],
            "count_check": after["grade_stats"]["matched_total"] + len(after["empty_holes"]),
            "instances": summarize_instances(after["opt_instances"]),
            "seedling_grades": after["grades"],
            "matches": after["matches"],
            "empty_holes": after["empty_holes"],
            "postprocess_debug": after["post_debug"],
            "assignment_debug": after["assign_debug"],
            "instance_mask": after["instance_mask"],
            "overlay": after["overlay"],
            "hole_map": after["hole_map"],
            "grade_map": after["grade_map"],
        },
    }, result_json)

    return {
        "image_stem": stem,
        "status": "ok",
        "raw_instances": len(raw_instances),
        "manual_empty": len(manual_empty_ids),
        "after_instances": len(after["opt_instances"]),
        "after_matches": len(after["matches"]),
        "after_empty_holes": len(after["empty_holes"]),
        "after_grade_I": after["grade_stats"]["I"],
        "after_grade_II": after["grade_stats"]["II"],
        "after_grade_III": after["grade_stats"]["III"],
        "after_total_check": after["grade_stats"]["matched_total"] + len(after["empty_holes"]),
        "after_overlay": after["overlay"],
        "after_hole_map": after["hole_map"],
        "after_grade_map": after["grade_map"],
        "result_json": str(result_json),
    }


def clear_outputs(out_dir):
    for pat in ["*_after_gnn_*.png", "*_after_gnn_result.json", "after_gnn_summary.csv"]:
        for p in out_dir.glob(pat):
            if p.is_file():
                p.unlink()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="/ssd/home/jiangyiyu/unit3code/mydata/output/step10_graph_merge_after_gnn_strict200")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--min-noise-area", type=int, default=120)
    parser.add_argument("--small-fragment-area", type=int, default=350)
    parser.add_argument("--merge-dist-ratio", type=float, default=0.45)
    parser.add_argument("--excessive-iou", type=float, default=0.40)
    parser.add_argument("--empty-radius-ratio", type=float, default=0.48)
    parser.add_argument("--gnn-iter", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        clear_outputs(out_dir)

    device = torch.device(args.device)
    json_files = sorted(s10.INPUT_DIR.glob("*.json"))

    if args.images:
        req = {Path(x).stem.replace(".json", "") for x in args.images}
        json_files = [p for p in json_files if p.stem in req or p.stem.replace("_instances_with_polygon", "") in req]

    print(f"device={device}")
    print(f"input json={len(json_files)}")
    print(f"output={out_dir}")

    rows = []
    for jp in json_files:
        try:
            row = process_one(jp, device, out_dir, args)
            rows.append(row)
            print(
                f"[OK] {jp.stem}: "
                f"seedling={row['after_matches']} "
                f"empty={row['after_empty_holes']} "
                f"total={row['after_total_check']} "
                f"manual_empty={row['manual_empty']}"
            )
        except Exception as e:
            row = {
                "image_stem": jp.stem,
                "status": "failed:" + str(e).replace(",", ";")[:220],
                "raw_instances": "",
                "manual_empty": "",
                "after_instances": "",
                "after_matches": "",
                "after_empty_holes": "",
                "after_grade_I": "",
                "after_grade_II": "",
                "after_grade_III": "",
                "after_total_check": "",
                "after_overlay": "",
                "after_hole_map": "",
                "after_grade_map": "",
                "result_json": "",
            }
            rows.append(row)
            print(f"[FAIL] {jp.stem}: {e}")

    csv_path = out_dir / "after_gnn_summary.csv"
    fieldnames = [
        "image_stem", "status", "raw_instances", "manual_empty",
        "after_instances", "after_matches", "after_empty_holes",
        "after_grade_I", "after_grade_II", "after_grade_III", "after_total_check",
        "after_overlay", "after_hole_map", "after_grade_map", "result_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print("--- SUMMARY ---")
    print(f"ok={len(ok)}/{len(rows)}")
    if ok:
        print(f"After total check: {sorted(set(int(r['after_total_check']) for r in ok))}")
        print(f"After: seedlings={sum(int(r['after_matches']) for r in ok)}, empty={sum(int(r['after_empty_holes']) for r in ok)}")
        print(f"Grade I:   {sum(int(r['after_grade_I']) for r in ok)}")
        print(f"Grade II:  {sum(int(r['after_grade_II']) for r in ok)}")
        print(f"Grade III: {sum(int(r['after_grade_III']) for r in ok)}")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
