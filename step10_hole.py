#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from dataclasses import dataclass


# =========================================================
# 路径配置
# =========================================================
INPUT_DIR = Path("/ssd/home/jiangyiyu/unit3code/mydata/output/step8_to_labelme/json_with_polygon")
OUTPUT_DIR = Path("/ssd/home/jiangyiyu/unit3code/mydata/output/step10_graph_merge_area_rank")
ANN_DIR = Path("/ssd/home/jiangyiyu/unit3code/mydata/labelme_data_ann")
STEP7_MASK_DIR = Path("/ssd/home/jiangyiyu/unit3code/mydata/output/step7_stitch")  # 真值 mask 目录
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# =========================================================
# 参数
# =========================================================
NUM_COLS = 10
NUM_ROWS = 20
TRUE_SINGLE_PLANT_COUNT = None

MATCH_RADIUS_MIN = 60
MATCH_RADIUS_MAX = 90
MATCH_RADIUS_DEFAULT = 72
EDGE_RADIUS_BONUS = 5

GRAPH_MAX_NEIGHBOR_DIST = 85.0
GRAPH_KNN = 6

SAME_HOLE_CENTER_DIST_MAX = 60.0
OVERLAP_IOU_MIN = 0.08
OVERLAP_DILATED_IOU_MIN = 0.18
MASK_DILATE_KERNEL = 5
MASK_DILATE_ITERS = 1
SMALL_FRAGMENT_AREA_RATIO = 0.45

HOLE_MATCH_MIN_OVERLAP_PIXELS = 20
HOLE_MATCH_MIN_OVERLAP_RATIO_TO_INSTANCE = 0.08
HOLE_MATCH_MIN_OVERLAP_RATIO_TO_HOLE = 0.05

VIS_LONG_SIDE = 800
MANUAL_EMPTY_MATCH_MAX_DIST_RATIO = 0.45  # 以孔间距为尺度：人工空穴点 -> 理论孔位的最大允许距离比例
RANK_MIN_AREA_ABS = 30                   # 面积排序时，过滤杂点的最小像素面积（绝对值）
RANK_MIN_AREA_PCTL = 10                  # 面积排序时，过滤杂点的面积分位数阈值（在已归属孔穴实例内计算）
STRICT_REQUIRE_MANUAL_EMPTY_JSON = False # 若为 True：找不到人工空穴 json 直接报错停止

# BGR
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
GRAY = (180, 180, 180)
SKY_BLUE = (235, 206, 135)   # 天蓝色方框（BGR近似 sky blue）
AREA_SMALL_RED = (0, 0, 255) # 红色方框

RED_RING_RADIUS = 7
RED_RING_THICKNESS = 2
BLUE_DOT_RADIUS = 5
BLACK_MATCH_THICKNESS = 2
EMPTY_HOLE_RADIUS = 8
AREA_BOX_THICKNESS = 3


# =========================================================
# 工具
# =========================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clear_output_dir(output_dir: Path):
    """
    清除之前运行历史：删除输出目录下所有文件（保留目录本身）。
    """
    ensure_dir(output_dir)
    for p in output_dir.glob("*"):
        if p.is_file():
            p.unlink()


def read_image(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    return img


def find_image_for_json(json_path: Path, data: dict) -> Optional[Path]:
    if data.get("imagePath", ""):
        p = json_path.parent / data["imagePath"]
        if p.exists():
            return p
    stem = json_path.stem
    base_stem = stem.replace("_instances_with_polygon", "")

    # 1) 同目录（常规 labelme）
    for ext in IMG_EXTS:
        for s in [stem, base_stem]:
            p = json_path.parent / f"{s}{ext}"
            if p.exists():
                return p

    # 2) 人工标注目录（当前你的 jpg 都在这里）
    for ext in IMG_EXTS:
        p = ANN_DIR / f"{base_stem}{ext}"
        if p.exists():
            return p
    return None


def find_ann_json_for_stem(stem: str, ann_dir: Path) -> Optional[Path]:
    """
    在人工标注目录中查找对应的 labelme json（支持若干常见命名）。
    """
    candidates = [
        ann_dir / f"{stem}.json",
        ann_dir / f"{stem}.JSON",
        ann_dir / f"{stem}_ann.json",
        ann_dir / f"{stem}_emptyholes.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def shape_to_center_xy(shape: dict) -> Optional[np.ndarray]:
    """
    将 labelme shape 转为一个中心点（用于标注空穴的位置）。
    支持: point/circle/rectangle/polygon。
    """
    pts = shape.get("points", [])
    st = shape.get("shape_type", "polygon")
    if not pts:
        return None

    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None

    if st == "point":
        return arr[0]
    if st in ["circle", "rectangle"]:
        # labelme circle: two points (center, edge). rectangle: two corners.
        return arr.mean(axis=0)
    if st == "polygon":
        return arr.mean(axis=0)
    # fallback
    return arr.mean(axis=0)


def load_manual_empty_hole_points(ann_json_path: Path) -> List[np.ndarray]:
    """
    读取人工标注 json 中的“空穴点”。默认将所有 shape 都视作空穴标注，
    如果你有更严格的 label 规范，可以在这里加过滤条件。
    """
    data = load_json(ann_json_path)
    shapes = data.get("shapes", [])
    pts = []
    for s in shapes:
        c = shape_to_center_xy(s)
        if c is None:
            continue
        pts.append(c)
    return pts


def map_manual_empty_points_to_holes(manual_pts_xy: List[np.ndarray], hole_centers, avg_spacing: float) -> set:
    """
    将人工标注空穴点映射到最近的理论孔位 hole_index（带距离门限，避免误配）。
    """
    if not manual_pts_xy:
        return set()
    hole_pts = np.asarray([h["center"] for h in hole_centers], dtype=np.float64)
    max_dist = max(20.0, float(avg_spacing) * MANUAL_EMPTY_MATCH_MAX_DIST_RATIO) if avg_spacing > 0 else 45.0

    used = set()
    mapped = set()
    for p in manual_pts_xy:
        d = np.linalg.norm(hole_pts - p[None, :], axis=1)
        idx = int(np.argmin(d))
        if float(d[idx]) <= max_dist and idx not in used:
            used.add(idx)
            mapped.add(int(hole_centers[idx]["hole_index"]))
    return mapped


def resize_long_side(img, target_long_side=800):
    h, w = img.shape[:2]
    scale = target_long_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale


def scale_point(pt, scale):
    return int(round(pt[0] * scale)), int(round(pt[1] * scale))


def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


# =========================================================
# Labelme -> mask
# =========================================================
def polygon_to_mask(points: List[List[float]], h: int, w: int):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.array(points, dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygons_to_mask(polygons, h: int, w: int) -> np.ndarray:
    """
    支持 step8 导出的 polygon 结构：通常是 [ [ [x,y], ... ], [ [x,y], ... ], ... ]
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygons is None:
        return mask
    # 兼容单个 contour: [ [x,y], ... ]
    if len(polygons) >= 3 and isinstance(polygons[0], (list, tuple)) and len(polygons[0]) == 2 and isinstance(polygons[0][0], (int, float)):
        return polygon_to_mask(polygons, h, w)

    for contour in polygons:
        if contour is None:
            continue
        if len(contour) < 3:
            continue
        m = polygon_to_mask(contour, h, w)
        mask = np.maximum(mask, m)
    return mask


def rectangle_to_mask(points: List[List[float]], h: int, w: int):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 2:
        return mask
    p1 = np.array(points[0], dtype=np.float32)
    p2 = np.array(points[1], dtype=np.float32)
    x1, y1 = np.minimum(p1, p2)
    x2, y2 = np.maximum(p1, p2)

    x1 = int(np.clip(round(x1), 0, w - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
    return mask


def shape_to_mask(shape: dict, h: int, w: int):
    st = shape.get("shape_type", "polygon")
    pts = shape.get("points", [])
    if st == "polygon":
        return polygon_to_mask(pts, h, w)
    elif st == "rectangle":
        return rectangle_to_mask(pts, h, w)
    return np.zeros((h, w), dtype=np.uint8)


def mask_to_props(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    area = int(len(xs))
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    return {
        "centroid": [cx, cy],
        "bbox": [x1, y1, x2, y2],
        "area": area
    }


# =========================================================
# 坐标系与网格恢复
# =========================================================
def world_to_local(points_xy: np.ndarray, origin: np.ndarray, e_u: np.ndarray, e_v: np.ndarray):
    d = points_xy - origin[None, :]
    u = d @ e_u
    v = d @ e_v
    return np.stack([u, v], axis=1)


def local_to_world(points_uv: np.ndarray, origin: np.ndarray, e_u: np.ndarray, e_v: np.ndarray):
    pts_uv = np.asarray(points_uv, dtype=np.float64)
    pts_xy = origin[None, :] + pts_uv[:, 0:1] * e_u[None, :] + pts_uv[:, 1:2] * e_v[None, :]
    return pts_xy


def pca_axes(points_xy: np.ndarray):
    pts = np.asarray(points_xy, dtype=np.float64)
    origin = pts.mean(axis=0)
    X = pts - origin
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    e1 = eigvecs[:, 0]
    e2 = eigvecs[:, 1]

    if e1[0] < 0:
        e1 = -e1
    if np.cross(np.append(e1, 0), np.append(e2, 0))[2] < 0:
        e2 = -e2

    angle_deg = float(np.degrees(np.arctan2(e1[1], e1[0])))
    return origin, e1, e2, angle_deg


def min_area_rect_axes_from_mask(union_mask: np.ndarray):
    ys, xs = np.where(union_mask > 0)
    if len(xs) < 10:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), _, _ = rect

    box = cv2.boxPoints(rect).astype(np.float64)
    edges = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        edges.append((length, vec))

    edges.sort(key=lambda x: x[0], reverse=True)
    long_vec = edges[0][1]
    long_vec = long_vec / (np.linalg.norm(long_vec) + 1e-12)
    short_vec = np.array([-long_vec[1], long_vec[0]], dtype=np.float64)
    origin = np.array([cx, cy], dtype=np.float64)
    return origin, long_vec, short_vec


def estimate_tray_axes(centroids_xy: np.ndarray, union_mask: np.ndarray):
    rect_res = min_area_rect_axes_from_mask(union_mask)
    if rect_res is not None:
        origin, a1, a2 = rect_res
        angle_deg = float(np.degrees(np.arctan2(a1[1], a1[0])))
    else:
        origin, a1, a2, angle_deg = pca_axes(centroids_xy)

    local_12 = world_to_local(centroids_xy, origin, a1, a2)
    span1 = local_12[:, 0].max() - local_12[:, 0].min()
    span2 = local_12[:, 1].max() - local_12[:, 1].min()

    if span1 >= span2:
        e_v = a1
        e_u = a2
    else:
        e_v = a2
        e_u = a1

    if e_u[0] < 0:
        e_u = -e_u
    if e_v[1] < 0:
        e_v = -e_v

    return origin, e_u, e_v, angle_deg


def kmeans_1d_torch(values: np.ndarray, k: int, device: torch.device, max_iter: int = 40):
    x = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1)
    init = np.percentile(values, np.linspace(0, 100, k)).astype(np.float32)
    centers = torch.as_tensor(init, dtype=torch.float32, device=device)

    for _ in range(max_iter):
        d2 = (x[:, None] - centers[None, :]) ** 2
        labels = torch.argmin(d2, dim=1)
        new_centers = centers.clone()
        for i in range(k):
            idx = labels == i
            if torch.any(idx):
                new_centers[i] = x[idx].mean()
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=0):
            centers = new_centers
            break
        centers = new_centers

    centers, _ = torch.sort(centers)
    return centers.detach().cpu().numpy()


def fit_equal_spacing_1d_edge_aware(values: np.ndarray, n_lines: int, device: torch.device):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    init_centers = kmeans_1d_torch(values, n_lines, device=device)

    idx = np.arange(n_lines, dtype=np.float64)
    ds_init, _ = np.polyfit(idx, init_centers, 1)
    if ds_init < 0:
        ds_init = -ds_init

    vmin = float(values.min())
    vmax = float(values.max())
    s0_from_min = vmin
    s0_from_max = vmax - (n_lines - 1) * ds_init
    s0_anchor = 0.5 * (s0_from_min + s0_from_max)

    ds_candidates = np.linspace(ds_init * 0.94, ds_init * 1.06, 21, dtype=np.float32)
    s0_candidates = np.linspace(s0_anchor - 0.25 * ds_init, s0_anchor + 0.25 * ds_init, 21, dtype=np.float32)

    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    idx_t = torch.arange(n_lines, dtype=torch.float32, device=device)

    best_cost = None
    best_s0 = None
    best_ds = None

    for ds in ds_candidates:
        ds_t = torch.tensor(ds, dtype=torch.float32, device=device)
        for s0 in s0_candidates:
            s0_t = torch.tensor(s0, dtype=torch.float32, device=device)
            centers = s0_t + idx_t * ds_t
            d2 = (x[:, None] - centers[None, :]) ** 2
            nearest = torch.min(d2, dim=1).values
            fit_err = float(nearest.mean().item())

            c0 = float(centers[0].item())
            cN = float(centers[-1].item())
            edge_cover = (abs(vmin - c0) + abs(vmax - cN)) / max(float(ds), 1e-6)

            cost = fit_err + 0.8 * edge_cover
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_s0 = float(s0)
                best_ds = float(ds)

    centers = best_s0 + np.arange(n_lines, dtype=np.float64) * best_ds
    return centers, float(best_ds), float(best_s0)


def build_theory_centers(row_centers_v, col_centers_u, origin, e_u, e_v):
    pts_uv = np.array([[u, v] for v in row_centers_v for u in col_centers_u], dtype=np.float64)
    pts_xy = local_to_world(pts_uv, origin, e_u, e_v)

    hole_centers = []
    idx = 0
    for r in range(len(row_centers_v)):
        for c in range(len(col_centers_u)):
            hole_centers.append({
                "hole_index": idx,
                "row": r,
                "col": c,
                "center": [float(pts_xy[idx, 0]), float(pts_xy[idx, 1])]
            })
            idx += 1
    return hole_centers


def compute_average_spacing(hole_centers, num_rows, num_cols):
    center_map = {(h["row"], h["col"]): np.array(h["center"], dtype=np.float64) for h in hole_centers}
    col_spacings = []
    row_spacings = []

    for r in range(num_rows):
        for c in range(num_cols - 1):
            col_spacings.append(float(np.linalg.norm(center_map[(r, c + 1)] - center_map[(r, c)])))
    for r in range(num_rows - 1):
        for c in range(num_cols):
            row_spacings.append(float(np.linalg.norm(center_map[(r + 1, c)] - center_map[(r, c)])))

    avg_col = float(np.mean(col_spacings)) if col_spacings else 0.0
    avg_row = float(np.mean(row_spacings)) if row_spacings else 0.0
    avg_all = float(np.mean([v for v in [avg_col, avg_row] if v > 0])) if (avg_col > 0 or avg_row > 0) else 0.0
    return avg_col, avg_row, avg_all


def choose_match_radius(avg_col_spacing, avg_row_spacing):
    vals = [v for v in [avg_col_spacing, avg_row_spacing] if v > 0]
    if not vals:
        return MATCH_RADIUS_DEFAULT
    r = int(round(min(vals) * 0.40))
    return max(MATCH_RADIUS_MIN, min(MATCH_RADIUS_MAX, r))


# =========================================================
# 图建模与合并
# =========================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def compute_adjacency(centroids_xy: np.ndarray, max_dist: float, knn: int, device: torch.device):
    pts = torch.as_tensor(centroids_xy, dtype=torch.float32, device=device)
    dmat_t = torch.cdist(pts, pts, p=2)
    dmat = dmat_t.detach().cpu().numpy()

    sigma = max(20.0, max_dist / 2.0)
    adj = np.zeros((len(centroids_xy), len(centroids_xy)), dtype=np.float32)

    order = np.argsort(dmat, axis=1)
    for i in range(len(centroids_xy)):
        used = 0
        for j in order[i]:
            if i == j:
                continue
            if dmat[i, j] > max_dist:
                continue
            adj[i, j] = math.exp(-float(dmat[i, j]) / sigma)
            used += 1
            if used >= knn:
                break

    adj = np.maximum(adj, adj.T)
    return adj, dmat


def assign_instances_to_candidate_holes(instances, hole_centers, base_radius, num_rows, num_cols, device: torch.device):
    inst_pts = np.array([ins["centroid"] for ins in instances], dtype=np.float32)
    hole_pts = np.array([h["center"] for h in hole_centers], dtype=np.float32)

    inst_t = torch.as_tensor(inst_pts, dtype=torch.float32, device=device)
    hole_t = torch.as_tensor(hole_pts, dtype=torch.float32, device=device)
    dmat = torch.cdist(inst_t, hole_t, p=2).detach().cpu().numpy()

    radii = np.full(len(hole_centers), base_radius, dtype=np.float32)
    for hi, h in enumerate(hole_centers):
        if h["row"] in [0, num_rows - 1] or h["col"] in [0, num_cols - 1]:
            radii[hi] = min(MATCH_RADIUS_MAX, base_radius + EDGE_RADIUS_BONUS)

    rows = []
    for ii in range(len(instances)):
        valid = np.where(dmat[ii] <= radii)[0]
        if len(valid) == 0:
            rows.append({
                "instance_index": ii,
                "best_hole_index": None,
                "best_distance": None,
                "second_hole_index": None,
                "second_distance": None,
                "candidate_count": 0,
            })
            continue

        dist_valid = dmat[ii, valid]
        sort_idx = np.argsort(dist_valid)
        valid_sorted = valid[sort_idx]
        dist_sorted = dist_valid[sort_idx]

        rows.append({
            "instance_index": ii,
            "best_hole_index": int(valid_sorted[0]),
            "best_distance": float(dist_sorted[0]),
            "second_hole_index": int(valid_sorted[1]) if len(valid_sorted) > 1 else None,
            "second_distance": float(dist_sorted[1]) if len(valid_sorted) > 1 else None,
            "candidate_count": int(len(valid_sorted)),
        })
    return rows


def bboxes_may_touch(b1, b2, margin=6):
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    return not (ax2 < bx1 - margin or bx2 < ax1 - margin or ay2 < by1 - margin or by2 < ay1 - margin)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def compute_dilated_iou(mask1: np.ndarray, mask2: np.ndarray, kernel_size=5, iters=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    d1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=iters)
    d2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=iters)
    inter = np.logical_and(d1 > 0, d2 > 0).sum()
    union = np.logical_or(d1 > 0, d2 > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def graph_rule_merge(instances, instance_masks, hole_assign_rows, adj, dmat):
    n = len(instances)
    uf = UnionFind(n)
    pair_rows = []
    bboxes = [ins["bbox"] for ins in instances]

    ii, jj = np.where(np.triu(adj > 0, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ai = hole_assign_rows[i]
        aj = hole_assign_rows[j]
        hi = ai["best_hole_index"]
        hj = aj["best_hole_index"]
        dij = float(dmat[i, j])

        if bboxes_may_touch(bboxes[i], bboxes[j], margin=6):
            iou = compute_mask_iou(instance_masks[i], instance_masks[j])
            dil_iou = compute_dilated_iou(instance_masks[i], instance_masks[j], kernel_size=MASK_DILATE_KERNEL, iters=MASK_DILATE_ITERS)
        else:
            iou = 0.0
            dil_iou = 0.0

        area_i = float(instances[i]["area"])
        area_j = float(instances[j]["area"])
        area_ratio = min(area_i, area_j) / max(max(area_i, area_j), 1e-6)

        same_hole = (hi is not None and hj is not None and hi == hj)
        should_merge = False
        reason = []

        if same_hole:
            if dij <= SAME_HOLE_CENTER_DIST_MAX:
                should_merge = True
                reason.append("same_hole_and_close_centroid")
            if iou >= OVERLAP_IOU_MIN or dil_iou >= OVERLAP_DILATED_IOU_MIN:
                should_merge = True
                reason.append("same_hole_and_mask_overlap")
            if area_ratio <= SMALL_FRAGMENT_AREA_RATIO and dij <= SAME_HOLE_CENTER_DIST_MAX:
                should_merge = True
                reason.append("same_hole_small_fragment")

        if hi is None and hj is None:
            if dij <= 25.0 and (iou > 0 or dil_iou >= 0.22):
                should_merge = True
                reason.append("unassigned_but_close_and_overlap")

        pair_rows.append({
            "instance_i": i,
            "instance_j": j,
            "best_hole_i": hi,
            "best_hole_j": hj,
            "centroid_dist": dij,
            "edge_weight": float(adj[i, j]),
            "mask_iou": iou,
            "dilated_iou": dil_iou,
            "area_ratio_small_over_large": area_ratio,
            "merge_decision": int(should_merge),
            "merge_reason": "|".join(reason)
        })

        if should_merge:
            uf.union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return groups, pair_rows


def merge_instance_group(group_indices, instances, instance_masks):
    merged_mask = np.zeros_like(instance_masks[0], dtype=np.uint8)
    source_ids = []
    labels = []
    for idx in group_indices:
        merged_mask = np.maximum(merged_mask, instance_masks[idx].astype(np.uint8))
        source_ids.append(int(instances[idx]["instance_index"]))
        labels.append(instances[idx].get("label", ""))

    props = mask_to_props(merged_mask)
    if props is None:
        return None

    return {
        "source_instance_indices": source_ids,
        "label_list": labels,
        "mask": merged_mask,
        "centroid": props["centroid"],
        "bbox": props["bbox"],
        "area": props["area"]
    }


# =========================================================
# 孔位匹配（去掉细茎判断）
# =========================================================
def make_hole_search_mask(h, w, center_xy, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(round(center_xy[0]))
    cy = int(round(center_xy[1]))
    cv2.circle(mask, (cx, cy), int(radius), 1, thickness=-1)
    return mask


def compute_instance_hole_relation(instance_mask, instance_centroid, hole_center, hole_search_mask):
    inst_mask = (instance_mask > 0).astype(np.uint8)
    hole_mask = (hole_search_mask > 0).astype(np.uint8)

    centroid_dist = float(np.linalg.norm(np.array(instance_centroid, dtype=np.float64) - np.array(hole_center, dtype=np.float64)))
    overlap = np.logical_and(inst_mask > 0, hole_mask > 0)
    overlap_pixels = int(overlap.sum())

    inst_area = int(inst_mask.sum())
    hole_area = int(hole_mask.sum())

    overlap_ratio_to_instance = float(overlap_pixels / inst_area) if inst_area > 0 else 0.0
    overlap_ratio_to_hole_region = float(overlap_pixels / hole_area) if hole_area > 0 else 0.0

    return {
        "centroid_dist": centroid_dist,
        "overlap_pixels": overlap_pixels,
        "overlap_ratio_to_instance": overlap_ratio_to_instance,
        "overlap_ratio_to_hole_region": overlap_ratio_to_hole_region
    }


def compute_instance_hole_relation_roi(instance_mask: np.ndarray, instance_bbox, instance_centroid, hole_center, radius: float):
    """
    仅在 hole 圆与实例 bbox 的交叠 ROI 内计算 overlap，避免构造整图 hole mask（显著提速）。
    """
    x1, y1, x2, y2 = instance_bbox
    h, w = instance_mask.shape[:2]
    cx = int(round(hole_center[0]))
    cy = int(round(hole_center[1]))
    r = int(round(radius))

    roi_x1 = max(0, min(w - 1, min(x1, cx - r)))
    roi_y1 = max(0, min(h - 1, min(y1, cy - r)))
    roi_x2 = max(0, min(w - 1, max(x2, cx + r)))
    roi_y2 = max(0, min(h - 1, max(y2, cy + r)))
    if roi_x2 < roi_x1 or roi_y2 < roi_y1:
        return {
            "centroid_dist": float(np.linalg.norm(np.array(instance_centroid, dtype=np.float64) - np.array(hole_center, dtype=np.float64))),
            "overlap_pixels": 0,
            "overlap_ratio_to_instance": 0.0,
            "overlap_ratio_to_hole_region": 0.0
        }

    inst_roi = (instance_mask[roi_y1:roi_y2 + 1, roi_x1:roi_x2 + 1] > 0).astype(np.uint8)
    inst_area = int((instance_mask > 0).sum())

    # 在 ROI 内画圆
    hole_roi = np.zeros_like(inst_roi, dtype=np.uint8)
    cv2.circle(hole_roi, (cx - roi_x1, cy - roi_y1), r, 1, thickness=-1)

    overlap_pixels = int(np.logical_and(inst_roi > 0, hole_roi > 0).sum())
    hole_area = int(hole_roi.sum())
    overlap_ratio_to_instance = float(overlap_pixels / inst_area) if inst_area > 0 else 0.0
    overlap_ratio_to_hole_region = float(overlap_pixels / hole_area) if hole_area > 0 else 0.0
    centroid_dist = float(np.linalg.norm(np.array(instance_centroid, dtype=np.float64) - np.array(hole_center, dtype=np.float64)))

    return {
        "centroid_dist": centroid_dist,
        "overlap_pixels": overlap_pixels,
        "overlap_ratio_to_instance": overlap_ratio_to_instance,
        "overlap_ratio_to_hole_region": overlap_ratio_to_hole_region
    }


def instance_matches_hole(relation, search_radius):
    if relation["centroid_dist"] <= search_radius:
        return True
    if relation["overlap_pixels"] >= HOLE_MATCH_MIN_OVERLAP_PIXELS:
        return True
    if relation["overlap_ratio_to_instance"] >= HOLE_MATCH_MIN_OVERLAP_RATIO_TO_INSTANCE:
        return True
    if relation["overlap_ratio_to_hole_region"] >= HOLE_MATCH_MIN_OVERLAP_RATIO_TO_HOLE:
        return True
    return False


def compute_match_score_torch(rel_rows, search_radius: float, device: torch.device):
    cd = torch.as_tensor([r["centroid_dist"] for r in rel_rows], dtype=torch.float32, device=device)
    op = torch.as_tensor([r["overlap_pixels"] for r in rel_rows], dtype=torch.float32, device=device)
    ori = torch.as_tensor([r["overlap_ratio_to_instance"] for r in rel_rows], dtype=torch.float32, device=device)
    orh = torch.as_tensor([r["overlap_ratio_to_hole_region"] for r in rel_rows], dtype=torch.float32, device=device)

    dist_score = torch.clamp(1.0 - cd / max(search_radius, 1e-6), min=0.0)
    overlap_score = 2.0 * ori + 1.5 * orh + 0.002 * op
    total_score = overlap_score + dist_score
    return total_score.detach().cpu().numpy()


def greedy_match_optimized_instances_with_mask_overlap(
    opt_instances,
    hole_centers,
    base_radius,
    num_rows,
    num_cols,
    image_h,
    image_w,
    device: torch.device,
):
    candidates = []
    hole_radii = []
    for h in hole_centers:
        radius = base_radius
        if h["row"] in [0, num_rows - 1] or h["col"] in [0, num_cols - 1]:
            radius = min(MATCH_RADIUS_MAX, base_radius + EDGE_RADIUS_BONUS)
        hole_radii.append(radius)

    for hi, h in enumerate(hole_centers):
        radius = hole_radii[hi]
        hole_center = h["center"]

        rel_rows = []
        valid_indices = []

        for ii, ins in enumerate(opt_instances):
            # 先用距离做快速筛选，降低 ROI overlap 计算次数
            cd = float(np.linalg.norm(np.array(ins["centroid"], dtype=np.float64) - np.array(hole_center, dtype=np.float64)))
            if cd > (radius + 12.0):
                continue
            relation = compute_instance_hole_relation_roi(ins["mask"], ins["bbox"], ins["centroid"], hole_center, radius=radius)
            if instance_matches_hole(relation, radius):
                rel_rows.append(relation)
                valid_indices.append(ii)

        if not rel_rows:
            continue

        scores = compute_match_score_torch(rel_rows, search_radius=radius, device=device)

        for k, ii in enumerate(valid_indices):
            relation = rel_rows[k]
            candidates.append({
                "score": float(scores[k]),
                "hole_index_local": hi,
                "instance_index_local": ii,
                "distance": relation["centroid_dist"],
                "overlap_pixels": relation["overlap_pixels"],
                "overlap_ratio_to_instance": relation["overlap_ratio_to_instance"],
                "overlap_ratio_to_hole_region": relation["overlap_ratio_to_hole_region"],
            })

    candidates.sort(key=lambda x: (-x["score"], x["distance"]))

    used_holes = set()
    used_instances = set()
    matches = []

    for cand in candidates:
        hi = cand["hole_index_local"]
        ii = cand["instance_index_local"]
        if hi in used_holes or ii in used_instances:
            continue

        used_holes.add(hi)
        used_instances.add(ii)

        h = hole_centers[hi]
        ins = opt_instances[ii]
        matches.append({
            "hole_index": int(h["hole_index"]),
            "row": int(h["row"]),
            "col": int(h["col"]),
            "theory_center_x": float(h["center"][0]),
            "theory_center_y": float(h["center"][1]),
            "opt_instance_index": int(ins["opt_instance_index"]),
            "instance_centroid_x": float(ins["centroid"][0]),
            "instance_centroid_y": float(ins["centroid"][1]),
            "distance": float(cand["distance"]),
            "overlap_pixels": int(cand["overlap_pixels"]),
            "overlap_ratio_to_instance": float(cand["overlap_ratio_to_instance"]),
            "overlap_ratio_to_hole_region": float(cand["overlap_ratio_to_hole_region"]),
            "match_score": float(cand["score"]),
            "source_instance_indices": ",".join(map(str, ins["source_instance_indices"]))
        })

    matched_holes = {m["hole_index"] for m in matches}
    empty_holes = []
    for h in hole_centers:
        if h["hole_index"] not in matched_holes:
            empty_holes.append({
                "hole_index": int(h["hole_index"]),
                "row": int(h["row"]),
                "col": int(h["col"]),
                "theory_center_x": float(h["center"][0]),
                "theory_center_y": float(h["center"][1]),
            })

    return matches, empty_holes


# =========================================================
# 面积排序与可视化
# =========================================================
def compute_area_rank_groups(opt_instances):
    if len(opt_instances) == 0:
        return set(), set(), []

    areas = np.array([ins["area"] for ins in opt_instances], dtype=np.float64)
    order_desc = np.argsort(-areas)
    order_asc = np.argsort(areas)

    k = max(1, int(np.ceil(len(opt_instances) * 0.05)))

    top_idx = set(order_desc[:k].tolist())
    bottom_idx = set(order_asc[:k].tolist())

    area_rows = []
    for rank_desc, idx in enumerate(order_desc, start=1):
        ins = opt_instances[idx]
        tag = ""
        if idx in top_idx:
            tag = "top_5_percent"
        elif idx in bottom_idx:
            tag = "bottom_5_percent"

        area_rows.append({
            "opt_instance_index": int(ins["opt_instance_index"]),
            "area": int(ins["area"]),
            "centroid_x": float(ins["centroid"][0]),
            "centroid_y": float(ins["centroid"][1]),
            "bbox_x1": int(ins["bbox"][0]),
            "bbox_y1": int(ins["bbox"][1]),
            "bbox_x2": int(ins["bbox"][2]),
            "bbox_y2": int(ins["bbox"][3]),
            "rank_desc": int(rank_desc),
            "tag": tag,
            "source_instance_indices": ",".join(map(str, ins["source_instance_indices"]))
        })

    return top_idx, bottom_idx, area_rows


def build_labeled_mask(instances_like, h, w, key="mask"):
    label_mask = np.zeros((h, w), dtype=np.int32)
    for i, ins in enumerate(instances_like):
        label_mask[ins[key] > 0] = i + 1
    return label_mask


def colorize_label_mask(label_mask: np.ndarray):
    h, w = label_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_mask)
    for k in ids:
        if k == 0:
            continue
        rng = np.random.default_rng(seed=int(k))
        vis[label_mask == k] = rng.integers(40, 255, size=3, dtype=np.uint8)
    return vis


def draw_tray_holes_presence_figure(image, hole_centers, matches, save_path: Path):
    """
    图1：整盘孔穴位置 + 有苗/无苗（仅依据匹配结果）。
    - 绿圈：有苗（hole 被匹配到实例）
    - 灰圈：无苗（empty hole）
    """
    vis, scale = resize_long_side(image, VIS_LONG_SIDE)
    matched = {m["hole_index"] for m in matches}
    for h in hole_centers:
        c = scale_point(h["center"], scale)
        color = GREEN if h["hole_index"] in matched else GRAY
        cv2.circle(vis, c, 7, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(vis, "GREEN=seedling, GRAY=empty", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 3, cv2.LINE_AA)
    cv2.putText(vis, "GREEN=seedling, GRAY=empty", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 1, cv2.LINE_AA)
    cv2.imwrite(str(save_path), vis)


def draw_ranked_instances_area_boxes_and_empty_holes_figure(
    image,
    opt_instances,
    matches,
    empty_holes,
    save_path: Path,
    alpha: float = 0.35,
):
    """
    图2：按实例面积从大到小对“实例像素”排序并上色（排名确定性）。
    同时标注：
    - top 5% bbox：天蓝色框
    - bottom 5% bbox：红色框
    - empty hole：蓝色圆圈
    """
    if len(opt_instances) == 0:
        vis, _ = resize_long_side(image, VIS_LONG_SIDE)
        for e in empty_holes:
            # 空穴蓝色圆圈
            # 这里 scale 后再画
            pass
        cv2.imwrite(str(save_path), vis)
        return

    # 仅统计“归属孔穴”的实例（matched 的 opt_instance_index）
    matched_opt_ids = {int(m["opt_instance_index"]) for m in matches if str(m.get("opt_instance_index", "")) != ""}
    assigned_instances = [ins for ins in opt_instances if int(ins["opt_instance_index"]) in matched_opt_ids]
    if len(assigned_instances) == 0:
        assigned_instances = opt_instances

    # 过滤过小杂点（只影响“面积排序像素上色”，不影响孔穴归属）
    areas = np.array([float(ins["area"]) for ins in assigned_instances], dtype=np.float64)
    pctl_thr = float(np.percentile(areas, RANK_MIN_AREA_PCTL)) if len(areas) >= 5 else float(areas.min())
    area_thr = max(float(RANK_MIN_AREA_ABS), pctl_thr)
    assigned_instances = [ins for ins in assigned_instances if float(ins["area"]) >= area_thr]
    if len(assigned_instances) == 0:
        # 极端情况：全被过滤，则退回不过滤
        assigned_instances = [ins for ins in opt_instances if int(ins["opt_instance_index"]) in matched_opt_ids] or opt_instances

    # 面积从大到小排序（决定像素“排名顺序”）
    assigned_sorted = sorted(assigned_instances, key=lambda x: float(x["area"]), reverse=True)

    # 生成排名 label_mask：面积最大 = 1，依次递增
    h, w = image.shape[:2]
    label_mask = np.zeros((h, w), dtype=np.int32)
    for rank, ins in enumerate(assigned_sorted, start=1):
        label_mask[ins["mask"] > 0] = rank

    color_mask = colorize_label_mask(label_mask)
    base = image.astype(np.float32)
    cm = color_mask.astype(np.float32)
    overlay = (base * (1.0 - alpha) + cm * alpha).clip(0, 255).astype(np.uint8)

    # 计算 top/bottom 5%（在 assigned_sorted 里）
    n = len(assigned_sorted)
    k = max(1, int(np.ceil(n * 0.05)))
    top_ids = {int(ins["opt_instance_index"]) for ins in assigned_sorted[:k]}
    bottom_ids = {int(ins["opt_instance_index"]) for ins in assigned_sorted[-k:]}

    vis, scale = resize_long_side(overlay, VIS_LONG_SIDE)

    # 画 empty hole 蓝圈
    for e in empty_holes:
        c = scale_point((e["theory_center_x"], e["theory_center_y"]), scale)
        cv2.circle(vis, c, EMPTY_HOLE_RADIUS, BLUE, thickness=2, lineType=cv2.LINE_AA)

    # 画 top/bottom bbox
    for ins in opt_instances:
        oid = int(ins["opt_instance_index"])
        if (oid not in top_ids) and (oid not in bottom_ids):
            continue
        x1, y1, x2, y2 = ins["bbox"]
        p1 = scale_point((x1, y1), scale)
        p2 = scale_point((x2, y2), scale)
        if oid in top_ids:
            cv2.rectangle(vis, p1, p2, SKY_BLUE, thickness=AREA_BOX_THICKNESS, lineType=cv2.LINE_AA)
        if oid in bottom_ids:
            cv2.rectangle(vis, p1, p2, AREA_SMALL_RED, thickness=AREA_BOX_THICKNESS, lineType=cv2.LINE_AA)

    cv2.putText(vis, "Top5%=SKY, Bottom5%=RED, EmptyHole=BLUE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 3, cv2.LINE_AA)
    cv2.putText(vis, "Top5%=SKY, Bottom5%=RED, EmptyHole=BLUE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 1, cv2.LINE_AA)
    cv2.imwrite(str(save_path), vis)


def draw_before_after_mask_compare(raw_label_mask, opt_label_mask, save_path: Path):
    raw_vis = colorize_label_mask(raw_label_mask)
    opt_vis = colorize_label_mask(opt_label_mask)

    h1, w1 = raw_vis.shape[:2]
    h2, w2 = opt_vis.shape[:2]
    H = max(h1, h2)

    if h1 != H:
        raw_vis = cv2.resize(raw_vis, (int(round(w1 * H / h1)), H), interpolation=cv2.INTER_NEAREST)
    if h2 != H:
        opt_vis = cv2.resize(opt_vis, (int(round(w2 * H / h2)), H), interpolation=cv2.INTER_NEAREST)

    gap = np.full((H, 20, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([raw_vis, gap, opt_vis], axis=1)

    cv2.putText(canvas, "Before merge", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 3, cv2.LINE_AA)
    cv2.putText(canvas, "After merge", (raw_vis.shape[1] + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 3, cv2.LINE_AA)
    cv2.imwrite(str(save_path), canvas)


# =========================================================
# 主流程
# =========================================================
def process_one_sample(json_path: Path, output_dir: Path, device: torch.device):
    data = load_json(json_path)

    img_path = find_image_for_json(json_path, data)
    if img_path is None:
        raise FileNotFoundError(f"找不到对应原图: {json_path}")

    image = read_image(img_path)
    h, w = image.shape[:2]

    if "imageHeight" in data and "imageWidth" in data:
        jh = int(data["imageHeight"])
        jw = int(data["imageWidth"])
        if jh != h or jw != w:
            raise RuntimeError(f"图像与 JSON 尺寸不一致: {json_path.name}, json=({jw},{jh}), image=({w},{h})")

    raw_instances = []
    raw_instance_masks = []
    union_mask = np.zeros((h, w), dtype=np.uint8)

    # ── 优先从 step7 真值 mask PNG 直接加载实例，保证无实例缺漏 ──────────────
    stem_base = json_path.stem.replace("_instances_with_polygon", "")
    step7_mask_path = STEP7_MASK_DIR / f"{stem_base}_instance_mask.png"

    if step7_mask_path.exists():
        # 直接从 uint16 instance mask 读取每个实例的像素区域
        inst_mask_img = np.array(Image.open(str(step7_mask_path)))
        if inst_mask_img.ndim == 3:
            inst_mask_img = inst_mask_img[:, :, 0]
        # 若 mask 尺寸与图像不一致则 resize
        if inst_mask_img.shape[0] != h or inst_mask_img.shape[1] != w:
            inst_mask_img = cv2.resize(
                inst_mask_img.astype(np.uint16), (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        unique_ids = np.unique(inst_mask_img)
        unique_ids = unique_ids[unique_ids > 0]
        raw_idx = 0
        for iid in unique_ids:
            binary = (inst_mask_img == iid).astype(np.uint8)
            props = mask_to_props(binary)
            if props is None:
                continue
            raw_instance_masks.append(binary)
            union_mask = np.maximum(union_mask, binary)
            # 尝试从 step8 JSON 获取 label/score 等附加信息
            label = "lettuce"
            if isinstance(data.get("instances", None), list):
                for ins in data["instances"]:
                    if int(ins.get("instance_id", -1)) == int(iid):
                        label = str(ins.get("category_name", "lettuce"))
                        break
            raw_instances.append({
                "instance_index": raw_idx,
                "label": label,
                "shape_type": "mask",
                "centroid": props["centroid"],
                "bbox": props["bbox"],
                "area": props["area"]
            })
            raw_idx += 1
        print(f"  [step7 mask] {stem_base}: loaded {raw_idx} instances from mask PNG")
    else:
        # 回退：使用 step8 导出的 instances（带 polygon）
        print(f"  [WARN] step7 mask not found for {stem_base}, falling back to step8 JSON polygons")
        if isinstance(data.get("instances", None), list) and len(data["instances"]) > 0:
            raw_idx = 0
            for ins in data["instances"]:
                poly = ins.get("polygon", None)
                mask = polygons_to_mask(poly, h, w)
                props = mask_to_props(mask)
                if props is None:
                    continue
                raw_instance_masks.append(mask.astype(np.uint8))
                union_mask = np.maximum(union_mask, mask.astype(np.uint8))
                raw_instances.append({
                    "instance_index": raw_idx,
                    "label": str(ins.get("category_name", "")),
                    "shape_type": "polygon",
                    "centroid": props["centroid"],
                    "bbox": props["bbox"],
                    "area": props["area"]
                })
                raw_idx += 1
        else:
            # 兼容 labelme 原生 shapes
            shapes = data.get("shapes", [])
            raw_idx = 0
            for shape in shapes:
                st = shape.get("shape_type", "polygon")
                if st not in ["polygon", "rectangle"]:
                    continue
                mask = shape_to_mask(shape, h, w)
                props = mask_to_props(mask)
                if props is None:
                    continue
                raw_instance_masks.append(mask.astype(np.uint8))
                union_mask = np.maximum(union_mask, mask.astype(np.uint8))
                raw_instances.append({
                    "instance_index": raw_idx,
                    "label": shape.get("label", ""),
                    "shape_type": st,
                    "centroid": props["centroid"],
                    "bbox": props["bbox"],
                    "area": props["area"]
                })
                raw_idx += 1

    if len(raw_instances) < 5:
        raise RuntimeError(f"有效实例过少，无法处理: {json_path.name}")

    raw_centroids_xy = np.array([ins["centroid"] for ins in raw_instances], dtype=np.float64)

    origin, e_u, e_v, angle_deg = estimate_tray_axes(raw_centroids_xy, union_mask)
    local_uv = world_to_local(raw_centroids_xy, origin, e_u, e_v)
    u_vals = local_uv[:, 0]
    v_vals = local_uv[:, 1]

    col_centers_u, du, u0 = fit_equal_spacing_1d_edge_aware(u_vals, NUM_COLS, device=device)
    row_centers_v, dv, v0 = fit_equal_spacing_1d_edge_aware(v_vals, NUM_ROWS, device=device)

    hole_centers = build_theory_centers(row_centers_v, col_centers_u, origin, e_u, e_v)

    avg_col_spacing, avg_row_spacing, avg_spacing = compute_average_spacing(hole_centers, NUM_ROWS, NUM_COLS)
    match_radius = choose_match_radius(avg_col_spacing, avg_row_spacing)

    adj, dmat = compute_adjacency(raw_centroids_xy, GRAPH_MAX_NEIGHBOR_DIST, GRAPH_KNN, device=device)
    hole_assign_rows = assign_instances_to_candidate_holes(raw_instances, hole_centers, match_radius, NUM_ROWS, NUM_COLS, device=device)

    groups, pair_rows = graph_rule_merge(raw_instances, raw_instance_masks, hole_assign_rows, adj, dmat)

    opt_instances = []
    for _, group_indices in groups.items():
        merged = merge_instance_group(group_indices, raw_instances, raw_instance_masks)
        if merged is not None:
            opt_instances.append(merged)

    for i, ins in enumerate(opt_instances):
        ins["opt_instance_index"] = i

    matches, empty_holes = greedy_match_optimized_instances_with_mask_overlap(
        opt_instances=opt_instances,
        hole_centers=hole_centers,
        base_radius=match_radius,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        image_h=h,
        image_w=w,
        device=device,
    )

    # =======================
    # 使用人工标注空穴校正孔穴归属
    # =======================
    stem_base = json_path.stem.replace("_instances_with_polygon", "")
    ann_json = find_ann_json_for_stem(stem_base, ANN_DIR)
    manual_empty_hole_ids = set()
    if ann_json is None:
        if STRICT_REQUIRE_MANUAL_EMPTY_JSON:
            raise RuntimeError(f"未找到人工标注空穴的 JSON（期望在 {ANN_DIR} 下存在 {stem_base}.json 等命名）")
    else:
        manual_pts = load_manual_empty_hole_points(ann_json)
        manual_empty_hole_ids = map_manual_empty_points_to_holes(manual_pts, hole_centers, avg_spacing=avg_spacing)

    if manual_empty_hole_ids:
        # 强约束：人工标注为空穴的 hole，不允许被匹配到实例
        matches = [m for m in matches if int(m["hole_index"]) not in manual_empty_hole_ids]

    matched_holes = {int(m["hole_index"]) for m in matches}
    empty_holes = []
    for hh in hole_centers:
        hid = int(hh["hole_index"])
        if (hid not in matched_holes) or (hid in manual_empty_hole_ids):
            empty_holes.append({
                "hole_index": hid,
                "row": int(hh["row"]),
                "col": int(hh["col"]),
                "theory_center_x": float(hh["center"][0]),
                "theory_center_y": float(hh["center"][1]),
            })

    stem = json_path.stem

    # 只保留两张图输出
    tray_presence_path = output_dir / f"{stem}_tray_holes_presence.png"
    ranked_area_path = output_dir / f"{stem}_ranked_instances_area_boxes_emptyholes.png"

    draw_tray_holes_presence_figure(image=image, hole_centers=hole_centers, matches=matches, save_path=tray_presence_path)
    draw_ranked_instances_area_boxes_and_empty_holes_figure(
        image=image,
        opt_instances=opt_instances,
        matches=matches,
        empty_holes=empty_holes,
        save_path=ranked_area_path,
        alpha=0.35,
    )

    print(f"[完成] {stem}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 清除之前运行历史（目录内旧文件全部删除），然后开始新一轮输出（只保留两张图）
    clear_output_dir(OUTPUT_DIR)
    json_files = sorted(INPUT_DIR.glob("*.json"))
    if len(json_files) == 0:
        print(f"未找到 JSON 文件: {INPUT_DIR}")
        return

    print(f"使用设备: {device}")
    print(f"共找到 {len(json_files)} 个 JSON，开始处理...")
    for json_path in json_files:
        try:
            process_one_sample(json_path, OUTPUT_DIR, device=device)
        except Exception as e:
            print(f"[失败] {json_path.name}: {e}")

    print("全部处理结束。")


if __name__ == "__main__":
    main()