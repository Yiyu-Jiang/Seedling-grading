#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step9_tray_grid_equal_spacing_edge_aware_v2.py

功能：
1. 读取 Labelme JSON 与对应原图
2. 将 polygon / rectangle 转为原图同尺寸 mask
3. 提取实例 centroid / bbox / area，并合并得到 union mask
4. 基于 union mask / 质心估计穴盘主方向
5. 在局部坐标系中拟合“严格等间距”的 10 列、20 行规则孔位中心
6. 使用边界锚定，避免最外侧行/列向内收缩
7. 生成理论穴位中心、网格线、四角点
8. 在每个红色网格交点附近 20–30 px 范围内判断是否有实例质心
9. 若检测到种苗，绘制蓝色圆点或蓝色圆环；空穴不绘制蓝色标记
10. 输出 PNG 和 JSON

关键约束：
- 所有几何计算基于原图分辨率
- 不允许 resize 后参与几何计算
- mask 与图像尺寸必须完全一致
- 图像与标注不能错位
- 10×20 行列线间隔严格一致
"""

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


# =========================================================
# 路径
# =========================================================
INPUT_DIR = Path("/ssd/home/jiangyiyu/unit3code/mydata/output/step8_to_labelme/labelme")
OUTPUT_DIR = Path("/ssd/home/jiangyiyu/unit3code/mydata/output/step9_exp")

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# =========================================================
# 参数
# =========================================================
NUM_COLS = 10
NUM_ROWS = 20

# 最新要求：交点附近 20–30 px 范围内检测到种苗
SEARCH_RADIUS_MIN = 40
SEARCH_RADIUS_MAX = 50
SEARCH_RADIUS_DEFAULT = 24

# 边缘行/列适当放宽
EDGE_RADIUS_BONUS = 2

VIS_LONG_SIDE = 800

LINE_COLOR = (0, 0, 255)    # 红色 BGR
POINT_COLOR = (255, 0, 0)   # 蓝色 BGR
LINE_THICKNESS = 5

# 蓝色标记样式
DRAW_MARKER_MODE = "ring_dot"   # "dot" / "ring" / "ring_dot"
RING_RADIUS = 7
RING_THICKNESS = 2
DOT_RADIUS = 4


# =========================================================
# 通用工具
# =========================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    for ext in IMG_EXTS:
        p = json_path.parent / f"{stem}{ext}"
        if p.exists():
            return p
    return None


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
    else:
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
# 坐标变换
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


# =========================================================
# 主方向估计
# =========================================================
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

    box = cv2.boxPoints(rect)
    box = np.asarray(box, dtype=np.float64)

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


# =========================================================
# 一维 KMeans（初始化）
# =========================================================
def kmeans_1d(values: np.ndarray, k: int, max_iter: int = 100):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < k:
        raise RuntimeError(f"样本数 {len(values)} 小于聚类数 {k}")

    centers = np.percentile(values, np.linspace(0, 100, k)).astype(np.float64)

    for _ in range(max_iter):
        d2 = (values[:, None] - centers[None, :]) ** 2
        labels = np.argmin(d2, axis=1)

        new_centers = centers.copy()
        for i in range(k):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                new_centers[i] = values[idx].mean()

        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    centers = np.sort(centers)
    d2 = (values[:, None] - centers[None, :]) ** 2
    labels = np.argmin(d2, axis=1)
    return centers, labels


# =========================================================
# 边界锚定的等间距拟合
# =========================================================
def fit_equal_spacing_1d_edge_aware(values: np.ndarray, n_lines: int):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    init_centers, _ = kmeans_1d(values, n_lines)

    idx = np.arange(n_lines, dtype=np.float64)
    ds_init, _ = np.polyfit(idx, init_centers, 1)

    if ds_init < 0:
        ds_init = -ds_init
        init_centers = init_centers[::-1]

    vmin = float(values.min())
    vmax = float(values.max())

    s0_from_min = vmin
    s0_from_max = vmax - (n_lines - 1) * ds_init
    s0_anchor = 0.5 * (s0_from_min + s0_from_max)

    best_cost = None
    best_s0 = None
    best_ds = None

    ds_candidates = np.linspace(ds_init * 0.90, ds_init * 1.10, 41)
    s0_candidates = np.linspace(s0_anchor - 0.5 * ds_init, s0_anchor + 0.5 * ds_init, 61)

    for ds in ds_candidates:
        if ds <= 1e-6:
            continue
        for s0 in s0_candidates:
            centers = s0 + idx * ds

            d2 = (values[:, None] - centers[None, :]) ** 2
            labels = np.argmin(d2, axis=1)
            nearest_dist = np.sqrt(np.min(d2, axis=1))
            fit_err = float(np.mean(nearest_dist ** 2))

            edge_cover = (abs(vmin - centers[0]) + abs(vmax - centers[-1])) / max(ds, 1e-6)

            cnt0 = np.sum(labels == 0)
            cntN = np.sum(labels == (n_lines - 1))
            edge_count_penalty = 0.0
            if cnt0 == 0:
                edge_count_penalty += 5.0
            if cntN == 0:
                edge_count_penalty += 5.0

            d_first = float(np.min(np.abs(values - centers[0])))
            d_last = float(np.min(np.abs(values - centers[-1])))
            edge_near_penalty = (d_first + d_last) / max(ds, 1e-6)

            cost = fit_err + 0.8 * edge_cover + 0.8 * edge_near_penalty + edge_count_penalty

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_s0 = float(s0)
                best_ds = float(ds)

    centers = best_s0 + idx * best_ds
    d2 = (values[:, None] - centers[None, :]) ** 2
    labels = np.argmin(d2, axis=1)

    return centers, labels, float(best_ds), float(best_s0)


# =========================================================
# 直线、交点、网格
# =========================================================
def line_from_points(points_xy: np.ndarray):
    pts = np.asarray(points_xy, dtype=np.float64)
    center = pts.mean(axis=0)

    X = pts - center
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    dir_vec = eigvecs[:, np.argmax(eigvals)]
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)

    n = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float64)
    A, B = n
    C = -(A * center[0] + B * center[1])

    return {
        "point": [float(center[0]), float(center[1])],
        "dir": [float(dir_vec[0]), float(dir_vec[1])],
        "abc": [float(A), float(B), float(C)]
    }


def intersect_lines_abc(l1_abc, l2_abc):
    A1, B1, C1 = l1_abc
    A2, B2, C2 = l2_abc

    M = np.array([[A1, B1], [A2, B2]], dtype=np.float64)
    b = np.array([-C1, -C2], dtype=np.float64)
    if abs(np.linalg.det(M)) < 1e-12:
        raise RuntimeError("两条线近似平行，无法求交")
    p = np.linalg.solve(M, b)
    return [float(p[0]), float(p[1])]


def build_theory_centers(row_centers_v, col_centers_u, origin, e_u, e_v):
    hole_centers = []
    pts_uv = []

    for r, v in enumerate(row_centers_v):
        for c, u in enumerate(col_centers_u):
            pts_uv.append([u, v])

    pts_uv = np.array(pts_uv, dtype=np.float64)
    pts_xy = local_to_world(pts_uv, origin, e_u, e_v)

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


def fit_grid_lines_from_theory_centers(hole_centers, num_rows, num_cols):
    center_map = {(h["row"], h["col"]): np.array(h["center"], dtype=np.float64) for h in hole_centers}

    vertical_lines = []
    for c in range(num_cols):
        pts = np.array([center_map[(r, c)] for r in range(num_rows)], dtype=np.float64)
        line = line_from_points(pts)
        vertical_lines.append({
            "index": c,
            "line": line
        })

    horizontal_lines = []
    for r in range(num_rows):
        pts = np.array([center_map[(r, c)] for c in range(num_cols)], dtype=np.float64)
        line = line_from_points(pts)
        horizontal_lines.append({
            "index": r,
            "line": line
        })

    corners = {
        "TL": intersect_lines_abc(horizontal_lines[0]["line"]["abc"], vertical_lines[0]["line"]["abc"]),
        "TR": intersect_lines_abc(horizontal_lines[0]["line"]["abc"], vertical_lines[-1]["line"]["abc"]),
        "BL": intersect_lines_abc(horizontal_lines[-1]["line"]["abc"], vertical_lines[0]["line"]["abc"]),
        "BR": intersect_lines_abc(horizontal_lines[-1]["line"]["abc"], vertical_lines[-1]["line"]["abc"]),
    }
    return vertical_lines, horizontal_lines, corners


def get_line_segment_for_draw(line_info, p_start, p_end, extend_ratio=0.01):
    p0 = np.array(line_info["point"], dtype=np.float64)
    d = np.array(line_info["dir"], dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)

    a = np.array(p_start, dtype=np.float64)
    b = np.array(p_end, dtype=np.float64)

    ta = np.dot(a - p0, d)
    tb = np.dot(b - p0, d)
    t1, t2 = min(ta, tb), max(ta, tb)

    ext = (t2 - t1) * extend_ratio
    t1 -= ext
    t2 += ext

    q1 = p0 + t1 * d
    q2 = p0 + t2 * d
    return [float(q1[0]), float(q1[1])], [float(q2[0]), float(q2[1])]


# =========================================================
# 间距与匹配
# =========================================================
def compute_average_spacing(hole_centers, num_rows, num_cols):
    center_map = {(h["row"], h["col"]): np.array(h["center"], dtype=np.float64) for h in hole_centers}

    col_spacings = []
    row_spacings = []

    for r in range(num_rows):
        for c in range(num_cols - 1):
            p1 = center_map[(r, c)]
            p2 = center_map[(r, c + 1)]
            col_spacings.append(float(np.linalg.norm(p2 - p1)))

    for r in range(num_rows - 1):
        for c in range(num_cols):
            p1 = center_map[(r, c)]
            p2 = center_map[(r + 1, c)]
            row_spacings.append(float(np.linalg.norm(p2 - p1)))

    avg_col = float(np.mean(col_spacings)) if col_spacings else 0.0
    avg_row = float(np.mean(row_spacings)) if row_spacings else 0.0
    vals = [v for v in [avg_col, avg_row] if v > 0]
    avg_all = float(np.mean(vals)) if vals else 0.0
    return avg_col, avg_row, avg_all


def choose_search_radius(avg_col_spacing, avg_row_spacing):
    vals = [v for v in [avg_col_spacing, avg_row_spacing] if v > 0]
    if not vals:
        return SEARCH_RADIUS_DEFAULT

    # 用平均间距的 0.28 自适应，再限制到 20–30
    r = int(round(min(vals) * 0.28))
    r = max(SEARCH_RADIUS_MIN, min(SEARCH_RADIUS_MAX, r))
    return r


def greedy_match_edge_aware(hole_centers, instances, base_radius, num_rows, num_cols):
    """
    仅用于判定：
    - 红色网格交点附近 20–30 px 范围内若有实例质心，则该穴位“检测到种苗”
    - 空穴不做蓝色标记
    """
    hole_pts = [np.array(h["center"], dtype=np.float64) for h in hole_centers]
    inst_pts = [np.array(ins["centroid"], dtype=np.float64) for ins in instances]

    candidates = []
    for hi, hp in enumerate(hole_pts):
        row = hole_centers[hi]["row"]
        col = hole_centers[hi]["col"]

        radius = base_radius
        if row in [0, num_rows - 1] or col in [0, num_cols - 1]:
            radius = min(SEARCH_RADIUS_MAX, base_radius + EDGE_RADIUS_BONUS)

        for ii, ip in enumerate(inst_pts):
            d = float(np.linalg.norm(hp - ip))
            if d <= radius:
                candidates.append((d, hi, ii))

    candidates.sort(key=lambda x: x[0])

    used_holes = set()
    used_instances = set()
    matches = []

    for d, hi, ii in candidates:
        if hi in used_holes or ii in used_instances:
            continue

        used_holes.add(hi)
        used_instances.add(ii)

        h = hole_centers[hi]
        ins = instances[ii]
        matches.append({
            "hole_index": h["hole_index"],
            "row": h["row"],
            "col": h["col"],
            "theory_center": h["center"],
            "instance_index": ins["instance_index"],
            "instance_centroid": ins["centroid"],
            "distance": d
        })

    matched_holes = {m["hole_index"] for m in matches}
    matched_instances = {m["instance_index"] for m in matches}

    empty_holes = []
    for h in hole_centers:
        if h["hole_index"] not in matched_holes:
            empty_holes.append({
                "hole_index": h["hole_index"],
                "row": h["row"],
                "col": h["col"],
                "theory_center": h["center"]
            })

    unmatched_instances = []
    for ins in instances:
        if ins["instance_index"] not in matched_instances:
            unmatched_instances.append({
                "instance_index": ins["instance_index"],
                "centroid": ins["centroid"],
                "bbox": ins["bbox"],
                "area": ins["area"]
            })

    return matches, empty_holes, unmatched_instances


# =========================================================
# 可视化
# =========================================================
def resize_long_side(img, target_long_side=800):
    h, w = img.shape[:2]
    scale = target_long_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale


def scale_point(pt, scale):
    return int(round(pt[0] * scale)), int(round(pt[1] * scale))


def draw_blue_marker(vis, center_xy, scale):
    c = scale_point(center_xy, scale)

    if DRAW_MARKER_MODE == "dot":
        cv2.circle(
            vis, c, DOT_RADIUS, POINT_COLOR,
            thickness=-1, lineType=cv2.LINE_AA
        )
    elif DRAW_MARKER_MODE == "ring":
        cv2.circle(
            vis, c, RING_RADIUS, POINT_COLOR,
            thickness=RING_THICKNESS, lineType=cv2.LINE_AA
        )
    else:  # ring_dot
        cv2.circle(
            vis, c, RING_RADIUS, POINT_COLOR,
            thickness=RING_THICKNESS, lineType=cv2.LINE_AA
        )
        cv2.circle(
            vis, c, DOT_RADIUS, POINT_COLOR,
            thickness=-1, lineType=cv2.LINE_AA
        )


def draw_result(image, vertical_lines, horizontal_lines, matches, save_path: Path):
    vis, scale = resize_long_side(image, VIS_LONG_SIDE)

    # 竖线
    for item in vertical_lines:
        line = item["line"]
        top_pt = intersect_lines_abc(line["abc"], horizontal_lines[0]["line"]["abc"])
        bottom_pt = intersect_lines_abc(line["abc"], horizontal_lines[-1]["line"]["abc"])
        p1, p2 = get_line_segment_for_draw(line, top_pt, bottom_pt, extend_ratio=0.01)

        cv2.line(
            vis,
            scale_point(p1, scale),
            scale_point(p2, scale),
            LINE_COLOR,
            LINE_THICKNESS,
            lineType=cv2.LINE_AA
        )

    # 横线
    for item in horizontal_lines:
        line = item["line"]
        left_pt = intersect_lines_abc(line["abc"], vertical_lines[0]["line"]["abc"])
        right_pt = intersect_lines_abc(line["abc"], vertical_lines[-1]["line"]["abc"])
        p1, p2 = get_line_segment_for_draw(line, left_pt, right_pt, extend_ratio=0.01)

        cv2.line(
            vis,
            scale_point(p1, scale),
            scale_point(p2, scale),
            LINE_COLOR,
            LINE_THICKNESS,
            lineType=cv2.LINE_AA
        )

    # 只对“检测到种苗”的穴位绘制蓝色标记
    for m in matches:
        draw_blue_marker(vis, m["theory_center"], scale)

    cv2.imwrite(str(save_path), vis)


# =========================================================
# 单样本处理
# =========================================================
def process_one_sample(json_path: Path, output_dir: Path):
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
            raise RuntimeError(
                f"图像与 JSON 尺寸不一致: {json_path.name}, "
                f"json=({jw},{jh}), image=({w},{h})"
            )

    shapes = data.get("shapes", [])
    instances = []
    union_mask = np.zeros((h, w), dtype=np.uint8)

    instance_index = 0
    for shape in shapes:
        shape_type = shape.get("shape_type", "polygon")
        if shape_type not in ["polygon", "rectangle"]:
            continue

        mask = shape_to_mask(shape, h, w)
        props = mask_to_props(mask)
        if props is None:
            continue

        union_mask = np.maximum(union_mask, mask)

        instances.append({
            "instance_index": instance_index,
            "label": shape.get("label", ""),
            "shape_type": shape_type,
            "centroid": props["centroid"],
            "bbox": props["bbox"],
            "area": props["area"]
        })
        instance_index += 1

    if len(instances) < 20:
        raise RuntimeError(f"实例数过少，无法稳定恢复 10x20 网格: {json_path.name}")

    centroids_xy = np.array([ins["centroid"] for ins in instances], dtype=np.float64)

    # 1) 主方向估计
    origin, e_u, e_v, angle_deg = estimate_tray_axes(centroids_xy, union_mask)

    local_uv = world_to_local(centroids_xy, origin, e_u, e_v)
    u_vals = local_uv[:, 0]
    v_vals = local_uv[:, 1]

    # 2) 边界锚定的等间距拟合
    col_centers_u, _, du, u0 = fit_equal_spacing_1d_edge_aware(u_vals, NUM_COLS)
    row_centers_v, _, dv, v0 = fit_equal_spacing_1d_edge_aware(v_vals, NUM_ROWS)

    if du < 0:
        col_centers_u = col_centers_u[::-1]
        du = -du
        u0 = float(col_centers_u[0])

    if dv < 0:
        row_centers_v = row_centers_v[::-1]
        dv = -dv
        v0 = float(row_centers_v[0])

    # 3) 理论穴位中心
    hole_centers = build_theory_centers(
        row_centers_v=row_centers_v,
        col_centers_u=col_centers_u,
        origin=origin,
        e_u=e_u,
        e_v=e_v
    )

    # 4) 从理论中心阵列拟合网格线
    vertical_lines, horizontal_lines, corners = fit_grid_lines_from_theory_centers(
        hole_centers=hole_centers,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS
    )

    # 5) 平均间距与匹配半径
    avg_col_spacing, avg_row_spacing, avg_spacing = compute_average_spacing(
        hole_centers, NUM_ROWS, NUM_COLS
    )
    search_radius = choose_search_radius(avg_col_spacing, avg_row_spacing)

    # 6) 按“交点附近 20–30 px 范围内有实例质心”进行判定
    matches, empty_holes, unmatched_instances = greedy_match_edge_aware(
        hole_centers=hole_centers,
        instances=instances,
        base_radius=search_radius,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS
    )

    # 7) 可视化：空穴不绘制蓝色标记
    stem = json_path.stem
    out_png = output_dir / f"{stem}_step9_match_on_image.png"
    draw_result(
        image=image,
        vertical_lines=vertical_lines,
        horizontal_lines=horizontal_lines,
        matches=matches,
        save_path=out_png
    )

    # 8) 导出 JSON
    out_json = output_dir / f"{stem}_step9_match.json"

    result = {
        "sample_name": stem,
        "image_path": str(img_path),
        "json_path": str(json_path),
        "image_size": {
            "width": w,
            "height": h
        },
        "constraints": {
            "all_geometry_computed_on_original_resolution": True,
            "no_resize_for_geometry": True,
            "mask_size_equals_image_size": True,
            "image_and_annotation_alignment_kept": True,
            "equal_spacing_grid_enforced": True,
            "blue_marker_only_for_detected_holes": True,
            "empty_holes_no_blue_marker": True,
            "hole_detection_radius_range_px": [SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX]
        },
        "instances": instances,
        "union_mask_summary": {
            "area": int((union_mask > 0).sum())
        },
        "local_coordinate_system": {
            "origin_xy": [float(origin[0]), float(origin[1])],
            "u_axis": [float(e_u[0]), float(e_u[1])],
            "v_axis": [float(e_v[0]), float(e_v[1])],
            "main_angle_deg": float(angle_deg)
        },
        "equal_spacing_model": {
            "num_cols": NUM_COLS,
            "num_rows": NUM_ROWS,
            "u0": float(u0),
            "du": float(du),
            "v0": float(v0),
            "dv": float(dv),
            "column_centers_u": [float(x) for x in col_centers_u.tolist()],
            "row_centers_v": [float(x) for x in row_centers_v.tolist()]
        },
        "grid": {
            "hole_centers": hole_centers,
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines
        },
        "corners": corners,
        "matching": {
            "average_col_spacing_px": avg_col_spacing,
            "average_row_spacing_px": avg_row_spacing,
            "average_spacing_px": avg_spacing,
            "base_search_radius_px": search_radius,
            "edge_radius_bonus_px": EDGE_RADIUS_BONUS,
            "matches": matches,
            "empty_holes": empty_holes,
            "unmatched_instances": unmatched_instances
        },
        "outputs": {
            "match_image_png": str(out_png),
            "match_json": str(out_json)
        }
    }

    save_json(result, out_json)

    print(f"[完成] {stem}")
    print(f"  PNG : {out_png}")
    print(f"  JSON: {out_json}")


# =========================================================
# 主程序
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if len(json_files) == 0:
        print(f"未找到 JSON 文件: {INPUT_DIR}")
        return

    print(f"共找到 {len(json_files)} 个 JSON，开始处理...")
    for json_path in json_files:
        try:
            process_one_sample(json_path, OUTPUT_DIR)
        except Exception as e:
            print(f"[失败] {json_path.name}: {e}")

    print("全部处理结束。")


if __name__ == "__main__":
    main()