#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# =========================
# 路径配置
# =========================
STEP7_DIR = "/ssd/home/jiangyiyu/unit3code/mydata/output/step7_stitch"
IMG_DIR = "/ssd/home/jiangyiyu/unit3code/mydata/org_img_label"

OUT_ROOT = "/ssd/home/jiangyiyu/unit3code/mydata/output/step8_to_labelme"
OUT_OVERLAY_DIR = os.path.join(OUT_ROOT, "overlay")
OUT_JSON_WITH_POLY_DIR = os.path.join(OUT_ROOT, "json_with_polygon")
OUT_LABELME_DIR = os.path.join(OUT_ROOT, "labelme")
OUT_LOG_PATH = os.path.join(OUT_ROOT, "check_log.json")

os.makedirs(OUT_OVERLAY_DIR, exist_ok=True)
os.makedirs(OUT_JSON_WITH_POLY_DIR, exist_ok=True)
os.makedirs(OUT_LABELME_DIR, exist_ok=True)

# =========================
# 参数
# =========================
RNG = np.random.default_rng(42)
COLOR_TABLE = RNG.integers(50, 255, size=(65536, 3), dtype=np.uint8)
COLOR_TABLE[0] = [0, 0, 0]

MIN_POLYGON_AREA = 5
POLY_EPS_RATIO = 0.002  # 轮廓近似精度


# =========================
# 工具函数
# =========================
def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_image_by_stem(stem: str, img_dir: str) -> Optional[str]:
    exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".tif", ".tiff"]
    for ext in exts:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def get_color(i: int) -> Tuple[int, int, int]:
    c = COLOR_TABLE[i % len(COLOR_TABLE)]
    return int(c[2]), int(c[1]), int(c[0])  # BGR


def draw_bbox(img: np.ndarray, bbox: List[float], color, text: Optional[str] = None) -> np.ndarray:
    x, y, w, h = bbox
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))
    if w <= 0 or h <= 0:
        return img

    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 2)

    if text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y1 = max(0, y - th - 4)
        y2 = y
        x2 = min(img.shape[1] - 1, x + tw + 4)
        cv2.rectangle(img, (x, y1), (x2, y2), color, -1)
        cv2.putText(
            img, text, (x + 2, y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def draw_polygon(img: np.ndarray, points: List[List[float]], color, text: Optional[str] = None) -> np.ndarray:
    if len(points) < 3:
        return img

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    if text:
        x = int(points[0][0])
        y = int(points[0][1])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y1 = max(0, y - th - 4)
        y2 = y
        x2 = min(img.shape[1] - 1, x + tw + 4)
        cv2.rectangle(img, (x, y1), (x2, y2), color, -1)
        cv2.putText(
            img, text, (x + 2, y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def bbox_from_binary(mask: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def centroid_from_binary(mask: np.ndarray) -> List[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [round(float(xs.mean()), 2), round(float(ys.mean()), 2)]


def mask_to_polygons(binary_mask: np.ndarray,
                     min_area: int = MIN_POLYGON_AREA,
                     eps_ratio: float = POLY_EPS_RATIO) -> List[List[List[float]]]:
    """
    从单个实例二值mask提取 polygon 列表。
    返回格式：
      [
        [[x1,y1],[x2,y2],...],   # contour 1
        [[x1,y1],[x2,y2],...]    # contour 2
      ]
    """
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        eps = eps_ratio * peri
        approx = cv2.approxPolyDP(c, eps, True)

        pts = approx.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            polygons.append([[float(x), float(y)] for x, y in pts])

    return polygons


def instance_mask_to_polygon_map(mask_img: np.ndarray) -> Dict[int, Dict]:
    """
    从实例 mask 图中按 instance_id 提取：
    - polygon
    - bbox
    - centroid
    - area
    """
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]

    ids = np.unique(mask_img)
    ids = ids[ids > 0]

    results = {}
    for iid in ids:
        iid = int(iid)
        binary = (mask_img == iid)
        area = int(binary.sum())
        if area <= 0:
            continue

        polygons = mask_to_polygons(binary)
        bbox = bbox_from_binary(binary)
        centroid = centroid_from_binary(binary)

        results[iid] = {
            "area": area,
            "bbox": bbox,
            "centroid": centroid,
            "polygons": polygons
        }

    return results


def polygon_to_labelme_shape(points: List[List[float]], label: str, group_id=None) -> Dict:
    return {
        "label": label,
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": group_id,
        "description": "",
        "shape_type": "polygon",
        "flags": {}
    }


def bbox_to_labelme_shape(bbox: List[float], label: str, group_id=None) -> Dict:
    x, y, w, h = bbox
    return {
        "label": label,
        "points": [
            [float(x), float(y)],
            [float(x + w), float(y + h)]
        ],
        "group_id": group_id,
        "description": "",
        "shape_type": "rectangle",
        "flags": {}
    }


def convert_to_labelme_json(step7_json_with_poly: Dict, image_filename: str, img_h: int, img_w: int) -> Dict:
    """
    优先使用 polygon 写成 Labelme。
    若某实例没有 polygon，则退化为 bbox。
    """
    shapes = []
    for ins in step7_json_with_poly.get("instances", []):
        instance_id = ins.get("instance_id", None)
        label = ins.get("category_name", "object")

        polygons = ins.get("polygon", [])
        if isinstance(polygons, list) and len(polygons) > 0:
            for poly in polygons:
                if isinstance(poly, list) and len(poly) >= 3:
                    shapes.append(polygon_to_labelme_shape(poly, label, group_id=instance_id))
        elif "bbox" in ins and isinstance(ins["bbox"], list) and len(ins["bbox"]) == 4:
            shapes.append(bbox_to_labelme_shape(ins["bbox"], label, group_id=instance_id))

    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w),
    }
    return labelme_json


def build_overlay(img: np.ndarray, json_data: Dict) -> np.ndarray:
    """
    将 polygon / bbox 覆盖到原图上。
    """
    vis = img.copy()
    instances = json_data.get("instances", [])

    for idx, ins in enumerate(instances, start=1):
        iid = int(ins.get("instance_id", idx))
        color = get_color(iid)
        text = f"id:{iid}"

        polygons = ins.get("polygon", [])
        drew_poly = False
        if isinstance(polygons, list) and len(polygons) > 0:
            for poly in polygons:
                if isinstance(poly, list) and len(poly) >= 3:
                    vis = draw_polygon(vis, poly, color, text=text)
                    drew_poly = True
                    text = None  # 避免一个实例多个polygon重复写字

        if not drew_poly and "bbox" in ins and isinstance(ins["bbox"], list) and len(ins["bbox"]) == 4:
            vis = draw_bbox(vis, ins["bbox"], color, text=text)

    return vis


# =========================
# 主逻辑
# =========================
def main():
    json_files = sorted(glob.glob(os.path.join(STEP7_DIR, "*_instances.json")))
    if len(json_files) == 0:
        print(f"在 {STEP7_DIR} 中未找到 *_instances.json")
        return

    logs = []

    for jf in json_files:
        base = os.path.basename(jf)
        stem = base.replace("_instances.json", "")

        mask_path = os.path.join(STEP7_DIR, f"{stem}_instance_mask.png")
        img_path = find_image_by_stem(stem, IMG_DIR)

        record = {
            "stem": stem,
            "json_path": jf,
            "mask_path": mask_path,
            "image_path": img_path,
        }

        if img_path is None:
            record["status"] = "image_not_found"
            logs.append(record)
            print(f"[WARN] {stem}: image not found")
            continue

        if not os.path.exists(mask_path):
            record["status"] = "mask_not_found"
            logs.append(record)
            print(f"[WARN] {stem}: instance mask not found")
            continue

        img = cv2.imread(img_path)
        if img is None:
            record["status"] = "image_read_failed"
            logs.append(record)
            print(f"[WARN] {stem}: image read failed")
            continue

        mask_img = np.array(Image.open(mask_path))
        if mask_img.ndim == 3:
            mask_img = mask_img[:, :, 0]

        h, w = img.shape[:2]
        mh, mw = mask_img.shape[:2]

        src_json = load_json(jf)

        json_w = src_json.get("imageWidth", None)
        json_h = src_json.get("imageHeight", None)

        size_match_img_json = (json_w == w and json_h == h)
        size_match_img_mask = (mw == w and mh == h)

        # 从 instance_mask 中提取 polygon / bbox / centroid / area
        polygon_map = instance_mask_to_polygon_map(mask_img)

        # 将 polygon 信息写回 JSON
        new_json = dict(src_json)
        new_instances = []
        for ins in src_json.get("instances", []):
            iid = int(ins.get("instance_id", -1))
            new_ins = dict(ins)

            if iid in polygon_map:
                new_ins["bbox"] = polygon_map[iid]["bbox"]
                new_ins["centroid"] = polygon_map[iid]["centroid"]
                new_ins["area"] = polygon_map[iid]["area"]
                new_ins["polygon"] = polygon_map[iid]["polygons"]
            else:
                # 若实例ID在mask中不存在，则保留原信息，但polygon为空
                new_ins["polygon"] = []

            new_instances.append(new_ins)

        new_json["imagePath"] = os.path.basename(img_path)
        new_json["imageWidth"] = int(w)
        new_json["imageHeight"] = int(h)
        new_json["instances"] = new_instances

        json_with_poly_path = os.path.join(OUT_JSON_WITH_POLY_DIR, f"{stem}_instances_with_polygon.json")
        save_json(json_with_poly_path, new_json)

        # 转成 Labelme 格式
        labelme_json = convert_to_labelme_json(new_json, os.path.basename(img_path), h, w)
        labelme_json_path = os.path.join(OUT_LABELME_DIR, f"{stem}.json")
        save_json(labelme_json_path, labelme_json)

        # 复制原图到 labelme 目录，方便直接打开
        target_img_path = os.path.join(OUT_LABELME_DIR, os.path.basename(img_path))
        if not os.path.exists(target_img_path):
            shutil.copy2(img_path, target_img_path)

        # overlay 可视化
        overlay = build_overlay(img, new_json)
        overlay_path = os.path.join(OUT_OVERLAY_DIR, f"{stem}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

        record.update({
            "status": "ok",
            "overlay_path": overlay_path,
            "json_with_polygon_path": json_with_poly_path,
            "labelme_json_path": labelme_json_path,
            "actual_width": int(w),
            "actual_height": int(h),
            "mask_width": int(mw),
            "mask_height": int(mh),
            "json_imageWidth": json_w,
            "json_imageHeight": json_h,
            "size_match_img_json": size_match_img_json,
            "size_match_img_mask": size_match_img_mask,
            "n_instances_json": len(src_json.get("instances", [])),
            "n_instances_mask": len(polygon_map),
        })
        logs.append(record)

        print(
            f"[OK] {stem} | "
            f"json-img match={size_match_img_json} | "
            f"mask-img match={size_match_img_mask} | "
            f"json_instances={len(src_json.get('instances', []))} | "
            f"mask_instances={len(polygon_map)}"
        )

    save_json(OUT_LOG_PATH, {"results": logs})

    print("\n处理完成：")
    print(f"overlay 输出目录: {OUT_OVERLAY_DIR}")
    print(f"带 polygon 的 JSON 输出目录: {OUT_JSON_WITH_POLY_DIR}")
    print(f"Labelme JSON 输出目录: {OUT_LABELME_DIR}")
    print(f"日志文件: {OUT_LOG_PATH}")


if __name__ == "__main__":
    main()