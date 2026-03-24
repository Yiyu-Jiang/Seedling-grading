#!/usr/bin/env python3
"""Convert per-image LabelMe polygon annotations to a single COCO instances json.

This dataset looks like:
  images: /ssd/home/jiangyiyu/unit3code/mydata/image_split-fu/*.jpg
  labelme: /ssd/home/jiangyiyu/unit3code/mydata/split_jeson-fu/*.json

Each json contains:
  - imagePath
  - imageHeight / imageWidth
  - shapes: [{label, points, shape_type: polygon}, ...]

We output COCO instances format:
  {images: [...], annotations: [...], categories: [...]}

Notes:
- Only polygon shapes are converted.
- Coordinates are kept as-is (float); COCO accepts float.
- A one-label-per-instance assumption is used.

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _polygon_area(poly_xy: List[Tuple[float, float]]) -> float:
    # Shoelace formula; returns absolute area.
    if len(poly_xy) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly_xy, poly_xy[1:] + poly_xy[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _polygon_bbox(poly_xy: List[Tuple[float, float]]) -> List[float]:
    xs = [p[0] for p in poly_xy]
    ys = [p[1] for p in poly_xy]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


@dataclass
class CocoIdGen:
    image_id: int = 1
    ann_id: int = 1


def convert(
    image_dir: Path,
    label_dir: Path,
    output_json: Path,
    categories: List[str] | None = None,
    strict: bool = False,
) -> Dict[str, Any]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        raise FileNotFoundError(f"No json files found under: {label_dir}")

    # If categories not specified, infer from labels across all json.
    if categories is None:
        labels = set()
        for p in label_files:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for shp in data.get("shapes", []):
                lab = shp.get("label")
                if lab:
                    labels.add(lab)
        categories = sorted(labels)

    cat_to_id = {name: i + 1 for i, name in enumerate(categories)}

    out: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cat_to_id[n], "name": n, "supercategory": "object"} for n in categories],
    }

    idgen = CocoIdGen()

    for label_path in label_files:
        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        img_name = data.get("imagePath")
        if not img_name:
            img_name = label_path.with_suffix(".jpg").name

        img_path = image_dir / Path(img_name).name
        if strict and not img_path.exists():
            raise FileNotFoundError(f"Image not found for {label_path}: {img_path}")

        w = int(data.get("imageWidth") or 0)
        h = int(data.get("imageHeight") or 0)
        if (w <= 0 or h <= 0) and strict:
            raise ValueError(f"Missing imageWidth/imageHeight in {label_path}")

        image_id = idgen.image_id
        idgen.image_id += 1

        out["images"].append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        shapes = data.get("shapes", [])
        for shp in shapes:
            if shp.get("shape_type") not in ("polygon", None):
                # labelme default is polygon; ignore others
                continue

            label = shp.get("label")
            if label not in cat_to_id:
                if strict:
                    raise ValueError(f"Unknown label '{label}' in {label_path}")
                # auto-add new category
                cat_to_id[label] = max(cat_to_id.values(), default=0) + 1
                out["categories"].append({"id": cat_to_id[label], "name": label, "supercategory": "object"})

            pts = shp.get("points")
            if not pts or len(pts) < 3:
                continue

            poly_xy = [(float(x), float(y)) for x, y in pts]
            area = _polygon_area(poly_xy)
            bbox = _polygon_bbox(poly_xy)
            segmentation = [[coord for xy in poly_xy for coord in xy]]

            out["annotations"].append(
                {
                    "id": idgen.ann_id,
                    "image_id": image_id,
                    "category_id": cat_to_id[label],
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            idgen.ann_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--categories", type=str, default="")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    out = convert(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_json=args.output,
        categories=cats,
        strict=args.strict,
    )

    print(
        f"Wrote: {args.output}\n"
        f"  images: {len(out['images'])}\n"
        f"  annotations: {len(out['annotations'])}\n"
        f"  categories: {len(out['categories'])} -> {[c['name'] for c in out['categories']]}"
    )


if __name__ == "__main__":
    main()
