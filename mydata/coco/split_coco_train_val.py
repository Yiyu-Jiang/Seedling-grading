#!/usr/bin/env python3
"""Split a COCO instances json into train/val by image.

Keeps categories unchanged; re-ids images & annotations for cleanliness.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def split_coco(
    coco: Dict[str, Any],
    val_ratio: float,
    seed: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    images: List[Dict[str, Any]] = coco.get("images", [])
    annos: List[Dict[str, Any]] = coco.get("annotations", [])
    cats = coco.get("categories", [])

    rng = random.Random(seed)
    img_ids = [img["id"] for img in images]
    rng.shuffle(img_ids)

    n_val = int(round(len(img_ids) * val_ratio))
    val_set = set(img_ids[:n_val])

    def build_subset(keep_ids: set[int]) -> Dict[str, Any]:
        old_to_new_img = {}
        subset_images = []
        next_img_id = 1
        for img in images:
            if img["id"] in keep_ids:
                old_to_new_img[img["id"]] = next_img_id
                new_img = dict(img)
                new_img["id"] = next_img_id
                subset_images.append(new_img)
                next_img_id += 1

        subset_annos = []
        next_ann_id = 1
        for ann in annos:
            if ann["image_id"] in keep_ids:
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = old_to_new_img[ann["image_id"]]
                subset_annos.append(new_ann)
                next_ann_id += 1

        return {"images": subset_images, "annotations": subset_annos, "categories": cats}

    train_keep = set(img_ids[n_val:])
    val_keep = val_set

    return build_subset(train_keep), build_subset(val_keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--train-out", type=Path, required=True)
    ap.add_argument("--val-out", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    train, val = split_coco(coco, val_ratio=args.val_ratio, seed=args.seed)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False)

    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    with args.val_out.open("w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False)

    print(
        f"train: images={len(train['images'])} annos={len(train['annotations'])}\n"
        f"val:   images={len(val['images'])} annos={len(val['annotations'])}\n"
        f"categories={len(train['categories'])}"
    )


if __name__ == "__main__":
    main()
