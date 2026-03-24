#!/usr/bin/env python3
"""Register the private lettuce dataset (COCO instances) to Detectron2."""

from __future__ import annotations

import os
import sys
from pathlib import Path

print("[REGISTER] Starting lettuce dataset registration...")

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    print("[REGISTER] Detectron2 imports successful")
except Exception as e:
    print(f"[REGISTER] ERROR importing detectron2: {e}")
    sys.exit(1)

DEFAULT_ROOT = Path("/ssd/home/jiangyiyu/unit3code/mydata")
DATA_ROOT = Path(os.environ.get("LETTUCE_DATA_ROOT", str(DEFAULT_ROOT / "coco"))).expanduser()
IMG_ROOT = Path(os.environ.get("LETTUCE_IMAGE_ROOT", str(DEFAULT_ROOT / "standardized_step1_half/merged/images"))).expanduser()
IMG_DIR = str(IMG_ROOT)
TRAIN_JSON = str(DATA_ROOT / "instances_train.json")
VAL_JSON = str(DATA_ROOT / "instances_val.json")

print(f"[REGISTER] IMG_DIR: {IMG_DIR}")
print(f"[REGISTER] TRAIN_JSON: {TRAIN_JSON}")
print(f"[REGISTER] VAL_JSON: {VAL_JSON}")

# 验证文件存在
if not os.path.exists(IMG_DIR):
    print(f"[REGISTER] ERROR: IMG_DIR does not exist: {IMG_DIR}")
    sys.exit(1)

if not os.path.exists(TRAIN_JSON):
    print(f"[REGISTER] ERROR: TRAIN_JSON does not exist: {TRAIN_JSON}")
    sys.exit(1)

if not os.path.exists(VAL_JSON):
    print(f"[REGISTER] ERROR: VAL_JSON does not exist: {VAL_JSON}")
    sys.exit(1)

print(f"[REGISTER] All files exist")

def register_all() -> None:
    print("[REGISTER] Removing existing datasets...")
    for name in ["lettuce_train", "lettuce_val"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
            print(f"[REGISTER] Removed {name}")
    
    print("[REGISTER] Registering lettuce_train...")
    register_coco_instances("lettuce_train", {}, TRAIN_JSON, IMG_DIR)
    print("[REGISTER] Registered lettuce_train")
    
    print("[REGISTER] Registering lettuce_val...")
    register_coco_instances("lettuce_val", {}, VAL_JSON, IMG_DIR)
    print("[REGISTER] Registered lettuce_val")

    print(f"[REGISTER] Registered datasets: {DatasetCatalog.list()}")

try:
    register_all()
    print("[REGISTER] Registration successful!")
except Exception as e:
    print(f"[REGISTER] ERROR during registration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
