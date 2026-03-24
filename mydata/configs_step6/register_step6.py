#!/usr/bin/env python3
"""Register step5_augmented lettuce datasets to Detectron2."""
from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT  = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step5_augmented')
IMG_DIR    = str(DATA_ROOT / 'images')
TRAIN_JSON = str(DATA_ROOT / 'instances_train.json')
VAL_JSON   = str(DATA_ROOT / 'instances_val.json')

# Allow env-var overrides
IMG_DIR    = os.environ.get('LETTUCE_IMAGE_ROOT', IMG_DIR)
TRAIN_JSON = os.environ.get('LETTUCE_TRAIN_JSON', TRAIN_JSON)
VAL_JSON   = os.environ.get('LETTUCE_VAL_JSON',   VAL_JSON)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def register_all():
    for name in ('lettuce_train', 'lettuce_val'):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
    register_coco_instances('lettuce_train', {}, TRAIN_JSON, IMG_DIR)
    register_coco_instances('lettuce_val',   {}, VAL_JSON,   IMG_DIR)
    print(f'[register] lettuce_train <- {TRAIN_JSON}')
    print(f'[register] lettuce_val   <- {VAL_JSON}')
    print(f'[register] images        <- {IMG_DIR}')

register_all()
