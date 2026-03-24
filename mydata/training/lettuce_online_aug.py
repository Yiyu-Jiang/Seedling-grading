from __future__ import annotations

import copy
import random
from typing import List

import cv2
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import AugInput


class RandomOcclusion(T.Augmentation):
    def __init__(self, prob: float = 0.5, max_holes: int = 3, max_ratio: float = 0.18):
        super().__init__()
        self.prob = prob
        self.max_holes = max_holes
        self.max_ratio = max_ratio

    def get_transform(self, image):
        h, w = image.shape[:2]
        if random.random() >= self.prob:
            return T.NoOpTransform()

        canvas = np.ones((h, w), dtype=np.uint8) * 255
        holes = random.randint(1, self.max_holes)
        for _ in range(holes):
            hole_w = max(8, int(random.uniform(0.05, self.max_ratio) * w))
            hole_h = max(8, int(random.uniform(0.05, self.max_ratio) * h))
            x0 = random.randint(0, max(0, w - hole_w))
            y0 = random.randint(0, max(0, h - hole_h))
            canvas[y0:y0 + hole_h, x0:x0 + hole_w] = 0
        return T.BlendTransform(src_image=np.zeros((h, w, 3), dtype=np.uint8), src_weight=1.0, dst_weight=0.0) if np.all(canvas == 255) else _MaskBlendTransform(canvas)


class _MaskBlendTransform(T.Transform):
    def __init__(self, mask: np.ndarray):
        super().__init__()
        self.mask = mask

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        out[self.mask == 0] = 0
        return out

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        out = segmentation.copy()
        out[self.mask == 0] = 0
        return out


def build_lettuce_train_augs(cfg) -> List[T.Augmentation]:
    min_size = cfg.INPUT.MIN_SIZE_TRAIN[0] if len(cfg.INPUT.MIN_SIZE_TRAIN) else 512
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    return [
        T.ResizeShortestEdge((min_size,), max_size, sample_style="choice"),
        T.RandomFlip(horizontal=True, vertical=False, prob=0.5),
        T.RandomFlip(horizontal=False, vertical=True, prob=0.3),
        T.RandomRotation(angle=[0, 90, 180, 270], expand=False, sample_style="choice"),
        T.RandomApply(T.RandomExtent(scale_range=(0.9, 1.1), shift_range=(0.08, 0.08)), prob=0.7),
        T.RandomApply(T.RandomBrightness(0.85, 1.15), prob=0.7),
        T.RandomApply(T.RandomContrast(0.85, 1.15), prob=0.7),
        T.RandomApply(T.RandomSaturation(0.9, 1.1), prob=0.3),
        T.RandomApply(T.RandomLighting(0.05), prob=0.2),
        T.RandomApply(T.RandomBlur(kernel_size=3), prob=0.25) if hasattr(T, "RandomBlur") else T.RandomApply(T.ResizeScale(1.0, 1.0, min_size, min_size), prob=0.0),
        RandomOcclusion(prob=0.35, max_holes=3, max_ratio=0.16),
    ]


class LettuceOnlineAugMapper:
    def __init__(self, cfg, is_train: bool = True):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT
        self.recompute_boxes = False
        self.augmentations = build_lettuce_train_augs(cfg) if is_train else []

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = AugInput(image)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        image = aug_input.image

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations", [])
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2], mask_format=self.instance_mask_format)
        instances = utils.filter_empty_instances(instances)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["instances"] = instances
        return dataset_dict