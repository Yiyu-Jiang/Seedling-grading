import sys
sys.path.insert(0, '/ssd/home/jiangyiyu/unit3code/mydata/coco')

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# 清除旧注册
for name in ['lettuce_train', 'lettuce_val']:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

# 注册数据集
register_coco_instances(
    "lettuce_train", 
    {}, 
    "/ssd/home/jiangyiyu/unit3code/mydata/standardized_step1_half/coco/instances_train.json",
    "/ssd/home/jiangyiyu/unit3code/mydata/standardized_step1_half/merged/images"
)
register_coco_instances(
    "lettuce_val", 
    {}, 
    "/ssd/home/jiangyiyu/unit3code/mydata/standardized_step1_half/coco/instances_val.json",
    "/ssd/home/jiangyiyu/unit3code/mydata/standardized_step1_half/merged/images"
)

print("✅ 数据集已注册: lettuce_train, lettuce_val")
