# 步骤7 评估与可视化 — 执行指南

## 当前状态
3个模型正在并行训练中（GPU 2,3,4,5,6,7）：
- **BlendMask-R50** (GPU 2,3): 运行中
- **MaskRCNN-R50** (GPU 4,5): 运行中  
- **TensorMask-R50** (GPU 6,7): 运行中

训练配置：
- 数据集：step5_augmented（1080张增强图，30,571标注）
- 迭代次数：40,000
- 学习率：0.005
- 检查点保存间隔：5000 iter

## 步骤7 执行流程

### 1. 等待训练完成
监控日志：
```bash
tail -f /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step6_blendmask_R50/train.log
tail -f /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step6_maskrcnn_R50/train.log
tail -f /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step6_tensormask_R50/train.log
```

### 2. 运行评估脚本
```bash
cd /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code
/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python step7_evaluate.py
```

### 3. 输出内容

#### 3.1 COCO指标
- **mAP**: 平均精度（所有IoU阈值）
- **AP50**: IoU=0.5时的精度
- **AP75**: IoU=0.75时的精度
- **APs**: 小目标精度
- **APm**: 中等目标精度
- **APl**: 大目标精度
- **Precision**: 精确率
- **Recall**: 召回率
- **F1**: F1分数

#### 3.2 推理可视化
- `BlendMask-R50_inference.jpg` — BlendMask在测试图上的分割结果
- `MaskRCNN-R50_inference.jpg` — MaskRCNN在测试图上的分割结果
- `TensorMask-R50_inference.jpg` — TensorMask在测试图上的分割结果

测试图像：`IMG_20230613_200341_r0_c1.jpg`（来自standardized_step1_half/merged/images）

#### 3.3 损失曲线
- `loss_curves.png` — 3个模型的训练损失曲线对比

#### 3.4 性能对比表
- `metrics_comparison.csv` — 3个模型的所有指标对比

## 输出目录
```
/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step7_evaluation/
├── BlendMask-R50_inference.jpg
├── MaskRCNN-R50_inference.jpg
├── TensorMask-R50_inference.jpg
├── evaluation_results.json
├── loss_curves.png
├── metrics_comparison.csv
└── [各模型的COCO评估结果]
```

## 预期完成时间
- BlendMask-R50: ~2h11m (iter 560/40000)
- MaskRCNN-R50: ~7h46m (iter 20/40000)
- TensorMask-R50: ~4h46m (iter 260/40000)

**总计：约8小时**

## 手动执行评估（如需提前测试）
```bash
# 使用现有的任何checkpoint
python step7_evaluate.py
```

脚本会自动查找 `model_final.pth`，如果不存在则跳过该模型。
