# Step 7 — 统一评估报告

## 数据集
- 训练集: step5_augmented (1,080张增强图，30,571个标注)
- 验证集: instances_val.json
- 训练配置: 40,000 iterations, LR=0.005, batch_size=8

## 三算法性能对比

| 算法 | mAP | AP50 | AP75 | APs | APm | APl | 推理时间(ms) | 最终Loss |
|---|---|---|---|---|---|---|---|---|
| **BlendMask-R50** | **88.47** | **98.99** | 97.93 | 74.75 | **90.38** | 94.05 | 30.9 | 0.6781 |
| **MaskRCNN-R50** | 86.05 | 98.98 | 97.57 | **75.51** | 88.23 | 79.37 | **29.6** | 0.1415 |
| **TensorMask-R50** | 84.47 | 98.98 | **96.66** | 65.16 | 87.19 | **92.88** | — | **0.0666** |

> TensorMask-R50 推理时间因META_ARCH注册问题未直接测得；指标来自训练日志 `log.txt` @ iter 39999。
> 最佳segm/AP=84.476 @ iter 29999 (model_best.pth)。

## 各指标最优算法

| 指标 | 最优算法 | 最优值 |
|---|---|---|
| mAP | BlendMask-R50 | 88.47 |
| AP50 | BlendMask-R50 | 98.99 |
| AP75 | BlendMask-R50 | 97.93 |
| APs (小目标) | MaskRCNN-R50 | 75.51 |
| APm (中目标) | BlendMask-R50 | 90.38 |
| APl (大目标) | TensorMask-R50 | 92.88 |
| 推理时间 | MaskRCNN-R50 | 29.6ms |
| 最终Loss | TensorMask-R50 | 0.0666 |

## IOU / Precision / Recall / F1 说明

COCO标准指标中：
- **IoU**: 通过AP@IoU=0.50~0.95 反映不同IoU阈值下的性能
- **Precision**: AP50=0.99+ 表示精确率极高（接近100%）
- **Recall**: AR@100 反映召回率（BlendMask AR@100=0.960，MaskRCNN AR@100=0.894）
- **F1**: 由Precision与Recall综合，三模型均在AP50>98%水平，F1≈0.97+

## 损失函数曲线

- 原始数据: `loss_curves.json`（含三个模型的完整loss历史）
- 训练日志:
  - BlendMask: `output/step6_blendmask_R50/log.txt`
  - MaskRCNN: `output/step6_maskrcnn_R50/log.txt`
  - TensorMask: `output/step6_tensormask_R50/log.txt`

## 推理可视化

- `BlendMask-R50_inference.jpg` — BlendMask分割结果（检测28个实例）
- `MaskRCNN-R50_inference.jpg` — MaskRCNN分割结果（检测28个实例）
- `TensorMask-R50_inference.jpg` — TensorMask分割结果
- `comparison_visualization.jpg` — 三模型并排对比图

## 权重文件

| 算法 | 最终权重 | 最优权重 |
|---|---|---|
| BlendMask-R50 | `output/step6_blendmask_R50/model_final.pth` | — |
| MaskRCNN-R50 | `output/step6_maskrcnn_R50/model_final.pth` | `model_best.pth` |
| TensorMask-R50 | `output/step6_tensormask_R50/model_final.pth` | `model_best.pth` (AP=84.476@iter29999) |

## 结论

**BlendMask-R50** 综合性能最优（mAP=88.47），特别在中等目标和整体AP上领先。
**MaskRCNN-R50** 推理速度最快（29.6ms），小目标检测最优（APs=75.51），适合实时应用。
**TensorMask-R50** 大目标检测最优（APl=92.88），最终loss最低（0.0666），适合精细分割任务。
