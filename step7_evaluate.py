#!/usr/bin/env python3
"""
Step 7: Evaluate 3 trained models and generate metrics + visualizations.
- Compute COCO metrics (mAP, AP50, AP75, APs, APm, APl)
- Plot loss curves
- Inference on test image with visualization
"""
import json, cv2, numpy as np, os, sys
from pathlib import Path
from collections import defaultdict

CODE = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code')
ADELAI = CODE / 'detectron2/AdelaiDet'
D2 = CODE / 'detectron2'
CFG = CODE / 'mydata/configs_step6'
STEP5 = CODE / 'mydata/step5_augmented'
OUT = CODE / 'mydata/step7_evaluation'
OUT.mkdir(parents=True, exist_ok=True)

# Test image
TEST_IMG = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/standardized_step1_half/merged/images/IMG_20230613_200341_r0_c1.jpg')

sys.path.insert(0, str(ADELAI))
sys.path.insert(0, str(CFG))

try:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
except ImportError as e:
    print(f'ERROR: {e}'); sys.exit(1)

try:
    import adet
    from adet.config import get_cfg as adet_get_cfg
except Exception as e:
    print(f'AdelaiDet import error: {e}'); sys.exit(1)

# Register datasets
os.environ['LETTUCE_IMAGE_ROOT'] = str(STEP5/'images')
os.environ['LETTUCE_TRAIN_JSON'] = str(STEP5/'instances_train.json')
os.environ['LETTUCE_VAL_JSON'] = str(STEP5/'instances_val.json')
os.environ['LETTUCE_REG_DIR'] = str(CFG)

try:
    import register_step6
except Exception as e:
    print(f'Register error: {e}')

def icolor(i):
    hue=int((i*137.508)%180)
    return cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

def draw_instances(img, instances):
    vis=img.copy()
    if len(instances)==0: return vis
    for i,inst in enumerate(instances):
        if not hasattr(inst,'pred_masks'): continue
        mask=inst.pred_masks[0].cpu().numpy().astype(np.uint8)*255
        c=icolor(i)
        ov=vis.copy()
        ov[mask>0]=c
        vis=cv2.addWeighted(ov,0.4,vis,0.6,0)
    return vis

def evaluate_model(model_name, cfg_path, model_weights):
    print(f'\n=== Evaluating {model_name} ===')
    if not Path(model_weights).exists():
        print(f'  Model not found: {model_weights}'); return None

    if 'blendmask' in model_name.lower():
        cfg=adet_get_cfg()
    else:
        cfg=get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(cfg_path))
    cfg.MODEL.WEIGHTS=str(model_weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.3
    cfg.DATASETS.TEST=('lettuce_val',)

    predictor=DefaultPredictor(cfg)

    # Inference on test image
    if TEST_IMG.exists():
        img=cv2.imread(str(TEST_IMG))
        out=predictor(img)
        vis=draw_instances(img,out['instances'])
        cv2.imwrite(str(OUT/f'{model_name}_inference.jpg'),vis)
        print(f'  Inference visualization saved')

    # COCO evaluation
    try:
        evaluator=COCOEvaluator('lettuce_val',cfg,False,output_dir=str(OUT/model_name))
        val_loader=build_detection_test_loader(cfg,'lettuce_val')
        results=inference_on_dataset(predictor.model,val_loader,evaluator)
        print(f'  Evaluation complete')
        return results
    except Exception as e:
        print(f'  Evaluation error: {e}')
        return None

def main():
    models=[
        ('BlendMask-R50',str(CFG/'blendmask_R50_step6.yaml'),
         '/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step6_blendmask_R50/model_final.pth'),
        ('MaskRCNN-R50',str(CFG/'maskrcnn_R50_step6.yaml'),
         '/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step6_maskrcnn_R50/model_final.pth'),
        ('TensorMask-R50',str(CFG/'tensormask_R50_step6.yaml'),
         '/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step6_tensormask_R50/model_final.pth'),
    ]

    results_all={}
    for name,cfg_path,weights in models:
        r=evaluate_model(name,cfg_path,weights)
        if r: results_all[name]=r

    # Save results
    with open(OUT/'evaluation_results.json','w') as f:
        json.dump({k:str(v) for k,v in results_all.items()},f,indent=2)
    print(f'\nEvaluation complete. Results saved to {OUT}')

if __name__=='__main__':
    main()
