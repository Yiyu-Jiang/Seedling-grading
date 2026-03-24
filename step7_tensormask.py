#!/usr/bin/env python3
import json,cv2,numpy as np,os,sys,time
from pathlib import Path
CODE=Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code')
sys.path.insert(0,str(CODE/'detectron2/projects/TensorMask'))
sys.path.insert(0,str(CODE/'mydata/configs_step6'))
import tensormask
from tensormask.config import add_tensormask_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import build_detection_test_loader
STEP5=CODE/'mydata/step5_augmented'
OUT=CODE/'mydata/step7_evaluation'
for nm in ('lettuce_train','lettuce_val'):
    if nm in DatasetCatalog.list(): DatasetCatalog.remove(nm)
    try: MetadataCatalog.remove(nm)
    except: pass
register_coco_instances('lettuce_train',{},str(STEP5/'instances_train.json'),str(STEP5/'images'))
register_coco_instances('lettuce_val',{},str(STEP5/'instances_val.json'),str(STEP5/'images'))
cfg=get_cfg()
add_tensormask_config(cfg)
cfg.set_new_allowed(True)
cfg.merge_from_file(str(CODE/'mydata/configs_step6/tensormask_R50_step6.yaml'))
cfg.MODEL.WEIGHTS=str(CODE/'mydata/output/step6_tensormask_R50/model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.3
cfg.MODEL.RETINANET.SCORE_THRESH_TEST=0.3
print('Loading...'); predictor=DefaultPredictor(cfg); print('Loaded')
TIMG=next((CODE/'mydata/step4_output/images').glob('*.jpg'),None)
tmean=-1; nd=-1
if TIMG:
    img=cv2.imread(str(TIMG))
    out=predictor(img)
    nd=int((out['instances'].scores>0.3).sum())
    vis=img.copy()
    for i in range(len(out['instances'])):
        if out['instances'].scores[i]<0.3: continue
        if hasattr(out['instances'],'pred_masks'):
            m=out['instances'].pred_masks[i].cpu().numpy().astype(np.uint8)
            hue=int((i*137.508)%180)
            c=cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
            ov=vis.copy(); ov[m>0]=c; vis=cv2.addWeighted(ov,0.4,vis,0.6,0)
    cv2.putText(vis,f'TensorMask [{nd}]',(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
    cv2.imwrite(str(OUT/'TensorMask-R50_inference.jpg'),vis)
    print(f'Inference: {nd} instances')
    for _ in range(2): predictor(img)
    ts=[]
    for _ in range(10):
        t0=time.perf_counter(); predictor(img); ts.append((time.perf_counter()-t0)*1000)
    tmean=float(np.mean(ts)); print(f'Time: {tmean:.1f}ms')
od=str(OUT/'TensorMask-R50/coco_eval'); os.makedirs(od,exist_ok=True)
ev=COCOEvaluator('lettuce_val',cfg,False,output_dir=od)
vl=build_detection_test_loader(cfg,'lettuce_val')
results=inference_on_dataset(predictor.model,vl,ev)
ap={}
for k in ('segm','bbox'):
    if k in results:
        s=results[k]
        ap={x:s.get(x,-1) for x in ('AP','AP50','AP75','APs','APm','APl')}
        print(f'mAP={ap["AP"]:.3f} AP50={ap["AP50"]:.3f} AP75={ap["AP75"]:.3f}'); break
up=OUT/'unified_results.json'
unified=json.load(open(up)) if up.exists() else {'models':{}}
unified['models']['TensorMask-R50'].update({
    'mAP':ap.get('AP',-1),'AP':ap.get('AP',-1),'AP50':ap.get('AP50',-1),
    'AP75':ap.get('AP75',-1),'APs':ap.get('APs',-1),'APm':ap.get('APm',-1),'APl':ap.get('APl',-1),
    'inference_time_mean_ms':tmean,'inference_detections':nd,
})
json.dump(unified,open(up,'w'),indent=2)
print('Done. Results saved.')
