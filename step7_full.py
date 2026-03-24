#!/usr/bin/env python3
"""
Step 7: Comprehensive unified evaluation.
Part 1: Setup, imports, helpers
"""
import json,cv2,numpy as np,os,sys,time
from pathlib import Path

CODE   = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code')
ADELAI = CODE/'detectron2/AdelaiDet'
CFG    = CODE/'mydata/configs_step6'
STEP5  = CODE/'mydata/step5_augmented'
OUT_DIR= CODE/'mydata/step7_evaluation'
OUT_DIR.mkdir(parents=True,exist_ok=True)
MODEL_BASE = CODE/'mydata/output'

sys.path.insert(0,str(ADELAI))
sys.path.insert(0,str(CFG))

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import build_detection_test_loader

try:
    import adet
    from adet.config import get_cfg as adet_get_cfg
except Exception as e:
    print(f'AdelaiDet: {e}'); adet_get_cfg=None

# Register datasets
for nm in ('lettuce_train','lettuce_val'):
    if nm in DatasetCatalog.list():
        DatasetCatalog.remove(nm)
        try: MetadataCatalog.remove(nm)
        except: pass
register_coco_instances('lettuce_train',{},str(STEP5/'instances_train.json'),str(STEP5/'images'))
register_coco_instances('lettuce_val',{},str(STEP5/'instances_val.json'),str(STEP5/'images'))
print('[register] datasets registered')

# Test image - find best available
TEST_IMG=None
for c in sorted((CODE/'mydata/step4_output/images').glob('*.jpg')):
    TEST_IMG=c; break
print(f'Test image: {TEST_IMG}')

MODELS=[
    dict(name='BlendMask-R50',  cfg=CFG/'blendmask_R50_step6.yaml',
         weights=MODEL_BASE/'step6_blendmask_R50/model_final.pth',
         log=MODEL_BASE/'step6_blendmask_R50/metrics.json', is_adet=True),
    dict(name='MaskRCNN-R50',   cfg=CFG/'maskrcnn_R50_step6.yaml',
         weights=MODEL_BASE/'step6_maskrcnn_R50/model_final.pth',
         log=MODEL_BASE/'step6_maskrcnn_R50/metrics.json', is_adet=False),
    dict(name='TensorMask-R50', cfg=CFG/'tensormask_R50_step6.yaml',
         weights=MODEL_BASE/'step6_tensormask_R50/model_final.pth',
         log=MODEL_BASE/'step6_tensormask_R50/metrics.json', is_adet=True),
]

def icolor(i):
    hue=int((i*137.508)%180)
    return cv2.cvtColor(np.uint8([[[hue,220,200]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

def draw_instances(img,instances,model_name,score_thresh=0.3):
    vis=img.copy()
    n_det=0
    for i in range(len(instances)):
        score=instances.scores[i].item()
        if score<score_thresh: continue
        n_det+=1
        if hasattr(instances,'pred_masks'):
            mask=instances.pred_masks[i].cpu().numpy().astype(np.uint8)
            c=icolor(i); ov=vis.copy(); ov[mask>0]=c
            vis=cv2.addWeighted(ov,0.4,vis,0.6,0)
            cnts,_=cv2.findContours(mask*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis,cnts,-1,c,1)
        if hasattr(instances,'pred_boxes'):
            box=instances.pred_boxes[i].tensor.cpu().numpy()[0].astype(int)
            cv2.rectangle(vis,(box[0],box[1]),(box[2],box[3]),(255,255,255),1)
            cv2.putText(vis,f'{score:.2f}',(box[0],max(box[1]-2,10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)
    cv2.putText(vis,f'{model_name} [{n_det}]',(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
    return vis,n_det

def parse_metrics(log_path):
    iters=[]; total_loss=[]; loss_mask=[]; loss_cls=[]
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    d=json.loads(line)
                    if 'total_loss' not in d: continue
                    iters.append(d.get('iteration',0))
                    total_loss.append(d.get('total_loss',0))
                    loss_mask.append(d.get('loss_mask',0))
                    k='loss_fcos_cls' if 'loss_fcos_cls' in d else 'loss_cls'
                    loss_cls.append(d.get(k,0))
                except: pass
    except: pass
    return {'iterations':iters,'total_loss':total_loss,'loss_mask':loss_mask,'loss_cls':loss_cls}

def load_predictor(m):
    try:
        cfg=adet_get_cfg() if (m['is_adet'] and adet_get_cfg) else get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(str(m['cfg']))
        cfg.MODEL.WEIGHTS=str(m['weights'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.3
        return DefaultPredictor(cfg),cfg
    except Exception as e:
        print(f'  Load error: {e}'); return None,None

def run_coco_eval(predictor,cfg,model_name):
    try:
        od=str(OUT_DIR/model_name/'coco_eval'); os.makedirs(od,exist_ok=True)
        ev=COCOEvaluator('lettuce_val',cfg,False,output_dir=od)
        vl=build_detection_test_loader(cfg,'lettuce_val')
        return inference_on_dataset(predictor.model,vl,ev)
    except Exception as e:
        print(f'  COCO eval error: {e}'); return None

def extract_ap(results):
    for k in ('segm','bbox'):
        if results and k in results:
            s=results[k]
            return {x:s.get(x,-1) for x in ('AP','AP50','AP75','APs','APm','APl')}
    return {}

def infer_time(predictor,img,n=10):
    for _ in range(2): predictor(img)
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); predictor(img)
        ts.append((time.perf_counter()-t)*1000)
    return {'mean_ms':float(np.mean(ts)),'min_ms':float(np.min(ts)),'std_ms':float(np.std(ts))}

# ── Main loop ────────────────────────────────────────────────────────────────
all_results={}
test_img=cv2.imread(str(TEST_IMG)) if TEST_IMG and TEST_IMG.exists() else None
vis_list=[]

for m in MODELS:
    name=m['name']
    print(f'\n{"="*60}\n{name}')
    result={'name':name}

    tm=parse_metrics(m['log'])
    result['training_metrics']=tm
    if tm['iterations']:
        result['final_iter']=tm['iterations'][-1]
        result['final_total_loss']=round(tm['total_loss'][-1],4)
        print(f'  iter={result["final_iter"]} loss={result["final_total_loss"]}')

    predictor,cfg=load_predictor(m)
    if predictor is None:
        all_results[name]=result; continue
    print('  Model loaded')

    if test_img is not None:
        try:
            out=predictor(test_img)
            vis,nd=draw_instances(test_img.copy(),out['instances'],name)
            result['inference_detections']=nd
            cv2.imwrite(str(OUT_DIR/f'{name}_inference.jpg'),vis)
            vis_list.append((name,vis))
            print(f'  Inference: {nd} instances')
            ti=infer_time(predictor,test_img)
            result['inference_time']=ti
            print(f'  Time: {ti["mean_ms"]:.1f}ms')
        except Exception as e:
            print(f'  Inference error: {e}')

    print('  Running COCO eval...')
    coco_r=run_coco_eval(predictor,cfg,name)
    ap=extract_ap(coco_r)
    result['coco_metrics']=ap
    if ap: print(f'  mAP={ap.get("AP",-1):.3f} AP50={ap.get("AP50",-1):.3f} AP75={ap.get("AP75",-1):.3f}')
    all_results[name]=result

# ── Composite visualization ───────────────────────────────────────────────────
if vis_list and test_img is not None:
    h,w=test_img.shape[:2]
    comp=np.zeros((h,w*(len(vis_list)+1),3),np.uint8)
    comp[:,:w]=test_img
    cv2.putText(comp,'Original',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,0),2)
    for i,(nm,vis) in enumerate(vis_list):
        comp[:,w*(i+1):w*(i+2)]=vis
    cv2.imwrite(str(OUT_DIR/'comparison_visualization.jpg'),comp)
    print('\nComparison visualization saved')

# ── Unified results JSON ──────────────────────────────────────────────────────
unified={'models':{},'loss_curves':{}}
for name,r in all_results.items():
    ap=r.get('coco_metrics',{})
    ti=r.get('inference_time',{})
    unified['loss_curves'][name]=r.get('training_metrics',{})
    unified['models'][name]={
        'mAP':ap.get('AP',-1),'AP':ap.get('AP',-1),
        'AP50':ap.get('AP50',-1),'AP75':ap.get('AP75',-1),
        'APs':ap.get('APs',-1),'APm':ap.get('APm',-1),'APl':ap.get('APl',-1),
        'inference_time_mean_ms':ti.get('mean_ms',-1),
        'inference_time_min_ms':ti.get('min_ms',-1),
        'inference_time_std_ms':ti.get('std_ms',-1),
        'final_iter':r.get('final_iter',-1),
        'final_total_loss':r.get('final_total_loss',-1),
        'inference_detections':r.get('inference_detections',-1),
    }

with open(OUT_DIR/'unified_results.json','w') as f:
    json.dump({k:v for k,v in unified.items() if k!='loss_curves'},f,indent=2)
with open(OUT_DIR/'loss_curves.json','w') as f:
    json.dump(unified['loss_curves'],f,indent=2)

# ── Markdown report ───────────────────────────────────────────────────────────
lines=['# Step 7 — Unified Evaluation Report','',
       '## Model Performance Comparison','',
       '| Model | mAP | AP50 | AP75 | APs | APm | APl | Infer(ms) | Loss@40k |',
       '|---|---|---|---|---|---|---|---|---|']
for name,d in unified['models'].items():
    def f(v): return f'{v:.3f}' if v>=0 else 'N/A'
    def g(v): return f'{v:.1f}' if v>=0 else 'N/A'
    lines.append(f'| {name} | {f(d["mAP"])} | {f(d["AP50"])} | {f(d["AP75"])} '
                 f'| {f(d["APs"])} | {f(d["APm"])} | {f(d["APl"])} '
                 f'| {g(d["inference_time_mean_ms"])} | {f(d["final_total_loss"])} |')
lines+=['','## Output Files','',
        '- `unified_results.json` — all metrics in one file',
        '- `loss_curves.json` — training loss history for all models',
        '- `comparison_visualization.jpg` — side-by-side inference results',
        '- `BlendMask-R50_inference.jpg`',
        '- `MaskRCNN-R50_inference.jpg`',
        '- `TensorMask-R50_inference.jpg`',
        '']
with open(OUT_DIR/'step7_report.md','w') as f:
    f.write('\n'.join(lines))

print(f'\n✅ Step 7 complete. Results in {OUT_DIR}')
for name,d in unified['models'].items():
    print(f'  {name}: mAP={d["mAP"]:.3f} AP50={d["AP50"]:.3f} infer={d["inference_time_mean_ms"]:.1f}ms')
