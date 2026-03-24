#!/usr/bin/env python3
"""
Step 4 (updated): Compute and output all required metrics from saved inference
results using pycocotools COCOeval:
  Precision, Recall, F1, mAP, AP, AP50, AP75, APs, APm, APl, infer_time_ms

All AP/AR values are reported in [0,100] % scale (COCO standard * 100).
mAP = AP @ IoU=0.50:0.95 (same as AP, the primary COCO metric).
Precision = AP @ IoU=0.50:0.95, area=all, maxDets=100  (stats[0])
Recall    = AR @ IoU=0.50:0.95, area=all, maxDets=100  (stats[8])
"""
from __future__ import annotations
import csv, json, io, contextlib
from pathlib import Path

UNIT3      = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code')
OUT_BASE   = UNIT3 / 'mydata/output'
RESULT_DIR = OUT_BASE / 'step4_comparison'
VAL_JSON   = UNIT3 / 'mydata/step2_output/coco/instances_val.json'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Inference times (s/iter per device, on 2 devices) from training logs
# Converted to ms/image = s_per_iter * 1000  (each iter = 2 images on 2 GPUs,
# so per-image time = (s/iter / 2) * 1000, but we report as logged: s/iter*1000)
INFER_TIME_MS = {
    'BlendMask-R50':  94.3,   # 0.094284 s/iter * 1000
    'MaskRCNN-R50':   75.3,   # 0.075253 s/iter * 1000
    'TensorMask-R50': 102.6,  # 0.102637 s/iter * 1000
}

MODELS = {
    'BlendMask-R50':  OUT_BASE / 'step3_blendmask_R50/inference/coco_instances_results.json',
    'MaskRCNN-R50':   OUT_BASE / 'step3_maskrcnn_R50/inference/coco_instances_results.json',
    'TensorMask-R50': OUT_BASE / 'step3_tensormask_R50/inference/coco_instances_results.json',
}


def run_cocoeval(val_json: Path, res_json: Path, iou_type: str = 'segm') -> dict:
    """
    Run COCOeval and return a dict of all 12 stats:
      AP    = stats[0]  IoU=0.50:0.95 area=all   maxDets=100
      AP50  = stats[1]  IoU=0.50      area=all   maxDets=100
      AP75  = stats[2]  IoU=0.75      area=all   maxDets=100
      APs   = stats[3]  IoU=0.50:0.95 area=small maxDets=100
      APm   = stats[4]  IoU=0.50:0.95 area=med   maxDets=100
      APl   = stats[5]  IoU=0.50:0.95 area=large maxDets=100
      AR1   = stats[6]  IoU=0.50:0.95 area=all   maxDets=1
      AR10  = stats[7]  IoU=0.50:0.95 area=all   maxDets=10
      AR100 = stats[8]  IoU=0.50:0.95 area=all   maxDets=100  <- Recall
      ARs   = stats[9]
      ARm   = stats[10]
      ARl   = stats[11]
    All values are in [0,1]; we multiply by 100 for output.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import math

    coco_gt = COCO(str(val_json))
    with open(str(res_json)) as f:
        res_data = json.load(f)

    if iou_type == 'segm':
        res_data = [r for r in res_data if 'segmentation' in r]

    if not res_data:
        print('  [warn] no segmentation results found')
        return {}

    coco_dt = coco_gt.loadRes(res_data)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    def s(i):
        v = float(ev.stats[i])
        return None if math.isnan(v) or v < 0 else round(v * 100, 3)

    return {
        'AP':   s(0),   # mAP = AP
        'AP50': s(1),
        'AP75': s(2),
        'APs':  s(3),
        'APm':  s(4),
        'APl':  s(5),
        'AR':   s(8),   # Recall @ maxDets=100
    }


def main():
    rows = []
    for model_name, res_json in MODELS.items():
        print(f'\n[eval] {model_name}')
        print(f'       {res_json}')

        if not res_json.exists():
            print('  [warn] inference results json not found, skipping')
            continue

        stats = run_cocoeval(VAL_JSON, res_json, 'segm')
        if not stats:
            continue

        ap   = stats['AP']    # same as mAP
        prec = stats['AP']    # Precision = AP (primary COCO metric)
        rec  = stats['AR']    # Recall
        f1   = None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = round(2 * prec * rec / (prec + rec), 3)

        row = {
            'Model':         model_name,
            'mAP':           ap,
            'AP':            ap,
            'AP50':          stats['AP50'],
            'AP75':          stats['AP75'],
            'APs':           stats['APs'],
            'APm':           stats['APm'],
            'APl':           stats['APl'],
            'Precision':     prec,
            'Recall':        rec,
            'F1':            f1,
            'infer_time_ms': INFER_TIME_MS[model_name],
        }
        rows.append(row)
        print(f'  mAP={ap}  AP50={stats["AP50"]}  AP75={stats["AP75"]}  '
              f'APs={stats["APs"]}  APm={stats["APm"]}  APl={stats["APl"]}')
        print(f'  Precision={prec}  Recall={rec}  F1={f1}  '
              f'infer_time_ms={INFER_TIME_MS[model_name]}')

    if not rows:
        print('No results to write.')
        return

    fields = ['Model', 'mAP', 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
              'Precision', 'Recall', 'F1', 'infer_time_ms']

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = RESULT_DIR / 'step4_results.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved CSV -> {csv_path}')

    # ── TXT ──────────────────────────────────────────────────────────────────
    txt_path = RESULT_DIR / 'step4_results.txt'
    with txt_path.open('w') as f:
        title = 'Step 4 – Instance Segmentation Comparison (Lettuce Dataset)'
        f.write(title + '\n')
        f.write('=' * 95 + '\n')
        f.write(f'  Val annotation : {VAL_JSON}\n')
        f.write(f'  Eval type      : COCO segm\n')
        f.write(f'  Scale          : all values in [0,100] % (COCO standard)\n')
        f.write(f'  mAP = AP @ IoU=0.50:0.95 | Precision=AP | Recall=AR @maxDets=100\n')
        f.write(f'  ms/img = inference time per iter (2 GPUs) from eval log\n')
        f.write('=' * 95 + '\n')
        # header
        hdr = (f'{"Model":<20} {"mAP":^6} {"AP":^6} {"AP50":^6} {"AP75":^6}'
               f' {"APs":^6} {"APm":^6} {"APl":^6}'
               f' {"Prec":^6} {"Rec":^6} {"F1":^6} {"ms/img":^8}')
        f.write(hdr + '\n')
        f.write('-' * 95 + '\n')

        def fmt(v):
            if v is None:
                return '-'
            return f'{float(v):.2f}'

        for row in rows:
            line = (
                f'{row["Model"]:<20}'
                f' {fmt(row["mAP"]):^6}'
                f' {fmt(row["AP"]):^6}'
                f' {fmt(row["AP50"]):^6}'
                f' {fmt(row["AP75"]):^6}'
                f' {fmt(row["APs"]):^6}'
                f' {fmt(row["APm"]):^6}'
                f' {fmt(row["APl"]):^6}'
                f' {fmt(row["Precision"]):^6}'
                f' {fmt(row["Recall"]):^6}'
                f' {fmt(row["F1"]):^6}'
                f' {fmt(row["infer_time_ms"]):^8}'
            )
            f.write(line + '\n')

        f.write('=' * 95 + '\n')
        best = max(rows, key=lambda r: r['mAP'] or 0)
        f.write(f'Best model: {best["Model"]}  '
                f'mAP={best["mAP"]}  F1={best["F1"]}  '
                f'infer_time_ms={best["infer_time_ms"]}\n')

    print(f'Saved TXT  -> {txt_path}')
    print()
    print(open(txt_path).read())


if __name__ == '__main__':
    main()
