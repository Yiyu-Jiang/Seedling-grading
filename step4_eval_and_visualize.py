#!/usr/bin/env python3
"""
Step 4:
  1. Parse best evaluation metrics (AP, AP50, AP75, Precision, Recall, F1,
     inference time) from each model's training log.
  2. Write unified CSV + TXT summary to output/step4_comparison/.
  3. Run inference on a single test image with all three trained models and
     save side-by-side visualizations.

Models: BlendMask-R50, Mask R-CNN-R50, TensorMask-R50
"""
from __future__ import annotations
import sys, os, re, json, csv, time
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
UNIT3      = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code')
ADELAIDET  = UNIT3 / 'detectron2/AdelaiDet'
DETECTRON2 = UNIT3 / 'detectron2'
CFG_DIR    = UNIT3 / 'mydata/configs_step3'
OUT_BASE   = UNIT3 / 'mydata/output'
RESULT_DIR = OUT_BASE / 'step4_comparison'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Source paths for custom modules
sys.path.insert(0, str(ADELAIDET))
sys.path.insert(0, str(DETECTRON2))
sys.path.insert(0, str(CFG_DIR))

# Test image for visualization
# The spec references standardized_step1_half which does not exist;
# use the equivalent tile from step1_output.
TEST_IMAGE = UNIT3 / 'mydata/step1_output/images/IMG_20230613_200341_r0_c1.jpg'

MODELS = {
    'BlendMask-R50': {
        'cfg':     str(CFG_DIR / 'blendmask_R50_step3.yaml'),
        'weights': str(OUT_BASE / 'step3_blendmask_R50/model_final.pth'),
        'outdir':  OUT_BASE / 'step3_blendmask_R50',
        'type':    'blendmask',
    },
    'MaskRCNN-R50': {
        'cfg':     str(CFG_DIR / 'maskrcnn_R50_step3.yaml'),
        'weights': str(OUT_BASE / 'step3_maskrcnn_R50/model_final.pth'),
        'outdir':  OUT_BASE / 'step3_maskrcnn_R50',
        'type':    'maskrcnn',
    },
    'TensorMask-R50': {
        'cfg':     str(CFG_DIR / 'tensormask_R50_step3.yaml'),
        'weights': str(OUT_BASE / 'step3_tensormask_R50/model_final.pth'),
        'outdir':  OUT_BASE / 'step3_tensormask_R50',
        'type':    'tensormask',
    },
}


# ─── Part 1: Parse metrics from log files ─────────────────────────────────────
def parse_log(log_path: Path) -> dict:
    """Extract segm AP/AP50/AP75, COCO Precision/Recall, F1, infer time."""
    r = dict(AP=None, AP50=None, AP75=None,
             Precision=None, Recall=None, F1=None, infer_time_ms=None)
    if not log_path.exists():
        print(f'  [warn] log not found: {log_path}')
        return r

    text = log_path.read_text(errors='replace')

    # --- segm AP from table: last occurrence (= final eval) ---
    # Pattern: "Evaluation results for segm:" then table row
    segm_tables = re.findall(
        r'Evaluation results for segm.*?'
        r'\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|',
        text, re.DOTALL)
    if segm_tables:
        ap, ap50, ap75 = map(float, segm_tables[-1])
        r['AP']   = round(ap,   4)
        r['AP50'] = round(ap50, 4)
        r['AP75'] = round(ap75, 4)

    # Overwrite with copypaste segm line if present (same numbers, more reliable)
    # Lines look like:
    #   copypaste: Task: segm
    #   copypaste: AP,AP50,AP75,APs,APm,APl
    #   copypaste: 47.7209,68.4071,52.1285,...
    for m in re.finditer(
            r'copypaste: Task: segm\s*\n'
            r'[^\n]+\n'
            r'copypaste: ([\d.]+),([\d.]+),([\d.]+)',
            text):
        r['AP']   = round(float(m.group(1)), 4)
        r['AP50'] = round(float(m.group(2)), 4)
        r['AP75'] = round(float(m.group(3)), 4)

    # --- COCO-style Precision / Recall (IoU=0.50:0.95, area=all, maxDets=100) ---
    pr_vals = re.findall(
        r'Average Precision  \(AP\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)',
        text)
    rc_vals = re.findall(
        r'Average Recall     \(AR\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)',
        text)
    if pr_vals:
        r['Precision'] = round(float(pr_vals[-1]), 4)
    if rc_vals:
        r['Recall']    = round(float(rc_vals[-1]), 4)
    if r['Precision'] is not None and r['Recall'] is not None:
        p, rec = r['Precision'], r['Recall']
        denom = p + rec
        r['F1'] = round(2 * p * rec / denom, 4) if denom > 0 else 0.0

    # --- Inference time: "X.XXXX s / iter per device" ---
    it_m = re.search(
        r'Total inference time:.*?\(([\d.]+) s / iter per device',
        text, re.IGNORECASE)
    if it_m:
        r['infer_time_ms'] = round(float(it_m.group(1)) * 1000, 1)

    return r


def aggregate_metrics():
    rows = []
    for model_name, info in MODELS.items():
        log_path = info['outdir'] / 'log.txt'
        if not log_path.exists():
            log_path = info['outdir'] / 'train.log'
        print(f'[metrics] parsing {model_name}: {log_path}')
        m = parse_log(log_path)
        row = {'Model': model_name, **m}
        rows.append(row)
        print(f'  -> {row}')

    fields = ['Model', 'AP', 'AP50', 'AP75',
              'Precision', 'Recall', 'F1', 'infer_time_ms']

    # CSV
    csv_path = RESULT_DIR / 'step4_results.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'\n[metrics] Saved CSV -> {csv_path}')

    # Plain-text table
    txt_path = RESULT_DIR / 'step4_results.txt'
    with txt_path.open('w') as f:
        f.write('Step 4 – Model Comparison (Segmentation)\n')
        f.write('=' * 72 + '\n')
        f.write(f'{"Model":<20} {"AP":>7} {"AP50":>7} {"AP75":>7} '
                f'{"Prec":>7} {"Rec":>7} {"F1":>7} {"ms/img":>9}\n')
        f.write('-' * 72 + '\n')
        for row in rows:
            def fmt(v):
                return f'{v:.4f}' if isinstance(v, float) else str(v or "-")
            f.write(
                f'{row["Model"]:<20} '
                f'{fmt(row["AP"]):>7} '
                f'{fmt(row["AP50"]):>7} '
                f'{fmt(row["AP75"]):>7} '
                f'{fmt(row["Precision"]):>7} '
                f'{fmt(row["Recall"]):>7} '
                f'{fmt(row["F1"]):>7} '
                f'{fmt(row["infer_time_ms"]):>9}\n'
            )
        f.write('=' * 72 + '\n')
        f.write('Units: AP/AP50/AP75/Precision/Recall/F1 are 0-100 COCO scale; '
                'ms/img = milliseconds per image.\n')
    print(f'[metrics] Saved TXT  -> {txt_path}')
    return rows


# ─── Part 2: Visualize on test image ──────────────────────────────────────────
def build_predictor(model_name, cfg_path, weights_path, model_type):
    """Build a Detectron2 DefaultPredictor for the given model config."""
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    if model_type == 'tensormask':
        cfg = get_cfg()
        try:
            from detectron2.projects.tensormask import add_tensormask_config
            add_tensormask_config(cfg)
        except ImportError:
            # Try from projects sub-directory
            tensormask_path = DETECTRON2 / 'projects/TensorMask'
            sys.path.insert(0, str(tensormask_path))
            from tensormask import add_tensormask_config
            add_tensormask_config(cfg)
    else:
        try:
            from adet.config import get_cfg as adet_get_cfg
            cfg = adet_get_cfg()
        except Exception as e:
            print(f'  [warn] adet get_cfg failed ({e}), using base cfg')
            cfg = get_cfg()

    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Score thresholds
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    try:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
    except Exception:
        pass
    try:
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.3
    except Exception:
        pass
    try:
        cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.3
    except Exception:
        pass

    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def visualize_all():
    import cv2
    import numpy as np
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog

    # Register datasets (provides metadata)
    import register_step3  # noqa – side effect: registers lettuce_train/val

    # Import adet to register its model heads
    try:
        import adet  # noqa
    except Exception as e:
        print(f'[warn] adet import: {e}')

    if not TEST_IMAGE.exists():
        print(f'[vis] ERROR: test image not found: {TEST_IMAGE}')
        return

    orig_img = cv2.imread(str(TEST_IMAGE))
    if orig_img is None:
        print(f'[vis] ERROR: cv2 cannot read: {TEST_IMAGE}')
        return

    vis_results = []   # (model_name, vis_image_np)

    for model_name, info in MODELS.items():
        print(f'[vis] Running {model_name} ...')
        try:
            predictor, cfg = build_predictor(
                model_name, info['cfg'], info['weights'], info['type'])

            img_bgr = orig_img.copy()
            t0 = time.perf_counter()
            outputs = predictor(img_bgr)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f'  inference: {elapsed_ms:.1f} ms  '
                  f'instances: {len(outputs["instances"])}')

            # Metadata
            dataset_name = cfg.DATASETS.TEST[0]
            meta = MetadataCatalog.get(dataset_name)
            if not hasattr(meta, 'thing_classes') or not meta.thing_classes:
                meta.set(thing_classes=['lettuce'])

            vis = Visualizer(
                img_bgr[:, :, ::-1],
                metadata=meta,
                scale=1.0,
                instance_mode=ColorMode.SEGMENTATION,
            )
            out = vis.draw_instance_predictions(
                outputs['instances'].to('cpu'))
            vis_img = out.get_image()   # RGB numpy

            # Save individual visualization
            safe_name = model_name.replace('/', '_').replace(' ', '_')
            save_path = RESULT_DIR / f'vis_{safe_name}.jpg'
            cv2.imwrite(str(save_path),
                        vis_img[:, :, ::-1])  # convert RGB->BGR for cv2
            print(f'  saved -> {save_path}')
            vis_results.append((model_name, vis_img))

        except Exception as e:
            import traceback
            print(f'  [ERROR] {model_name}: {e}')
            traceback.print_exc()

    # ── Compose side-by-side comparison image ──────────────────────────────
    if vis_results:
        try:
            import numpy as np
            import cv2 as cv2_
            from PIL import Image, ImageDraw, ImageFont

            panels = []
            label_h = 40
            for name, img_rgb in vis_results:
                h, w = img_rgb.shape[:2]
                # Add label bar at top
                panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
                panel[label_h:] = img_rgb
                # Draw label with PIL for better font rendering
                pil = Image.fromarray(panel)
                draw = ImageDraw.Draw(pil)
                try:
                    font = ImageFont.truetype(
                        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
                except Exception:
                    font = ImageFont.load_default()
                draw.rectangle([0, 0, w, label_h], fill=(30, 30, 30))
                draw.text((10, 8), name, fill=(255, 255, 100), font=font)
                panels.append(np.array(pil))

            composite = np.concatenate(panels, axis=1)
            comp_path = RESULT_DIR / 'vis_comparison.jpg'
            Image.fromarray(composite).save(str(comp_path), quality=95)
            print(f'[vis] Composite saved -> {comp_path}')
        except Exception as e:
            print(f'[vis] Could not build composite: {e}')


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('STEP 4 – Metrics aggregation')
    print('=' * 60)
    rows = aggregate_metrics()

    print()
    print('=' * 60)
    print('STEP 4 – Visualization on test image')
    print(f'  image: {TEST_IMAGE}')
    print('=' * 60)
    visualize_all()

    print()
    print('Done. Results in:', RESULT_DIR)
