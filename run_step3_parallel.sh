#!/usr/bin/env bash
# =============================================================================
# Step 3: Train BlendMask-R50, MaskRCNN-R50, TensorMask-R50 in PARALLEL
# Each model gets 2 dedicated GPUs:
#   BlendMask-R50  -> GPU 0,1  (tmux: step3_blendmask)
#   MaskRCNN-R50   -> GPU 2,3  (tmux: step3_maskrcnn)
#   TensorMask-R50 -> GPU 4,5  (tmux: step3_tensormask)
# 40000 iters, LR=0.005, IMS_PER_BATCH=4
# =============================================================================

PY="/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python"
PROJECT="/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code"
ADELAI="${PROJECT}/detectron2/AdelaiDet"
TMASK="${PROJECT}/detectron2/projects/TensorMask"
CFG_DIR="${PROJECT}/mydata/configs_step3"
DATA_ROOT="${PROJECT}/mydata/step2_output"
OUT_BASE="${PROJECT}/mydata/output"

export LETTUCE_IMAGE_ROOT="${DATA_ROOT}/images"
export LETTUCE_TRAIN_JSON="${DATA_ROOT}/coco/instances_train.json"
export LETTUCE_VAL_JSON="${DATA_ROOT}/coco/instances_val.json"
export LETTUCE_REG_DIR="${CFG_DIR}"

BLEND_OUT="${OUT_BASE}/step3_blendmask_R50"
MRCNN_OUT="${OUT_BASE}/step3_maskrcnn_R50"
TMASK_OUT="${OUT_BASE}/step3_tensormask_R50"

mkdir -p "${BLEND_OUT}" "${MRCNN_OUT}" "${TMASK_OUT}" "${OUT_BASE}/step3_comparison"

SOLVER="SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 SOLVER.MAX_ITER 40000 SOLVER.STEPS (28000,36000) SOLVER.WARMUP_ITERS 1000 SOLVER.CHECKPOINT_PERIOD 5000 TEST.EVAL_PERIOD 5000"

# Kill any existing step3 sessions
tmux kill-session -t step3_blendmask  2>/dev/null || true
tmux kill-session -t step3_maskrcnn   2>/dev/null || true
tmux kill-session -t step3_tensormask 2>/dev/null || true
tmux kill-session -t step3_maskrcnn_condinst 2>/dev/null || true

# Also kill old sequential nohup job if still running
kill $(pgrep -f 'run_step3') 2>/dev/null || true

# ---------------------------------------------------------------------------
# BlendMask-R50 on GPU 0,1
# ---------------------------------------------------------------------------
tmux new-session -d -s step3_blendmask "bash -c '
  export CUDA_VISIBLE_DEVICES=0,1
  export LETTUCE_IMAGE_ROOT=${LETTUCE_IMAGE_ROOT}
  export LETTUCE_TRAIN_JSON=${LETTUCE_TRAIN_JSON}
  export LETTUCE_VAL_JSON=${LETTUCE_VAL_JSON}
  export LETTUCE_REG_DIR=${LETTUCE_REG_DIR}
  export PYTHONPATH=${CFG_DIR}:${ADELAI}
  mkdir -p ${BLEND_OUT}
  ${PY} ${ADELAI}/tools/train_net.py \
    --num-gpus 2 \
    --config-file ${CFG_DIR}/blendmask_R50_step3.yaml \
    --dist-url tcp://127.0.0.1:29520 \
    ${SOLVER} OUTPUT_DIR ${BLEND_OUT} \
    2>&1 | tee ${BLEND_OUT}/train.log
  echo EXIT:$? >> ${BLEND_OUT}/train.log
'"

# ---------------------------------------------------------------------------
# MaskRCNN-R50 on GPU 2,3
# ---------------------------------------------------------------------------
tmux new-session -d -s step3_maskrcnn "bash -c '
  export CUDA_VISIBLE_DEVICES=2,3
  export LETTUCE_IMAGE_ROOT=${LETTUCE_IMAGE_ROOT}
  export LETTUCE_TRAIN_JSON=${LETTUCE_TRAIN_JSON}
  export LETTUCE_VAL_JSON=${LETTUCE_VAL_JSON}
  export LETTUCE_REG_DIR=${LETTUCE_REG_DIR}
  export PYTHONPATH=${CFG_DIR}:${ADELAI}
  mkdir -p ${MRCNN_OUT}
  ${PY} ${ADELAI}/tools/train_net.py \
    --num-gpus 2 \
    --config-file ${CFG_DIR}/maskrcnn_R50_step3.yaml \
    --dist-url tcp://127.0.0.1:29521 \
    ${SOLVER} OUTPUT_DIR ${MRCNN_OUT} \
    2>&1 | tee ${MRCNN_OUT}/train.log
  echo EXIT:$? >> ${MRCNN_OUT}/train.log
'"

# ---------------------------------------------------------------------------
# TensorMask-R50 on GPU 4,5
# ---------------------------------------------------------------------------
tmux new-session -d -s step3_tensormask "bash -c '
  export CUDA_VISIBLE_DEVICES=4,5
  export LETTUCE_IMAGE_ROOT=${LETTUCE_IMAGE_ROOT}
  export LETTUCE_TRAIN_JSON=${LETTUCE_TRAIN_JSON}
  export LETTUCE_VAL_JSON=${LETTUCE_VAL_JSON}
  export LETTUCE_REG_DIR=${LETTUCE_REG_DIR}
  export PYTHONPATH=${CFG_DIR}:${TMASK}
  mkdir -p ${TMASK_OUT}
  ${PY} ${TMASK}/train_net.py \
    --num-gpus 2 \
    --config-file ${CFG_DIR}/tensormask_R50_step3.yaml \
    --dist-url tcp://127.0.0.1:29522 \
    ${SOLVER} OUTPUT_DIR ${TMASK_OUT} \
    2>&1 | tee ${TMASK_OUT}/train.log
  echo EXIT:$? >> ${TMASK_OUT}/train.log
'"

echo ""
echo "======================================================"
echo " 3 training sessions launched in parallel:"
echo "   step3_blendmask  (GPU 0,1) -> ${BLEND_OUT}"
echo "   step3_maskrcnn   (GPU 2,3) -> ${MRCNN_OUT}"
echo "   step3_tensormask (GPU 4,5) -> ${TMASK_OUT}"
echo ""
echo " Monitor: tmux attach -t step3_blendmask"
echo "          tmux attach -t step3_maskrcnn"
echo "          tmux attach -t step3_tensormask"
echo "======================================================"
