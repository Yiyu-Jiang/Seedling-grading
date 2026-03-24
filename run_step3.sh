#!/usr/bin/env bash
# =============================================================================
# Step 3: Train 3 models in PARALLEL
# GPU 0 -> BlendMask-R50 (1 GPU)
# GPU 1 -> MaskRCNN-R50  (1 GPU)
# GPU 0,1 -> CondInst-R50 (2 GPU, starts after both above finish)
# =============================================================================

PY="/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python"
PROJECT="/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code"
ADELAI="${PROJECT}/detectron2/AdelaiDet"
CFG_DIR="${PROJECT}/mydata/configs_step3"
DATA_ROOT="${PROJECT}/mydata/step2_output"
OUT_BASE="${PROJECT}/mydata/output"
TRAIN_NET="${ADELAI}/tools/train_net.py"

export LETTUCE_IMAGE_ROOT="${DATA_ROOT}/images"
export LETTUCE_TRAIN_JSON="${DATA_ROOT}/coco/instances_train.json"
export LETTUCE_VAL_JSON="${DATA_ROOT}/coco/instances_val.json"
export LETTUCE_REG_DIR="${CFG_DIR}"
export PYTHONPATH="${CFG_DIR}:${ADELAI}:${PYTHONPATH}"

BLEND_OUT="${OUT_BASE}/step3_blendmask_R50"
MRCNN_OUT="${OUT_BASE}/step3_maskrcnn_R50"
COND_OUT="${OUT_BASE}/step3_condinst_R50"

mkdir -p "${BLEND_OUT}" "${MRCNN_OUT}" "${COND_OUT}" "${OUT_BASE}/step3_comparison"

SOLVER="SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 40000 SOLVER.STEPS (28000,36000) SOLVER.WARMUP_ITERS 1000 SOLVER.CHECKPOINT_PERIOD 5000 TEST.EVAL_PERIOD 5000"

echo "[$(date)] Launching BlendMask-R50 on GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 \
  LETTUCE_IMAGE_ROOT=${LETTUCE_IMAGE_ROOT} \
  LETTUCE_TRAIN_JSON=${LETTUCE_TRAIN_JSON} \
  LETTUCE_VAL_JSON=${LETTUCE_VAL_JSON} \
  LETTUCE_REG_DIR=${LETTUCE_REG_DIR} \
  PYTHONPATH=${PYTHONPATH} \
  ${PY} ${TRAIN_NET} \
    --num-gpus 1 \
    --config-file ${CFG_DIR}/blendmask_R50_step3.yaml \
    --dist-url tcp://127.0.0.1:29510 \
    ${SOLVER} \
    OUTPUT_DIR ${BLEND_OUT} \
    > ${BLEND_OUT}/train.log 2>&1 &
PID_BLEND=$!

echo "[$(date)] Launching MaskRCNN-R50 on GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 \
  LETTUCE_IMAGE_ROOT=${LETTUCE_IMAGE_ROOT} \
  LETTUCE_TRAIN_JSON=${LETTUCE_TRAIN_JSON} \
  LETTUCE_VAL_JSON=${LETTUCE_VAL_JSON} \
  LETTUCE_REG_DIR=${LETTUCE_REG_DIR} \
  PYTHONPATH=${PYTHONPATH} \
  ${PY} ${TRAIN_NET} \
    --num-gpus 1 \
    --config-file ${CFG_DIR}/maskrcnn_R50_step3.yaml \
    --dist-url tcp://127.0.0.1:29511 \
    ${SOLVER} \
    OUTPUT_DIR ${MRCNN_OUT} \
    > ${MRCNN_OUT}/train.log 2>&1 &
PID_MRCNN=$!

echo "[$(date)] Waiting for BlendMask (PID=${PID_BLEND}) and MaskRCNN (PID=${PID_MRCNN}) ..."
wait ${PID_BLEND}
echo "[$(date)] BlendMask done. EXIT:$?"
wait ${PID_MRCNN}
echo "[$(date)] MaskRCNN done. EXIT:$?"

echo "[$(date)] Launching CondInst-R50 on GPU 0,1 ..."
CUDA_VISIBLE_DEVICES=0,1 \
  LETTUCE_IMAGE_ROOT=${LETTUCE_IMAGE_ROOT} \
  LETTUCE_TRAIN_JSON=${LETTUCE_TRAIN_JSON} \
  LETTUCE_VAL_JSON=${LETTUCE_VAL_JSON} \
  LETTUCE_REG_DIR=${LETTUCE_REG_DIR} \
  PYTHONPATH=${PYTHONPATH} \
  ${PY} ${TRAIN_NET} \
    --num-gpus 2 \
    --config-file ${CFG_DIR}/condinst_R50_step3.yaml \
    --dist-url tcp://127.0.0.1:29512 \
    SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 SOLVER.MAX_ITER 40000 \
    SOLVER.STEPS "(28000,36000)" SOLVER.WARMUP_ITERS 1000 \
    SOLVER.CHECKPOINT_PERIOD 5000 TEST.EVAL_PERIOD 5000 \
    OUTPUT_DIR ${COND_OUT} \
    > ${COND_OUT}/train.log 2>&1
echo "[$(date)] CondInst done. EXIT:$?"

echo "[$(date)] All 3 models done. Aggregating results..."
${PY} ${CFG_DIR}/aggregate_results.py
echo "[$(date)] Results saved to ${OUT_BASE}/step3_comparison/"
