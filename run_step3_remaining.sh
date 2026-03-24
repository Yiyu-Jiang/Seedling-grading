#!/usr/bin/env bash
# Step 3: Run MaskRCNN-R50 then CondInst-R50 sequentially (BlendMask already done or running)
# Each uses 2 GPUs, 40000 iters, LR=0.005

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
export CUDA_VISIBLE_DEVICES="0,1"

SOLVER="SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 SOLVER.MAX_ITER 40000 SOLVER.STEPS (28000,36000) SOLVER.WARMUP_ITERS 1000 SOLVER.CHECKPOINT_PERIOD 5000 TEST.EVAL_PERIOD 5000"

run_model() {
    local NAME="$1" CFG="$2" OUT="$3" PORT="$4"
    mkdir -p "${OUT}"
    echo "[$(date)] === Starting ${NAME} ==="
    ${PY} ${TRAIN_NET} \
        --num-gpus 2 \
        --config-file "${CFG}" \
        --dist-url "tcp://127.0.0.1:${PORT}" \
        ${SOLVER} \
        OUTPUT_DIR "${OUT}" \
        2>&1 | tee "${OUT}/train.log"
    echo "[$(date)] === ${NAME} finished EXIT:$? ==="
}

# Wait for BlendMask to finish if still running
echo "[$(date)] Waiting for any running BlendMask process to finish..."
while pgrep -f 'step3_blendmask_R50' > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] BlendMask done (or not running). Starting MaskRCNN-R50..."

run_model "MaskRCNN-R50" \
    "${CFG_DIR}/maskrcnn_R50_step3.yaml" \
    "${OUT_BASE}/step3_maskrcnn_R50" \
    "29511"

run_model "CondInst-R50" \
    "${CFG_DIR}/condinst_R50_step3.yaml" \
    "${OUT_BASE}/step3_condinst_R50" \
    "29512"

echo "[$(date)] All models done. Aggregating results..."
${PY} "${CFG_DIR}/aggregate_results.py"
echo "[$(date)] Done. Results at ${OUT_BASE}/step3_comparison/"
