#!/usr/bin/env bash
# =============================================================================
# BlendMask Backbone Comparison - 40k iters, 2-GPU, LR=0.005, IMS_PER_BATCH=4
# Configs: R_50_3x, R_101_3x, 550_R_50_3x, 550_R_50_dcni3_5x,
#          R_101_dcni3_5x, RT_R_50_4x_bn-head_syncbn_shtw
# =============================================================================
set -e

CONDA_BASE="/ldap_shared/home/t_jyy/miniconda3"
ENV_NAME="detectron2"
PROJECT="/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code"
ADELAI_DIR="${PROJECT}/detectron2/AdelaiDet"
DATA_ROOT="${PROJECT}/mydata"
CONFIG_DIR="${DATA_ROOT}/configs_40k"
COCO_DIR="${DATA_ROOT}/coco"
IMAGE_DIR="${DATA_ROOT}/standardized_step1_half/merged/images"

export LETTUCE_DATA_ROOT="${COCO_DIR}"
export LETTUCE_IMAGE_ROOT="${IMAGE_DIR}"

# --------------------------------------------------------------------------
# 0. Activate conda env
# --------------------------------------------------------------------------
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "======================================"
echo " Python : $(which python)"
echo " Started: $(date)"
echo "======================================"

# --------------------------------------------------------------------------
# 1. Install PyTorch (CUDA 12.1) if not present
# --------------------------------------------------------------------------
if ! python -c "import torch" 2>/dev/null; then
    echo "[SETUP] Installing PyTorch 2.1.2 + CUDA 12.1 ..."
    pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121
fi
PYTORCH_VER=$(python -c 'import torch; print(torch.__version__)')
echo "[SETUP] PyTorch ${PYTORCH_VER}, CUDA=$(python -c 'import torch; print(torch.version.cuda)')"

# --------------------------------------------------------------------------
# 2. Install detectron2 v0.6 if not present
# --------------------------------------------------------------------------
if ! python -c "import detectron2" 2>/dev/null; then
    echo "[SETUP] Installing detectron2 v0.6 ..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
fi
echo "[SETUP] detectron2 $(python -c 'import detectron2; print(detectron2.__version__)')"

# --------------------------------------------------------------------------
# 3. Clone AdelaiDet if not present, then install
# --------------------------------------------------------------------------
mkdir -p "${PROJECT}/detectron2"
if [ ! -d "${ADELAI_DIR}" ]; then
    echo "[SETUP] Cloning AdelaiDet ..."
    git clone https://github.com/aim-uofa/AdelaiDet.git "${ADELAI_DIR}"
fi
if ! python -c "import adet" 2>/dev/null; then
    echo "[SETUP] Installing AdelaiDet ..."
    pip install -e "${ADELAI_DIR}"
fi
echo "[SETUP] AdelaiDet: $(python -c 'import adet; print(adet.__file__)')"

# --------------------------------------------------------------------------
# 4. Install remaining deps
# --------------------------------------------------------------------------
pip install pycocotools opencv-python-headless -q 2>/dev/null || true

TRAIN_NET="${ADELAI_DIR}/tools/train_net.py"

# --------------------------------------------------------------------------
# 5. Helper: run one experiment (2 GPUs, datasets pre-registered via wrapper)
# --------------------------------------------------------------------------
run_exp() {
    local NAME="$1"
    local CFG="$2"
    local OUT_DIR="$3"

    echo ""
    echo "############################################################"
    echo " EXPERIMENT : ${NAME}"
    echo " Config     : ${CFG}"
    echo " Output     : ${OUT_DIR}"
    echo " Start      : $(date)"
    echo "############################################################"

    mkdir -p "${OUT_DIR}"

    CUDA_VISIBLE_DEVICES=0,1 \
    LETTUCE_DATA_ROOT="${LETTUCE_DATA_ROOT}" \
    LETTUCE_IMAGE_ROOT="${LETTUCE_IMAGE_ROOT}" \
    python "${PROJECT}/mydata/training/launch_train.py" \
        --num-gpus 2 \
        --config-file "${CFG}" \
        --dist-url "tcp://127.0.0.1:$((29500 + RANDOM % 1000))" \
        SOLVER.IMS_PER_BATCH 4 \
        SOLVER.BASE_LR 0.005 \
        SOLVER.MAX_ITER 40000 \
        SOLVER.STEPS "(28000,36000)" \
        SOLVER.WARMUP_ITERS 1000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        TEST.EVAL_PERIOD 2000 \
        OUTPUT_DIR "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/train.log"

    echo "[DONE] ${NAME} finished at $(date)"
    echo ""
}

# --------------------------------------------------------------------------
# 6. Run all 6 experiments sequentially
# --------------------------------------------------------------------------
run_exp \
    "R_50_3x" \
    "${CONFIG_DIR}/blendmask_lettuce_R_50_3x_40k.yaml" \
    "${DATA_ROOT}/output/blendmask_40k_R_50_3x"

run_exp \
    "R_101_3x" \
    "${CONFIG_DIR}/blendmask_lettuce_R_101_3x_40k.yaml" \
    "${DATA_ROOT}/output/blendmask_40k_R_101_3x"

run_exp \
    "550_R_50_3x" \
    "${CONFIG_DIR}/blendmask_lettuce_550_R_50_3x_40k.yaml" \
    "${DATA_ROOT}/output/blendmask_40k_550_R_50_3x"

run_exp \
    "550_R_50_dcni3_5x" \
    "${CONFIG_DIR}/blendmask_lettuce_550_R_50_dcni3_5x_40k.yaml" \
    "${DATA_ROOT}/output/blendmask_40k_550_R_50_dcni3_5x"

run_exp \
    "R_101_dcni3_5x" \
    "${CONFIG_DIR}/blendmask_lettuce_R_101_dcni3_5x_40k.yaml" \
    "${DATA_ROOT}/output/blendmask_40k_R_101_dcni3_5x"

run_exp \
    "RT_R_50_4x_bn_head_syncbn_shtw" \
    "${CONFIG_DIR}/blendmask_lettuce_RT_R_50_4x_bn_head_syncbn_shtw_40k.yaml" \
    "${DATA_ROOT}/output/blendmask_40k_RT_R_50_4x_bn_head_syncbn_shtw"

echo ""
echo "============================================================"
echo " ALL 6 EXPERIMENTS COMPLETE"
echo " Finished: $(date)"
echo "============================================================"
