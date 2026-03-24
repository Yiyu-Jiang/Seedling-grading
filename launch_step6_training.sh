#!/bin/bash
# Step6: Launch 3 parallel training runs in tmux
# BlendMask-R50 on GPU 2,3 | MaskRCNN-R50 on GPU 4,5 | TensorMask-R50 on GPU 6,7

PY=/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python
CODE=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code
ADELAI=$CODE/detectron2/AdelaiDet
D2=$CODE/detectron2
CFG=$CODE/mydata/configs_step6
STEP5=$CODE/mydata/step5_augmented

mkdir -p $CODE/mydata/output/step6_blendmask_R50
mkdir -p $CODE/mydata/output/step6_maskrcnn_R50
mkdir -p $CODE/mydata/output/step6_tensormask_R50

# Dataset env vars - explicitly point to step5
DATA_ENV="LETTUCE_IMAGE_ROOT=$STEP5/images LETTUCE_TRAIN_JSON=$STEP5/instances_train.json LETTUCE_VAL_JSON=$STEP5/instances_val.json LETTUCE_REG_DIR=$CFG"
PYPATH="PYTHONPATH=$CFG:$ADELAI"

# ── BlendMask-R50 on GPU 2,3 ─────────────────────────────────────────────────
tmux new-session -d -s step6_blendmask -x 220 -y 50
tmux send-keys -t step6_blendmask "cd $ADELAI && \
$DATA_ENV $PYPATH CUDA_VISIBLE_DEVICES=2,3 \
$PY tools/train_net.py \
  --config-file $CFG/blendmask_R50_step6.yaml \
  --num-gpus 2 \
  SOLVER.BASE_LR 0.005 \
  SOLVER.MAX_ITER 40000 \
  2>&1 | tee $CODE/mydata/output/step6_blendmask_R50/train.log" Enter
echo "Started step6_blendmask on GPU 2,3"

# ── MaskRCNN-R50 on GPU 4,5 ──────────────────────────────────────────────────
tmux new-session -d -s step6_maskrcnn -x 220 -y 50
tmux send-keys -t step6_maskrcnn "cd $D2 && \
$DATA_ENV $PYPATH CUDA_VISIBLE_DEVICES=4,5 \
$PY tools/train_net.py \
  --config-file $CFG/maskrcnn_R50_step6.yaml \
  --num-gpus 2 \
  SOLVER.BASE_LR 0.005 \
  SOLVER.MAX_ITER 40000 \
  2>&1 | tee $CODE/mydata/output/step6_maskrcnn_R50/train.log" Enter
echo "Started step6_maskrcnn on GPU 4,5"

# ── TensorMask-R50 on GPU 6,7 ────────────────────────────────────────────────
tmux new-session -d -s step6_tensormask -x 220 -y 50
tmux send-keys -t step6_tensormask "cd $D2/projects/TensorMask && \
$DATA_ENV $PYPATH CUDA_VISIBLE_DEVICES=6,7 \
$PY train_net.py \
  --config-file $CFG/tensormask_R50_step6.yaml \
  --num-gpus 2 \
  SOLVER.BASE_LR 0.005 \
  SOLVER.MAX_ITER 40000 \
  2>&1 | tee $CODE/mydata/output/step6_tensormask_R50/train.log" Enter
echo "Started step6_tensormask on GPU 6,7"

echo ""
echo "All 3 training runs launched in tmux sessions:"
echo "  step6_blendmask  (GPU 2,3): tmux attach -t step6_blendmask"
echo "  step6_maskrcnn   (GPU 4,5): tmux attach -t step6_maskrcnn"
echo "  step6_tensormask (GPU 6,7): tmux attach -t step6_tensormask"
echo ""
echo "Monitor logs:"
echo "  tail -f $CODE/mydata/output/step6_blendmask_R50/train.log"
echo "  tail -f $CODE/mydata/output/step6_maskrcnn_R50/train.log"
echo "  tail -f $CODE/mydata/output/step6_tensormask_R50/train.log"
