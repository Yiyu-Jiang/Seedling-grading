#!/bin/bash
# Step 8: BlendMask backbone comparison
# 6 configs x 2 GPUs each
# Run 2 experiments in parallel at a time (no GPU overlap)
# Pair1: GPU2,3 + GPU4,6  (experiments 1,2)
# Pair2: GPU7,2 + GPU3,4  (experiments 3,4) -- wait for pair1
# Pair3: GPU6,7 + GPU2,3  (experiments 5,6) -- wait for pair2
# Simplest: just run all 6 sequentially on GPU2,3 to avoid conflicts
# OR run 3 pairs simultaneously with non-overlapping GPUs:
# Exp1: GPU2,3  Exp2: GPU4,6  -- simultaneous (pair A)
# Exp3: GPU7,2 conflicts. Use: Exp3: GPU7,0 if free
# Best safe plan: 3 pairs non-overlapping:
# Pair A (simultaneous): GPU2,3 and GPU4,6
# Pair B (after A): GPU2,3 and GPU4,6
# Pair C (after B): GPU2,3 and GPU4,6

PY=/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python
CODE=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code
ADELAI=$CODE/detectron2/AdelaiDet
CFG=$CODE/mydata/configs_step8
STEP5=$CODE/mydata/step5_augmented
REGDIR=$CODE/mydata/configs_step6

DATA_ENV="LETTUCE_IMAGE_ROOT=$STEP5/images LETTUCE_TRAIN_JSON=$STEP5/instances_train.json LETTUCE_VAL_JSON=$STEP5/instances_val.json LETTUCE_REG_DIR=$REGDIR"
PYPATH="PYTHONPATH=$REGDIR:$ADELAI"

launch_exp() {
    local name=$1 cfg=$2 gpu0=$3 gpu1=$4 port=$5
    local outdir=$CODE/mydata/output/step8_${name}
    mkdir -p "$outdir"
    tmux new-session -d -s "s8_${name}" -x 220 -y 50
    tmux send-keys -t "s8_${name}" \
        "cd $ADELAI && $DATA_ENV $PYPATH CUDA_VISIBLE_DEVICES=${gpu0},${gpu1} $PY tools/train_net.py --config-file $cfg --num-gpus 2 --dist-url tcp://127.0.0.1:${port} 2>&1 | tee ${outdir}/train.log" Enter
    echo "  Launched s8_${name} on GPU ${gpu0},${gpu1}"
}

echo "=== Launching Step8 Pair A: R50_3x + R101_3x ==="
launch_exp R50_3x          $CFG/blendmask_R50_3x.yaml        2 3 29701
launch_exp R101_3x         $CFG/blendmask_R101_3x.yaml       4 6 29702

echo "=== Launching Step8 Pair B: 550_R50_3x + 550_R50_dcni3 ==="
launch_exp 550_R50_3x      $CFG/blendmask_550_R50_3x.yaml    5 7 29703
launch_exp 550_R50_dcni3   $CFG/blendmask_550_R50_dcni3.yaml 0 1 29704

echo "Note: Pair C (R101_dcni3 + RT_R50) will be launched after Pair A/B complete."
echo "Launch them with:"
echo "  bash $CODE/launch_step8_pairC.sh"

echo ""
echo "Active sessions:"
tmux ls | grep s8_
echo ""
echo "Monitor logs:"
for name in R50_3x R101_3x 550_R50_3x 550_R50_dcni3; do
    echo "  tail -f $CODE/mydata/output/step8_${name}/train.log"
done
