#!/bin/bash
# Step8 Pair C: R101_dcni3 + RT_R50 (run after Pair A/B complete)
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
echo "=== Launching Step8 Pair C: R101_dcni3 + RT_R50 ==="
launch_exp R101_dcni3 $CFG/blendmask_R101_dcni3.yaml 2 3 29705
launch_exp RT_R50     $CFG/blendmask_RT_R50.yaml     4 6 29706
tmux ls | grep s8_
