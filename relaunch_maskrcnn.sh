#!/bin/bash
CODE=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code
D2=$CODE/detectron2
ADELAI=$CODE/detectron2/AdelaiDet
CFG=$CODE/mydata/configs_step6
STEP5=$CODE/mydata/step5_augmented
PY=/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python

tmux kill-session -t step6_maskrcnn 2>/dev/null
sleep 1
tmux new-session -d -s step6_maskrcnn -x 220 -y 50
tmux send-keys -t step6_maskrcnn \
  "cd $D2 && LETTUCE_IMAGE_ROOT=$STEP5/images LETTUCE_TRAIN_JSON=$STEP5/instances_train.json LETTUCE_VAL_JSON=$STEP5/instances_val.json LETTUCE_REG_DIR=$CFG PYTHONPATH=$CFG:$ADELAI CUDA_VISIBLE_DEVICES=4,5 $PY tools/train_net.py --config-file $CFG/maskrcnn_R50_step6.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:29602 2>&1 | tee $CODE/mydata/output/step6_maskrcnn_R50/train.log" Enter
echo MaskRCNN_relaunched
tmux ls
