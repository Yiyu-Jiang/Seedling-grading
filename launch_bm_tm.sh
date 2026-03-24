#!/bin/bash
CODE=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code
ADELAI=$CODE/detectron2/AdelaiDet
D2=$CODE/detectron2
CFG=$CODE/mydata/configs_step6
STEP5=$CODE/mydata/step5_augmented
PY=/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python

# BlendMask on GPU 2,3 - port 29601
tmux new-session -d -s step6_blendmask -x 220 -y 50
tmux send-keys -t step6_blendmask \
  "cd $ADELAI && LETTUCE_IMAGE_ROOT=$STEP5/images LETTUCE_TRAIN_JSON=$STEP5/instances_train.json LETTUCE_VAL_JSON=$STEP5/instances_val.json LETTUCE_REG_DIR=$CFG PYTHONPATH=$CFG:$ADELAI CUDA_VISIBLE_DEVICES=2,3 $PY tools/train_net.py --config-file $CFG/blendmask_R50_step6.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:29601 2>&1 | tee $CODE/mydata/output/step6_blendmask_R50/train.log" Enter
echo "BlendMask launched on GPU 2,3"

# TensorMask on GPU 6,7 - port 29603
tmux new-session -d -s step6_tensormask -x 220 -y 50
tmux send-keys -t step6_tensormask \
  "cd $D2/projects/TensorMask && LETTUCE_IMAGE_ROOT=$STEP5/images LETTUCE_TRAIN_JSON=$STEP5/instances_train.json LETTUCE_VAL_JSON=$STEP5/instances_val.json LETTUCE_REG_DIR=$CFG PYTHONPATH=$CFG:$ADELAI CUDA_VISIBLE_DEVICES=6,7 $PY train_net.py --config-file $CFG/tensormask_R50_step6.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:29603 2>&1 | tee $CODE/mydata/output/step6_tensormask_R50/train.log" Enter
echo "TensorMask launched on GPU 6,7"

tmux ls
