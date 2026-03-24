#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
export LETTUCE_IMAGE_ROOT=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/images
export LETTUCE_TRAIN_JSON=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/coco/instances_train.json
export LETTUCE_VAL_JSON=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/coco/instances_val.json
export LETTUCE_REG_DIR=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/configs_step3
export PYTHONPATH=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/configs_step3:/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/detectron2/AdelaiDet
OUT=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step3_blendmask_R50
mkdir -p $OUT
exec /ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python \
  /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/detectron2/AdelaiDet/tools/train_net.py \
  --num-gpus 2 \
  --config-file /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/configs_step3/blendmask_R50_step3.yaml \
  --dist-url tcp://127.0.0.1:29520 \
  SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 SOLVER.MAX_ITER 40000 \
  SOLVER.STEPS "(28000,36000)" SOLVER.WARMUP_ITERS 1000 \
  SOLVER.CHECKPOINT_PERIOD 5000 TEST.EVAL_PERIOD 5000 \
  OUTPUT_DIR $OUT \
  2>&1 | tee $OUT/train.log
