#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/configs_step3:/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/detectron2/AdelaiDet:/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/detectron2/projects/TensorMask
export LETTUCE_IMAGE_ROOT=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/images
export LETTUCE_TRAIN_JSON=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/coco/instances_train.json
export LETTUCE_VAL_JSON=/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/coco/instances_val.json
mkdir -p /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step3_tensormask_R50
exec /ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python \
  /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/detectron2/projects/TensorMask/train_net.py \
  --num-gpus 2 \
  --config-file /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/configs_step3/tensormask_R50_step3.yaml \
  --dist-url tcp://127.0.0.1:29522 \
  OUTPUT_DIR /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step3_tensormask_R50
