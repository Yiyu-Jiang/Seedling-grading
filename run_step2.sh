#!/bin/bash
mkdir -p /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output
/ldap_shared/home/t_jyy/miniconda3/envs/detectron2/bin/python \
  /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/step2_augment.py \
  --overwrite \
  > /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/step2.log 2>&1
echo "EXIT:$?" >> /ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/step2_output/step2.log
