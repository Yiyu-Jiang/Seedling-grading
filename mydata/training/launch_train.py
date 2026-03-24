#!/usr/bin/env python3
"""Wrapper: registers lettuce datasets then runs AdelaiDet train_net via subprocess."""
import os
import sys
import subprocess

ADELAI_TRAIN = '/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/detectron2/AdelaiDet/tools/train_net.py'
COCO_DIR = '/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/coco'

# Build env for subprocess: prepend coco dir to PYTHONPATH so register script auto-runs
env = os.environ.copy()
existing_pp = env.get('PYTHONPATH', '')
env['PYTHONPATH'] = COCO_DIR + (':' + existing_pp if existing_pp else '')
env.setdefault('LETTUCE_DATA_ROOT', COCO_DIR)
env.setdefault('LETTUCE_IMAGE_ROOT',
    '/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/standardized_step1_half/merged/images')

# Build a small sitecustomize snippet to register datasets at interpreter start
registration_code = '''
import sys, os
sys.path.insert(0, os.environ.get("LETTUCE_DATA_ROOT",
    "/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/coco"))
try:
    import register_lettuce_coco
except Exception as e:
    print("[register] WARNING:", e)
'''

# Write a sitecustomize.py into a temp dir and prepend it to PYTHONPATH
import tempfile
tmpdir = tempfile.mkdtemp(prefix='blendmask_reg_')
with open(os.path.join(tmpdir, 'sitecustomize.py'), 'w') as f:
    f.write(registration_code)
env['PYTHONPATH'] = tmpdir + ':' + env['PYTHONPATH']

# Forward all sys.argv arguments (train_net.py args)
cmd = [sys.executable, ADELAI_TRAIN] + sys.argv[1:]
print('[launch_train] Running:', ' '.join(cmd))
result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)
