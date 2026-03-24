import sys, os
# Force correct registration in ALL processes (including mp.spawn workers)
_reg_dir = os.environ.get('LETTUCE_REG_DIR', '')
if _reg_dir:
    sys.path.insert(0, _reg_dir)
try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    for _n in ('lettuce_train', 'lettuce_val'):
        if _n in DatasetCatalog:
            DatasetCatalog.remove(_n)
            try: MetadataCatalog.remove(_n)
            except Exception: pass
except Exception:
    pass
try:
    import register_step6
except Exception as e:
    print('[sitecustomize] WARNING:', e)
