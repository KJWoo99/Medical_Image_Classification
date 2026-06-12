"""Lung Cancer - Data Preparation
Splits raw images (class/image structure) into train/val/test folders and generates YAML.
Place original images under DATA_ROOT/raw/<class>/*.jpg
"""
import os, random, shutil, yaml

DATA_ROOT = './data/lung_cancer'   # train/ val/ test/ will be created here
RAW_DIR   = os.path.join(DATA_ROOT, 'raw')  # raw/<Benign>/ <Malignant>/ <Normal>/
SPLIT     = (0.8, 0.1, 0.1)
SEED      = 2024

cls_list = sorted(os.listdir(RAW_DIR))
for split in ['train', 'val', 'test']:
    for cls in cls_list:
        os.makedirs(os.path.join(DATA_ROOT, split, cls), exist_ok=True)

random.seed(SEED)
for cls in cls_list:
    files = os.listdir(os.path.join(RAW_DIR, cls))
    random.shuffle(files)
    n      = len(files)
    n_test = int(n * SPLIT[2])
    n_val  = int(n * SPLIT[1])
    splits = {
        'test' : files[:n_test],
        'val'  : files[n_test:n_test + n_val],
        'train': files[n_test + n_val:],
    }
    for split, flist in splits.items():
        for fname in flist:
            shutil.copy(os.path.join(RAW_DIR, cls, fname),
                        os.path.join(DATA_ROOT, split, cls, fname))
    print(f"{cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

yaml_path = os.path.join(DATA_ROOT, 'lung_cancer.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump({'train': os.path.join(DATA_ROOT, 'train'),
               'val'  : os.path.join(DATA_ROOT, 'val'),
               'test' : os.path.join(DATA_ROOT, 'test'),
               'nc'   : len(cls_list),
               'names': cls_list}, f)
print(f'YAML saved: {yaml_path}')
