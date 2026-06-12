"""Lung Cancer Classification - Inference Visualization"""
import os, glob, random
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

DATA_ROOT = './data/lung_cancer'
BEST_PT   = './runs/classify/lung_cancer_cls/weights/best.pt'
N_IMAGES  = 10

model     = YOLO(BEST_PT)
test_imgs = glob.glob(os.path.join(DATA_ROOT, 'test', '*', '*.jpg'))
random.shuffle(test_imgs)

plt.figure(figsize=(20, 8))
for idx, img_path in enumerate(test_imgs[:N_IMAGES]):
    gt   = os.path.basename(os.path.dirname(img_path))
    r    = model(img_path, verbose=False)[0]
    pred = r.names[r.probs.top1]
    ax   = plt.subplot(2, 5, idx + 1)
    ax.imshow(Image.open(img_path).convert('RGB'))
    color = 'green' if gt == pred else 'red'
    ax.set_title(f'GT: {gt}\nPred: {pred}', color=color, fontsize=9)
    ax.axis('off')
plt.suptitle('Lung Cancer Classification - Inference')
os.makedirs('result', exist_ok=True)
plt.tight_layout(); plt.savefig('result/inference_result.png', dpi=120); plt.close()
