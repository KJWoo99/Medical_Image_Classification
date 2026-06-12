"""Lung Cancer Classification - Evaluation"""
import os, glob
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ultralytics import YOLO

DATA_ROOT   = './data/lung_cancer'
BEST_PT     = './runs/classify/lung_cancer_cls/weights/best.pt'

if __name__ == '__main__':
    model       = YOLO(BEST_PT)
    metrics     = model.val(split='test', workers=0)
    print(f'Top-1: {metrics.top1:.4f}  Top-5: {metrics.top5:.4f}')

    test_root   = os.path.join(DATA_ROOT, 'test')
    class_names = sorted(os.listdir(test_root))
    c2i         = {c: i for i, c in enumerate(class_names)}

    all_preds, all_labels = [], []
    for cls in class_names:
        for img_path in glob.glob(os.path.join(test_root, cls, '*.jpg')):
            all_preds.append(model(img_path, verbose=False)[0].probs.top1)
            all_labels.append(c2i[cls])

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title('Confusion Matrix - Lung Cancer')
    os.makedirs('result', exist_ok=True)
    plt.tight_layout(); plt.savefig('result/confusion_matrix.png', dpi=120); plt.close()
