"""Skin Cancer Classification - Evaluation"""
import os
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B5_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'BKL', 'VASC']
DATA_ROOT   = './data/skin_cancer'
MODEL_PATH  = './best.pt'
BATCH_SIZE  = 16
IMG_SIZE    = 456
IMG_EXT     = '.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df        = pd.read_csv(csv_file)
        self.img_dir   = img_dir
        self.transform = transform
        self.labels    = self.df.iloc[:, 1:].values.argmax(axis=1).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0] + IMG_EXT)
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds = SkinCancerDataset(
    os.path.join(DATA_ROOT, 'test.csv'),
    os.path.join(DATA_ROOT, 'test'),
    transform=val_tf,
)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(2048, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for x, y in test_ld:
        out   = model(x.to(device))
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)

print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# AUROC (OvR) — standard metric for HAM10000
try:
    from sklearn.preprocessing import label_binarize
    labels_bin = label_binarize(all_labels, classes=list(range(len(CLASS_NAMES))))
    auc = roc_auc_score(labels_bin, all_probs, multi_class='ovr', average=None)
    print('Per-class AUROC:')
    for name, score in zip(CLASS_NAMES, auc):
        print(f'  {name}: {score:.4f}')
    print(f'  Macro avg: {auc.mean():.4f}')
except Exception as e:
    print(f'AUROC skipped: {e}')

os.makedirs('result', exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
plt.title('Confusion Matrix - Skin Cancer')
plt.tight_layout(); plt.savefig('result/confusion_matrix.png', dpi=120); plt.close()
print('Saved → result/confusion_matrix.png')
