"""Evaluation - load best.pt and evaluate on test set."""
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B5_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision.datasets import ImageFolder
import os

CLASS_NAMES = ['abnormal', 'normal']
DATA_ROOT   = './data/gastric_polyp'
MODEL_PATH  = './best.pt'
BATCH_SIZE  = 16
IMG_SIZE    = 456

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageFolder(os.path.join(DATA_ROOT, 'test'), transform=val_tf)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(2048, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        all_preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())

print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
plt.title('Confusion Matrix - Gastric Polyp')
os.makedirs('result', exist_ok=True)
plt.tight_layout(); plt.savefig('result/confusion_matrix.png', dpi=120); plt.close()
