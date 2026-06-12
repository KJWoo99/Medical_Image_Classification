"""Skin Cancer Classification - Inference visualization"""
import os
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B5_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'BKL', 'VASC']
DATA_ROOT   = './data/skin_cancer'
MODEL_PATH  = './best.pt'
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
loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)

model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(2048, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for idx, (x, y) in enumerate(loader):
    if idx >= 10:
        break
    with torch.no_grad():
        out   = model(x.to(device))
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
    pred = int(probs.argmax())
    top3 = probs.argsort()[::-1][:3]

    img = x.squeeze().numpy().transpose(1, 2, 0)
    img = np.clip(img * std + mean, 0, 1)
    axes[idx].imshow(img)

    top3_str = '\n'.join(f'{CLASS_NAMES[i]}: {probs[i]*100:.1f}%' for i in top3)
    color = 'green' if pred == y.item() else 'red'
    axes[idx].set_title(f'GT: {CLASS_NAMES[y.item()]}\n{top3_str}',
                        color=color, fontsize=8)
    axes[idx].axis('off')

plt.suptitle('Skin Cancer - Inference (Top-3 Probabilities)', fontsize=12)
os.makedirs('result', exist_ok=True)
plt.tight_layout(); plt.savefig('result/inference_result.png', dpi=120); plt.close()
print('Saved → result/inference_result.png')
