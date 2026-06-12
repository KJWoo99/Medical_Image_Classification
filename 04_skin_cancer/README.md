# Skin Cancer Classification

EfficientNet-B5 기반 피부 병변 이미지 5클래스 분류 모델입니다.  
HAM10000 데이터셋의 심각한 클래스 불균형(NV ~67%)을 WeightedRandomSampler로 처리하고,  
의료 표준 평가 지표인 AUROC를 클래스별로 산출합니다.

## Dataset

| 항목 | 내용 |
|------|------|
| 기반 데이터셋 | HAM10000 (Human Against Machine with 10000 training images) |
| 총 이미지 | 9,573장 (train 6,360 / val 1,922 / test 1,291) |
| 클래스 | MEL (흑색종) / NV (모반) / BCC (기저세포암) / BKL (양성각화증) / VASC (혈관병변) |
| 구조 | train/ val/ test/ + CSV 파일 (one-hot 라벨) |
| 입력 크기 | 456×456 |

### 클래스 설명

| 클래스 | 한국어 | 특징 |
|--------|--------|------|
| MEL | 흑색종 | 가장 위험한 피부암, 멜라닌 세포에서 발생 |
| NV | 모반 (점) | 양성, 데이터셋의 약 67% 차지 |
| BCC | 기저세포암 | 가장 흔한 피부암, 성장 느림 |
| BKL | 양성각화증 | 지루각화증 포함, 양성 병변 |
| VASC | 혈관병변 | 혈관종 등 혈관 관련 병변 |

## Model

- **EfficientNet-B5** (`EfficientNet_B5_Weights.IMAGENET1K_V1`)
- Phase 1: 백본 동결, 헤드 학습 (20 epochs, lr=3e-4)
- Phase 2: 전체 파인튜닝 (50 epochs, lr=3e-5)

## Training Strategy

| 기법 | 내용 |
|------|------|
| WeightedRandomSampler | NV 클래스 67% 불균형 대응 — 클래스별 역빈도 가중치 |
| AMP (fp16) | `torch.cuda.amp.autocast` + `GradScaler` |
| Gradient Accumulation | accumulation_steps=2 |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` |
| Label Smoothing | `CrossEntropyLoss(label_smoothing=0.1)` |
| RAdam Optimizer | 적응형 학습률 + 안정적 수렴 |
| ReduceLROnPlateau | mode='max', factor=0.5, patience=5, min_lr=1e-7 |
| EarlyStopping | Phase 2 patience=10 |

## Augmentation

```python
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(30)
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # 피부 털/아티팩트 simulate
```

## Evaluation

- Classification Report (precision / recall / F1 per class)
- Confusion Matrix
- **AUROC per class** (HAM10000 표준 평가 지표, OvR 방식)

## Data Format

CSV 형식 (one-hot 인코딩):
```
image_name, MEL, NV, BCC, BKL, VASC
ISIC_0000000, 0, 1, 0, 0, 0
...
```

## Usage

```bash
pip install torch torchvision tqdm scikit-learn matplotlib pandas

# 1. 학습 (Phase 1 → Phase 2, best.pt 자동 저장)
python 1_train.py

# 2. 평가 (Classification Report + AUROC + Confusion Matrix)
python 2_evaluate.py

# 3. 추론 시각화 (Top-3 확률 표시)
python 3_inference.py
```
