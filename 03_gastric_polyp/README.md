# Gastric Polyp Classification

EfficientNet-B5 기반 내시경 이미지 이진 분류 모델입니다.  
WeightedRandomSampler로 클래스 불균형을 처리하고, 2단계 파인튜닝으로 학습합니다.

## Dataset

| 항목 | 내용 |
|------|------|
| 클래스 | abnormal (이상) / normal (정상) |
| 구조 | train / val / test 폴더 분리 (ImageFolder 형식) |
| 입력 크기 | 456×456 |
| 데이터 출처 | 위 내시경 이미지 |

> 데이터셋 규모가 소규모(전체 334장)로, 성능의 한계가 있음.

## Model

- **EfficientNet-B5** (`EfficientNet_B5_Weights.IMAGENET1K_V1`)
- Phase 1: 백본 동결, 헤드 학습 (20 epochs, lr=3e-4)
- Phase 2: 전체 파인튜닝 (50 epochs, lr=3e-5)

## Training Strategy

| 기법 | 내용 |
|------|------|
| WeightedRandomSampler | 클래스 불균형 대응 |
| AMP (fp16) | `torch.cuda.amp.autocast` + `GradScaler` |
| Gradient Accumulation | accumulation_steps=2 |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` |
| Label Smoothing | `CrossEntropyLoss(label_smoothing=0.1)` |
| RAdam Optimizer | 적응형 학습률 + 안정적 수렴 |
| ReduceLROnPlateau | mode='max', factor=0.5, patience=5, min_lr=1e-7 |
| EarlyStopping | Phase 2 patience=10 |

## Usage

```bash
pip install torch torchvision tqdm scikit-learn matplotlib

# 1. 학습 (Phase 1 → Phase 2, best.pt 자동 저장)
python 1_train.py

# 2. 평가 (Classification Report + Confusion Matrix)
python 2_evaluate.py

# 3. 추론 시각화
python 3_inference.py
```
