# Medical Image Classification

딥러닝 기반 의료 이미지 분류 프로젝트 4종.  
YOLOv8n-cls, EfficientNet-B5를 활용해 실제 의료 데이터셋에 적용하고, 도메인 특화 전처리 및 학습 전략을 구현했습니다.

---

## 프로젝트

### 1. 폐암 분류 (Lung Cancer Classification)

CT 스캔 이미지에서 폐암 여부를 3클래스로 분류.

- **모델**: YOLOv8n-cls
- **클래스**: Malignant / Benign / Normal
- **데이터**: 1,097장 (Train 80% / Val 10% / Test 10%)
- **설정**: Epochs 100, imgsz 512, patience 20, cos_lr
- **결과**: Top-1 Accuracy **98%대** (Malignant recall 100%)

### 2. 치과 질환 분류 (Dental Disease Classification)

구강 이미지에서 치과 질환 5종을 분류.

- **모델**: EfficientNet-B5 (2단계 파인튜닝)
- **클래스**: Calculus / Caries / Discoloration / Hypodontia / Ulcers
- **데이터**: 3,269장 (train / val / test 312장)
- **특징**: 2단계 파인튜닝, WeightedRandomSampler, AMP, EarlyStopping, ReduceLROnPlateau
- **결과**: Test Accuracy **88%** — Hypodontia F1 0.97 / Calculus F1 0.90 / Discoloration F1 0.10 (테스트 샘플 18개로 신뢰도 낮음)

### 3. 위·내시경 용종 분류 (Gastric Polyp Classification)

내시경 이미지에서 위 용종 이상 유무를 이진 분류.

- **모델**: EfficientNet-B5 (2단계 파인튜닝)
- **클래스**: abnormal / normal
- **특징**: WeightedRandomSampler, AMP, EarlyStopping, ReduceLROnPlateau
- **결과**: Test Accuracy **84%** — normal Recall 0.95 / abnormal Recall 0.67 (테스트 샘플 31개)

### 4. 피부암 분류 (Skin Cancer Classification)

피부 병변 이미지에서 피부암 5종을 분류.

- **모델**: EfficientNet-B5 (2단계 파인튜닝)
- **클래스**: MEL (흑색종) / NV (모반) / BCC (기저세포암) / BKL (양성각화증) / VASC (혈관병변)
- **데이터**: HAM10000 기반, 9,573장 (train 6,360 / val 1,922 / test 1,291)
- **특징**: WeightedRandomSampler (NV 클래스 67% 불균형 대응), AUROC per class 평가, RandomErasing
- **결과**: Test Accuracy **79%** — Macro AUROC **0.9403** (BCC 0.9896 / VASC 0.9957 / MEL 0.8694)

---

## 공통 학습 전략

| 기법 | 내용 |
|------|------|
| 2단계 파인튜닝 | Phase 1: 백본 동결, 헤드만 학습 (20 에폭, lr=3e-4) → Phase 2: 전체 파인튜닝 (50 에폭, lr=3e-5) |
| WeightedRandomSampler | 클래스 불균형 대응 — 배치 내 클래스 비율 균등화 |
| AMP (fp16) | `torch.cuda.amp.autocast` + `GradScaler` — VRAM 절감 및 학습 가속 |
| Gradient Accumulation | accumulation_steps=2 — 유효 배치 크기 2배 |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` — 학습 안정성 |
| Label Smoothing | `CrossEntropyLoss(label_smoothing=0.1)` — 과적합 방지 |
| RAdam Optimizer | 적응형 학습률 + 안정적 수렴 |
| ReduceLROnPlateau | mode='max', factor=0.5, patience=5 — val acc 정체 시 LR 자동 감소 |
| EarlyStopping | Phase 2 patience=10 — 과적합 방지 및 학습 자원 절약 |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Framework | PyTorch |
| Models | YOLOv8n-cls, EfficientNet-B5 |
| Augmentation | torchvision transforms |
| Optimizer | RAdam |
| Evaluation | scikit-learn (classification_report, AUROC) |
| Library | ultralytics, torchvision, scikit-learn, matplotlib |

---

## 디렉토리 구조

```
.
├── 01_lung_cancer/
│   ├── README.md
│   ├── 1_prepare_data.py   # 데이터 분할 + YAML 생성
│   ├── 2_train.py          # YOLOv8n-cls 학습
│   ├── 3_evaluate.py       # Top-1/5 + Confusion Matrix
│   └── 4_inference.py      # 추론 시각화
├── 02_dental_disease/
│   ├── README.md
│   ├── 1_train.py          # EfficientNet-B5 2-phase + WeightedRandomSampler
│   ├── 2_evaluate.py       # Classification Report + Confusion Matrix
│   ├── 3_inference.py      # 추론 시각화
│   ├── training_curves.png # 학습 Loss/Acc 곡선
│   └── result/             # confusion_matrix.png, inference_result.png
├── 03_gastric_polyp/
│   ├── README.md
│   ├── 1_train.py          # EfficientNet-B5 + WeightedRandomSampler
│   ├── 2_evaluate.py
│   ├── 3_inference.py
│   ├── training_curves.png
│   └── result/
└── 04_skin_cancer/
    ├── README.md
    ├── 1_train.py          # EfficientNet-B5 + WeightedRandomSampler + RandomErasing
    ├── 2_evaluate.py       # Classification Report + AUROC per class
    ├── 3_inference.py      # Top-3 확률 시각화
    ├── training_curves.png
    └── result/             # confusion_matrix.png, inference_result.png
```
