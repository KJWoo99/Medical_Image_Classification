# Lung Cancer Classification

YOLOv8n-cls 기반 폐 CT 이미지 3클래스 분류 모델입니다.

## Dataset

| 항목 | 내용 |
|------|------|
| 총 이미지 | 1,097장 |
| 클래스 | Malignant (악성) / Benign (양성) / Normal (정상) |
| 분할 | Train 80% / Val 10% / Test 10% |
| 입력 크기 | 512×512 |

## Model

- **YOLOv8n-cls** (ultralytics)
- Epochs: 100 (patience=20, EarlyStopping)
- Batch size: 8
- cos_lr: True (코사인 어닐링 스케줄러)
- imgsz: 512

## Results

**Test Top-1 Accuracy: 97.25%**  
**Test Top-5 Accuracy: 100%**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Benign | 0.91 | 0.83 | 0.87 |
| Malignant | 1.00 | 1.00 | 1.00 |
| Normal | 0.95 | 0.98 | 0.96 |

> Malignant(악성) recall 100% — 의료 태스크에서 가장 중요한 악성 미탐지 0건.

## Usage

```bash
pip install ultralytics torch torchvision tqdm scikit-learn matplotlib

# 1. 데이터 분할 + YAML 생성
python 1_prepare_data.py

# 2. 학습
python 2_train.py

# 3. 평가 (Top-1/5 + Confusion Matrix)
python 3_evaluate.py

# 4. 추론 시각화
python 4_inference.py
```
