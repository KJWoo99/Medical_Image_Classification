"""Lung Cancer Classification - Training  (YOLOv8s-cls)"""
import os
from ultralytics import YOLO
import ultralytics

DATA_ROOT = './data/lung_cancer'
MODEL     = 'yolov8n-cls.pt'
EPOCHS    = 100
BATCH     = 8
IMG_SIZE  = 512

if __name__ == '__main__':
    ultralytics.checks()
    model = YOLO(MODEL)
    model.train(
        data    = DATA_ROOT,
        epochs  = EPOCHS,
        batch   = BATCH,
        imgsz   = IMG_SIZE,
        device  = 0,
        patience= 20,
        name    = 'lung_cancer_cls',
        cos_lr  = True,
    )
