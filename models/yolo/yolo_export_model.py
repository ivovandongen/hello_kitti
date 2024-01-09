#!/usr/bin/env python3
from ultralytics import YOLO

for file in ['yolov8n.pt', 'yolov8n-seg.pt']:
    model = YOLO(file)
    model.export(format='coreml')
    model.export(format='onnx')
