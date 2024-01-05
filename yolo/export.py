#!/usr/bin/env python3
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='coreml')
model.export(format='onnx')
