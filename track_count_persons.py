

from ultralytics import YOLO
import os


import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

results = model("pics/imageTest.webp", save=True, conf=0.5, classes = [0])



class_names = ['person']

person_counter = 0

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    probs = result.probs  # Class probabilities for classification outputs
    cls = boxes.cls.tolist()  # Convert tensor to list
    xyxy = boxes.xyxy
    xywh = boxes.xywh  # box with xywh format, (N, 4)
    conf = boxes.conf

    for i in range(len(cls)):
        class_index = cls[i]
        confidence = conf[i]

        if confidence > 0.5 and class_index == 0:
            person_counter += 1
            class_name = class_names[int(class_index)]
            print("Total persons detected:", person_counter)
            print("Class:", class_name, " Confidence:", round(confidence.item(), 2),"%")