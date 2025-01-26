import cv2
import numpy as np
from ultralytics import YOLO
import torchvision
import torch
import PIL
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from torchvision.models import detection
import numpy as np


class Detector:
    def __init__(self):
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            "__background__",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "N/A",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "N/A",
            "backpack",
            "umbrella",
            "N/A",
            "N/A",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "N/A",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "N/A",
            "dining table",
            "N/A",
            "N/A",
            "toilet",
            "N/A",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "N/A",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        self.MODELS = {
            "frcnn-resnet": detection.fasterrcnn_resnet50_fpn(pretrained=True),
            "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=True
            ),
            "retinanet": detection.retinanet_resnet50_fpn(pretrained=True),
            "COCO": "yolo11n.pt",
            "Open Images": "yolov8n-oiv7.pt",
        }

        COLORS = {}
        for name in self.COCO_INSTANCE_CATEGORY_NAMES:
            COLORS[name] = (random.random(), random.random(), random.random())
        self.colors = COLORS

    def get_prediction(self, img, model_name="COCO", treshold=0.5):
        if model_name in ["COCO", "Open Images"]:
            model = YOLO(self.MODELS[model_name])
            result = model([img])[0]
            result.save(filename="result.jpg")

        else:
            image = transforms.ToTensor()(img)
            image = image.unsqueeze(0)
            predictions = self.MODELS[model_name].eval()(image)
            self.process_predictions(image, predictions, treshold=treshold)

        result = cv2.imread("result.jpg")
        return result

    def process_predictions(self, image, predictions, treshold=0.5):
       
        image = image.squeeze(0).permute(1, 2, 0).numpy()
        
        boxes = predictions[0]["boxes"].detach().cpu().numpy()
        scores = predictions[0]["scores"].detach().cpu().numpy()
        labels = predictions[0]["labels"].detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(8, 8), dpi=100)
        ax.imshow(image)
        mask = scores > treshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        for box, score, label in zip(boxes, scores, labels):
            color = self.colors[self.COCO_INSTANCE_CATEGORY_NAMES[label]]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            
            ax.text(
                box[0],
                box[1],
                f"{self.COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}",
                color=color,
                fontsize=8,
            )

        plt.axis("off")
        plt.savefig("result.jpg", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)  
