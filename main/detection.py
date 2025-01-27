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
            # Convert RGB to BGR for YOLO processing
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            model = YOLO(self.MODELS[model_name])
            results = model(img_bgr)[0]
            
            # Plot directly using cv2 instead of YOLO's built-in plotting
            plotted_img = img_bgr.copy()
            
            for box in results.boxes:
                # Get box coordinates, confidence and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                if conf > treshold:
                    # Get random color for this class
                    color = [int(c * 255) for c in self.colors.get(str(cls), (1.0, 1.0, 1.0))]
                    
                    # Draw rectangle
                    cv2.rectangle(
                        plotted_img, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        color, 
                        2
                    )
                    
                    # Add label
                    label = f"{results.names[cls]}: {conf:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    cv2.rectangle(
                        plotted_img,
                        (int(x1), int(y1) - label_height - 5),
                        (int(x1) + label_width, int(y1)),
                        color,
                        -1,
                    )
                    
                    cv2.putText(
                        plotted_img,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            
            return plotted_img

        else:
            image = transforms.ToTensor()(img)
            image = image.unsqueeze(0)
            predictions = self.MODELS[model_name].eval()(image)
            self.process_predictions(image, predictions, treshold=treshold)
            result = cv2.imread("result.jpg")
            return result

    def process_predictions(self, image, predictions, treshold=0.5):
        # Convert tensor to numpy array
        image = (image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get predictions
        boxes = predictions[0]["boxes"].detach().cpu().numpy()
        scores = predictions[0]["scores"].detach().cpu().numpy()
        labels = predictions[0]["labels"].detach().cpu().numpy()

        # Filter predictions by threshold
        mask = scores > treshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Draw boxes and labels using OpenCV
        for box, score, label in zip(boxes, scores, labels):
            # Get color for this class
            color = [int(c * 255) for c in self.colors[self.COCO_INSTANCE_CATEGORY_NAMES[label]]]
            
            # Draw rectangle
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color,
                2
            )
            
            # Add label
            label_text = f"{self.COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1]) - label_height - 5),
                (int(box[0]) + label_width, int(box[1])),
                color,
                -1,
            )
            
            # Draw label text
            cv2.putText(
                image,
                label_text,
                (int(box[0]), int(box[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Save result directly using OpenCV
        cv2.imwrite("result.jpg", image)
        return image  
