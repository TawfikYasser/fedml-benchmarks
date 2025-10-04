import torch
from torch.serialization import safe_globals
from pathlib import Path
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys, os
import numpy as np

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
YOLOV5_ROOT = os.path.join(PROJECT_ROOT, "model/yolov5")
sys.path.insert(0, YOLOV5_ROOT)

MODEL_WEIGHTS = "./runs/train/exp/weights/model_client_0.pt"
DATA_YAML = "./data/coco128.yaml"  # update if your coco128.yaml is elsewhere
YOLO_CFG = "./model/yolov5/models/yolov5s.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(PROJECT_ROOT) / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Import YOLOv5 modules
# -----------------------------
from model.yolov5.models.yolo import DetectionModel, Model
from model.yolov5.models.common import Conv, C3, SPPF
from model.yolov5.utils.general import non_max_suppression, scale_coords as scale_boxes
from model.yolov5.utils.augmentations import letterbox

# -----------------------------
# Load dataset info
# -----------------------------
with open(DATA_YAML, 'r') as f:
    data_dict = yaml.safe_load(f)

IMG_DIR = Path(os.path.expanduser(data_dict.get('train', '/home/tawfik/fedcv_data/coco128/images/train2017')))
print(f"Image dir: {IMG_DIR}")
LABEL_DIR = Path(os.path.expanduser(data_dict.get('val', '/home/tawfik/fedcv_data/coco128/labels/train2017')))
# LABEL_DIR = Path(data_dict.get('labels', '/home/tawfik/fedcv_data/coco128/labels/train2017'))
CLASS_NAMES = list(data_dict['names'].values()) if isinstance(data_dict['names'], dict) else data_dict['names']

# -----------------------------
# Initialize model & load weights
# -----------------------------
model = Model(YOLO_CFG, ch=3, nc=len(CLASS_NAMES)).to(DEVICE)

from torch.serialization import safe_globals
from model.yolov5.models.yolo import DetectionModel

with safe_globals([DetectionModel]):
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=False)

# If the checkpoint is already a DetectionModel, use it directly
if isinstance(checkpoint, DetectionModel):
    model = checkpoint.to(DEVICE)
else:
    # Else assume it's a state_dict or dict with "model" key
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

model.eval()

# -----------------------------
# Plot helper
# -----------------------------
def plot_image(img, gt_boxes=None, pred_boxes=None, class_names=None, save_path=None):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    # Ground truth boxes
    if gt_boxes:
        for box in gt_boxes:
            cls, x_c, y_c, w, h = box
            x1 = (x_c - w/2) * img.width
            y1 = (y_c - h/2) * img.height
            rect = patches.Rectangle((x1, y1), w*img.width, h*img.height, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            if class_names:
                ax.text(x1, y1, class_names[int(cls)], color='green', fontsize=12, weight='bold')

    # Predictions
    if pred_boxes is not None:
        for *box, conf, cls in pred_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if class_names:
                ax.text(x1, y1, f"{class_names[int(cls)]} {conf:.2f}", color='red', fontsize=12, weight='bold')

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

# -----------------------------
# Run inference
# -----------------------------
test_images = list(IMG_DIR.glob("*.jpg"))[:5]
print(f"Running inference on {len(test_images)} images...")
for img_path in test_images:
    img = Image.open(img_path).convert("RGB")

    # Letterbox resize
    img_resized = letterbox(np.array(img), new_shape=640)[0]
    img_tensor = torch.tensor(img_resized).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(img_tensor)      # returns tuple
        output = preds[0]              # take first element
        pred_nms = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)[0]

    print(f"{img_path.name} predictions: {pred_nms}")
    print(f"Classes: {CLASS_NAMES}, num: {len(CLASS_NAMES)}")
    print(f"Using weights: {MODEL_WEIGHTS}")
    print(f"Device: {DEVICE}")

    # Convert to pixel xyxy
    pred_boxes = []
    if pred_nms is not None and len(pred_nms):
        pred_nms = scale_boxes(img_tensor.shape[2:], pred_nms[:, :4], img.size)
        for det in pred_nms.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            pred_boxes.append([x1, y1, x2, y2, conf, cls])

    # Load GT
    gt_boxes = []
    label_file = LABEL_DIR / (img_path.stem + ".txt")
    if label_file.exists():
        with open(label_file) as f:
            for line in f:
                gt_boxes.append([float(p) if i>0 else int(p) for i,p in enumerate(line.strip().split())])

    # Save plot
    save_path = RESULTS_DIR / f"{img_path.stem}_pred.png"
    plot_image(img, gt_boxes=gt_boxes, pred_boxes=pred_boxes, class_names=CLASS_NAMES, save_path=save_path)
    print(f"Saved: {save_path}")
