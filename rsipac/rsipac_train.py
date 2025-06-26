import sys

sys.path.append('/home/zwcheng/Work/competition/rsipac/ultralytics/')

from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11l.yaml").load('yolo11l-cls.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="ultralytics/cfg/datasets/rsipac.yaml", 
    epochs=100, 
    imgsz=640,
    batch=8,
    save_period=50,
    cache="disk",
    optimizer="Adam",
    cos_lr=True,
    lr0=0.01,
    close_mosaic=True,
    single_cls=True,
    amp=True,
    plots=True,
    deterministic=False,
)