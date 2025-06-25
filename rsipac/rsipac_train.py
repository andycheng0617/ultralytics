from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11l.yaml")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="ultralytics/cfg/datasets/rsipac.yaml", 
    epochs=100, 
    imgsz=1024,
    batch=2,
    save_period=50,
    # cache="ram",
    optimizer="Adam",
    cos_lr=True,
    lr0=0.001)