

# Build a new model from YAML and start training from scratch
yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

# Start training from a pretrained *.pt model
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

# Build a new model from YAML, transfer pretrained weights to it and start training
yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640


# -------------------------------------------------------------------------------------------------------------------------





yolo detect train data=ultralytics/cfg/datasets/qiyuan.yaml model=yolo11l.yaml pretrained=yolo11l.pt epochs=100 imgsz=640 batch=4 save_period=50 cache=ram optimizer=Adam cos_lr=True lr0=0.001

















