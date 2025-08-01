

from ultralytics import YOLO

# Load an official or custom model
# model = YOLO("yolo11n.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

model = YOLO("yolo11l.pt")

# Perform tracking with the model
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
results = model.track("../datasets/2/RSIPAC2025/Preliminary/val/22-3.avi", show=True, tracker="bytetrack.yaml")  # with ByteTrack