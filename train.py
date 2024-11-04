from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")

# Train the model
train_results = model.train(
    data="/home/derek_austin/ultralytics/drone_v7_synthetic/dataset.yaml",  # path to dataset YAML
    epochs=20,  # number of training epochs
    imgsz=256,  # training image size
    device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
