from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")

# Train the model
train_results = model.train(
    data="/Users/derek/Desktop/dataset_collection/dataset_collection/yolo_format/dataset.yaml",  # path to dataset YAML
    epochs=20,  # number of training epochs
    imgsz=256,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
