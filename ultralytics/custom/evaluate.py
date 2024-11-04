from torch.utils.data import DataLoader
from ultralytics import YOLO

import os

from .dataset import YoloFormatDataset, rt_detr_val_transform

# os.environ["TORCH_USE_CUDA_DSA"] = "1"

if __name__ == "__main__":
    DATSET_DIR = "/home/derek_austin/ultralytics/drone_v7_synthetic"
    eval_spatial_size = 256
    val_dataset = YoloFormatDataset(
        root_dir=DATSET_DIR,
        split="valid",
        transform=rt_detr_val_transform(eval_spatial_size),
    )
    device = "cuda:0"

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=YoloFormatDataset.collate_fn,
        num_workers=8,
    )
    model = YOLO("/home/derek_austin/ultralytics/runs/detect/train4/weights/best.pt")

    
