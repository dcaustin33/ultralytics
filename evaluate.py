import os

import torch
import wandb
from torch.utils.data import DataLoader

from ultralytics import YOLO
from ultralytics.custom.dataset import YoloFormatDataset, rt_detr_val_transform
from ultralytics.custom.hungarian_matcher import HungarianMatcher
from ultralytics.custom.trainer import Trainer

if __name__ == "__main__":
    DATSET_DIR = "/home/derek_austin/ultralytics/drone_v7_synthetic"
    eval_spatial_size = 256
    val_dataset = YoloFormatDataset(
        root_dir=DATSET_DIR,
        split="valid",
        transform=rt_detr_val_transform(eval_spatial_size),
    )
    device = "cuda"

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=YoloFormatDataset.collate_fn,
        num_workers=8,
    )
    model = YOLO("/home/derek_austin/ultralytics/runs/detect/train4/weights/best.pt")
    model.to(device)
    _ = model.model.eval()
    # model.predictor.args.verbose = False
    
    wandb.init(project="rt-detr-aerial", name=f"yolo_medium_v7_synthetic", )


    matcher = HungarianMatcher()
    trainer = Trainer(model=model, matcher=matcher)
    trainer.validate_loop(
        val_dataloader=val_dataloader,
        device=device,
        epoch=0,
        save_output_file_path="test.jsonl",
    )