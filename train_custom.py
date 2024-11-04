import os
import torch
from torch.utils.data import DataLoader
from ultralytics.custom.hungarian_matcher import HungarianMatcher
import wandb

from ultralytics import YOLO
from ultralytics.custom.dataset import YoloFormatDataset, rt_detr_train_transform, rt_detr_val_transform
from ultralytics.custom.trainer import Trainer

if __name__ == "__main__":
    model_name = "yolov8m"
    eval_spatial_size = 256
    batch_size = 16
    dataset_path = (
        "/Users/derek/Desktop/dataset_collection/dataset_collection/yolo_format"
    )
    epochs = 10
    no_wandb = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_model_path = f"/Users/derek/Desktop/ultralytics/custom_runs/{model_name}"
    config = {
        "model": model_name,
        "batch_size": batch_size,
        "eval_spatial_size": eval_spatial_size,
        "dataset_path": dataset_path,
        "no_wandb": no_wandb,
        "epochs": epochs,
    }
    
    if not no_wandb:
        wandb = wandb.init(project="rt-detr-aerial", config=config)
    
    model = YOLO(f"{model_name}.pt")
    dataset = YoloFormatDataset(
        dataset_path,
        split="train",
        transform=rt_detr_train_transform(eval_spatial_size),
    )
    val_dataset = YoloFormatDataset(
        dataset_path,
        split="val",
        transform=rt_detr_val_transform(eval_spatial_size),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=YoloFormatDataset.ultralytics_collate_fn,
        num_workers=12,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=YoloFormatDataset.ultralytics_collate_fn,
        num_workers=12,
    )
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    if save_model_path is not None:
        if os.path.exists(save_model_path):
            # if input("Checkpoint path exists, delete? [y/n]") == "y":
            # shutil.rmtree(args.save_model_path)
            pass
        os.makedirs(save_model_path, exist_ok=True)
    

    trainer = Trainer(model, matcher=HungarianMatcher())
    trainer.train_loop(
        train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        epochs=epochs,
        val_dataloader=val_dataloader,
        val_freq=3,
        wandb=wandb,
        save_model_path=save_model_path,
    )
