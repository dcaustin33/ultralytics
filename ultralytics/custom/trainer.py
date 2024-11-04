import json
import math
import os
from typing import Any, Iterable, List, Optional

import torch
import tqdm

from ultralytics.custom.postprocessing import box_metrics
from ultralytics.custom.schema import Metrics


class Trainer:
    
    def __init__(self, model: torch.nn.Module, matcher: torch.nn.Module = None):
        self.model = model
        self.keys_to_keep = ["labels", "boxes"]
        self.matcher = matcher
        
    def run_metrics_loop(
        self,
        current_metrics: Metrics,
        model_output: torch.Tensor,
        targets: torch.Tensor,
        loss_value: float,
        confidence_threshold: float = 0.5,
    ) -> Metrics:
        """
        Update the current metrics with the model output and targets.

        Args:
            current_metrics (Metrics): The current metrics to be updated.
            model_output (torch.Tensor): The output from the model, containing
                predicted logits and boxes.
            targets (torch.Tensor): The ground truth targets, containing
                labels and boxes.
            loss_value (float): The loss value for the current batch.
            confidence_threshold (float, optional): The confidence threshold
                for considering a prediction. Defaults to 0.5.
        """
        for im_idx in range(len(model_output)):
            pred_logits = model_output[im_idx].boxes.conf
            pred_boxes = model_output[im_idx].boxes.xywhn
            pred_labels = model_output[im_idx].boxes.cls
            
            metrics = box_metrics(
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                target_classes=targets[im_idx]["labels"],
                target_boxes=targets[im_idx]["boxes"],
                matcher=self.matcher,
                confidence_threshold=confidence_threshold,
                loss_value=loss_value,
            )
            if current_metrics is None:
                current_metrics = metrics
            else:
                current_metrics += metrics
        return current_metrics
    
    
    def write_output(
        self,
        model_output: torch.Tensor,
        targets: torch.Tensor,
        img_names: List[str],
        save_output_paths: str,
    ) -> None:
        """
        Write the output boxes to disk for examination.

        Args:
            outputs (torch.Tensor): The output from the model, containing
                predicted logits and boxes.
            targets (torch.Tensor): The ground truth targets, containing
                labels and boxes.
            img_names (List[str]): The list of image names corresponding
                to the outputs.
            save_output_paths (str): The path to save the output files.
                Should be a jsonl file.
        """
        for im_idx in range(len(model_output)):
            pred_logits = model_output[im_idx].boxes.conf
            pred_boxes = model_output[im_idx].boxes.xywhn
            pred_labels = model_output[im_idx].boxes.cls
            # write to jsonl file
            output_dict = {
                "img_name": img_names[im_idx],
                "predictions": {
                    "bboxes": pred_boxes.tolist(),
                    "class_ids": pred_labels.tolist(),
                    "confidences": pred_logits.tolist(),
                },
                "targets": {
                    "bboxes": targets[im_idx]["boxes"].tolist(),
                    "class_ids": targets[im_idx]["labels"].tolist(),
                },
            }
            if os.path.exists(save_output_paths):
                with open(save_output_paths, "a") as f:
                    f.write(json.dumps(output_dict) + "\n")
            else:
                with open(save_output_paths, "w") as f:
                    f.write(json.dumps(output_dict) + "\n")
        return

 
    
    def validate_loop(
            self,
            val_dataloader: Iterable,
            device: torch.device,
            epoch: int,
            steps: Optional[int] = None,
            save_metrics_path: Optional[str] = None,
            save_output_file_path: Optional[str] = None,
            wandb: Optional[Any] = None,
        ) -> None:
            """
            Run the validation loop for one epoch.

            Args:
                val_dataloader (Iterable): DataLoader providing the validation data.
                device (torch.device): Device to run the validation on
                    (e.g., 'cpu' or 'cuda').
                epoch (int): Current epoch number.
                steps (Optional[int], optional): Number of steps to run in the validation
                    loop. Defaults to None (run all steps).
                save_metrics_path (Optional[str], optional): Path to save the validation
                    metrics. Defaults to None. Should be jsonl file if provided.
                save_output_file_path (Optional[str], optional): Path to save the output
                    files. Defaults to None.
                wandb (Optional[Any], optional): Weights and Biases logging object.
                    Defaults to None.
            """
            self.model.model.eval()
            print(f"The number of steps are {steps}")

            if save_output_file_path and os.path.exists(save_output_file_path):
                os.remove(save_output_file_path)

            current_metrics = None
            old_wandb_output = None
            for idx, (samples, targets) in tqdm.tqdm(enumerate(val_dataloader)):
                samples = samples.to(device)
                if save_output_file_path:
                    img_names = [t["img_name"] for t in targets]
                targets = [
                    {k: v.to(device) for k, v in t.items() if k in self.keys_to_keep}
                    for t in targets
                ]

                with torch.no_grad():
                    with torch.autocast(device_type=str(device), cache_enabled=True):
                        outputs = self.model.predict(samples)

                    current_metrics = self.run_metrics_loop(
                        current_metrics=current_metrics,
                        model_output=outputs,
                        targets=targets,
                        loss_value=0,
                    )
                    if idx % 100 == 0 and idx != 0:
                        if wandb is None:
                            current_metrics = None
                        else:
                            if old_wandb_output is not None:
                                wandb_output = current_metrics.add_wandb_outputs(
                                    other_wandb_output=old_wandb_output,
                                    split="Val",
                                )
                            else:
                                wandb_output = current_metrics.wandb_loggable_output(
                                    split="Val"
                                )
                            wandb.log(wandb_output)
                            current_metrics = None
                            old_wandb_output = wandb_output

                    if save_output_file_path:
                        self.write_output(
                            model_output=outputs,
                            targets=targets,
                            img_names=img_names,
                            save_output_paths=save_output_file_path,
                        )
                if steps is not None and idx == steps:
                    break
                
                
    def _setup_train(self):
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = always_freeze_names
        for k, v in self.model.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                v.requires_grad = True


    def train_one_epoch(
        self,
        dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
        lr_scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None,
        save_model_path: Optional[str] = None,
        save_metrics_path: Optional[str] = None,
        log_grad_norm: bool = True,
        wandb: Optional[Any] = None,
    ) -> None:
        """
        Train the model for one epoch.

        Args:
            dataloader (Iterable): DataLoader providing the training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            device (torch.device): Device to run the training on (e.g., 'cpu' or
                'cuda').
            epoch (int): Current epoch number.
            max_norm (float): Maximum norm for gradient clipping.
                Defaults to 0 (no clipping).
            lr_scheduler (Optional[torch.optim.lr_scheduler.CosineAnnealingLR]):
                Learning rate scheduler. Defaults to None.
            save_model_path (Optional[str]): Path to save the model checkpoint.
                Defaults to None.
            save_metrics_path (Optional[str]): Path to save the training metrics.
                Defaults to None.
            log_grad_norm (bool): Whether to log gradient norms. Defaults to True.
            wandb (Optional[Any]): Weights and Biases logging object. Defaults to None.
        """

        current_metrics = None
        self._setup_train()

        for idx, batch in tqdm.tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            for key in batch:
                if type(batch[key]) == torch.Tensor:
                    batch[key] = batch[key].to(device)

            with torch.autocast(
                device_type=str(device), cache_enabled=True, dtype=torch.bfloat16
            ):
                loss_value, individual_losses = self.model.model(batch)

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, skipping batch")
                print(individual_losses)
                optimizer.zero_grad()
                continue

            loss_value.backward()

            if log_grad_norm:
                grad_norm = (
                    sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self.model.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            if wandb:
                wandb.log(
                    {
                        "loss": loss_value,
                        "grad_norm": grad_norm,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                )
            if idx % 50 == 0 and idx > 0:
                if save_model_path is not None:
                    checkpoint_name = f"checkpoint_{epoch}.pth"
                    optimizer_checkpoint = os.path.join(
                        save_model_path, "optimizer.pth"
                    )
                    checkpoint_path = os.path.join(save_model_path, checkpoint_name)
                    torch.save(self.model.state_dict(), checkpoint_path)
                    torch.save(optimizer.state_dict(), optimizer_checkpoint)

        if save_model_path is not None:
            checkpoint_name = f"checkpoint_{epoch}.pth"
            optimizer_checkpoint = os.path.join(save_model_path, "optimizer.pth")
            checkpoint_path = os.path.join(save_model_path, checkpoint_name)
            torch.save(self.model.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_checkpoint)
            print(f"Saved model to {checkpoint_path}")

        return
    
    def train_loop(
        self,
        train_dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None,
        val_dataloader: Optional[Iterable] = None,
        val_freq: int = 5,
        val_steps: Optional[int] = None,
        max_norm: float = 0,
        wandb: Optional[Any] = None,
        save_model_path: Optional[str] = None,
        save_metrics_path: Optional[str] = None,
    ) -> None:
        """
        Run the training loop for a specified number of epochs.

        Args:
            train_dataloader (Iterable): DataLoader providing the training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            device (torch.device): Device to run the training on
                (e.g., 'cpu' or 'cuda').
            epochs (int): Number of epochs to train the model.
            lr_scheduler (Optional[torch.optim.lr_scheduler.CosineAnnealingLR],
                optional): Learning rate scheduler. Defaults to None.
            val_dataloader (Optional[Iterable], optional): DataLoader providing
                the validation data. Defaults to None.
            val_freq (int, optional): Frequency of validation (in epochs).
                Defaults to 5.
            max_norm (float, optional): Maximum norm for gradient clipping.
                Defaults to 0 (no clipping).
            wandb (Optional[Any], optional): Weights and Biases logging object.
                Defaults to None.
            save_model_path (Optional[str], optional): Path to save the model checkpoint.
                Defaults to None.
            save_metrics_path (Optional[str], optional): Path to save the training
                metrics. Defaults to None.
        """
        print("Starting training loop runnning for {} epochs".format(epochs))
        for epoch in range(epochs):
            self.train_one_epoch(
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                max_norm=max_norm,
                save_model_path=save_model_path,
                save_metrics_path=save_metrics_path,
                wandb=wandb,
            )
            if epoch % val_freq == 0 and val_dataloader is not None:
                self.validate_loop(
                    val_dataloader=val_dataloader,
                    steps=val_steps,
                    device=device,
                    epoch=epoch,
                    save_metrics_path=save_metrics_path,
                    wandb=wandb,
                )
        if epoch % val_freq != 0 and val_dataloader is not None:
            self.validate_loop(
                val_dataloader=val_dataloader,
                device=device,
                epoch=epoch,
                steps=val_steps,
                save_metrics_path=save_metrics_path,
                wandb=wandb,
            )
        return
