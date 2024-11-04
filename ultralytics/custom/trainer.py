import json
import os
from typing import Any, Iterable, List, Optional

import torch
import tqdm

from ultralytics.custom.postprocessing import box_metrics
from ultralytics.custom.schema import Metrics


class Trainer:
    
    def __init__(self, model: torch.nn.Module, matcher: torch.nn.Module):
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
