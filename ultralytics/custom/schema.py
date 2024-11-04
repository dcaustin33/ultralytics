from typing import Any, Dict, List, Optional

import torch
from pydantic.dataclasses import dataclass


@dataclass
class ConfusionMetrics:
    tp_binary: int
    tp_normal: int
    total_gold_boxes: int
    total_pred_boxes: int

    @property
    def recall_binary(self):
        return self.tp_binary / self.total_gold_boxes

    @property
    def recall_normal(self):
        return self.tp_normal / self.total_gold_boxes

    @property
    def precision_binary(self):
        return self.tp_binary / self.total_pred_boxes

    @property
    def precision_normal(self):
        return self.tp_normal / self.total_gold_boxes

    @property
    def f1_binary(self):
        return (
            2
            * (self.precision_binary * self.recall_binary)
            / (self.precision_binary + self.recall_binary)
        )

    @property
    def f1_normal(self):
        return (
            2
            * (self.precision_normal * self.recall_normal)
            / (self.precision_normal + self.recall_normal)
        )


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    loss: float
    empty_images: int

    unmatched_pred_areas: List[float]
    unmatched_pred_x_locations: List[float]
    unmatched_pred_y_locations: List[float]
    unmatched_gold_areas: List[float]
    unmatched_gold_x_locations: List[float]
    unmatched_gold_y_locations: List[float]

    matched_pred_areas: List[float]
    matched_pred_x_locations: List[float]
    matched_pred_y_locations: List[float]
    matched_gold_areas: List[float]
    matched_gold_x_locations: List[float]
    matched_gold_y_locations: List[float]
    l1_matched_center_diff: List[float]
    l2_matched_center_diff: List[float]

    matched_ious: List[float]
    matched_l1_errors: List[float]
    matched_classification_accuracy: List[bool]
    
    map50: List[float]
    map75: List[float]
    map50_95: List[float]
    small_map_50: List[float]
    small_map_75: List[float]
    small_map_50_95: List[float]
    medium_map_50: List[float]
    medium_map_75: List[float]
    medium_map_50_95: List[float]
    large_map_50: List[float]
    large_map_75: List[float]
    large_map_50_95: List[float]
    

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-6)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-6)

    @property
    def f1(self):
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-6)
        )

    @property
    def total_pred_boxes(self):
        return self.tp + self.fp

    @property
    def total_gold_boxes(self):
        return self.tp + self.fn
    

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            empty_images=self.empty_images + other.empty_images,
            loss=self.loss + other.loss,
            unmatched_pred_areas=self.unmatched_pred_areas + other.unmatched_pred_areas,
            unmatched_pred_x_locations=self.unmatched_pred_x_locations
            + other.unmatched_pred_x_locations,
            unmatched_pred_y_locations=self.unmatched_pred_y_locations
            + other.unmatched_pred_y_locations,
            unmatched_gold_areas=self.unmatched_gold_areas + other.unmatched_gold_areas,
            unmatched_gold_x_locations=self.unmatched_gold_x_locations
            + other.unmatched_gold_x_locations,
            unmatched_gold_y_locations=self.unmatched_gold_y_locations
            + other.unmatched_gold_y_locations,
            matched_pred_areas=self.matched_pred_areas + other.matched_pred_areas,
            matched_pred_x_locations=self.matched_pred_x_locations
            + other.matched_pred_x_locations,
            matched_pred_y_locations=self.matched_pred_y_locations
            + other.matched_pred_y_locations,
            matched_gold_areas=self.matched_gold_areas + other.matched_gold_areas,
            matched_gold_x_locations=self.matched_gold_x_locations
            + other.matched_gold_x_locations,
            matched_gold_y_locations=self.matched_gold_y_locations
            + other.matched_gold_y_locations,
            matched_ious=self.matched_ious + other.matched_ious,
            matched_l1_errors=self.matched_l1_errors + other.matched_l1_errors,
            matched_classification_accuracy=self.matched_classification_accuracy
            + other.matched_classification_accuracy,
            l1_matched_center_diff=self.l1_matched_center_diff
            + other.l1_matched_center_diff,
            l2_matched_center_diff=self.l2_matched_center_diff
            + other.l2_matched_center_diff,
            map50=self.map50 + other.map50,
            map75=self.map75 + other.map75,
            map50_95=self.map50_95 + other.map50_95,
            small_map_50=self.small_map_50 + other.small_map_50,
            small_map_75=self.small_map_75 + other.small_map_75,
            small_map_50_95=self.small_map_50_95 + other.small_map_50_95,
            medium_map_50=self.medium_map_50 + other.medium_map_50,
            medium_map_75=self.medium_map_75 + other.medium_map_75,
            medium_map_50_95=self.medium_map_50_95 + other.medium_map_50_95,
            large_map_50=self.large_map_50 + other.large_map_50,
            large_map_75=self.large_map_75 + other.large_map_75,
            large_map_50_95=self.large_map_50_95 + other.large_map_50_95,
        )
        

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "unmatched_pred_areas": self.unmatched_pred_areas,
            "unmatched_pred_x_locations": self.unmatched_pred_x_locations,
            "unmatched_pred_y_locations": self.unmatched_pred_y_locations,
            "unmatched_gold_areas": self.unmatched_gold_areas,
            "unmatched_gold_x_locations": self.unmatched_gold_x_locations,
            "unmatched_gold_y_locations": self.unmatched_gold_y_locations,
            "matched_pred_areas": self.matched_pred_areas,
            "matched_pred_x_locations": self.matched_pred_x_locations,
            "matched_pred_y_locations": self.matched_pred_y_locations,
            "matched_gold_areas": self.matched_gold_areas,
            "matched_gold_x_locations": self.matched_gold_x_locations,
            "matched_gold_y_locations": self.matched_gold_y_locations,
            "matched_ious": self.matched_ious,
            "matched_l1_errors": self.matched_l1_errors,
            "matched_classification_accuracy": self.matched_classification_accuracy,
            "l1_matched_center_diff": self.l1_matched_center_diff,
            "l2_matched_center_diff": self.l2_matched_center_diff,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "l1_matched_center_diff": self.l1_matched_center_diff,
            "l2_matched_center_diff": self.l2_matched_center_diff,
            "map50": self.map50,
            "map75": self.map75,
            "map50_95": self.map50_95,
            "small_map_50": self.small_map_50,
            "small_map_75": self.small_map_75,
            "small_map_50_95": self.small_map_50_95,
            "medium_map_50": self.medium_map_50,
            "medium_map_75": self.medium_map_75,
            "medium_map_50_95": self.medium_map_50_95,
            "large_map_50": self.large_map_50,
            "large_map_75": self.large_map_75,
            "large_map_50_95": self.large_map_50_95,
        }

    def wandb_loggable_output(
        self,
        split: str = "Train",
    ) -> Dict[str, Any]:
        epsilon = 1e-10  # Small value to avoid division by zero
        output = {
            "total_true_positives": self.tp,
            "total_false_positives": self.fp,
            "total_false_negatives": self.fn,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "average_unmatched_pred_area": sum(self.unmatched_pred_areas)
            / (len(self.unmatched_pred_areas) + epsilon),
            "average_unmatched_pred_x_locations": sum(self.unmatched_pred_x_locations)
            / (len(self.unmatched_pred_x_locations) + epsilon),
            "average_unmatched_pred_y_locations": sum(self.unmatched_pred_y_locations)
            / (len(self.unmatched_pred_y_locations) + epsilon),
            "average_unmatched_gold_areas": sum(self.unmatched_gold_areas)
            / (len(self.unmatched_gold_areas) + epsilon),
            "average_unmatched_gold_x_locations": sum(self.unmatched_gold_x_locations)
            / (len(self.unmatched_gold_x_locations) + epsilon),
            "average_unmatched_gold_y_locations": sum(self.unmatched_gold_y_locations)
            / (len(self.unmatched_gold_y_locations) + epsilon),
            "average_matched_pred_areas": sum(self.matched_pred_areas)
            / (len(self.matched_pred_areas) + epsilon),
            "average_matched_pred_x_locations": sum(self.matched_pred_x_locations)
            / (len(self.matched_pred_x_locations) + epsilon),
            "average_matched_pred_y_locations": sum(self.matched_pred_y_locations)
            / (len(self.matched_pred_y_locations) + epsilon),
            "average_matched_gold_areas": sum(self.matched_gold_areas)
            / (len(self.matched_gold_areas) + epsilon),
            "average_matched_gold_x_locations": sum(self.matched_gold_x_locations)
            / (len(self.matched_gold_x_locations) + epsilon),
            "average_matched_gold_y_locations": sum(self.matched_gold_y_locations)
            / (len(self.matched_gold_y_locations) + epsilon),
            "average_matched_ious": sum(self.matched_ious)
            / (len(self.matched_ious) + epsilon),
            "average_matched_l1_errors": sum(self.matched_l1_errors)
            / (len(self.matched_l1_errors) + epsilon),
            "average_matched_classification_accuracy": sum(
                self.matched_classification_accuracy
            )
            / (len(self.matched_classification_accuracy) + epsilon),
            "total_pred_boxes": self.total_pred_boxes,
            "total_gold_boxes": self.total_gold_boxes,
            "total_divisors": 1,
            "average_l1_matched_center_diff": sum(self.l1_matched_center_diff)
            / (len(self.l1_matched_center_diff) + epsilon),
            "average_l2_matched_center_diff": sum(self.l2_matched_center_diff)
            / (len(self.l2_matched_center_diff) + epsilon),
            "map50": sum(self.map50) / (len(self.map50) + epsilon),
            "map75": sum(self.map75) / (len(self.map75) + epsilon),
            "map50_95": sum(self.map50_95) / (len(self.map50_95) + epsilon),
            "small_map_50": sum(self.small_map_50) / (len(self.small_map_50) + epsilon),
            "small_map_75": sum(self.small_map_75) / (len(self.small_map_75) + epsilon),
            "small_map_50_95": sum(self.small_map_50_95) / (len(self.small_map_50_95) + epsilon),
            "medium_map_50": sum(self.medium_map_50) / (len(self.medium_map_50) + epsilon),
            "medium_map_75": sum(self.medium_map_75) / (len(self.medium_map_75) + epsilon),
            "medium_map_50_95": sum(self.medium_map_50_95) / (len(self.medium_map_50_95) + epsilon),
            "large_map_50": sum(self.large_map_50) / (len(self.large_map_50) + epsilon),
            "large_map_75": sum(self.large_map_75) / (len(self.large_map_75) + epsilon),
            "large_map_50_95": sum(self.large_map_50_95) / (len(self.large_map_50_95) + epsilon),
        }
        return {f"{split}_{k}": v for k, v in output.items()}

    def add_wandb_outputs(
        self, other_wandb_output: Dict[str, Any], split: str = "train"
    ) -> Dict[str, Any]:
        wandb_output = self.wandb_loggable_output(split)
        divisor = (
            wandb_output[f"{split}_total_divisors"]
            + other_wandb_output[f"{split}_total_divisors"]
        )
        for key in wandb_output:
            if "total" not in key:
                wandb_output[key] = (
                    wandb_output[key]  + other_wandb_output[key] * (divisor - 1)
                ) / divisor
            else:
                wandb_output[key] += other_wandb_output[key]
        return wandb_output

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metrics":
        return cls(
            tp=data["tp"],
            fp=data["fp"],
            fn=data["fn"],
            unmatched_pred_areas=data["unmatched_pred_areas"],
            unmatched_pred_x_locations=data["unmatched_pred_x_locations"],
            unmatched_pred_y_locations=data["unmatched_pred_y_locations"],
            unmatched_gold_areas=data["unmatched_gold_areas"],
            unmatched_gold_x_locations=data["unmatched_gold_x_locations"],
            unmatched_gold_y_locations=data["unmatched_gold_y_locations"],
            matched_pred_areas=data["matched_pred_areas"],
            matched_pred_x_locations=data["matched_pred_x_locations"],
            matched_pred_y_locations=data["matched_pred_y_locations"],
            matched_gold_areas=data["matched_gold_areas"],
            matched_gold_x_locations=data["matched_gold_x_locations"],
            matched_gold_y_locations=data["matched_gold_y_locations"],
            matched_ious=data["matched_ious"],
            matched_l1_errors=data["matched_l1_errors"],
            matched_classification_accuracy=data["matched_classification_accuracy"],
            l1_matched_center_diff=data["l1_matched_center_diff"],
            l2_matched_center_diff=data["l2_matched_center_diff"],
            map50=data["map50"],
            map75=data["map75"],
            map50_95=data["map50_95"],
            small_map_50=data["small_map_50"],
            small_map_75=data["small_map_75"],
            small_map_50_95=data["small_map_50_95"],
            medium_map_50=data["medium_map_50"],
            medium_map_75=data["medium_map_75"],
            medium_map_50_95=data["medium_map_50_95"],
            large_map_50=data["large_map_50"],
            large_map_75=data["large_map_75"],
            large_map_50_95=data["large_map_50_95"],
        )


class TrainingState:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_func: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
