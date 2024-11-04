"""
Creating a dataloader that assumes the data is in a folder with the following structure (yolo format):

test/
    images/
        image1.jpg
    labels/
        image1.txt
train/
    images/
        image1.jpg
    labels/
        image1.txt
val/
    images/
        image1.jpg
    labels/
        image1.txt
"""

from collections import defaultdict
import os
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


def rt_detr_val_transform(
    img_size: int = 640,
) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((img_size, img_size)),
            v2.ToTensor(),
        ]
    )
    

def rt_detr_train_transform(
    img_size: int = 640,
) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((img_size, img_size)),
            v2.RandomZoomOut(side_range=(1, 2)),
            v2.RandomPhotometricDistort(
                brightness=(0.875, 1.125),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=(-0.05, 0.05),
            ),
            v2.RandomIoUCrop(
                min_scale=0.2,
                max_scale=1.0,
                min_aspect_ratio=0.5,
                max_aspect_ratio=2.0,
                sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                trials=40,
            ),
            v2.SanitizeBoundingBoxes(min_size=1),
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
        ]
    )



class YoloFormatDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split: str = "train",
        transform: v2.Compose = rt_detr_val_transform,
        num_samples: int = None,
        empty_labels: bool = False,
    ):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.images_paths = os.listdir(os.path.join(self.root_dir, "images"))[:100]
        self.labels_paths = os.listdir(os.path.join(self.root_dir, "labels"))

        # create the image path to label path dict
        self.image_to_label = {}
        for img in self.images_paths:
            file_ext = os.path.splitext(img)[1]
            img_base = os.path.basename(img)[: len(file_ext) * -1]
            if img_base in self.image_to_label:
                raise ValueError(f"Duplicate image {img_base} found")
            self.image_to_label[img] = f"{img_base}.txt"
            assert (
                f"{img_base}.txt" in self.labels_paths
            ), (f"Label for image {img_base} not found"
            f"Label path: {os.path.join(self.root_dir, 'labels', self.image_to_label[img])}")
        # assert len(self.images_paths) == len(
        #     self.labels_paths
        # ), "Number of images and labels must be the same"
        if not empty_labels:
            self.remove_empty_images()
        self.images_paths = self.images_paths[:num_samples]

    def remove_empty_images(self):
        """Function to remove those images without any labels"""
        total_removed = 0
        imgs_to_remove = set()
        for image in self.images_paths:
            label_path = os.path.join(
                self.root_dir, "labels", self.image_to_label[image]
            )
            with open(label_path, "r") as f:
                labels = f.readlines()
            if len(labels) == 0:
                imgs_to_remove.add(image)
        for img in imgs_to_remove:
            self.images_paths.remove(img)
            self.labels_paths.remove(self.image_to_label[img])
            total_removed += 1
        print(f"Removed {total_removed} images without labels")
        print(f"Total images: {len(self.images_paths)}")

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "images", self.images_paths[idx])
        img = Image.open(image_path).convert("RGB")
        height = img.height
        width = img.width
        label_path = os.path.join(
            self.root_dir, "labels", self.image_to_label[self.images_paths[idx]]
        )
        with open(label_path, "r") as f:
            labels = f.readlines()
        
        if len(labels) == 0:
            # Handle empty labels file
            labels = []
            bboxes = tv_tensors.BoundingBoxes(
                torch.empty((0, 4)),
                format=tv_tensors.BoundingBoxFormat.CXCYWH,
                canvas_size=(height, width),
            )
            final_annotation = {
                "boxes": bboxes,
                "labels": torch.empty(0, dtype=torch.long),
            }
        else:
            labels = [label.strip().split() for label in labels]
            labels = [
                [
                    int(label[0]),
                    float(label[1]) * width,
                    float(label[2]) * height,
                    float(label[3]) * width,
                    float(label[4]) * height,
                ]
                for label in labels
            ]
            bboxes = tv_tensors.BoundingBoxes(
                torch.Tensor(
                    [[label[1], label[2], label[3], label[4]] for label in labels]
                ),
                format=tv_tensors.BoundingBoxFormat.CXCYWH,
                canvas_size=(height, width),
            )
            final_annotation = {
                "boxes": bboxes,
                "labels": torch.Tensor([label[0] for label in labels]).long(),
            }

        img, labels = self.transform((img, final_annotation))
        height = img.shape[-2]
        width = img.shape[-1]
        if len(labels["boxes"]) > 0:
            labels["boxes"][:, [0, 2]] /= width
            labels["boxes"][:, [1, 3]] /= height
        labels["img_name"] = self.images_paths[idx]
        sample = {
            "image": img.contiguous(),
            "annotations": labels,
        }
        return sample

    @staticmethod
    def collate_fn(batch: List[dict]):
        images = [item["image"] for item in batch]
        annotations = [item["annotations"] for item in batch]
        return torch.stack(images), annotations
    
    @staticmethod
    def ultralytics_collate_fn(batch: List[dict]):
        batch_return = defaultdict(list)
        for idx, item in enumerate(batch):
            batch_return["im_files"].append(item["annotations"]["img_name"])
            batch_return["ori_shape"].append(item["image"].shape[-2:])
            batch_return["resized_shape"].append(item["image"].shape[-2:])
            batch_return['img'].append(item["image"])
            if len(item["annotations"]["labels"]) > 0:
                if "cls" not in batch_return:
                    batch_return['cls'] = item["annotations"]["labels"]
                else:
                    batch_return['cls'] = torch.cat((batch_return['cls'], item["annotations"]["labels"]))
                if "bboxes" not in batch_return:
                    batch_return['bboxes'] = item["annotations"]["boxes"]
                else:
                    batch_return['bboxes'] = torch.cat((batch_return['bboxes'], item["annotations"]["boxes"]))
                for i in range(len(item["annotations"]["labels"])):
                    batch_return['batch_idx'].append(idx)
                
        batch_return["batch_idx"] = torch.tensor(batch_return["batch_idx"])
        batch_return["img"] = torch.stack(batch_return["img"])
        return batch_return


if __name__ == "__main__":
    # example
    dataloader = DataLoader(
        YoloFormatDataset(
            "/Users/derek/Desktop/dataset_collection/yolo_format", "train"
        ),
        batch_size=1,
        collate_fn=YoloFormatDataset.collate_fn,
    )
