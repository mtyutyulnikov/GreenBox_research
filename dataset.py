from typing import Dict
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import cv2
import torch

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from preproccesing import (
    get_test_transforms,
    get_train_top_transforms,
    get_train_side_transforms,
)
import pandas as pd
import random
import math
from torchvision.transforms import Grayscale
import numpy as np


def choose_labels(all_files, label_list):
    files = [file for file in all_files for label in label_list if label in file.name]
    return files


class GarbageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        dataset_info_file_path,
        labels_set,
        side_transform=None,
        top_transform=None,
        metal_randomize=False,
        weight_randomize=False,
        weight_divider_coef=1000,
        metal_divider_coef=300,
        class_to_int_dict={"plastic": 0, "alum": 1, "glass": 2, "other": 3},
        background_image_path="background.jpg",
        allow_ood_shift=False,
    ):
        self.dataset_info = pd.read_csv(dataset_info_file_path)
        self.dataset_info["weight"] /= weight_divider_coef
        self.dataset_info["metal_median"] /= metal_divider_coef

        self.dataset_info["weight"] = self.dataset_info["weight"].astype(np.float32)
        self.dataset_info["metal_median"] = self.dataset_info["metal_median"].astype(
            np.float32
        )

        self.side_photos = choose_labels(
            sorted(list(data_dir.glob("*_side.png"))), labels_set
        )
        self.top_photos = choose_labels(
            sorted(list(data_dir.glob("*_top.png"))), labels_set
        )
        # print(self.top_photos)
        self.side_transform = side_transform
        self.top_transform = top_transform
        self.metal_randomize = metal_randomize
        self.weight_randomize = weight_randomize
        self.weight_divider_coef = weight_divider_coef
        self.metal_divider_coef = metal_divider_coef
        self.class_to_int_dict = class_to_int_dict

        self.background_image = cv2.cvtColor(
            cv2.imread(background_image_path), cv2.COLOR_BGR2RGB
        )
        self.background_image = cv2.resize(self.background_image, (640, 360))
        self.background_image = self.top_transform(image=self.background_image)["image"]

        assert len(self.side_photos) == len(self.top_photos)
        self.to_grayscale = Grayscale()
        self.allow_ood_shift = allow_ood_shift

    def __len__(self):
        return len(self.top_photos)

    def __getitem__(self, index):
        top_photo = self.top_photos[index]
        side_photo = self.side_photos[index]

        top_img = cv2.cvtColor(cv2.imread(str(top_photo)), cv2.COLOR_BGR2RGB)
        side_img = cv2.cvtColor(cv2.imread(str(side_photo)), cv2.COLOR_BGR2RGB)

        if self.top_transform is not None:
            top_img = self.top_transform(image=top_img)["image"]

        if self.side_transform is not None:
            side_img = self.side_transform(image=side_img)["image"]

        top_delta_mask = abs(top_img - self.background_image) ** 2
        top_delta_mask = top_delta_mask.mean(0).unsqueeze(0)

        df = self.dataset_info
        row = df[
            (df["side_filename"] == side_photo.name)
            & (df["top_filename"] == top_photo.name)
        ]
        row = row.iloc[0]
        metal_median = row["metal_median"]
        weight = row["weight"]
        bottle_class = row["class"]

        if self.weight_randomize:
            if self.allow_ood_shift:  # Бутылка с жидкостью
                if bottle_class == "plastic" and random.random() < 0.10:
                    weight = random.uniform(100, 500) / self.weight_divider_coef
                    bottle_class = "other"
                elif bottle_class == "alum" and random.random() < 0.05:
                    weight = random.uniform(100, 200) / self.weight_divider_coef
                    bottle_class = "other"

            if random.random() < 0.2:  # Шум
                weight += random.normalvariate(0, weight / 6) / self.weight_divider_coef

            weight = max(0, weight)

        if self.metal_randomize:
            if bottle_class == "alum":
                if random.random() < 0.10:  # Не полностью сработал датчик
                    metal_median = (
                        random.normalvariate(100, 50) / self.metal_divider_coef
                    )
                elif random.random() < 0.05:  # Не сработал датчик
                    metal_median = (
                        random.normalvariate(270, 40) / self.metal_divider_coef
                    )

            else:  # Шум
                metal_median += random.normalvariate(0, 3) / self.metal_divider_coef
            metal_median = max(metal_median, 0)

        y = self.class_to_int_dict[bottle_class]

        x = {
            "top_img": top_img,
            "side_img": side_img,
            "top_delta_mask": top_delta_mask,
            "metal": np.float32(metal_median),
            "weight": np.float32(weight),
        }

        return x, y, index


class GarbageDataModule(LightningDataModule):
    def __init__(
        self,
        labels_set: Dict,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 3
        self.labels_set = labels_set

    def setup(self, stage=None):
        train_path = self.data_dir / "train"
        val_path = self.data_dir / "val"
        test_path = self.data_dir / "test"
        csv_path = self.data_dir / "full_dataset_info.csv"

        if stage == "fit" or stage is None:
            self.train_dataset = GarbageDataset(
                train_path,
                csv_path,
                labels_set=self.labels_set["train_labels"],
                side_transform=get_train_side_transforms(),
                top_transform=get_train_top_transforms(),
                metal_randomize=True,
                weight_randomize=True,
                allow_ood_shift=True,
            )
            self.train_dataset_no_augs_indom = GarbageDataset(
                train_path,
                csv_path,
                labels_set=self.labels_set["train_labels_indom"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
            )
            self.train_dataset_indom = GarbageDataset(
                train_path,
                csv_path,
                labels_set=self.labels_set["train_labels_indom"],
                side_transform=get_train_side_transforms(),
                top_transform=get_train_top_transforms(),
                allow_ood_shift=False,
            )
            self.val_dataset = GarbageDataset(
                val_path,
                csv_path,
                labels_set=self.labels_set["val_labels"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
                allow_ood_shift=True,
            )
            self.val_dataset_in_domain = GarbageDataset(
                val_path,
                csv_path,
                labels_set=self.labels_set["val_labels_indom"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
                allow_ood_shift=False,
            )
            self.val_dataset_ood = GarbageDataset(
                val_path,
                csv_path,
                labels_set=self.labels_set["val_labels"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
                class_to_int_dict={"plastic": 0, "alum": 1, "glass": 2, "other": -1},
                allow_ood_shift=True,
            )
            self.train_dataset_binary = GarbageDataset(
                train_path,
                csv_path,
                labels_set=self.labels_set["train_labels"],
                side_transform=get_train_side_transforms(),
                top_transform=get_train_top_transforms(),
                class_to_int_dict={"plastic": 0, "alum": 0, "glass": 0, "other": 1},
                allow_ood_shift=True,
            )
            self.val_dataset_binary = GarbageDataset(
                val_path,
                csv_path,
                labels_set=self.labels_set["val_labels"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
                class_to_int_dict={"plastic": 0, "alum": 0, "glass": 0, "other": 1},
                allow_ood_shift=True,
            )

        self.test_dataset = GarbageDataset(
            test_path,
            csv_path,
            labels_set=self.labels_set["test_labels"],
            side_transform=get_test_transforms(),
            top_transform=get_test_transforms(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def train_dataloader_indom(self):
        return DataLoader(
            self.train_dataset_indom,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def train_dataloader_no_augs_indom(self):
        return DataLoader(
            self.train_dataset_no_augs_indom,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader_indom(self):
        return DataLoader(
            self.val_dataset_in_domain,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader_ood(self):
        return DataLoader(
            self.val_dataset_ood,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def aluminium_test_dataloader(self):
        return DataLoader(
            self.test_aluminium_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def others_dataloader(self):
        return DataLoader(
            self.test_others_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def train_dataloader_binary(self) :
        return DataLoader(
            self.train_dataset_binary,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    def val_dataloader_binary(self) :
        return DataLoader(
            self.val_dataset_binary,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

