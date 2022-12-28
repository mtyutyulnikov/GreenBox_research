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

from preproccesing import get_test_transforms, get_train_transforms
import pandas as pd
import random
import math


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
        metal_randomize = False,
        weight_randomize =False,
        weight_divider_coef = 1,
        metal_divider_coef = 1,
        class_to_int_dict = {'plastic' : 0, 'alum' : 1, 'glass' : 2, 'other' : 3}
    ):
        self.dataset_info = pd.read_csv(dataset_info_file_path)
        self.dataset_info['weight'] /= weight_divider_coef
        self.dataset_info['metal_median'] /= metal_divider_coef

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


        assert len(self.side_photos) == len(self.top_photos)

    def __len__(self):
        return len(self.top_photos)

    def __getitem__(self, index):
        top_photo = self.top_photos[index]
        side_photo = self.side_photos[index]

        top_img = cv2.imread(str(top_photo))
        side_img = cv2.imread(str(side_photo))

        if self.top_transform is not None:
            top_img = self.top_transform(image=top_img)
        if self.side_transform is not None:
            side_img = self.side_transform(image=side_img)

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
            if random.random() < 0.15:
                weight = random.normalvariate(2*weight, weight) 
        
        if self.metal_randomize:
            if random.random() < 0.35:
                if bottle_class == 'alum':
                    metal_median = random.normalvariate(100, 50) / self.metal_divider_coef
                else:
                    metal_median += random.normalvariate(0, 30) / self.metal_divider_coef


        y = self.class_to_int_dict[bottle_class]
        return top_img, side_img, metal_median, weight, y, index


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

        # ----define dataset-----#
        if stage == "fit" or stage is None:
            self.train_dataset = GarbageDataset(
                train_path,
                csv_path,
                labels_set=self.labels_set["train_labels"],
                side_transform=get_train_transforms(),
                top_transform=get_train_transforms(),
                metal_randomize = True,
                weight_randomize =True,
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
                side_transform=get_train_transforms(),
                top_transform=get_train_transforms(),
            )
            self.val_dataset = GarbageDataset(
                val_path,
                csv_path,
                labels_set=self.labels_set["val_labels"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
            )
            self.val_dataset_in_domain = GarbageDataset(
                val_path,
                csv_path,
                labels_set=self.labels_set["val_labels_indom"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
            )
            self.val_dataset_binary = GarbageDataset(
                val_path,
                csv_path,
                labels_set=self.labels_set["val_labels"],
                side_transform=get_test_transforms(),
                top_transform=get_test_transforms(),
                class_to_int_dict = {'plastic' : 0, 'alum' : 0, 'glass' : 0, 'other' : -1}
            )


        self.test_dataset = GarbageDataset(
            test_path,
            csv_path,
            labels_set=self.labels_set["test_labels"],
            side_transform=get_test_transforms(),
            top_transform=get_test_transforms(),
        )
        self.test_dataset_no_transforms = GarbageDataset(
            test_path,
            csv_path,
            labels_set=self.labels_set["test_labels"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def train_dataloader_indom(self):
        return DataLoader(
            self.train_dataset_indom, batch_size=self.batch_size, num_workers=self.num_workers
        )
    def train_dataloader_no_augs_indom(self):
        return DataLoader(
            self.train_dataset_no_augs_indom, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    # def val_dataloader(self):
    #     return self.val_dataloader_indom()

    def val_dataloader_indom(self):
        return DataLoader(
            self.val_dataset_in_domain,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    def val_dataloader_binary(self):
        return DataLoader(
            self.val_dataset_binary,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def aluminium_test_dataloader(self):
        return DataLoader(
            self.test_aluminium_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def others_dataloader(self):
        return DataLoader(
            self.test_others_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
