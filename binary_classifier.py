from ast import Dict
from sklearn.manifold import TSNE
from torch import nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pytorch_lightning import LightningModule
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import GarbageDataModule, GarbageDataset
from pathlib import Path
import wandb


class BinaryMobileNet(LightningModule):
    def __init__(
        self,
        model_hparams: Dict,
        optimizer_name: str,
        optimizer_hparams: Dict,
    ):
        """Initialize an image classification model

        Args:
            model_hparams (Dict): keys:  num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()

        num_classes = model_hparams["num_classes"]

        self.loss_module = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4))

        self.softmax = nn.Softmax(dim=-1)
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(
            7, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )  # 7 input channels instead of 3

        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, num_classes
        )  # output channels setting

        self.model.classifier[0] = nn.Linear(
            self.model.classifier[0].in_features + 2,
            self.model.classifier[0].out_features,
        )  # weight and metal features

        self.example_input_array = (
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 1, 360, 640),
            torch.ones(1, 2),
        )
        self.sigmoid = nn.Sigmoid()
        self.threshold = 0.5

    def forward(self, top_img, side_img, top_delta_mask, sensors):
        img = torch.cat((top_img, side_img, top_delta_mask), dim=1)
        img_features = self.model.features(img)
        img_features = self.model.avgpool(img_features).squeeze(-1).squeeze(-1)
        x = torch.cat((img_features, sensors), dim=1)
        logits = self.model.classifier(x)
        logits = logits.squeeze()
        return logits

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)

        probs = self.sigmoid(logits)
        preds = torch.zeros_like(probs)
        preds[probs > self.threshold] = 1
        return preds, labels, probs

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)
        loss = self.loss_module(logits, labels.float())

        probs = self.sigmoid(logits)
        preds = torch.zeros_like(probs)
        preds[probs > self.threshold] = 1

        acc = (preds == labels).float().mean()

        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)
        loss = self.loss_module(logits, labels.float())

        probs = self.sigmoid(logits)
        preds = torch.zeros_like(probs)
        preds[probs > self.threshold] = 1

        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[20, 30], gamma=0.1
        # )  # TODO fix scheduler
        lmbda = lambda epoch: 0.8**epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    default_net = BinaryMobileNet(
        {"num_classes": 1}, "Adam", {"lr": 1e-3, "weight_decay": 1e-4}
    )

    wandb_logger = WandbLogger(log_model=True, project="Greedbox_ood")
    labels_set = {
        "train_labels": ["alum", "plastic", "glass", "other"],
        "val_labels": ["alum", "plastic", "glass", "other"],
        "val_labels_indom": ["alum", "plastic", "glass"],
        "test_labels": ["alum", "plastic", "glass", "other"],
        "train_labels_indom": ["alum", "plastic", "glass"],
    }
    batch_size = 24

    dm = GarbageDataModule(
        labels_set,
        data_dir=Path("dataset"),
        batch_size=batch_size,
        num_workers=10,
    )
    dm.setup()

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=20000,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                monitor="val_acc",
                mode="max",
            ),
            EarlyStopping(monitor="val_acc", mode="max", patience=100),
        ],
        default_root_dir="ood_models",
        logger=wandb_logger,
    )

    trainer.fit(
        default_net,
        train_dataloaders=dm.train_dataloader_binary(),
        val_dataloaders=dm.val_dataloader_binary(),
    )
