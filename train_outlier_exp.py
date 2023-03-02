from ast import Dict
from sklearn.manifold import TSNE
from torch import nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pytorch_lightning import LightningModule
import torch

from train_3_class_model import DefaultMobileNet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import GarbageDataModule, GarbageDataset
from pathlib import Path
import wandb



class OEMobileNet(LightningModule):
    # https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/oe_tune.py
    def __init__(
        self,
        model_hparams: Dict,
        optimizer_name: str,
        optimizer_hparams: Dict,
        base_model,
    ):
        """Initialize an image classification model

        Args:
            model_hparams (Dict): keys:  num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters(ignore=["base_model"])

        num_classes = model_hparams["num_classes"]

        self.loss_module = nn.CrossEntropyLoss()
        self.ood_loss = (
            lambda oodlogits: 0.5
            * -(oodlogits.mean(1) - torch.logsumexp(oodlogits, dim=1)).mean()
        )

        self.softmax = nn.Softmax(dim=-1)
        self.model = base_model

        self.example_input_array = (
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 3, 360, 640), 
            torch.ones(1, 1, 360, 640),
            torch.ones(1, 2),
        )

    def forward(self, top_img, side_img, top_delta_mask, sensors):
        img = torch.cat((top_img, side_img), dim=1)
        img_features = self.model.features(img)
        img_features = self.model.avgpool(img_features).squeeze(-1).squeeze(-1)
        x = torch.cat((img_features, sensors), dim=1)
        logits = self.model.classifier(x)
        return logits

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x['top_delta_mask'], sensors)
        probs = self.softmax(logits)
        preds = logits.argmax(dim=-1)
        return preds, labels, probs, idx

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x['top_delta_mask'], sensors)

        indom_idxs = labels != 3
        ood_idxs = labels == 3

        loss = 0
        if indom_idxs.sum() > 0:
            loss += self.loss_module(logits[indom_idxs], labels[indom_idxs])
        if ood_idxs.sum() > 0:
            loss += self.ood_loss(logits[ood_idxs])

        # acc = (logits.argmax(dim=-1) == labels).float().mean()
        indom_acc = ((logits.argmax(dim=-1) == labels) & (labels != 3)).sum() / (
            labels != 3
        ).sum()

        self.log(
            "train_indom_acc", indom_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train_loss", loss)
        return {"loss": loss, "indom_acc": indom_acc}

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x['top_delta_mask'], sensors)

        indom_idxs = labels != 3
        ood_idxs = labels == 3
        loss = 0
        if indom_idxs.sum() > 0:
            loss += self.loss_module(logits[indom_idxs], labels[indom_idxs])
        if ood_idxs.sum() > 0:
            loss += self.ood_loss(logits[ood_idxs])

        indom_acc = ((logits.argmax(dim=-1) == labels) & (labels != 3)).sum() / (
            labels != 3
        ).sum()

        self.log("val_indom_acc", indom_acc)
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
    base_model = DefaultMobileNet(
        {"num_classes": 3}, "Adam", {"lr": 1e-3, "weight_decay": 1e-4}
    )
    # base_model = base_model.load_from_checkpoint(
    #     "ood_models/Greedbox_ood/105fonth/checkpoints/epoch=27-step=980.ckpt"
    # )
    base_model = base_model.model
    model = OEMobileNet(
        {"num_classes": 3},
        "Adam",
        {"lr": 1e-3, "weight_decay": 1e-4},
        base_model=base_model,
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
                monitor="val_loss",
                mode="min",
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=100),
        ],
        default_root_dir="ood_models",
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
