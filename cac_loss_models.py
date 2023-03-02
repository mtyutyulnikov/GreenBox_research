from ast import Dict
from sklearn.manifold import TSNE
from torch import nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pytorch_lightning import LightningModule
import torch
from pytorch_ood.loss import CACLoss
from pytorch_ood.utils import OODMetrics

from supcontrast_resnet import SupConResNet
from torchvision.models import MobileNet_V3_Small_Weights


class CACLossNet(LightningModule):
    def __init__(
        self,
        model_hparams: Dict,
        cac_loss_params: Dict,
        optimizer_name: str,
        optimizer_hparams: Dict,
    ):
        """Initialize an image classification model

        Args:
            model_hparams (Dict): keys:  num_classes: int
            cac_loss_params (Dict) : magnitude : float, alpha: float
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()

        num_classes = model_hparams["num_classes"]

        self.loss_module = CACLoss(
            num_classes,
            **cac_loss_params
            # magnitude=cac_loss_params["magnitude"],
            # alpha=cac_loss_params["alpha"],
        )
        self.softmax = nn.Softmax(dim=-1)
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(
            7, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )  # 7 input channels instead of 3

        self.model.classifier[0] = nn.Linear(
            self.model.classifier[0].in_features + 2,
            self.model.classifier[0].out_features,
        )  # weight and metal features

        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, num_classes
        )  # output channels setting

        self.oodmetrics = OODMetrics()

        self.example_input_array = (
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 1, 360, 640),
            torch.ones(1, 2),
        )
        self.save_hyperparameters()

    def forward(self, top_img, side_img, top_delta_img, sensors):
        input_img = torch.cat((top_img, side_img, top_delta_img), dim=1)
        img_features = self.model.features(input_img)
        img_features = self.model.avgpool(img_features).squeeze(-1).squeeze(-1)
        # print(img_features.dtype, sensors.dtype)
        x = torch.cat((img_features, sensors), dim=1)
        logits = self.model.classifier(x)
        return logits

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)

        preds = logits.argmax(dim=-1)
        return preds, labels, logits

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)

        distances = self.loss_module.calculate_distances(logits)
        loss = self.loss_module(distances, labels)

        acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)

        distances = self.loss_module.calculate_distances(logits)

        indom_acc = (logits.argmax(dim=-1) == labels).sum() / (labels != 3).sum()
 

        self.log("val_acc", indom_acc, on_epoch=True, on_step=False)
        # self.log("val_ood_acc", ood_acc)

        labels[labels == 3] = -1
        score = CACLoss.score(distances)
        self.oodmetrics.update(score, labels)

    def on_validation_epoch_end(self):
        metrics_res = self.oodmetrics.compute()
        self.log("val_auroc", metrics_res["AUROC"])

    # def test_step(self, batch, batch_idx):
    #     x, labels, idx = batch
    #     input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)
    #     sensors = torch.cat((x["metal"], x["weight"]))
    #     logits = self.model(input_img, sensors)

    #     distances = self.loss_module.calculate_distances(logits)
    #     loss = self.loss_module(distances, labels)

    #     acc = (logits.argmax(dim=-1) == labels).float().mean()

    #     self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    #     return {'loss' : loss, 'acc':acc}

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


#################################################
#################################################
#################################################
#################################################
#################################################


class CacLossClassifierNet(LightningModule):
    def __init__(
        self,
        base_model,
        optimizer_name: str,
        optimizer_hparams: Dict,
    ):
        """Initialize an image classification model

        Args:

            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()

        self.softmax = nn.Softmax(dim=-1)

        self.base_model = base_model
        self.loss_module = nn.CrossEntropyLoss()
        self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, 4)
        # self.classifier = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 4))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, top_img, side_img, top_delta_img, sensors):
        input_img = torch.cat((top_img, side_img, top_delta_img), dim=1)
        img_features = self.base_model.features(input_img)
        img_features = self.base_model.avgpool(img_features).squeeze(-1).squeeze(-1)
        x = torch.cat((img_features, sensors), dim=1)
        four_dim_logits = self.base_model.classifier(x)

        # dists = CACLoss(3, magnitude=3, alpha=2.25).calculate_distances(
        #     three_dim_logits.detach().cpu()
        # ).to(self.device)

        # four_dim_logits = self.classifier(torch.cat((three_dim_logits, dists), dim=1))

        return four_dim_logits

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)
        probs = self.softmax(logits)
        preds = probs.argmax(-1)
        return preds, probs, labels, idx

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)

        probs = self.softmax(logits)
        loss = self.loss_module(logits, labels)

        total_acc = (probs.argmax(-1) == labels).float().mean()
        # ood_acc = ((probs.argmax(-1) == labels) & (labels == 3)).sum() / (labels == 3).sum()

        self.log("train_loss", loss)
        self.log(
            "train_acc",
            total_acc,
            on_step=False,
            on_epoch=True,
        )
        # self.log('train_ood_acc', ood_acc, on_step=False, on_epoch=True, )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], x["top_delta_mask"], sensors)

        probs = self.softmax(logits)
        loss = self.loss_module(logits, labels)

        total_acc = (probs.argmax(-1) == labels).float().mean()
        ood_acc = ((probs.argmax(-1) == labels) & (labels == 3)).sum() / (
            labels == 3
        ).sum()

        self.log("val_loss", loss)
        self.log(
            "val_acc",
            total_acc,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_ood_acc",
            ood_acc,
            on_step=False,
            on_epoch=True,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(
                [
                    {
                        "params": self.classifier.parameters(),
                        "lr": self.hparams.optimizer_hparams["lr"],
                        "weight_decay": self.hparams.optimizer_hparams["weight_decay"],
                    },
                    {
                        "params": self.base_model.parameters(),
                        "lr": self.hparams.optimizer_hparams["lr"]/10,
                        "weight_decay": self.hparams.optimizer_hparams["weight_decay"],
                    },
                    # {"params": self.base_model.parameters(), "lr": 1e-5},
                ]
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(
                [
                    {
                        "params": self.classifier.parameters(),
                        "lr": self.hparams.optimizer_hparams["lr"],
                        "weight_decay": self.hparams.optimizer_hparams["weight_decay"],
                    },
                    # {"params": self.base_model.parameters(), "lr": 1e-5},
                ]
            )
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[20, 30], gamma=0.1
        # )  # TODO fix scheduler
        lmbda = lambda epoch: 0.65**epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        return [optimizer], [scheduler]
