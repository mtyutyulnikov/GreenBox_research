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




####################################################################
####################################################################
####################################################################
####################################################################

from supcontrast_loss import SupConLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb


def plot_tsne(X, labels):
    X = X.cpu().detach().numpy()
    X_points = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=3
    ).fit_transform(X)
    fig = plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        x=X_points[:, 0],
        y=X_points[:, 1],
        c=labels.cpu().detach().numpy(),
    )
    plt.legend(*scatter.legend_elements())
    return fig


class SupContrastMobileNet(LightningModule):
    def __init__(
        self,
        model_hparams: Dict,
        optimizer_name: str,
        optimizer_hparams: Dict,
    ):
        """Initialize an image classification model

        Args:
            model_hparams (Dict): keys:  features_num: int

            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()
        self.loss_module = SupConLoss()

        self.features_num = model_hparams["features_num"]

        self.side_model_features = mobilenet_v3_small(
            MobileNet_V3_Small_Weights.DEFAULT
        ).features
        self.side_model_avg_pool = mobilenet_v3_small(
            MobileNet_V3_Small_Weights.DEFAULT
        ).avgpool

        self.top_model_features = mobilenet_v3_small(
            MobileNet_V3_Small_Weights.DEFAULT
        ).features
        self.top_model_avg_pool = mobilenet_v3_small(
            MobileNet_V3_Small_Weights.DEFAULT
        ).avgpool

        # self.model.features[0][0] = nn.Conv2d(
        #     7, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        # )  # 7 input channels instead of 3

        # self.model.classifier[0] = nn.Linear(
        #     self.model.classifier[0].in_features + 2,
        #     self.model.classifier[0].out_features,
        # )  # weight and metal features

        # self.model.classifier[-1] = nn.Linear(
        #     self.model.classifier[-1].in_features, self.features_num
        # )  # output channels setting

        # self.example_input_array = (torch.ones(1, 7, 360, 640), torch.ones(1, 2))

        self.example_input_array = (
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 2),
        )
        # self.example_input_array = (
        #     torch.ones(1, 3, 360, 640),
        #     torch.ones(1, 3, 360, 640),
        # )

        self.mlp_features = nn.Sequential(
            nn.Linear(
                576 * 2, self.features_num
            ),  # nn.Hardswish(), nn.Linear(576, self.features_num)
        )

        # self.model = SupConResNet(name='resnet18', feat_dim=256)
        # self.model.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, top_img, side_img, sensors):
        top_img_features = self.top_model_features(top_img)
        top_img_features = (
            self.top_model_avg_pool(top_img_features).squeeze(-1).squeeze(-1)
        )

        side_img_features = self.side_model_features(side_img)
        side_img_features = (
            self.side_model_avg_pool(side_img_features).squeeze(-1).squeeze(-1)
        )

        x = torch.cat((top_img_features, side_img_features), dim=1)

        logits = self.mlp_features(x)
        features = F.normalize(logits, dim=1)
        return features

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)

        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], sensors)
        # logits = self.forward(input_img)
        return logits, labels

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        # logits = self.forward(x["top_img"], x["side_img"], sensors)
        features = self.forward(x["top_img"], x["side_img"], sensors)
        loss = self.loss_module(features.unsqueeze(1), labels)

        self.log("train_loss", loss)

        return {"loss": loss, "features": features, "labels": labels}

    def training_epoch_end(self, outputs):
        features = torch.cat([o["features"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        if self.current_epoch % 1 == 0:
            features = features.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            tsne_features = TSNE(
                n_components=2, learning_rate="auto", init="random", perplexity=3
            ).fit_transform(features)
            plt.clf()
            scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels)
            plt.legend(*scatter.legend_elements())
            plt.savefig("train_tsne.png")
            wandb.log({"train_tsne": wandb.Image("train_tsne.png")})

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        features = self.forward(x["top_img"], x["side_img"], sensors)

        # logits = self.forward(x["top_img"], x["side_img"], sensors)
        # logits = self.forward(input_img)

        loss = self.loss_module(features.unsqueeze(1), labels)

        self.log("val_loss", loss)
        return {"loss": loss, "features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        features = torch.cat([o["features"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        if self.current_epoch % 1 == 0:
            features = features.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            tsne_features = TSNE(
                n_components=2, learning_rate="auto", init="random", perplexity=3
            ).fit_transform(features)
            plt.clf()
            scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels)
            plt.legend(*scatter.legend_elements())
            plt.savefig("val_tsne.png")
            wandb.log({"val_tsne": wandb.Image("val_tsne.png")})

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
        lmbda = lambda epoch: 0.65**epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        return [optimizer], [scheduler]


####################################################################
####################################################################
####################################################################
####################################################################


####################################################################
####################################################################
####################################################################
####################################################################


class SupContrastClassifier(LightningModule):
    def __init__(
        self,
        model_hparams: Dict,
        optimizer_name: str,
        optimizer_hparams: Dict,
    ):
        """Initialize an image classification model

        Args:
            model_hparams (Dict): keys:  features_num: int

            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()
        self.loss_module = SupConLoss()

        self.features_num = model_hparams["features_num"]

        self.softmax = nn.Softmax(dim=-1)

        self.side_model_features = mobilenet_v3_small(
            MobileNet_V3_Small_Weights
        ).features
        self.side_model_avg_pool = mobilenet_v3_small(
            MobileNet_V3_Small_Weights
        ).avgpool

        self.top_model_features = mobilenet_v3_small(
            MobileNet_V3_Small_Weights
        ).features
        self.top_model_avg_pool = mobilenet_v3_small(MobileNet_V3_Small_Weights).avgpool

        # self.model.features[0][0] = nn.Conv2d(
        #     7, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        # )  # 7 input channels instead of 3

        # self.model.classifier[0] = nn.Linear(
        #     self.model.classifier[0].in_features + 2,
        #     self.model.classifier[0].out_features,
        # )  # weight and metal features

        # self.model.classifier[-1] = nn.Linear(
        #     self.model.classifier[-1].in_features, self.features_num
        # )  # output channels setting

        # self.example_input_array = (torch.ones(1, 7, 360, 640), torch.ones(1, 2))

        self.example_input_array = (
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 2),
        )

        self.mlp_features = nn.Sequential(
            nn.Linear(576 * 2 + 2, 576),
            nn.Hardswish(),
            nn.Linear(576, self.features_num),
        )

        self.celoss = nn.CrossEntropyLoss()

        self.classifier = nn.Sequential(
            nn.Linear(self.features_num, 256), nn.Hardswish(), nn.Linear(256, 4)
        )

        # self.model = SupConResNet(name='resnet18', feat_dim=256)
        # self.model.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, top_img, side_img, sensors):
        top_img_features = self.top_model_features(top_img)
        top_img_features = (
            self.top_model_avg_pool(top_img_features).squeeze(-1).squeeze(-1)
        )

        side_img_features = self.side_model_features(side_img)
        side_img_features = (
            self.side_model_avg_pool(side_img_features).squeeze(-1).squeeze(-1)
        )

        x = torch.cat((top_img_features, side_img_features, sensors), dim=1)

        features = self.mlp_features(x)
        # features = F.normalize(features, dim=1)

        logits = self.classifier(features)
        self.softmax = nn.Softmax(dim=-1)
        return features, logits

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        features, logits = self.forward(x["top_img"], x["side_img"], sensors)
        probs = self.softmax(logits)
        return probs.argmax(-1), probs, features, labels

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        features, logits = self.forward(x["top_img"], x["side_img"], sensors)
        supcon_loss = self.loss_module(features.unsqueeze(1), labels)
        ce_loss = self.celoss(logits, labels)
        loss = ce_loss + 0.05 * supcon_loss

        probs = self.softmax(logits)
        acc = (probs.argmax(-1) == labels).float().mean()

        self.log("train_acc", acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        features, logits = self.forward(x["top_img"], x["side_img"], sensors)

        supcon_loss = self.loss_module(features.unsqueeze(1), labels)
        ce_loss = self.celoss(logits, labels)
        # loss = ce_loss + 0.05 * supcon_loss

        probs = self.softmax(logits)
        acc = (probs.argmax(-1) == labels).float().mean()

        self.log("val_acc", acc, on_epoch=True, on_step=False)
        self.log("val_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        lmbda = lambda epoch: 0.8**epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        return [optimizer], [scheduler]


###############################################################
###############################################################
###############################################################

from supcontrast_resnet import resnet18


class SupContrastResnet(LightningModule):
    def __init__(
        self,
        model_hparams: Dict,
        optimizer_name: str,
        optimizer_hparams: Dict,
    ):
        """Initialize an image classification model

        Args:
            model_hparams (Dict): keys:  features_num: int

            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()
        self.loss_module = SupConLoss()

        self.features_num = model_hparams["features_num"]

        self.side_model_features = resnet18()

        self.top_model_features = resnet18()

        # self.model.features[0][0] = nn.Conv2d(
        #     7, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        # )  # 7 input channels instead of 3

        # self.model.classifier[0] = nn.Linear(
        #     self.model.classifier[0].in_features + 2,
        #     self.model.classifier[0].out_features,
        # )  # weight and metal features

        # self.model.classifier[-1] = nn.Linear(
        #     self.model.classifier[-1].in_features, self.features_num
        # )  # output channels setting

        # self.example_input_array = (torch.ones(1, 7, 360, 640), torch.ones(1, 2))

        self.example_input_array = (
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 3, 360, 640),
            torch.ones(1, 2),
        )
        # self.example_input_array = (
        #     torch.ones(1, 3, 360, 640),
        #     torch.ones(1, 3, 360, 640),
        # )

        self.mlp_features = nn.Sequential(
            nn.Linear(
                512 * 2 + 2, self.features_num
            ),  # nn.Hardswish(), nn.Linear(576, self.features_num)
        )

        # self.model = SupConResNet(name='resnet18', feat_dim=256)
        # self.model.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, top_img, side_img, sensors):
        top_img_features = self.top_model_features(top_img)
        side_img_features = self.side_model_features(side_img)
        x = torch.cat((top_img_features, side_img_features, sensors), dim=1)

        logits = self.mlp_features(x)
        features = F.normalize(logits, dim=1)
        return features

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)

        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        logits = self.forward(x["top_img"], x["side_img"], sensors)
        # logits = self.forward(input_img)
        return logits, labels

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch
        # input_img = torch.cat((x["top_img"], x["side_img"], x["top_delta_mask"]), dim=1)
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        # logits = self.forward(x["top_img"], x["side_img"], sensors)
        features = self.forward(x["top_img"], x["side_img"], sensors)
        loss = self.loss_module(features.unsqueeze(1), labels)

        self.log("train_loss", loss)

        return {"loss": loss, "features": features, "labels": labels}

    def training_epoch_end(self, outputs):
        features = torch.cat([o["features"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        if self.current_epoch % 1 == 0:
            features = features.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            tsne_features = TSNE(
                n_components=2, learning_rate="auto", init="random", perplexity=3
            ).fit_transform(features)
            scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels)
            plt.legend(*scatter.legend_elements())
            plt.savefig("train_tsne.png")
            wandb.log({"train_tsne": wandb.Image("train_tsne.png")})

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch
        sensors = torch.stack((x["metal"], x["weight"]), dim=1)
        features = self.forward(x["top_img"], x["side_img"], sensors)

        # logits = self.forward(x["top_img"], x["side_img"], sensors)
        # logits = self.forward(input_img)

        loss = self.loss_module(features.unsqueeze(1), labels)

        self.log("val_loss", loss)
        return {"loss": loss, "features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        features = torch.cat([o["features"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        if self.current_epoch % 1 == 0:
            features = features.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            tsne_features = TSNE(
                n_components=2, learning_rate="auto", init="random", perplexity=3
            ).fit_transform(features)
            scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels)
            plt.legend(*scatter.legend_elements())
            plt.savefig("val_tsne.png")
            wandb.log({"val_tsne": wandb.Image("val_tsne.png")})

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
        lmbda = lambda epoch: 0.65**epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        return [optimizer], [scheduler]
