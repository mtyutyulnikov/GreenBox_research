import torch
import numpy as np
from dataset import GarbageDataModule, GarbageDataset
from pathlib import Path
import wandb
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceGraph_ELBO
from tqdm.auto import trange, tqdm
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    resnet18,
    ResNet18_Weights,
)
from pyro.distributions import Normal, Categorical

torch.manual_seed(0)
np.random.seed(0)


device = "cuda:0"


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileNetSensors(PyroModule):
    def __init__(self, num_classes=3, use_top_mask = False):
        super().__init__()

        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # features_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # features_model.fc = Identity()

        self.features = model.features
        self.avgpool = model.avgpool
        img_features_space = 576
        self.use_top_mask = use_top_mask

        self.features[0][0] = nn.Conv2d(
            6+int(use_top_mask), 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )  # 7 input channels instead of 3

        # self.features.conv1 = nn.Conv2d(
        #     6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        # )


        self.classifier = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](img_features_space + 2, 1024),
            PyroModule[nn.Hardswish](),
            # PyroModule[nn.Dropout](),
            PyroModule[nn.Linear](1024, num_classes),
        )

        self.classifier[0].weight = PyroSample(
            prior=dist.Normal(0.0, torch.tensor(1.0, device=device))
            .expand([1024, img_features_space + 2])
            .to_event(2)
        )
        self.classifier[0].bias = PyroSample(
            prior=dist.Normal(0.0, torch.tensor(10.0, device=device))
            .expand([1024])
            .to_event(1)
        )

        self.classifier[-1].weight = PyroSample(
            prior=dist.Normal(0.0, torch.tensor(1.0, device=device))
            .expand([num_classes, 1024])
            .to_event(2)
        )
        self.classifier[-1].bias = PyroSample(
            prior=dist.Normal(0.0, torch.tensor(10.0, device=device))
            .expand([num_classes])
            .to_event(1)
        )
        self.to(device)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, top_img, side_img, top_delta_mask, sensors, y=None):
        if self.use_top_mask:
            input_img = torch.cat((top_img, side_img, top_delta_mask), dim=1)
        else:
            input_img = torch.cat((top_img, side_img), dim=1)
        
        img_features = self.features(input_img)
        img_features = self.avgpool(img_features).squeeze(-1).squeeze(-1)

        features = torch.cat((img_features, sensors), dim=1)
        mu = self.classifier(features)

        lhat = self.log_softmax(mu)

        with pyro.plate("data", top_img.shape[0]):
            obs = pyro.sample("obs", Categorical(logits=lhat), obs=y)
        return mu


if __name__ == "__main__":

    labels_set = {
        "train_labels": ["alum", "plastic", "glass", "other"],
        "val_labels": ["alum", "plastic", "glass", "other"],
        "val_labels_indom": ["alum", "plastic", "glass"],
        "test_labels": ["alum", "plastic", "glass", "other"],
        "train_labels_indom": ["alum", "plastic", "glass"],
    }
    batch_size = 32

    dm = GarbageDataModule(
        labels_set,
        data_dir=Path("dataset"),
        batch_size=batch_size,
        num_workers=15,
    )

    dm.setup()

    model = MobileNetSensors(use_top_mask=True)
    guide = AutoDiagonalNormal(model).to(device)
    adam = pyro.optim.Adam({"lr": 1e-2})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    pyro.clear_param_store()

    model = model.to(device)
    best_loss = 1e9

    for epoch in range(1000):
        train_loss = 0
        for x, y, idx in dm.train_dataloader_indom():
            sensors = torch.stack((x["metal"], x["weight"]), dim=1).to(device)
            top_img, side_img, top_delta_mask, y = (
                x["top_img"].to(device),
                x["side_img"].to(device),
                x["top_delta_mask"].to(device),
                y.to(device),
            )
            loss = svi.step(top_img, side_img, top_delta_mask, sensors, y)
            train_loss += loss
        train_loss /= len(dm.train_dataset_indom)
        torch.cuda.empty_cache()

        val_loss = 0
        for x, y, idx in dm.val_dataloader_indom():
            sensors = torch.stack((x["metal"], x["weight"]), dim=1).to(device)
            top_img, side_img, top_delta_mask, y = (
                x["top_img"].to(device),
                x["side_img"].to(device),
                x["top_delta_mask"].to(device), 
                y.to(device),
            )
            loss = svi.evaluate_loss(top_img, side_img, top_delta_mask, sensors, y)
            val_loss += loss
        val_loss /= len(dm.val_dataset_in_domain)
        torch.cuda.empty_cache()

        if val_loss < best_loss:
            print("saving new best")
            save_name = "bnn_saved_models/bnn_best_5"
            best_loss = val_loss
            torch.save(model.state_dict(), save_name + ".model")
            torch.save(guide, save_name + ".guide")

        print(f"Epoch {epoch} :  Train_Loss = {train_loss} ; Val_loss = {val_loss}")
