from dataset import GarbageDataModule, GarbageDataset
from pathlib import Path
import wandb
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from cac_loss_models import CACLossNet

cac_net = CACLossNet({'num_classes' : 3}, {'magnitude' : 2, 'alpha' : 2}, 'Adam', {'lr' : 1e-3, 'weight_decay' : 1e-4})

wandb_logger = WandbLogger(log_model=True, project='Greedbox_ood')
labels_set = {
    "train_labels": ["alum", "plastic", "glass", 'other'],
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


trainer = Trainer(
    accelerator='gpu', 
    devices=[0],
    max_epochs=20000,
    callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                monitor="val_auroc",
                mode="max",
            ),
            EarlyStopping(monitor="val_auroc", mode="max", patience=200)
        ],
    default_root_dir='ood_models',
    logger=wandb_logger,
)


trainer.fit(cac_net, train_dataloaders=dm.train_dataloader_indom(), val_dataloaders=dm.val_dataloader())
