from dataset import GarbageDataModule, GarbageDataset
from pathlib import Path
import wandb
import torch
from ood_models import SupContrastMobileNet, CACLossNet, DefaultMobileNet, SupContrastClassifier, SupContrastResnet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# sup_net = SupContrastClassifier({'features_num' : 256}, 'Adam', {'lr' : 1e-4, 'weight_decay' : 1e-5})
sup_net = SupContrastMobileNet({'features_num' : 256}, 'Adam', {'lr' : 1e-4, 'weight_decay' : 1e-5})

wandb_logger = WandbLogger(log_model=True, project='Greedbox_ood')

labels_set = {
    "train_labels": ["alum", "plastic", "glass", 'other'],
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
    accelerator='gpu', 
    devices=[0],
    max_epochs=20000,
    callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=30)
        ],
    default_root_dir='ood_models',
    logger=wandb_logger,
    log_every_n_steps=30,
)

trainer.fit(sup_net, train_dataloaders=dm.train_dataloader_indom(), val_dataloaders=dm.val_dataloader_indom())
