from pytorch_lightning import Trainer
from cac_loss_models import CACLossNet

from dataset import GarbageDataModule, GarbageDataset
from pathlib import Path
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from cac_loss_models import CacLossClassifierNet


cac_net = CACLossNet({'num_classes' : 3}, {'magnitude' : 3, 'alpha' : 2.25}, 'Adam', {'lr' : 1e-3, 'weight_decay' : 1e-4})
cac_net = cac_net.load_from_checkpoint('ood_models/Greedbox_ood/djy9xrwi/checkpoints/epoch=187-step=5076.ckpt')

classifier_net = CacLossClassifierNet(cac_net.model, 'Adam', {'lr' : 1e-3, 'weight_decay' : 1e-4})


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
                monitor="val_ood_acc",
                mode="max",
            ),
            EarlyStopping(monitor="val_ood_acc", mode="max", patience=200)
        ],
    default_root_dir='ood_models',
    logger=wandb_logger,
    log_every_n_steps = 5
)


trainer.fit(classifier_net, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

