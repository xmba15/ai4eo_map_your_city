import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from torchvision.utils import make_grid

from src.models import DomainClsLoss, FocalLoss, MultiModalNet
from src.utils import get_object_from_dict

__all__ = (
    "MultiModalNetPl",
    "MultiModalNetFullModalityPl",
)


class MultiModalNetPl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        self.model = MultiModalNet(
            self.hparams["model"]["encoder_name"],
            self.hparams["model"]["num_classes"],
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams["model"]["num_classes"])
        self.losses = [
            ("focal", 1.0, FocalLoss()),
            ("domain_cls", 0.02, DomainClsLoss()),
            ("distribution", 0.1, nn.L1Loss()),
        ]

    def forward(self, batch):
        return self.model(batch[0], batch[1], batch[2])

    def common_step(self, batch, batch_idx, is_val: bool = False):
        _, _, _, label = batch
        logits, spec_logits, shared_feats = self.forward(batch)
        batch_size = logits.shape[0]
        num_modal = spec_logits.shape[1]

        total_loss = 0.0
        losses_dict = {}
        for loss_name, weight, loss_class in self.losses:
            if loss_name == "focal":
                cur_loss = loss_class(logits, label)
            elif loss_name == "domain_cls":
                spec_labels = torch.arange(num_modal).repeat_interleave(batch_size).to(spec_logits.device)
                cur_loss = loss_class(spec_logits, spec_labels)
            else:
                assert loss_name == "distribution"
                cur_loss = (
                    loss_class(shared_feats[0], shared_feats[1])
                    + loss_class(shared_feats[1], shared_feats[2])
                    + loss_class(shared_feats[2], shared_feats[0])
                )

            total_loss += weight * cur_loss

            losses_dict[loss_name] = cur_loss

        acc = None
        if is_val:
            _, pred = logits.max(1)
            acc = self.accuracy(pred, label)

        return total_loss, losses_dict, acc

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            self.logger.experiment.add_image(
                "train_ortho",
                make_grid(
                    batch[0][:, :3, :, :],
                    nrow=batch[0].shape[0],
                ),
                global_step=self.current_epoch * self.trainer.num_training_batches + batch_idx,
            )

            self.logger.experiment.add_image(
                "train_street",
                make_grid(
                    batch[0][:, 3:, :, :],
                    nrow=batch[0].shape[0],
                ),
                global_step=self.current_epoch * self.trainer.num_training_batches + batch_idx,
            )

        total_loss, losses_dict, _ = self.common_step(batch, batch_idx, is_val=False)

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for loss_name in losses_dict:
            self.log(
                f"train_loss_{loss_name}",
                losses_dict[loss_name],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, losses_dict, acc = self.common_step(batch, batch_idx, is_val=True)

        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for loss_name in losses_dict:
            self.log(
                f"val_loss_{loss_name}",
                losses_dict[loss_name],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return acc

    def configure_optimizers(self):
        optimizer = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.parameters() if x.requires_grad],
        )

        scheduler = get_object_from_dict(
            self.hparams["scheduler"],
            optimizer=optimizer,
        )

        return [optimizer], [scheduler]


class MultiModalNetFullModalityPl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        self.model = get_object_from_dict(
            self.hparams["model"],
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams["model"]["num_classes"])
        self.loss = FocalLoss()

    def forward(self, images, s2_data, country_id):
        return self.model(images, s2_data, country_id)

    def common_step(self, batch, batch_idx, is_val: bool = False):
        images, s2_data, country_id, label = batch
        logits = self.forward(images, s2_data, country_id)
        batch_size = logits.shape[0]

        loss = self.loss(logits, label)

        acc = None
        if is_val:
            _, pred = logits.max(1)
            acc = self.accuracy(pred, label)

        return loss, acc

    def training_step(self, batch, batch_idx):
        if batch_idx % 1000 == 0:
            self.logger.experiment.add_image(
                "train_ortho",
                make_grid(
                    batch[0][:, :3, :, :],
                    nrow=batch[0].shape[0],
                ),
                global_step=self.current_epoch * self.trainer.num_training_batches + batch_idx,
            )

            self.logger.experiment.add_image(
                "train_street",
                make_grid(
                    batch[0][:, 3:, :, :],
                    nrow=batch[0].shape[0],
                ),
                global_step=self.current_epoch * self.trainer.num_training_batches + batch_idx,
            )

            self.logger.experiment.add_image(
                "train_s2",
                make_grid(
                    batch[1][:, [3, 2, 1], :, :],
                    nrow=batch[0].shape[0],
                ),
                global_step=self.current_epoch * self.trainer.num_training_batches + batch_idx,
            )

        loss, _ = self.common_step(batch, batch_idx, is_val=False)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_step(batch, batch_idx, is_val=True)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return acc

    def configure_optimizers(self):
        optimizer = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.parameters() if x.requires_grad],
        )

        scheduler = {
            "scheduler": get_object_from_dict(
                self.hparams["scheduler"],
                optimizer=optimizer,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]
