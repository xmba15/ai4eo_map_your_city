import argparse
import os
import sys

import albumentations as alb
import cv2
import numpy as np
import pytorch_lightning as pl
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CountryCode, CustomSubset, MapYourCityDataset, S2RandomRotation
from src.integrated import MultiModalNetFullModalityPl, MultiModalNetPl
from src.utils import fix_seed, worker_init_fn


def get_args():
    parser = argparse.ArgumentParser("train multimodal")
    parser.add_argument("--config_path", type=str, default="./config/base_full_modality_1.yaml")

    return parser.parse_args()


def get_transforms(hparams):
    image_size = hparams["image_size"]

    all_transforms = {}
    all_transforms["street"] = {
        "train": alb.Compose(
            [
                alb.RandomCropFromBorders(crop_left=0.05, crop_right=0.05, crop_top=0.05, crop_bottom=0.05, p=0.5),
                alb.OneOf(
                    [
                        alb.Compose(
                            [
                                alb.Resize(height=image_size, width=image_size, p=1.0),
                                alb.Rotate(limit=(-5, 5), p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
                            ]
                        ),
                        alb.Compose(
                            [
                                alb.Rotate(limit=(-5, 5), p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
                                alb.Resize(height=image_size, width=image_size, p=1.0),
                            ]
                        ),
                    ],
                    p=1,
                ),
                alb.ColorJitter(p=0.5),
                alb.AdvancedBlur(p=0.5),
                alb.HorizontalFlip(p=0.5),
                alb.OneOf(
                    [
                        alb.CoarseDropout(min_holes=200, max_holes=400),
                        alb.GridDropout(),
                        alb.Spatter(),
                    ],
                    p=0.5,
                ),
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        ),
        "val": alb.Compose(
            [
                alb.Resize(height=image_size, width=image_size),
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        ),
    }

    all_transforms["ortho"] = {
        "train": alb.Compose(
            [
                alb.RandomCropFromBorders(crop_left=0.01, crop_right=0.01, crop_top=0.01, crop_bottom=0.01, p=0.6),
                alb.OneOf(
                    [
                        alb.Compose(
                            [
                                alb.Resize(height=image_size, width=image_size),
                                alb.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
                            ]
                        ),
                        alb.Compose(
                            [
                                alb.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
                                alb.Resize(height=image_size, width=image_size),
                            ]
                        ),
                    ],
                    p=1,
                ),
                alb.ColorJitter(p=0.5),
                alb.AdvancedBlur(p=0.5),
                alb.Flip(p=0.7),
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        ),
        "val": alb.Compose(
            [
                alb.Resize(height=image_size, width=image_size),
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        ),
    }

    def clip_s2(image, **params):
        return np.clip(image, 0, 10000)

    all_transforms["s2"] = {
        "train": alb.Compose(
            [
                S2RandomRotation(limits=(0, 360), always_apply=False, p=0.7),
                alb.Flip(p=0.7),
                alb.Lambda(image=clip_s2),
                alb.ToFloat(max_value=10000.0),
                ToTensorV2(),
            ]
        ),
        "val": alb.Compose(
            [
                alb.Lambda(image=clip_s2),
                alb.ToFloat(max_value=10000.0),
                ToTensorV2(),
            ]
        ),
    }

    return all_transforms


def setup_train_val_split(
    original_dataset,
    hparams,
):
    kf = StratifiedKFold(
        n_splits=hparams["dataset"]["n_splits"],
        shuffle=True,
        random_state=hparams["seed"],
    )

    train_indices, val_indices = list(
        kf.split(
            range(len(original_dataset)),
            original_dataset.labels,
        )
    )[hparams["dataset"]["fold_th"]]

    return train_indices, val_indices


def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    os.makedirs(hparams["output_root_dir"], exist_ok=True)
    fix_seed(hparams["seed"])
    pl.seed_everything(hparams["seed"])

    dataset = MapYourCityDataset(
        csv_path=hparams["dataset"]["train_csv"],
        data_dir=hparams["dataset"]["train_dir"],
        train=True,
    )

    train_indices, val_indices = setup_train_val_split(dataset, hparams)

    transforms_dict = get_transforms(hparams)
    train_dataset = CustomSubset(
        Subset(dataset, train_indices),
        transforms_dict={
            "street": transforms_dict["street"]["train"],
            "ortho": transforms_dict["ortho"]["train"],
            "s2": transforms_dict["s2"]["train"],
        },
    )

    val_dataset = CustomSubset(
        Subset(dataset, val_indices),
        transforms_dict={
            "street": transforms_dict["street"]["val"],
            "ortho": transforms_dict["ortho"]["val"],
            "s2": transforms_dict["s2"]["val"],
        },
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["train_parameters"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=hparams["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams["val_parameters"]["batch_size"],
        num_workers=hparams["num_workers"],
    )

    model = MultiModalNetFullModalityPl(hparams)
    trainer = Trainer(
        default_root_dir=hparams["output_root_dir"],
        max_epochs=hparams["trainer"]["max_epochs"],
        devices=hparams["trainer"]["devices"],
        accelerator=hparams["trainer"]["accelerator"],
        gradient_clip_val=hparams["trainer"]["gradient_clip_val"],
        accumulate_grad_batches=hparams["trainer"]["accumulate_grad_batches"],
        deterministic=True,
        logger=TensorBoardLogger(
            save_dir=hparams["output_root_dir"],
            version=f"{hparams['experiment_name']}_{hparams['model']['encoder_name']}_"
            f"{hparams['train_parameters']['batch_size']*hparams['trainer']['accumulate_grad_batches']}_"
            f"{hparams['optimizer']['lr']}",
            name=f"{hparams['experiment_name']}_{hparams['model']['encoder_name']}",
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    if hparams["trainer"]["resume_from_checkpoint"] is not None and os.path.isfile(
        hparams["trainer"]["resume_from_checkpoint"]
    ):
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=hparams["trainer"]["resume_from_checkpoint"],
        )
    else:
        trainer.fit(
            model,
            train_loader,
            val_loader,
        )


if __name__ == "__main__":
    main()
