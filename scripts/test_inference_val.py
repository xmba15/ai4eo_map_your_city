import argparse
import os
import sys

import albumentations as alb
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CountryCode, CustomSubset, MapYourCityDataset, S2RandomRotation
from src.integrated import MultiModalNetFullModalityPl, MultiModalNetPl
from src.models import MultiModalNet
from src.utils import fix_seed, worker_init_fn


def get_args():
    parser = argparse.ArgumentParser("test inference")
    parser.add_argument("--config_path", type=str, default="./config/base_missing_modality_1.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    return parser.parse_args()


def get_transforms(hparams):
    image_size = hparams["image_size"]

    all_transforms = {}
    all_transforms["street"] = {
        "val": alb.Compose(
            [
                alb.Resize(height=image_size, width=image_size),
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        ),
    }

    all_transforms["ortho"] = {
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
    assert os.path.isfile(args.checkpoint_path)
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

    _, val_indices = setup_train_val_split(dataset, hparams)

    transforms_dict = get_transforms(hparams)
    val_dataset = CustomSubset(
        Subset(dataset, val_indices),
        transforms_dict={
            "street": transforms_dict["street"]["val"],
            "ortho": transforms_dict["ortho"]["val"],
            "s2": transforms_dict["s2"]["val"],
        },
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams["val_parameters"]["batch_size"],
        num_workers=hparams["num_workers"],
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    accuracy = Accuracy(task="multiclass", num_classes=hparams["model"]["num_classes"])
    accuracy_missing_modality = Accuracy(task="multiclass", num_classes=hparams["model"]["num_classes"])
    model = MultiModalNetFullModalityPl.load_from_checkpoint(
        args.checkpoint_path,
        hparams=hparams,
        map_location=device,
    )
    model.eval()

    for batch in tqdm.tqdm(val_loader):
        images, s2_data, country_id, label = batch
        images = images.to(device)
        s2_data = s2_data.to(device)
        country_id = country_id.to(device)

        with torch.no_grad():
            logits = model(images, s2_data, country_id)
        logits = logits.cpu()
        preds = torch.argmax(logits.cpu(), dim=1)

        images[:, 3:, :, :] = 0
        with torch.no_grad():
            logits_missing_modality = model(images, s2_data, country_id)

        logits_missing_modality = logits_missing_modality.cpu()
        preds_missing_modality = torch.argmax(logits_missing_modality.cpu(), dim=1)

        accuracy.update(preds, label)
        accuracy_missing_modality.update(preds_missing_modality, batch[3])

    final_accuracy = accuracy.compute()
    final_accuracy_missing_modality = accuracy_missing_modality.compute()
    print(f"final accuracy full modality: {final_accuracy}")
    print(f"final accuracy missing modality: {final_accuracy_missing_modality}")


if __name__ == "__main__":
    main()
