import argparse
import os
import sys

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
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
from src.integrated import (
    MultiModalNetFullModalityAttentionFusionPl,
    MultiModalNetFullModalityPl,
)
from src.models import MultiModalNet
from src.utils import fix_seed, worker_init_fn


def get_args():
    parser = argparse.ArgumentParser("test inference")
    parser.add_argument("--result_config_path", type=str, default="./config/result_config.yaml")
    parser.add_argument("--current_submit_csv", type=str, default="./shared_encoder_example_result.csv")
    parser.add_argument("--update_submit_csv", type=str, default="./plus_attention_example_result.csv")

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


def _hflip(image_tensor):
    return torch.flip(image_tensor, [3])


def _vflip(image_tensor):
    return torch.flip(image_tensor, [2])


def _hvflip(image_tensor):
    return torch.flip(image_tensor, [2, 3])


def main():
    args = get_args()
    with open(args.result_config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    test_dataset = MapYourCityDataset(
        csv_path=hparams["dataset"]["test_csv"],
        data_dir=hparams["dataset"]["test_dir"],
        train=False,
    )
    transforms_dict = get_transforms(hparams)
    transformed_test_dataset = CustomSubset(
        Subset(test_dataset, range(len(test_dataset))),
        transforms_dict={
            "street": transforms_dict["street"]["val"],
            "ortho": transforms_dict["ortho"]["val"],
            "s2": transforms_dict["s2"]["val"],
        },
        train=False,
    )

    test_loader = DataLoader(
        transformed_test_dataset,
        batch_size=1,
        num_workers=4,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    full_modality_hparams = hparams["full_modality"]
    full_modality_models = []
    for model_config_name in full_modality_hparams:
        model_config_path = full_modality_hparams[model_config_name]["config_path"]
        model_weights_path = full_modality_hparams[model_config_name]["weights_path"]
        with open(model_config_path, encoding="utf-8") as f:
            model_hparams = yaml.load(f, Loader=yaml.SafeLoader)

        model = MultiModalNetFullModalityPl.load_from_checkpoint(
            model_weights_path,
            hparams=model_hparams,
            map_location=device,
        )
        model.eval()
        full_modality_models.append(model)

    attention_fusion_hparams = hparams["attention_fusion"]
    attention_fusion_models = []
    for model_config_name in attention_fusion_hparams:
        model_config_path = attention_fusion_hparams[model_config_name]["config_path"]
        model_weights_path = attention_fusion_hparams[model_config_name]["weights_path"]
        with open(model_config_path, encoding="utf-8") as f:
            model_hparams = yaml.load(f, Loader=yaml.SafeLoader)

        model = MultiModalNetFullModalityAttentionFusionPl.load_from_checkpoint(
            model_weights_path,
            hparams=model_hparams,
            map_location=device,
        )
        model.eval()
        attention_fusion_models.append(model)

    df = pd.read_csv(args.current_submit_csv)
    pid_probs_list = {pid: [] for pid in test_dataset.subfolder_names}
    pid_preds_list = dict(zip(df["pid"], df["predicted_label"]))

    output_pids = []
    output_preds = []

    for idx, batch in enumerate(tqdm.tqdm(test_loader)):
        if idx >= 1000:
            continue

        images, s2_data, country_id = batch
        images = images.to(device)
        s2_data = s2_data.to(device)
        country_id = country_id.to(device)

        pid = test_dataset.subfolder_names[idx]
        output_pids.append(pid)

        is_input_drop = images.shape[1] == 3
        if is_input_drop:
            output_preds.append(pid_preds_list[pid])
            continue

        def _process_one_pattern(_images):
            for _model in full_modality_models:
                with torch.no_grad():
                    logits = _model(_images, s2_data, country_id)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu()
                del logits

                pid_probs_list[pid].append(probs)

            for _model in attention_fusion_models:
                with torch.no_grad():
                    logits, _, _, _ = _model(_images, s2_data, country_id)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu()
                del logits

                pid_probs_list[pid].append(probs)

        ortho_data = images[:, :3, :, :]
        street_data = images[:, 3:, :, :]

        for ortho_op in [torch.nn.Identity(), _vflip, _hflip, _hvflip]:
            for street_op in [torch.nn.Identity(), _hflip]:
                with torch.no_grad():
                    _process_one_pattern(
                        torch.cat(
                            (
                                ortho_op(ortho_data),
                                street_op(street_data),
                            ),
                            dim=1,
                        )
                    )

        pid_probs = torch.stack(pid_probs_list[pid], dim=0)
        pid_probs = torch.mean(pid_probs, dim=0)
        preds = torch.argmax(pid_probs, dim=1)
        pid_preds_list[pid] = preds.item()
        output_preds.append(pid_preds_list[pid])

    df = pd.DataFrame(list(zip(output_pids, output_preds)), columns=["pid", "predicted_label"])
    df.to_csv(args.update_submit_csv, index=False)


if __name__ == "__main__":
    main()
