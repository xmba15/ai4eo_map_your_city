import argparse
import os
import sys

import numpy as np
import yaml
from matplotlib import pyplot as plt

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CountryCode, MapYourCityDataset  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser("test read dataset")
    parser.add_argument("--config_path", type=str, default="./config/base.yaml")

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    test_dataset = MapYourCityDataset(
        csv_path=hparams["dataset"]["test_csv"],
        data_dir=hparams["dataset"]["test_dir"],
        train=False,
    )

    train_dataset = MapYourCityDataset(
        csv_path=hparams["dataset"]["train_csv"],
        data_dir=hparams["dataset"]["train_dir"],
        train=True,
    )

    # distribution by labels
    plt.figure(figsize=(8, 5))
    num_classes = 7
    plt.bar(np.arange(num_classes), np.bincount(train_dataset.labels, minlength=num_classes), color="skyblue")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.title("Distribution of Train Dataset Labels")
    plt.xticks(np.arange(6))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("train_dataset_label_distribution.jpg")

    # distribution by countries for train dataset
    plt.figure(figsize=(8, 5))
    num_countries = len(CountryCode)
    plt.bar(
        range(num_countries),
        np.bincount([CountryCode[c_id].value for c_id in train_dataset.country_ids], minlength=num_countries),
        color="skyblue",
    )
    plt.xlabel("Countries")
    plt.ylabel("Count")
    plt.title("Distribution of Train Dataset Labels by Countries")
    plt.xticks(np.arange(6))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("train_dataset_label_distribution_by_countries.jpg")

    # distribution by countries for test dataset
    plt.figure(figsize=(8, 5))
    num_countries = len(CountryCode)
    plt.bar(
        range(num_countries),
        np.bincount([CountryCode[c_id].value for c_id in test_dataset.country_ids], minlength=num_countries),
        color="skyblue",
    )
    plt.xlabel("Countries")
    plt.ylabel("Count")
    plt.title("Distribution of Test Dataset Labels by Countries")
    plt.xticks(np.arange(6))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("test_dataset_label_distribution_by_countries.jpg")


if __name__ == "__main__":
    main()
