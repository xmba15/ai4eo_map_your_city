import os
from enum import Enum

import cv2
import pandas as pd
import rasterio as rs
import torch
from torch.utils.data import Dataset

__all__ = (
    "MapYourCityDataset",
    "CountryCode",
    "CustomSubset",
)


class CountryCode(Enum):
    FMW = 0
    PNN = 1
    EMA = 2
    HUN = 3
    NEA = 4
    QCD = 5


class MapYourCityDataset(Dataset):
    def __init__(
        self,
        csv_path,
        data_dir,
        train: bool = True,
    ):
        super().__init__()

        assert os.path.isfile(os.path.expanduser(csv_path)), f"{csv_path} is not a valid file"
        assert os.path.isdir(os.path.expanduser(data_dir)), f"{data_dir} is not a valid dir"

        self.data_dir = os.path.expanduser(data_dir)
        self.csv_path = os.path.expanduser(csv_path)
        self.train = train
        self._process_gt()

    def _process_gt(self):
        df = pd.read_csv(self.csv_path)
        self.country_ids = df["country_id"].to_list()
        self.city_ids = df["city_id"].to_list()
        self.subfolder_names = df["pid"].to_list()

        if self.train:
            self.labels = df["label"].to_list()

    def __len__(self):
        return len(self.subfolder_names)

    def __getitem__(self, idx):
        assert idx < self.__len__()

        country_id = self.country_ids[idx]
        country_id_int = CountryCode[country_id].value

        subfolder_name = self.subfolder_names[idx]
        street_path = os.path.join(self.data_dir, f"{subfolder_name}/street.jpg")
        orthophoto_path = os.path.join(self.data_dir, f"{subfolder_name}/orthophoto.tif")
        s2_l2a = os.path.join(self.data_dir, f"{subfolder_name}/s2_l2a.tif")

        with rs.open(s2_l2a) as ds:
            s2_data = ds.read().transpose(1, 2, 0)

        street_data = cv2.imread(street_path)
        if self.train:
            assert street_data is not None, f"{street_path} is not a street valid path"
        if street_data is not None:
            street_data = street_data[..., ::-1]

        orthophoto_data = cv2.imread(orthophoto_path)
        assert orthophoto_data is not None, f"{orthophoto_path} is not a orthophoto valid path"
        orthophoto_data = orthophoto_data[..., ::-1]

        out = s2_data, orthophoto_data, street_data, country_id_int
        if self.train:
            out = *out, self.labels[idx]

        return out


class CustomSubset(Dataset):
    def __init__(
        self,
        subset,
        transforms_dict=None,
    ):
        self.subset = subset

        self.transforms_dict = transforms_dict

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        s2_data, orthophoto_data, street_data, country_id_int, label = self.subset[idx]

        assert orthophoto_data is not None
        assert s2_data is not None

        if self.transforms_dict is not None:
            if street_data is not None:
                street_data = self.transforms_dict["street"](image=street_data)["image"]

            orthophoto_data = self.transforms_dict["ortho"](image=orthophoto_data)["image"]
            s2_data = self.transforms_dict["s2"](image=s2_data)["image"]

        if street_data is None:
            return (
                orthophoto_data,
                s2_data,
                country_id_int,
                label,
            )

        return (
            torch.cat(
                [
                    orthophoto_data,
                    street_data,
                ],
                dim=0,
            ),
            s2_data,
            country_id_int,
            label,
        )
