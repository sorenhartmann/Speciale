from pathlib import Path

import pandas as pd
import requests
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from pytorch_lightning import LightningDataModule

data_dir = Path(__file__).parents[2] / "data"

from sklearn.decomposition import PCA


class IrisDataset(Dataset):

    url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
    file = data_dir / "raw" / "iris" / "iris.csv"

    def __init__(self, n_dims=None, train=True, download=False):

        if not train:
            raise ValueError("No test set available")

        if download:
            self.download_data()

        self.df = pd.read_csv(self.file)
        self.df["species"] = self.df["species"].astype("category")

        self.X = torch.tensor(
            self.df[
                ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            ].to_numpy(),
            dtype=torch.float,
        )

        if n_dims is not None:
            self.X = torch.tensor(PCA(n_dims).fit_transform(self.X), dtype=torch.float)

        self.y = torch.tensor(self.df["species"].cat.codes, dtype=torch.long)

    def download_data(self):

        if self.file.exists():
            return

        # download
        r = requests.get(self.url)

        self.file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.file, "wb") as f:
            f.write(r.content)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
