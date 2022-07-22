import gzip
import torch
import numpy as np
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import argparse
import time
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datetime import timedelta

import pandas as pd
import glob
import os.path
from pathlib import Path
from PIL import Image


class LiveCellImageDataset(torch.utils.data.Dataset):
    """Dataset that reads in various features"""

    def __init__(self, dir_path, ext="tif"):

        dir_path = Path(
            "D:\\xing-vimentin-dic-pipeline\\src\\cxa_segmentation\\cxa-data\\june_2022_data\\day0_Notreat_Group1_wellA1_RI_MIP_stitched"
        )
        self.img_path_list = sorted(glob.glob(str(dir_path / "*tif")))
        self.img_cache = {}
        print("%d %s img file paths loaded: " % (len(self.img_path_list), ext))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        if idx in self.img_cache:
            return self.img_cache[idx]

        img = Image.open(self.img_path_list[idx])
        img = np.array(img)
        self.img_cache[idx] = img
        return img
