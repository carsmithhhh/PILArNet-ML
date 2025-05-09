import sys
sys.path.append('..')

import MinkowskiEngine as ME
import torch
from torchvision import transforms
import random

import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from data_utils import *
from polarmae.datasets import PILArNetDataModule

from model import *
from loss import *
import torch.optim as optim

dataset = PILArNetDataModule(
    #data_path="../pilarnet/train/*.h5",
    data_path="../../pilarnet/train/*.h5",
    batch_size=48,  # 24 events per batch
    num_workers=0,
    dataset_kwargs={
        "emin": 1.0e-2,  # min energy for log transform
        "emax": 20.0,  # max energy for log transform
        "energy_threshold": 0.13,  # remove points with energy < 0.13
        "remove_low_energy_scatters": True,  # remove low energy scatters (PID=4)
        "maxlen": -1,  # max number of events to iterate over
        "min_points": 1024, # minimum number of points in an event
    },
)
dataset.setup()

# DataLoader
train_loader = dataset.train_dataloader()
val_loader = dataset.val_dataloader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

