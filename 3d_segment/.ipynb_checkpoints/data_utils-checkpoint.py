import MinkowskiEngine as ME
import torch
from custom_transforms import *
from torchvision import transforms
import random
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np

    
class SimCLRPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, point_clouds, transform, device="cuda"):
        self.point_clouds = point_clouds  # list of (N, 4) arrays or tensors
        self.transform = transform
        self.device = device
        np.random.seed(42)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        pc = self.point_clouds[idx]  # (N, 4)

        x1 = self.transform(pc) # (N, 4)
        x2 = self.transform(pc) # (N, 4)

        coords1, coords2 = x1[:, :3], x2[:, :3]
        feats1, feats2 = x1[:, 3:], x2[:, 3:] # energy

        coords1 = torch.tensor(coords1, dtype=torch.float32)
        coords2 = torch.tensor(coords2, dtype=torch.float32)
        coords1 = coords1.floor().int() # already voxelized
        coords2 = coords2.floor().int()
        feats1 = torch.tensor(feats1, dtype=torch.float32)
        feats2 = torch.tensor(feats2, dtype=torch.float32)

        # batch = torch.zeros((coords1.shape[0], 1), dtype=torch.int32)  # single batch index
        # coords1_full = torch.cat([batch, coords1], dim=1)  # (N, 4)
        # coords2_full = torch.cat([batch, coords2], dim=1)  # (N, 4)

        return coords1, coords2, feats1, feats2
    
class PilarnetDataset(torch.utils.data.Dataset): # returns dataset on cpu
    def __init__(self, point_clouds, ground_truth_ids):
        self.point_clouds = point_clouds  # list of (N_i, 4) arrays
        self.labels = ground_truth_ids  # list of (N_i,) label arrays
        np.random.seed(42)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        pc = self.point_clouds[idx]  # (N, 4)
        # length = self.lengths[idx]
        # pc = self.point_clouds[idx][:length] # (N, 4)
        labels = self.labels[idx]    # (N,)
    
        coords = pc[:, :3]
        feats  = pc[:, 3:] # energy
    
        coords = torch.tensor(coords, dtype=torch.float32)
        feats  = torch.tensor(feats, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
    
        # already voxelized
        coords = coords.floor().int()
        batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32)  # single batch index
        coords_full = torch.cat([batch, coords], dim=1)  # (N, 4)
    
        return coords, feats, labels # why not coords_full? seems to be working anyways....

def compute_train_transform(seed=123456):
    '''
    Data is 3d point clouds with dimension (n_points, xyz)
    Transform before converting to sparse tensors w/ ME
    
    Current Transforms:
        - 3d rotation
        - dropping of random points

    Could Add:
        - jitter energy values
    
    Must make sure all transformed data have same dimension / resolution before training
    '''
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    train_transform = transforms.Compose([
        # Step 1: rotate 0, 90, 180, or 270 degrees
        RandomRotate90Z(),
        # Step 2: dropout some points
        RandomPointDropout(dropout_p=0.3), # might be a problem - returns a pointcloud with a smaller number of points
    ])

    return train_transform

def compute_test_transform():
    # test_transform = transforms.Compose([
        
    # ])
    # return test_transform
    pass

class MNIST3DExtrudedDataset(Dataset):
    def __init__(self, root='./data', train=True, download=True, depth=3, voxel_size=1.0, device='cuda'):
        self.dataset = MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())
        self.depth = depth
        self.voxel_size = voxel_size
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]     # img: (1, 28, 28), label: int
        img = img.squeeze(0).numpy()       # (28, 28)

        coords, feats = [], []
        for z in range(self.depth):
            ys, xs = np.nonzero(img)
            intensity = img[ys, xs]        # (N,)
            coords_z = np.stack([xs, ys, np.full_like(xs, z)], axis=1)  # (N, 3)
            coords.append(coords_z)
            feats.append(intensity[:, None])  # (N, 1)

        coords = np.concatenate(coords, axis=0)

        feats = np.concatenate(feats, axis=0)

        # Quantize coords
        coords = torch.tensor(coords, dtype=torch.float32)  # (M, 3)
        coords = torch.floor(coords / self.voxel_size).int()


        # batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32)  # single sample per batch
        # coords = torch.cat([batch, coords], dim=1)  # (M, 4)
        feats = torch.tensor(feats, dtype=torch.float32)

        # stensor = ME.SparseTensor(features=feats, coordinates=coords, device=self.device)

        return coords, feats, label


