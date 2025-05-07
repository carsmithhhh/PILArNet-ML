import MinkowskiEngine as ME
import torch
from custom_transforms import *
from torchvision import transforms
import random

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

