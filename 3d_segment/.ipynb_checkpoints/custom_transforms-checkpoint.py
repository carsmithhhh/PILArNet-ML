import numpy as np
import random

class RandomRotate90Z: # should still associate energies with the correct coordinates
    def __init__(self):
        self.rot_mats = [
            np.eye(3),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        ]
    
    def __call__(self, pc): # should be cloud with dimension (N, 4)
        R = random.choice(self.rot_mats)
        coords = pc[:, :3] @ R.T
        # coords = pc @ R.T
        return np.hstack([coords, pc[:, 3:]])  # retain energy
        # return coords

class RandomPointDropout:
    def __init__(self, dropout_p):
        self.dropout_p = dropout_p

    def __call__(self, pc):
        N = pc.shape[0]
        mask = np.random.rand(N) > self.dropout_p
        return pc[mask]
