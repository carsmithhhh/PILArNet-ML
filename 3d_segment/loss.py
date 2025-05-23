import torch
import torch.nn.functional as F

def simclr_loss_vectorized(z_i, z_j, temperature=0.07):
    """
    Vectorized SimCLR contrastive loss for batch embeddings.

    Inputs:
    - z_i: (B, D) Tensor, first view embeddings
    - z_j: (B, D) Tensor, second view embeddings

    Returns:
    - scalar contrastive loss
    """
    B, D = z_i.shape

    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

    # Compute cosine similarity between all 2B vectors
    sim_matrix = torch.matmul(z, z.T)  # (2B, 2B)
    sim_matrix = sim_matrix / temperature

    # Remove self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_sim = torch.sum(z_i * z_j, dim=1) / temperature  # (B,)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)       # (2B,)

    # Denominator: sum over all non-self similarities
    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)  # (2B,)
    return loss.mean()


# import torch
# import numpy as np

# def sim(z_i, z_j):
#     """Normalized dot product between two vectors.

#     Inputs:
#     - z_i: 1xD tensor.
#     - z_j: 1xD tensor.
    
#     Returns:
#     - A scalar value that is the normalized dot product between z_i and z_j.
#     """
#     norm_dot_product = None
              
#     dot_product = torch.dot(z_i, z_j)                                    #
#     norm_dot_product = dot_product / (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))      
    
#     return norm_dot_product

# def compute_sim_matrix(out):
#     """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

#     Inputs:
#     - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
#     There are a total of 2N augmented examples in the batch.
    
#     Returns:
#     - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
#     """

#     # L2 normalizing each row
#     out_norm = out / out.norm(dim=1, keepdim=True)  # [2N, D]
#     sim_matrix = torch.mm(out_norm, out_norm.T)

#     return sim_matrix

# def sim_positive_pairs(out_left, out_right):
#     """Normalized dot product between positive pairs.

#     Inputs:
#     - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
#     - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
#     Each row is a z-vector for an augmented sample in the batch.
#     The same row in out_left and out_right form a positive pair.
    
#     Returns:
#     - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
#     """
#     pos_pairs = None
    
#     dot_product = torch.sum(out_left * out_right, dim=1, keepdim=True)
#     norm_left = torch.linalg.norm(out_left, dim=1, keepdim=True)
#     norm_right = torch.linalg.norm(out_right, dim=1, keepdim=True)

#     pos_pairs = dot_product / (norm_left * norm_right)

#     return pos_pairs

# def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'): #cuda
#     """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
#     Inputs and output are the same as in simclr_loss_naive.
#     """
#     N = out_left.shape[0]
    
#     # Concatenate out_left and out_right into a 2*N x D tensor.
#     out = torch.cat([out_left, out_right], dim=0).to(device)  # [2*N, D]
    
#     # Compute similarity matrix between all pairs of augmented examples in the batch.
#     sim_matrix = compute_sim_matrix(out).to(device) # [2*N, 2*N]

#     # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
#     exponential = torch.exp(sim_matrix / tau).to(device) # 2N x 2N
    
#     # define mask to zero out terms where k=i.
#     mask = (torch.ones_like(exponential) - torch.eye(2 * N, device=exponential.device)).bool()
    
#     # binary mask
#     # exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
#     # exponential = exponential.clone().masked_select(mask).view(2 * N, -1)
#     exponential = exponential.masked_fill(~mask, 0.0)
    
#     # 2N x 1 vector.
#     denom = torch.sum(exponential, dim=1, keepdim=True)

#     # Step 2: Compute similarity between positive pairs.
#     similarities = sim_positive_pairs(out_left, out_right) # Shape: Nx1
    
#     # Step 3: Compute the numerator value for all augmented samples.
#     pos_indices = torch.arange(N, device=device)
#     similarities = sim_matrix[pos_indices, pos_indices + N]
#     numerator = torch.exp(torch.cat([similarities, similarities]) / tau).unsqueeze(1)
    
#     # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss
#     loss = -torch.log(numerator / denom).mean()
    
#     return loss