import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from MinkowskiEngine.utils import batch_sparse_collate
from data_utils import *
import os
from loss import *
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

class ResidualBlock(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, dimension=3):
        super().__init__(dimension)
        self.conv1 = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=1, dimension=dimension)
        self.bn1 = ME.MinkowskiInstanceNorm(num_features=out_channels)
        self.conv2 = ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, stride=1, dimension=dimension)
        self.bn2 = ME.MinkowskiInstanceNorm(num_features=out_channels)

        # downsampling sometimes
        if in_channels != out_channels:
            self.downsample = ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=dimension)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x) # downsample all
        out = MF.relu(self.bn1(self.conv1(x))) # applying layer to input
        out = self.bn2(self.conv2(out))
        return MF.relu(out + identity) # residual connection (adding input to output)

class UNet_Encoder(ME.MinkowskiNetwork): # all layers use Kaiming initialization by default 
    def __init__(self, in_channels=1, out_features=128, dimension=3): # out_channels is for contrastive loss projections
        super().__init__(dimension)

        # Input layers
        self.conv0 = ME.MinkowskiConvolution(in_channels=in_channels, out_channels=32, kernel_size=5, dimension=3)
        self.bn0 = ME.MinkowskiInstanceNorm(32)

        self.conv1 = ME.MinkowskiConvolution(in_channels=32, out_channels=32, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn1 = ME.MinkowskiInstanceNorm(32)
        # Residual blocks
        self.block1a = ResidualBlock(in_channels=32, out_channels=64, dimension=3)
        self.block1b = ResidualBlock(in_channels=64, out_channels=64, dimension=3)
        
        
        self.conv2 = ME.MinkowskiConvolution(in_channels=64, out_channels=64, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn2 = ME.MinkowskiInstanceNorm(num_features=64)
        self.block2a = ResidualBlock(in_channels=64, out_channels=128, dimension=3)
        self.block2b = ResidualBlock(in_channels=128, out_channels=128, dimension=3)

        
        self.conv3 = ME.MinkowskiConvolution(in_channels=128, out_channels=128, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn3 = ME.MinkowskiInstanceNorm(num_features=128)
        self.block3a = ResidualBlock(in_channels=128, out_channels=256, dimension=3)
        self.block3b = ResidualBlock(in_channels=256, out_channels=256, dimension=3)
        
        self.conv4 = ME.MinkowskiConvolution(in_channels=256, out_channels=256, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn4 = ME.MinkowskiInstanceNorm(num_features=256)
        self.block4a = ResidualBlock(in_channels=256, out_channels=512, dimension=3)
        self.block4b = ResidualBlock(in_channels=512, out_channels=512, dimension=3)
        
        # I also think we don't want max pooling when we train entire UNet together
        self.gmaxpool = ME.MinkowskiGlobalMaxPooling()

        # projection head for doing contrastive loss with DENSE tensors
        # taking away when training full UNet together
        self.proj_linear = nn.Linear(512, 256)
        self.proj_layernorm = nn.LayerNorm(256)
        self.out_final = nn.Linear(256, out_features)

    def forward(self, x, return_embedding=True): # want to use return_embeddings = True for training whole UNet together
        feats = []

        out = MF.relu(self.bn0(self.conv0(x))) 
        feats.append(out) #f32

        out = MF.relu(self.bn1(self.conv1(out))) # conv --> bn --> relu     
        out1 = self.block1a(out)
        out1 = self.block1b(out1)
        feats.append(out1) #f64
        
        out1 = MF.relu(self.bn2(self.conv2(out1)))
        out2 = self.block2a(out1)
        out2 = self.block2b(out2)
        feats.append(out2) #f128
        
        out2 = MF.relu(self.bn3(self.conv3(out2)))
        out3 = self.block3a(out2)
        out3 = self.block3b(out3)
        feats.append(out3) #f256
        
        out3 = MF.relu(self.bn4(self.conv4(out3)))
        out4 = self.block4a(out3) # (B, n_points_down3, 512) # relu is applied in residual block
        out4 = self.block4b(out4)
        feats.append(out4) #f512

        # if return_embedding: # do this for validation
        #     return out4.F, feats # (dense tensor of shape B, n_points_down3, 512)
        if return_embedding:
            batch_indices = out4.C[:, 0]
            pooled = []
        
            for b in range(batch_indices.max().item() + 1):  # loop over batch indices
                mask = batch_indices == b
                pooled_embedding = out4.F[mask].mean(dim=0)  # shape (D,)
                pooled.append(pooled_embedding)
        
            pooled = torch.stack(pooled, dim=0)  # shape (batch_size, D)
            return pooled, feats

        # dense projection layers
        else:
            pooled = self.gmaxpool(out4)
            x = pooled.F
            x = self.proj_linear(x) # (1, 256)
            x = F.relu(self.proj_layernorm(x)) # (1, 256)
            return self.out_final(x)
                
# Classification head for validating embeddings
class LinearProbe(torch.nn.Module):
    def __init__(self, out_dim, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(out_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

    
# trains unet encoder for 1 epoch with SIMCLR loss
def train_encoder(model, train_loader, optimizer, epoch, epochs, temperature=0.07, device='cuda'):
    # model = model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    log_file_path = os.path.join('./', f'train_loss_no_norm.txt')
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, position=0)

    total_loss = 0.0
    total_num = 0

    with open(log_file_path, 'a') as log_file:
        # for x_i_arr, x_j_arr in progress:  # lists of SparseTensors
        #     reps_i = []
        #     reps_j = []

        #     for x_i, x_j in zip(x_i_arr, x_j_arr):
        #         z_i = model(x_i, return_embedding=False) # access contrastive projection head
        #         z_j = model(x_j, return_embedding=False)

        #         reps_i.append(z_i.squeeze(0))
        #         reps_j.append(z_j.squeeze(0))

        #     reps_i = torch.stack(reps_i)  # (B, D)
        #     reps_j = torch.stack(reps_j)  # (B, D)
        for x_i_batch, x_j_batch in progress:
        
            # Move to device if needed
            # x_i_batch = x_i_batch.to(device)
            # x_j_batch = x_j_batch.to(device)
        
            # Model returns (B, D) after internal pooling
            reps_i, _ = model(x_i_batch, return_embedding=True)
            reps_j, _ = model(x_j_batch, return_embedding=True)

            cosine_sim = F.cosine_similarity(reps_i, reps_j, dim=1).mean().item()

            loss = simclr_loss_vectorized(reps_i, reps_j, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += 1
            avg_loss = total_loss / total_num

            log_file.write(f"{epoch},{total_num},{avg_loss:.6f},{cosine_sim:.6f}\n")
            log_file.flush()
            progress.set_postfix(loss=f"{avg_loss:.4f}")

    return avg_loss


def mnist_validate(encoder, classifier, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda'):
    for epoch in range(epochs):
        train_loss = mnist_train(encoder, classifier, train_loader, criterion, optimizer, epoch, epochs, device)
    eval_loss, eval_acc = mnist_evaluate(encoder, classifier, val_loader, criterion, epoch='final')
    return eval_loss, eval_acc
    

def mnist_train(encoder, classifier, train_loader, criterion, optimizer, epoch, epochs=5, device='cuda'):
    encoder.eval()         # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    classifier.train()     # train classifier
    classifier.to(device)

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, position=0)

    for x_batch, y_batch in progress:
        #x_batch = [x.to(device) for x in x_batch]
        # y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

        # Get embeddings (no gradients through encoder)
        with torch.no_grad():
            # print("x_batch type:", type(x_batch))
            # print("x_batch.D:", x_batch.D if isinstance(x_batch, ME.SparseTensor) else "not a SparseTensor")
            # print("x_batch coordinate shape:", x_batch.C.shape if isinstance(x_batch, ME.SparseTensor) else "no C")
            embeddings, _ = encoder(x_batch, return_embedding=True)
       
        #z = F.normalize(embeddings, dim=1)
        logits = classifier(embeddings)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)
        avg_loss = total_loss / total_samples
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    return avg_loss
            
def mnist_evaluate(encoder, classifier, val_loader, criterion, epoch=None, device='cuda'):
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader, desc="Evaluating", leave=False):
            #x_batch = [x.to(device) for x in x_batch]
            # y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

            embeddings, _ = encoder(x_batch, return_embedding=True)
            #z = F.normalize(embeddings, dim=1)
            logits = classifier(embeddings)

            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"[Eval] Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
        





    

