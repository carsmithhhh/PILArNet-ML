import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
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
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, stride=1, dimension=dimension)
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)

        # downsampling sometimes
        if in_channels != out_channels:
            self.downsample = ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=dimension)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = MF.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return MF.relu(out + identity) # residual connection (adding input to output)

class UNet_Encoder(ME.MinkowskiNetwork): # all layers use Kaiming initialization by default 
    def __init__(self, in_channels=1, out_features=128, dimension=3): # out_channels is for contrastive loss projections
        super().__init__(dimension)

        # Input layers
        self.conv0 = ME.MinkowskiConvolution(in_channels=in_channels, out_channels=32, kernel_size=5, dimension=3)
        self.bn0 = ME.MinkowskiBatchNorm(32)

        self.conv1 = ME.MinkowskiConvolution(in_channels=32, out_channels=32, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn1 = ME.MinkowskiBatchNorm(32)

        # Residual blocks
        self.block1 = ResidualBlock(in_channels=32, out_channels=64, dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=64, out_channels=64, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn2 = ME.MinkowskiBatchNorm(64)
        
        self.block2 = ResidualBlock(in_channels=64, out_channels=128, dimension=3)
        self.conv3 = ME.MinkowskiConvolution(in_channels=128, out_channels=128, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn3 = ME.MinkowskiBatchNorm(128)

        self.block3 = ResidualBlock(in_channels=128, out_channels=256, dimension=3)
        self.conv4 = ME.MinkowskiConvolution(in_channels=256, out_channels=256, kernel_size=2, stride=2, dimension=3) # Downsampling
        self.bn4 = ME.MinkowskiBatchNorm(256)
        
        self.block4 = ResidualBlock(in_channels=256, out_channels=512, dimension=3)
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
        out1 = self.block1(out)
        feats.append(out1) #f64
        out1 = MF.relu(self.bn2(self.conv2(out1)))

        out2 = self.block2(out1)
        feats.append(out2) #f128
        out2 = MF.relu(self.bn3(self.conv3(out2)))

        out3 = self.block3(out2)
        feats.append(out3) #f256
        out3 = MF.relu(self.bn4(self.conv4(out3)))

        # removing max pooling
        out4 = self.block4(out3) # (B, n_points_down3, 512) # relu is applied in residual block
        # out4 = self.gmaxpool(out4)
        feats.append(out4) #f512

        if return_embedding: # do this for training whole UNet!!
            return out4, feats # (dense tensor of shape B, n_points_down3, 512)

        # dense projection layers
        else:
            pooled = self.gmaxpool(out4)
            proj = F.relu(self.proj_layernorm(self.proj_linear(pooled.F)))
            return self.out_final(x)
            # x = out4.F  # (1, 512)
            # x = self.proj_linear(x) # (1, 256)
            # x = F.relu(self.proj_layernorm(x)) # (1, 256)
            # final_out = self.out_final(x) 

        # return final_out, feats # for 1 tensor, returns (1, 128) feature vector for contrastive loss

    
# Classification head for validating embeddings
class LinearProbe(torch.nn.Module):
    def __init__(self, out_dim, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(out_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

    
# trains unet encoder for 1 epoch with SIMCLR loss
def train_unet(model, train_loader, optimizer, epoch, epochs, temperature=0.07, device='cuda'):
    model = model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    log_file_path = os.path.join('./', f'train_loss_no_norm.txt')
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, position=0)

    total_loss = 0.0
    total_num = 0

    with open(log_file_path, 'a') as log_file:
        for batch in progress:
            x_i_batch, x_j_batch = batch  # both are lists of SparseTensors
            out_left = torch.cat([model(x_i) for x_i in x_i_batch], dim=0)
            out_right = torch.cat([model(x_j) for x_j in x_j_batch], dim=0)
            cosine_sim = F.cosine_similarity(out_left, out_right).mean().item()

            # out_left = F.normalize(out_left, dim=1)
            # out_right = F.normalize(out_right, dim=1)       
            loss = simclr_loss_vectorized(out_left, out_right, temperature).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x_i_batch)
            total_num += len(x_i_batch)
            avg_loss = total_loss / total_num

            log_file.write(f"{epoch},{total_num},{avg_loss:.6f}, {cosine_sim}\n")
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
        y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

        # Get embeddings (no gradients through encoder)
        with torch.no_grad():
            embeddings = torch.cat([encoder(x, return_embedding=True) for x in x_batch], dim=0)
       
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
            y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

            embeddings = torch.cat([encoder(x, return_embedding=True) for x in x_batch], dim=0)
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
        





    

