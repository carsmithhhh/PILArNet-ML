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
from collections import defaultdict

class UpsamplingBlock(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, dimension=3, upsample=True):
        super().__init__(dimension)

        self.upsample = upsample
        if self.upsample:
            self.conv1T = ME.MinkowskiConvolutionTranspose(
                in_channels, out_channels, kernel_size=2, stride=2, dimension=dimension
            )
        else:
            self.conv1T = ME.MinkowskiConvolution(  # no upsampling
                in_channels, out_channels, kernel_size=3, stride=1, dimension=dimension
            )

        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels * 2 if self.upsample else out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dimension=dimension
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x, skip=None):
        x = MF.relu(self.bn1(self.conv1T(x)))
        if skip is not None:
            x = ME.cat(x, skip)
        x = MF.relu(self.bn2(self.conv2(x)))
        return x

class UNet_Decoder(ME.MinkowskiNetwork):
    # input is last output of encoder: (B, n_points_down3, 512)
    def __init__(self, in_channels=512, out_channels=1, num_classes=4, dimension=3):
        super().__init__(dimension)

        # Layers
        self.up1 = UpsamplingBlock(in_channels=in_channels, out_channels=256)
        self.up2 = UpsamplingBlock(in_channels=256, out_channels=128)
        self.up3 = UpsamplingBlock(in_channels=128, out_channels=64)
        self.up4 = UpsamplingBlock(in_channels=64, out_channels=32)
        self.up5 = UpsamplingBlock(in_channels=32, out_channels=16, upsample=False)

    # def forward(self, encoder_feats):
    #     x = encoder_feats[-1] # from 512-channel bottleneck
    #     x = self.up1(torch.cat([x, encoder_feats[-2]], dim=1)) # to 256
    #     x = self.up2(torch.cat([x, encoder_feats[-3]], dim=1)) # to 128
    #     x = self.up3(torch.cat([x, encoder_feats[-4]], dim=1)) # to 64
    #     x = self.up4(torch.cat([x, encoder_feats[-5]], dim=1)) # to 32
    #     x = self.up5(x) # to 16

    #     return x
    def forward(self, encoder_feats):
        x = encoder_feats[-1]  # f512
    
        x = self.up1(x, encoder_feats[-2])  # → f256
        x = self.up2(x, encoder_feats[-3])  # → f128
        x = self.up3(x, encoder_feats[-4])  # → f64
        x = self.up4(x, encoder_feats[-5])  # → f32
        x = self.up5(x, None)               # no skip connection for last block

        return x
    
class Segmentation_Head(ME.MinkowskiNetwork):
    def __init__(self, in_channels=16, num_classes=5, dimension=3):
        super().__init__(dimension)

        self.seg_head = ME.MinkowskiConvolution(in_channels=16, out_channels=num_classes, kernel_size=1, dimension=3)

    def forward(self, x):
        return self.seg_head(x)
    
class Full_UNet(ME.MinkowskiNetwork):
    def __init__(self, encoder, decoder, segmentation_head=None, dimension=3):
        super().__init__(dimension)
        self.encoder = encoder
        self.decoder = decoder
        self.seg_head = segmentation_head

    def forward(self, x):
        out, feats = self.encoder(x)
        x = self.decoder(feats)
        if self.seg_head:
            x = self.seg_head(x)
        return x
    
# supervised training full UNet
def supervised_train_unet(model, train_loader, optimizer, criterion, epoch, epochs, device='cuda'):
    model = model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    log_file_path = os.path.join('./', f'supervised_loss.txt')
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, position=0)

    total_loss = 0.0
    total_num = 0

    correct = 0
    total = 0

    with open(log_file_path, 'a') as log_file:
       for points_batch, labels_batch in progress:
            # points_batch = points_batch.to(device) 
            # labels_batch = labels_batch.to(device).view(-1)  # (N_total,)
            labels_batch = labels_batch.view(-1)
    
            logits = model(points_batch).F  # shape: (N_total, num_classes)
            assert logits.shape[0] == labels_batch.shape[0], \
                f"Mismatch: logits={logits.shape[0]}, labels={labels_batch.shape[0]}"

            # evaluate accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.numel()
           
            # evalaute loss
            loss = criterion(logits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels_batch.size(0)
            total_num += labels_batch.size(0)
            avg_loss = total_loss / total_num
            accuracy = correct / total

            log_file.write(f"{epoch},{total_num},{avg_loss:.6f}, {accuracy:.6f}\n")
            log_file.flush()
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.4f}")
           
    print(f"[Epoch {epoch}] Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def supervised_val_unet(model, val_loader, criterion, epoch=None, device='cuda'):
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_num = 0
    all_preds = []
    all_labels = []

    progress = tqdm(val_loader, desc=f"Validation", leave=False, position=0)

    with torch.no_grad():
        for points_batch, labels_batch in progress:
            # points_batch = points_batch.to(device)
            # labels_batch = labels_batch.to(device).view(-1)  # (N_total,)
            labels_batch = labels_batch.view(-1)

            logits = model(points_batch).F  # (N_total, num_classes)
            assert logits.shape[0] == labels_batch.shape[0], \
                f"Mismatch: logits={logits.shape[0]}, labels={labels_batch.shape[0]}"

            loss = criterion(logits, labels_batch)
            total_loss += loss.item() * labels_batch.size(0)
            total_num += labels_batch.size(0)

            # maybe: track predictions for metrics
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels_batch.cpu())

            avg_loss = total_loss / total_num
            progress.set_postfix(val_loss=f"{avg_loss:.4f}")

    # concatenate predictions and labels for evaluation
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy = (all_preds == all_labels).float().mean().item()

    if epoch is not None:
        print(f"[Epoch {epoch}] Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    else:
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    log_file_path = os.path.join('./', f'supervised_val_stats.txt')

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{epoch},{total_num},{avg_loss:.6f},{accuracy:.6f}\n")
        log_file.flush()

    return avg_loss, accuracy

def test_supervised_unet(model, test_loader, num_classes=4, ignore_index=-1, device='cuda'):
    model = model.to(device)
    model.eval()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for points_batch, labels_batch in tqdm(test_loader, desc="Testing"):
            # points_batch = points_batch.to(device=device)
            labels_batch = labels_batch.view(-1)

            logits = model(points_batch).F  # (N, num_classes)
            preds = logits.argmax(dim=1) # (N,)

            valid_mask = labels_batch != ignore_index
            preds = preds[valid_mask]
            labels = labels_batch[valid_mask]

            # accumulate correct predictions per class
            for c in range(num_classes):
                class_mask = labels == c
                class_total[c] += class_mask.sum().item()
                class_correct[c] += (preds[class_mask] == c).sum().item()

            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    #  final accuracies
    class_accuracies = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            class_accuracies[c] = class_correct[c] / class_total[c]
        else:
            class_accuracies[c] = float('nan')  # incase of unused classes

    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    print(f"\nTest Overall Accuracy: {overall_accuracy:.4f}")
    print("Per-class Accuracy:")
    for c in range(num_classes):
        acc = class_accuracies[c]
        print(f"Class {c}: {acc:.4f}" if not np.isnan(acc) else f"  Class {c}: N/A (no samples)")

    return overall_accuracy, class_accuracies, class_total
