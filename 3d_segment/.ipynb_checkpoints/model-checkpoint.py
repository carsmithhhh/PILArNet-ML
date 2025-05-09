import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from data_utils import *
import os
from loss import *
import tqdm

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
        self.gmaxpool = ME.MinkowskiGlobalMaxPooling()

        # projection head for doing contrastive loss 
        self.proj_linear = ME.MinkowskiLinear(in_features=512, out_features=256)
        self.proj_bn = ME.MinkowskiBatchNorm(256)
        self.out_final = ME.MinkowskiLinear(in_features=256, out_features=out_features)
        
        '''
        simclr projection head:
         nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),  # good catch!
            nn.Linear(512, feature_dim, bias=True)
        '''

    def forward(self, x):
        out = MF.relu(self.bn0(self.conv0(x)))
        out = MF.relu(self.bn1(self.conv1(out))) # conv --> bn --> relu
                      
        out1 = self.block1(out)
        out1 = MF.relu(self.bn2(self.conv2(out1)))

        out2 = self.block2(out1)
        out2 = MF.relu(self.bn3(self.conv3(out2)))

        out3 = self.block3(out2)
        out3 = MF.relu(self.bn4(self.conv4(out3)))

        out4 = self.block4(out3)
        out4 = self.gmaxpool(out4)

        linear_out_1 = self.proj_linear(out4)
        out5 = MF.relu(self.proj_bn(linear_out_1))
        final_out = self.out_final(out5)

        return final_out # for 1 tensor, returns (1, 128) feature vector
    
# trains unet encoder for 1 epoch
def train_unet(model, train_loader, train_optimizer: torch.optim.Optimizer, epoch: int, epochs: int, temperature: float = 0.05, voxel_size=0.01, device: str = 'cuda') -> float:
    # logging
    log_dir = './'
    log_file_path = os.path.join(log_dir, f'train_loss.txt')

    model = model.to(device=device)
    model.train()
    print("Data augmentation")
    transform = compute_train_transform(seed=45)

    pc_coords = []
    feats = []
    sparse_tensors = [] # will contain 2 views of each image in the batch [a, a, b, b, c, c, etc.]

    # making sparse tensors
    progress = tqdm(train_loader, desc=f'Converting to sparse tensors...', leave=False)
    for batch in train_loader:
        for pc in batch['points']:
            x1, x2 = transform(pc), transform(pc)
            x1 = torch.tensor(x1, dtype=torch.float32, device=device)
            x2 = torch.tensor(x2, dtype=torch.float32, device=device)

            pc_coords.append(x1[:, :3]) # separating spatial coords
            pc_coords.append(x2[:, :3])

            feats.append(x1[:, :3]) # energies for each point cloud
            feats.append(x2[:, 3:])
    batch_size = batch['points'].shape[0]

    for i, pc in enumerate(pc_coords):
        quantized_coords = torch.floor(pc / voxel_size).int()
        batch_index = torch.full((quantized_coords.shape[0], 1), i, dtype=torch.int32, device=quantized_coords.device)
        coords_with_batch = torch.cat([batch_index, quantized_coords], dim=1)  # shape (n, 4)

        sparse_tensor = ME.SparseTensor(
            features=feats[i],           # shape (n, C)
            coordinates=coords_with_batch      # shape (n, 1 + 3)
        )
        sparse_tensors.append(sparse_tensor)

    progress = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False)
    
    with open(log_file_path, 'a') as log_file:
        total_num = 0
        total_loss = 0.0
        for i in range(len(sparse_tensors) // 2):
            x_i, x_j = sparse_tensors[i], sparse_tensors[i+1]
            x_i, x_j = x_i.to(device), x_j.to(device) # maybe redundant??

            out_left, out_right, loss = None, None, None

            # embedding two views of event
            out_left = model(x_i).to(device)
            out_right = model(x_j).to(device)

            # evaluating loss
            loss = simclr_loss_vectorized(out_left, out_right, temperature)
            loss = loss.to(device)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size
            avg_loss = total_loss / total_num
            progress.set_postfix(loss=f'{avg_loss:.4f}')
            log_file.write(f'{epoch},{i},{avg_loss:.6f}\n')

            if (i+1) % 200 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': train_optimizer.state_dict(),
                        'loss': loss,
                    }, 'checkpoint.pth')

    return avg_loss
        





    

