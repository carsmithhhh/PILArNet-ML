import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

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
        self.out_final = ME.MinkowskiLinear(in_features=256, out_features=features)
        
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


        