import torch
import torch.nn as nn

import torchvision.models as models
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x



class encoder(nn.Module):
    def __init__(self, num_classes):
        super(encoder, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        # x 224
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        return e1, e2, e3, e4, e5



class encoder18(nn.Module):
    def __init__(self, num_classes):
        super(encoder18, self).__init__()


        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):

        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        return e1, e2, e3, e4, e5

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, feature):
        e1, e2, e3, e4, e5 = feature
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out1 = self.outconv(d1)  # 224

        return torch.sigmoid(out1)









class ResNet34U_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet34U_f, self).__init__()

        self.encoder1 = encoder(num_classes)

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )


    def forward(self, x,fp=False):
        e1, e2, e3, e4, e5 = self.encoder1(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out1 = self.outconv(d1)
        if fp:
            return F.sigmoid(out1), e5
        else:
            return F.sigmoid(out1)

class ResNet18U_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet18U_f, self).__init__()


        self.encoder1 = encoder18(num_classes)
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )


    def forward(self, x,fp=False):
        e1, e2, e3, e4, e5 = self.encoder1(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out1 = self.outconv(d1)
        if fp:
            return F.sigmoid(out1), e5
        else:
            return F.sigmoid(out1)


class CNNFusionBlock(nn.Module):
    def __init__(self, c1, c2, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1 + c2, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))

class SEFusionBlock(nn.Module):
    def __init__(self, c1, c2, out_c):
        super().__init__()
        total_c = c1 + c2
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_c, 32, 1), nn.ReLU(),
            nn.Conv2d(32, total_c, 1), nn.Sigmoid()
        )
        
        self.conv = nn.Conv2d(total_c, out_c, 3, padding=1)
    
    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)  # [B,1024,H,W]
        
        attn = self.se(cat)
        enhanced = cat * attn
        
        return self.conv(enhanced)


class Depth_W_CNNFusion_ResNet34U_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.rgb_branch = ResNet34U_f(num_classes, dropout)        
        self.depth_encoder = encoder(num_classes=None)
        
        # Fusion blocks
        self.fusion_block5 = CNNFusionBlock(512, 512, 512)
        self.fusion_block4 = CNNFusionBlock(256, 256, 256)
        self.fusion_block3 = CNNFusionBlock(128, 128, 128)
        self.fusion_block2 = CNNFusionBlock(64, 64, 64)
        self.fusion_block1 = CNNFusionBlock(64, 64, 64)
        
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64, 64)
        self.decoder1 = DecoderBlock(64 + 64, 64)
        
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def forward(self, x, depth=None, fp=False):
        e1, e2, e3, e4, e5 = self.rgb_branch.encoder1(x)
        output = {'rgb': self.rgb_branch(x, fp=fp), 'rgb_depth': None}
        
        if depth is None:
            return output
        
        # RGB-D fusion
        e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        f5 = self.fusion_block5(e5, e5_d)
        f4 = self.fusion_block4(e4, e4_d)
        f3 = self.fusion_block3(e3, e3_d)
        f2 = self.fusion_block2(e2, e2_d)
        f1 = self.fusion_block1(e1, e1_d)
        
        # Decoder
        d5 = self.decoder5(f5)
        d4 = self.decoder4(torch.cat([d5, f4], dim=1))
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))
        d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
        out1 = self.outconv(d1)
        output['rgb_depth'] = (F.sigmoid(out1), f5) if fp else F.sigmoid(out1)
        return output


class Depth_W_SEFusion_ResNet34U_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.rgb_branch = ResNet34U_f(num_classes, dropout)        
        self.depth_encoder = encoder(num_classes=None)
        
        # Fusion blocks
        self.fusion_block5 = SEFusionBlock(512, 512, 512)
        self.fusion_block4 = SEFusionBlock(256, 256, 256)
        self.fusion_block3 = SEFusionBlock(128, 128, 128)
        self.fusion_block2 = SEFusionBlock(64, 64, 64)
        self.fusion_block1 = SEFusionBlock(64, 64, 64)
        
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64, 64)
        self.decoder1 = DecoderBlock(64 + 64, 64)
        
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def forward(self, x, depth=None, fp=False):
        output = {'rgb': self.rgb_branch(x, fp=fp), 'rgb_depth': None}
        
        if depth is None:
            return output
        
        # RGB-D fusion
        e1, e2, e3, e4, e5 = self.rgb_branch.encoder1(x)
        e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        f5 = self.fusion_block5(e5, e5_d)
        f4 = self.fusion_block4(e4, e4_d)
        f3 = self.fusion_block3(e3, e3_d)
        f2 = self.fusion_block2(e2, e2_d)
        f1 = self.fusion_block1(e1, e1_d)
        
        # Decoder
        d5 = self.decoder5(f5)
        d4 = self.decoder4(torch.cat([d5, f4], dim=1))
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))
        d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
        out1 = self.outconv(d1)
        output['rgb_depth'] = (F.sigmoid(out1), f5) if fp else F.sigmoid(out1)
        return output


class DepthFusion_ResNet34U_f_EMAEncoderOnly(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.rgb_encoder = encoder(num_classes=None)

        self.depth_encoder = encoder(num_classes=None)
        
        # Fusion blocks
        self.fusion_block5 = SEFusionBlock(512, 512, 512)
        self.fusion_block4 = SEFusionBlock(256, 256, 256)
        self.fusion_block3 = SEFusionBlock(128, 128, 128)
        self.fusion_block2 = SEFusionBlock(64, 64, 64)
        self.fusion_block1 = SEFusionBlock(64, 64, 64)
        
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64, 64)
        self.decoder1 = DecoderBlock(64 + 64, 64)
        
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def forward(self, x, depth, fp=False):

        # RGB-D fusion
        e1, e2, e3, e4, e5 = self.rgb_encoder(x)
        e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        
        f5 = self.fusion_block5(e5, e5_d)
        f4 = self.fusion_block4(e4, e4_d)
        f3 = self.fusion_block3(e3, e3_d)
        f2 = self.fusion_block2(e2, e2_d)
        f1 = self.fusion_block1(e1, e1_d)
        
        # Decoder
        d5 = self.decoder5(f5)
        d4 = self.decoder4(torch.cat([d5, f4], dim=1))
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))
        d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
        out1 = self.outconv(d1)
        final_output = F.sigmoid(out1)
        if fp:
            return final_output, f5
        else:
            return final_output





if __name__ == "__main__":
    rgb = torch.randn(1, 3, 320, 320)
    depth = torch.randn(1, 3, 320, 320)
    mask = torch.randn(1, 2, 320, 320)

        # Training example (Mean Teacher)
    teacher = Depth_W_SEFusion_ResNet34U_f(num_classes=1)

    student = ResNet34U_f(num_classes=1)    

