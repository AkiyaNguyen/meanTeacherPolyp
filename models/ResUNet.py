import torch
import torch.nn as nn

import torchvision.models as models
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

from torchinfo import summary

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

        return e1, e2, e3, e4, e5 ## 64, 64, 128, 256, 512



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

class SEBlock(nn.Module):
    def __init__(self,input_channels, mid_channels=32):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, mid_channels, 1), nn.ReLU(),
            nn.Conv2d(mid_channels, input_channels, 1), nn.Sigmoid()   
        )
    def forward(self, x):
        return x * self.se(x)

class SEFusionBlock(nn.Module):
    def __init__(self, c1, c2, out_c):
        super().__init__()
        total_c = c1 + c2
        self.se = SEBlock(total_c)
        self.conv = nn.Conv2d(total_c, out_c, 3, padding=1)
    
    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)  # [B,1024,H,W]
        
        enhanced = self.se(cat)
        
        return self.conv(enhanced)

class ACFusionBlock(nn.Module):
    """
    implementation of attention complementary module 
    from "https://arxiv.org/abs/1905.10089"

    """
    def __init__(self, channel):
        super().__init__()
        self.se1 = SEBlock(input_channels=channel)
        self.se2 = SEBlock(input_channels=channel)
    def forward(self, x1, x2, preceding_feature=None):
        """
        ensure that the shape of x1 and x2 are the same and previous feature (if exists) are the same
        """
        se1 = self.se1(x1)
        se2 = self.se2(x2)
        if preceding_feature is not None:
            return preceding_feature + se1 + se2
        else:
            return se1 + se2


class Depth_W_ACM_ResNet34U_f_EMAEncoderOnly(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.rgb_encoder = encoder(num_classes=None)

        self.depth_encoder = encoder(num_classes=None)
        
        # Fusion blocks
        self.fusion_block5 = ACFusionBlock(512)
        self.fusion_block4 = ACFusionBlock(256)
        self.fusion_block3 = ACFusionBlock(128)
        self.fusion_block2 = ACFusionBlock(64)
        self.fusion_block1 = ACFusionBlock(64)

        res_merge = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.merge_layer1 = res_merge.layer1
        self.merge_layer2 = res_merge.layer2
        self.merge_layer3 = res_merge.layer3
        self.merge_layer4 = res_merge.layer4

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
        
        # f5 = self.fusion_block5(e5, e5_d)
        # f5 = ...
        # f4 = self.fusion_block4(e4, e4_d, preceding_feature=f5)
        # f4 = ...
        # f3 = self.fusion_block3(e3, e3_d, preceding_feature=f4)
        # f3 = ...
        # f2 = self.fusion_block2(e2, e2_d, preceding_feature=f3)
        # f2 = ...
        # f1 = self.fusion_block1(e1, e1_d, preceding_feature=f2)

        f1 = self.fusion_block1(e1, e1_d)
        m = nn.MaxPool2d(3, stride=2, padding=1)
        merge_f1 = self.merge_layer1(m(f1))
        f2 = self.fusion_block2(e2, e2_d, preceding_feature=merge_f1)
        merge_f2 = self.merge_layer2(f2)
        f3 = self.fusion_block3(e3, e3_d, preceding_feature=merge_f2)
        merge_f3 = self.merge_layer3(f3)
        f4 = self.fusion_block4(e4, e4_d, preceding_feature=merge_f3)
        merge_f4 = self.merge_layer4(f4)
        f5 = self.fusion_block5(e5, e5_d, preceding_feature=merge_f4)

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

class DepthFusion_ResNet34U_f_EMAEncoderOnly1(nn.Module):
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
            fea = {'rgb_encode': e5, 'depth_encode': e5_d, 'fusion': f5}
            return final_output, fea
        else:
            return final_output


class ResidualSEFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1 + c2, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        return self.conv(concat)


class DepthResidualSEFusion_ResNet34U_f_EMAEncoderOnly(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.rgb_encoder = encoder(num_classes=None)

        self.depth_encoder = encoder(num_classes=None)
        
        # Fusion blocks
        self.fusion_block5 = ResidualSEFusion(512, 512)
        self.fusion_block4 = ResidualSEFusion(256, 256)
        self.fusion_block3 = ResidualSEFusion(128, 128)
        self.fusion_block2 = ResidualSEFusion(64, 64)
        self.fusion_block1 = ResidualSEFusion(64, 64)
        
        # up_channel_maker = lambda c_in, c_out: nn.Sequential(
        #     nn.Conv2d(c_in, c_out, kernel_size=1),
        #     nn.BatchNorm2d(c_out),
        #     nn.ReLU(inplace=True)
        # )
        # self.up_channels = nn.ModuleList([
        #     up_channel_maker(512, 512),
        #     up_channel_maker(256, 256),
        #     up_channel_maker(128, 128),
        #     up_channel_maker(64, 64),
        #     up_channel_maker(64, 64),
        # ])

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
        
        res_f5 = self.fusion_block5(e5, e5_d)
        res_f4 = self.fusion_block4(e4, e4_d)
        res_f3 = self.fusion_block3(e3, e3_d)
        res_f2 = self.fusion_block2(e2, e2_d)
        res_f1 = self.fusion_block1(e1, e1_d)
        
        # f5 = res_f5 + self.up_channels[0](e5)
        # f4 = res_f4 + self.up_channels[1](e4)
        # f3 = res_f3 + self.up_channels[2](e3)
        # f2 = res_f2 + self.up_channels[3](e2)
        # f1 = res_f1 + self.up_channels[4](e1)
        f5 = res_f5 + e5
        f4 = res_f4 + e4
        f3 = res_f3 + e3
        f2 = res_f2 + e2
        f1 = res_f1 + e1
        
        
        # Decoder
        d5 = self.decoder5(f5)
        d4 = self.decoder4(torch.cat([d5, f4], dim=1))
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))
        d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
        out1 = self.outconv(d1)
        final_output = F.sigmoid(out1)
        if fp:
            fea = {'rgb_encode': e5, 'depth_encode': e5_d, 'fusion': f5, 'res_fusion': res_f5}
            return final_output, fea
        else:
            return final_output


# class DepthFusion_ResNet34U_f_EMAEncoderOnly_Disentanglement(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super().__init__()
#         self.rgb_encoder = encoder(num_classes=None)

#         self.depth_encoder = encoder(num_classes=None)
        
#         # Fusion blocks
#         self.fusion_block5 = SEFusionBlock(512, 512, 1024)
#         self.fusion_block4 = SEFusionBlock(256, 256, 256)
#         self.fusion_block3 = SEFusionBlock(128, 128, 128)
#         self.fusion_block2 = SEFusionBlock(64, 64, 64)
#         self.fusion_block1 = SEFusionBlock(64, 64, 64)
        
#         self.decoder5 = DecoderBlock(512, 512)
#         self.decoder4 = DecoderBlock(512 + 256, 256)
#         self.decoder3 = DecoderBlock(256 + 128, 128)
#         self.decoder2 = DecoderBlock(128 + 64, 64)
#         self.decoder1 = DecoderBlock(64 + 64, 64)

#         self.fuse_before_split = nn.Sequential(
#             nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        
#         self.outconv = nn.Sequential(
#             ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(32, num_classes, 1),
#         )
    
#     def forward(self, x, depth, fp=False):

#         # RGB-D fusion
#         e1, e2, e3, e4, e5 = self.rgb_encoder(x)
#         e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        
#         f5 = self.fusion_block5(e5, e5_d)
#         f4 = self.fusion_block4(e4, e4_d)
#         f3 = self.fusion_block3(e3, e3_d)
#         f2 = self.fusion_block2(e2, e2_d)
#         f1 = self.fusion_block1(e1, e1_d)

#         middle_feature = self.fuse_before_split(f5)
#         inv, dom = torch.chunk(middle_feature, 2, dim=1)
#         features = {'inv': inv, 'dom': dom}
#         inv_pooled = torch.flatten(self.avgpool(inv), 1)
#         dom_pooled = torch.flatten(self.avgpool(dom), 1)

#         features['inv-pool'] = inv_pooled
#         features['dom-pool'] = dom_pooled


#         # Decoder
#         d5 = self.decoder5(f5)
#         d4 = self.decoder4(torch.cat([d5, f4], dim=1))
#         d3 = self.decoder3(torch.cat([d4, f3], dim=1))
#         d2 = self.decoder2(torch.cat([d3, f2], dim=1))
#         d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
#         out1 = self.outconv(d1)
#         final_output = F.sigmoid(out1)
#         if fp:
#             return final_output, features
#         else:
#             return final_output

class DAv2Fusion_ResNet34U_f_EMAEncoderOnly(nn.Module):
    """
    Teacher model replacing the ResNet-34 depth encoder with a frozen DAv2 ViT-S encoder.
    
    Key design decisions:
    - DAv2 encoder is fully frozen; only projection layers + fusion blocks + decoder are trained.
    - 4 intermediate ViT layers are extracted and projected to match ResNet-34 channel dims.
    - e1 (160x160) is skipped for DAv2 fusion: upsampling from 22x22 by 7x is too aggressive.
    - Inference uses RGB-only ResNet-34 student — no DAv2 at deployment time.
    """
    def __init__(self, num_classes, dropout=0.1, dav2_model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        
        # RGB encoder — EMA-updated during SSL training (same as current architecture)
        self.rgb_encoder = encoder(num_classes=None)
        
        # DAv2 ViT-S backbone — frozen, used as privileged geometric feature extractor
        
        dav2 = AutoModelForDepthEstimation.from_pretrained(dav2_model_name)
        self.dav2_encoder = dav2.backbone 
        for p in self.dav2_encoder.parameters():
            p.requires_grad = False
        
        # Project DAv2 patch tokens [B, 484, 384] to match ResNet-34 channel dims at each level.
        # ViT-S hidden dim = 384. We map 4 intermediate layers to e2..e5 (skip e1, see docstring).
        # Layer indices used: 3, 6, 9, 12 (evenly spaced across 12 transformer blocks).
        dav2_dim = 384
        self.proj5 = self._make_proj(dav2_dim, 512)  # layer 12 → matches e5 (512ch, 10x10)
        self.proj4 = self._make_proj(dav2_dim, 256)  # layer  9 → matches e4 (256ch, 20x20)
        self.proj3 = self._make_proj(dav2_dim, 128)  # layer  6 → matches e3 (128ch, 40x40)
        self.proj2 = self._make_proj(dav2_dim, 64)   # layer  3 → matches e2 ( 64ch, 80x80)
        
        # SE-guided fusion blocks — same as DepthFusion_ResNet34U_f_EMAEncoderOnly
        self.fusion_block5 = SEFusionBlock(512, 512, 512)
        self.fusion_block4 = SEFusionBlock(256, 256, 256)
        self.fusion_block3 = SEFusionBlock(128, 128, 128)
        self.fusion_block2 = SEFusionBlock(64,  64,  64)
        
        # Standard U-Net decoder with skip connections
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64,  64)
        self.decoder1 = DecoderBlock(64  + 64,  64)  # e1 passed directly, no DAv2 fusion
        
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def _make_proj(self, in_dim, out_channels):
        """
        1x1 conv projection: maps DAv2 channel dim to ResNet-34 channel dim at a given level.
        Spatial resizing is handled separately via F.interpolate in forward().
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _extract_dav2_features(self, x):
        """
        Extract 4 intermediate hidden states from frozen DAv2 ViT-S.
        
        Input : x        [B, 3, 320, 320]
        Output: list of 4 tensors, each [B, 384, 22, 22]
        
        Spatial grid: floor(320 / 14) = 22  →  22x22 = 484 patch tokens
        HuggingFace returns 13 hidden states (index 0 = patch embedding, 1..12 = transformer layers).
        CLS token at position 0 is dropped before reshape.
        """
        with torch.no_grad():
            hidden_states = self.dav2_encoder(
                pixel_values=x,
                output_hidden_states=True
            ).hidden_states  # tuple of 13 x [B, 485, 384]
        
        feats = []
        for idx in [3, 6, 9, 12]:
            h = hidden_states[idx]       # [B, 485, 384]
            h = h[:, 1:, :]             # drop CLS token → [B, 484, 384]
            B, N, D = h.shape
            h = h.permute(0, 2, 1)      # [B, 384, 484]
            h = h.reshape(B, D, 22, 22) # [B, 384, 22, 22]
            feats.append(h)
        
        # feats[0] = layer 3  (low-level, closer to edges/textures)
        # feats[3] = layer 12 (high-level, semantic/geometric)
        return feats
    
    def forward(self, x, fp=False):
        # --- RGB encoder ---
        e1, e2, e3, e4, e5 = self.rgb_encoder(x)
        # e1: [B,  64, 160, 160]
        # e2: [B,  64,  80,  80]
        # e3: [B, 128,  40,  40]
        # e4: [B, 256,  20,  20]
        # e5: [B, 512,  10,  10]

        # --- DAv2 feature extraction (no gradient) ---
        dav2_feats = self._extract_dav2_features(x)
        # each: [B, 384, 22, 22]

        # --- Project + bilinear resize to match ResNet spatial dims ---
        def proj_and_resize(proj_layer, feat, target_feat):
            out = proj_layer(feat)  # [B, out_c, 22, 22]
            return F.interpolate(
                out, size=target_feat.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        d2 = proj_and_resize(self.proj2, dav2_feats[0], e2)  # [B,  64,  80,  80]
        d3 = proj_and_resize(self.proj3, dav2_feats[1], e3)  # [B, 128,  40,  40]
        d4 = proj_and_resize(self.proj4, dav2_feats[2], e4)  # [B, 256,  20,  20]
        d5 = proj_and_resize(self.proj5, dav2_feats[3], e5)  # [B, 512,  10,  10]

        # --- SE-guided fusion at each encoder level ---
        f5 = self.fusion_block5(e5, d5)
        f4 = self.fusion_block4(e4, d4)
        f3 = self.fusion_block3(e3, d3)
        f2 = self.fusion_block2(e2, d2)
        # e1 has no DAv2 counterpart — passed directly to decoder

        # --- U-Net decoder with skip connections ---
        dec5 = self.decoder5(f5)
        dec4 = self.decoder4(torch.cat([dec5, f4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, f3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, f2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, e1], dim=1))

        out = self.outconv(dec1)
        final_output = torch.sigmoid(out)

        if fp:
            return final_output, dec1
        return final_output



if __name__ == "__main__":
    # rgb = torch.randn(1, 3, 320, 320)
    # depth = torch.randn(1, 3, 320, 320)
    # mask = torch.randn(1, 2, 320, 320)

    #     # Training example (Mean Teacher)
    # # teacher = Depth_W_SEFusion_ResNet34U_f(num_classes=1)

    # # student = ResNet34U_f(num_classes=1)    

    # teacher = Depth_W_ACM_ResNet34U_f_EMAEncoderOnly(num_classes=1)
    # pred = teacher(rgb, depth)
    # print(pred.shape)

    
    def count(model):
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = total - trainable
        return total, trainable, frozen

    def fmt(n):
        return f"{n/1e6:.2f}M"

    rgb   = torch.randn(1, 3, 320, 320)
    depth = torch.randn(1, 3, 320, 320)

    print("=" * 58)
    print(f"{'Model':<12} {'Total':>10} {'Trainable':>12} {'Frozen':>10}")
    print("=" * 58)

    models_to_test = [
        ("A (RGB)",    ResNet34U_f(num_classes=1),                      (rgb,)),
        ("B (Depth)",  DepthFusion_ResNet34U_f_EMAEncoderOnly(num_classes=1), (rgb, depth)),
        ("C (DAv2)",   DAv2Fusion_ResNet34U_f_EMAEncoderOnly(num_classes=1),  (rgb,)),
    ]

    for name, model, inp in models_to_test:
        total, trainable, frozen = count(model)
        print(f"{name:<12} {fmt(total):>10} {fmt(trainable):>12} {fmt(frozen):>10}")

    print()
    print("=" * 58)
    print("FLOPs — input 320x320, batch=1")
    print("=" * 58)
    for name, model, inp in models_to_test:
        s = summary(model, input_data=inp, verbose=0)
        macs_g = s.total_mult_adds / 1e9
        print(f"{name:<12}  GMACs: {macs_g:.2f}")