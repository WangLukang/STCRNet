import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16
from torchsummary import summary

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()

        # Squeeze module
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation module
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Squeeze module
        squeezed = self.squeeze(x).view(batch_size, channels)

        # Excitation module
        weights = self.excitation(squeezed).view(batch_size, channels, 1, 1)

        # Scale the input features
        scaled_features = x * weights

        return scaled_features

class SENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SENet, self).__init__()

        self.se_block = SEBlock(in_channels, reduction_ratio)

    def forward(self, x):
        scaled_features = self.se_block(x)

        return scaled_features

class UResNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UResNet, self).__init__()

        resnet = models.resnet50(pretrained=True)
        # -------------Encoder--------------
        # inputs 64*256*256
        self.encoder0 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # stage 1
        self.encoder1 = resnet.layer1  
        # stage 2
        self.encoder2 = resnet.layer2  
        # stage 3
        self.encoder3 = resnet.layer3  
        # stage 4
        self.encoder4 = resnet.layer4 

        # -------------SENET----------------
        self.fusion4 = SENet(4096)
        self.fusion3 = SENet(4096)
        self.fusion2 = SENet(2048)
        self.fusion1 = SENet(1024)

        # -------------Decoder--------------
        self.decoder4 = nn.Sequential(
            nn.Conv2d(4096, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(4096, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder0 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -------------Upsample------------
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # -------------Final Output---------
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x1, x2):
        # Encoder
        hx0 = self.encoder0(x1)
        hx1 = self.encoder1(hx0)
        hx2 = self.encoder2(hx1)
        hx3 = self.encoder3(hx2)
        hx4 = self.encoder4(hx3)

        hx0_2 = self.encoder0(x2)
        hx1_2 = self.encoder1(hx0_2)
        hx2_2 = self.encoder2(hx1_2)
        hx3_2 = self.encoder3(hx2_2)
        hx4_2 = self.encoder4(hx3_2)
        
        
        # Decoder
        feature4 = self.fusion4(torch.cat((hx4, hx4_2), 1))
        feature4 = self.decoder4(feature4)
        feature4 = self.upsample(feature4)

        
        feature3 = self.fusion3(torch.cat((feature4, torch.cat((hx3, hx3_2), 1)), 1))
        feature3 = self.decoder3(feature3)
        feature3 = self.upsample(feature3)

        feature2 = self.fusion2(torch.cat((feature3, torch.cat((hx2, hx2_2), 1)), 1))
        feature2 = self.decoder2(feature2)
        feature2 = self.upsample(feature2)

        feature1 = self.fusion1(torch.cat((feature2, torch.cat((hx1, hx1_2), 1)), 1))
        feature1 = self.decoder1(feature1)

        feature0 = self.decoder0(feature1)

        # Final Output
        output = self.final(feature0)
        
        return F.sigmoid(output)
    


class UResNet_VGG(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UResNet_VGG, self).__init__()

        vgg = vgg16(pretrained=True)

        # -------------Encoder--------------
        # inputs 64*256*256
        self.encoder0 = vgg.features[:4]
        self.encoder1 = vgg.features[4:9]
        self.encoder2 = vgg.features[9:16]
        self.encoder3 = vgg.features[16:23]
        self.encoder4 = vgg.features[23:30]

        # -------------SENET----------------
        self.fusion4 = SENet(1024)
        self.fusion3 = SENet(1536)
        self.fusion2 = SENet(1024)
        self.fusion1 = SENet(512)
        self.fusion0 = SENet(384)

        # -------------Decoder--------------
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(1536, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder0 = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -------------Upsample------------
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # -------------Final Output---------
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x1, x2):
        # Encoder
        hx0 = self.encoder0(x1)
        hx1 = self.encoder1(hx0)
        hx2 = self.encoder2(hx1)
        hx3 = self.encoder3(hx2)
        hx4 = self.encoder4(hx3)

        hx0_2 = self.encoder0(x2)
        hx1_2 = self.encoder1(hx0_2)
        hx2_2 = self.encoder2(hx1_2)
        hx3_2 = self.encoder3(hx2_2)
        hx4_2 = self.encoder4(hx3_2)
        # print(hx4.shape)
        # print(hx3.shape)
        # print(hx2.shape)
        # print(hx1.shape)
        # print(hx0.shape)
        # exit()

        # Decoder
        feature4 = self.fusion4(torch.cat((hx4, hx4_2), 1))
        feature4 = self.decoder4(feature4)
        feature4 = self.upsample(feature4)
        # print(feature4.shape)
        # xxxx = torch.cat((hx3, hx3_2), 1)
        # print(xxxx.shape)
        # result = torch.cat((feature4, torch.cat((hx3, hx3_2), 1)), dim=1)
        # print(result.shape)
        # xx11 = self.fusion3(result)
        # print(xx11.shape)
        # exit()
       

        feature3 = self.fusion3(torch.cat((feature4, torch.cat((hx3, hx3_2), 1)), dim=1))
        feature3 = self.decoder3(feature3)
        feature3 = self.upsample(feature3)

        feature2 = self.fusion2(torch.cat((feature3, torch.cat((hx2, hx2_2), 1)), 1))
        feature2 = self.decoder2(feature2)
        feature2 = self.upsample(feature2)

        feature1 = self.fusion1(torch.cat((feature2, torch.cat((hx1, hx1_2), 1)), 1))
        feature1 = self.decoder1(feature1)
        feature1 = self.upsample(feature1)

        feature0 = self.fusion0(torch.cat((feature1, torch.cat((hx0, hx0_2), 1)), 1))
        feature0 = self.decoder0(feature0)

        # Final Output
        output = self.final(feature0)

        return torch.sigmoid(output), feature0


class UResNet_34(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UResNet_34, self).__init__()

        resnet = models.resnet34(pretrained=True)
        # -------------Encoder--------------
        
        # self.encoder0 = nn.Sequential(
        #     nn.Conv2d(n_channels, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        
        self.encoder0 = resnet.conv1
        # stage 1
        self.encoder1 = resnet.layer1  
        # stage 2
        self.encoder2 = resnet.layer2  
        # stage 3
        self.encoder3 = resnet.layer3  
        # stage 4
        self.encoder4 = resnet.layer4  

        # -------------SENET----------------
        self.fusion4 = SENet(1024)
        self.fusion3 = SENet(1024)
        self.fusion2 = SENet(512)
        self.fusion1 = SENet(256)
        self.fusion0 = SENet(192)

        # -------------Decoder--------------
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder0 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -------------Upsample------------
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # -------------Final Output---------
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x1, x2):
        # Encoder
        hx0 = self.encoder0(x1)
        hx1 = self.encoder1(hx0)
        hx2 = self.encoder2(hx1)
        hx3 = self.encoder3(hx2)
        hx4 = self.encoder4(hx3)

        hx0_2 = self.encoder0(x2)
        hx1_2 = self.encoder1(hx0_2)
        hx2_2 = self.encoder2(hx1_2)
        hx3_2 = self.encoder3(hx2_2)
        hx4_2 = self.encoder4(hx3_2)
        
        
        # Decoder
        feature4 = self.fusion4(torch.cat((hx4, hx4_2), 1))
        feature4 = self.decoder4(feature4)
        feature4 = self.upsample(feature4)

        
        feature3 = self.fusion3(torch.cat((feature4, torch.cat((hx3, hx3_2), 1)), 1))
        feature3 = self.decoder3(feature3)
        feature3 = self.upsample(feature3)

        feature2 = self.fusion2(torch.cat((feature3, torch.cat((hx2, hx2_2), 1)), 1))
        feature2 = self.decoder2(feature2)
        feature2 = self.upsample(feature2)

        feature1 = self.fusion1(torch.cat((feature2, torch.cat((hx1, hx1_2), 1)), 1))
        feature1 = self.decoder1(feature1)
        # feature1 = self.upsample(feature1)
       
        
        feature0 = self.fusion0(torch.cat((feature1, torch.cat((hx0, hx0_2), 1)), 1))
        feature0 = self.decoder0(feature0)
        
        feature0 = self.upsample(feature0)
       
        
        # Final Output
        output = self.final(feature0)
        
        return F.sigmoid(output)

import torchvision.models as models
from thop import profile   

if __name__ == '__main__':
    model = UResNet_34(3, 1)
    
    in_batch, inchannel, in_h, in_w = 1, 3, 256, 256
   
    x1 = torch.randn(in_batch, inchannel, in_h, in_w)

    x2 = torch.randn(in_batch, inchannel, in_h, in_w)

    y = model(x1, x2)

 

   
    input_data = torch.randn(1, 3, 224, 224)  # 假设输入图像尺寸为224x224，3个通道

    # 计算FLOPs和参数数量
    flops, params = profile(model, inputs=(x1, x2))
    print(f"FLOPs: {flops / 1e9} G")  # 以十亿次（Giga）FLOPs为单位输出
    print(f"Parameters: {params / 1e6} M")  # 以百万（Mega）参数数量为单位输出



    


        
        

       

     

   

  