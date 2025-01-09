from numpy import dtype

from cross_attention_swin_1.model import SwinTransformer_2
from cross_attention_swin_1.cross_attention import CrossAttentionBlock1, CrossAttentionBlock
from cross_attention_swin_1.cross_attention import CrossAttentionBlock2
from cross_attention_swin_1.swin_transformer import SwinTransformer
from cross_attention_swin_1.model2 import MobileViT
from cross_attention_swin_1.model_config import get_config
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from cross_attention_swin_1.jiaquan import ImageTransformNet

import sys

class Fusioncls(nn.Module):
    def __init__(self,):
        super(Fusioncls, self).__init__()
        self.model1 = SwinTransformer_2(in_chans=3,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=1)
        self.model2 = MobileViT(get_config("x_small"), num_classes=1)
        self.model3 = SwinTransformer(in_chans=1,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=1)
        self.cross_attn1 = CrossAttentionBlock1(dim=768, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.cross_attn2 = CrossAttentionBlock2(dim=384, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.cross_attn3 = CrossAttentionBlock(dim=1152, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.projs1 = nn.Sequential(
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 384),
        )
        self.projs2 = nn.Sequential(
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 768),
        )
        # self.jiaquan = nn.Sequential()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1152, 1)
        # self.rtt = reshape_to_target(x=input, target_shape=(36, 1536, 1))
        # self.rrr = ImageTransformNet()



    # def forward(self, x1, x2):
    def forward(self, inputs):

        inputs_1 = inputs[:, 0, :, :]
        inputs_1 = inputs_1.unsqueeze(1)
        inputs_2 = inputs[:, 1, :, :]
        inputs_2 = inputs_2.unsqueeze(1)
        inputs_3 = inputs[:, 2, :, :]
        inputs_3 = inputs_3.unsqueeze(1)
        x1 = torch.cat((inputs_1, inputs_2, inputs_3), 1) #[2, 3, 128, 128]
        # print('x1', x1.shape)
        A1 = x1
        A1 = self.pool(F.relu(self.conv1(A1)))  # 大小变为: [batch_size, 16, 64, 64]
        A1 = self.pool(F.relu(self.conv2(A1)))  # 大小变为: [batch_size, 32, 32, 32]
        A1 = self.pool(F.relu(self.conv3(A1)))  # 大小变为: [batch_size, 1, 16, 16]

        # 为了达到 [48, 48, 1]，可以在最后添加一个上采样层
        A1 = F.interpolate(A1, size=(48, 48), mode='bilinear', align_corners=False)  # 大小变为: [batch_size, 1, 48, 48]

        # 去除多余的维度 [batch_size, 1, 48, 48] -> [batch_size, 48, 48]
        A1 = torch.squeeze(A1, dim=1)
        # print('A1', A1.shape)
        # print(A1.shape)
        # print('A1.dtype', A1.dtype)
        # print('dtype(A1)',dtype(A1))
        # A1 = ImageTransformNet()
        # A1 = A1(x1)
        # print('A1', A1.shape)
        inputs_4 = inputs[:, 3, :, :]
        inputs_4 = inputs_4.unsqueeze(1)
        inputs_5 = inputs[:, 3, :, :]
        inputs_5 = inputs_5.unsqueeze(1)
        inputs_6 = inputs[:, 3, :, :]
        inputs_6 = inputs_6.unsqueeze(1)
        x2 = torch.cat((inputs_4, inputs_5, inputs_6), 1)   #[2, 3, 128, 128]
        # print('x2', x2.shape)
        B1 = x2
        B1 = self.pool(F.relu(self.conv1(B1)))  # 大小变为: [batch_size, 16, 64, 64]
        B1 = self.pool(F.relu(self.conv2(B1)))  # 大小变为: [batch_size, 32, 32, 32]
        B1 = self.pool(F.relu(self.conv3(B1)))  # 大小变为: [batch_size, 1, 16, 16]

        # 为了达到 [48, 48, 1]，可以在最后添加一个上采样层
        B1 = F.interpolate(B1, size=(48, 48), mode='bilinear', align_corners=False)  # 大小变为: [batch_size, 1, 48, 48]

        # 去除多余的维度 [batch_size, 1, 48, 48] -> [batch_size, 48, 48]
        B1 = torch.squeeze(B1, dim=1)
        # print('B1', B1.shape)
        # print('B1.dtype', B1.dtype)
        # print('dtype(B1)', dtype(B1))

        # B1 = ImageTransformNet()
        # print('B1', B1.shape)
        x1 = self.model1(x1)        #[36,1,768]
        # print('model1(X1)', x1.shape)
        x2 = self.model2(x2)        #[36,1,384]
        # print('model2(X2)', x2.shape)

        tokens = []
        tokens.append(x1)
        tokens.append(x2)

        x11 = self.projs1(x1)       #[36,1,384]
        x22 = self.projs2(x2)       #[36,1,768]

        cls_proj =[]
        cls_proj.append(x11)
        cls_proj.append(x22)

        fusion1 = torch.cat((tokens[0], cls_proj[1][:, 1:, ...]), dim=1)        #[36, 1, 768]
        # print('fusion1', fusion1.shape)
        fusion1 = self.cross_attn1(fusion1)                         #[36, 1, 768]
        # print('cross_attention(fusion)', fusion1.shape)
        fusion2 = torch.cat((tokens[1], cls_proj[0][:, 1:, ...]), dim=1)        #[36, 1, 384]
        # print('fusion2', fusion2.shape)
        fusion2 = self.cross_attn2(fusion2)                         #[36, 1, 384]
        # print('cross_attention(fusion)', fusion2.shape)

        fusion3 = torch.cat((tokens[0][:, 1:, ...], cls_proj[1]), dim=1)        #[36, 1, 768] [B,L,C]
        fusion3 = self.cross_attn1(fusion3)
        fusion4 = torch.cat((tokens[1][:, 1:, ...], cls_proj[0]), dim=1)        #[36, 1, 384] [B,L,C]
        fusion4 = self.cross_attn2(fusion4)

        A = torch.cat((fusion1, fusion2), dim=2) #[36, 1, 1152]
        # print('A', A.shape)
        A = A.reshape(A.shape[0], 24, 48)
        # print('A.reshape', A.shape)
        attn1 = (A1  @ A.transpose(-2, -1)) #[36,48,24]
        # print('attn1', attn1.shape)
        attn1 = torch.add(A.transpose(1,2), attn1)
        # print('add_attn1', attn1.shape)
        attn2 = (B1 @ A.transpose(-2, -1))  #[36,48,24]
        # print('attn2', attn2.shape)
        attn2 = torch.add(A.transpose(1,2), attn2) #[36,48,24]
        # print('add_attn2', attn2.shape)
        atten3 = (attn1  @ attn2.transpose(-2, -1))
        A1B1 = torch.add(A1, B1)
        atten3A1B1 = (atten3  @ A1B1)
        # print('atten3A1B1', atten3A1B1.shape)
        atten1 =(atten3A1B1  @ attn1.transpose(1,2).transpose(-2, -1))
        # print('XXXXXX', atten1.shape)
        atten2 = (atten3A1B1 @ attn2.transpose(1, 2).transpose(-2, -1))
        # print('YYYYYY', atten1.shape)
        attn1 = attn1.reshape(attn1.shape[0], 1, 1152)
        # print('attn1.reshape', attn1.shape)
        attn2 = attn2.reshape(attn2.shape[0], 1, 1152)
        # print('attn2.reshape', attn2.shape)
        attn1 = self.cross_attn3(attn1)
        # print('cross_attn1', attn1.shape)
        attn2 = self.cross_attn3(attn2)
        # print('cross_attn2', attn2.shape)
        x = torch.cat((attn1, attn2), dim=2)
        # print('x_connect', x.shape)


        x = torch.cat((fusion1, fusion2, fusion3, fusion4), dim=2)          # [36, 1, 2304] [B,L,C]
        x = torch.cat((fusion3, fusion4), dim=2)
        x = self.avgpool(x.transpose(1, 2))
        x = x.transpose(1, 2)                           #[36, 2304, 1] [B,C,L]
        x = self.rtt(x)                              #[36, 1, 1] [B,C,L]
        x = x.reshape_to_target(x)
        target_shape = (36, 1536, 1)
        x = reshape_to_target(x, target_shape)
        x = x.resize(36, 1536, 1)
        x = x.reshape(x.shape[0], 1, 48, 48)
        x = self.model3(x)                                #[1, 768, 1] [B,C,L]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

