import torch
import torch.nn as nn
import torch.nn.functional as F
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # 输入大小: [batch_size, 3, 128, 128]
        x = self.pool(F.relu(self.conv1(x)))  # 大小变为: [batch_size, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # 大小变为: [batch_size, 32, 32, 32]
        x = self.pool(F.relu(self.conv3(x)))  # 大小变为: [batch_size, 1, 16, 16]

        # 为了达到 [48, 48, 1]，可以在最后添加一个上采样层
        x = F.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)  # 大小变为: [batch_size, 1, 48, 48]

        # 去除多余的维度 [batch_size, 1, 48, 48] -> [batch_size, 48, 48]
        x = torch.squeeze(x, dim=1)

        return x

