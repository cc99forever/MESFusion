import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个函数生成负无穷大的矩阵，用于注意力机制中遮罩操作
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

# 定义一个交叉注意力模块
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels# 输入通道数
        self.channels = in_channels // 8# 缩减的通道数，为输入通道数的1/8
        # 定义三个1x1卷积层用于生成query、key和value
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)# 定义一个softmax层，用于计算注意力权重
        self.INF = INF # 引用之前定义的INF函数
        self.gamma = nn.Parameter(torch.zeros(1)) # 定义一个学习参数gamma，用于调节注意力的影响

    def forward(self, x):
        b, _, h, w = x.size()

        # 生成query
        query = self.ConvQuery(x)

        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)

        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # 生成key
        key = self.ConvKey(x)

        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)

        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # 生成value
        value = self.ConvValue(x)

        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)

        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # 计算水平和垂直方向的注意力分数，并应用负无穷大遮罩
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)

        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)

        # 合并水平和垂直方向的注意力分数，并通过softmax归一化
        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))
        # 分离水平和垂直方向的注意力，并应用到value上
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        # 根据注意力分数加权value，并将水平和垂直方向的结果相加
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

if __name__ == "__main__":
    model = CrissCrossAttention(32)# 实例化模型，假设输入通道数为512
    x = torch.randn(1,32, 14, 14)# 生成一个随机的输入张量，形状为[2, 512, 28, 28]
    model.cuda() # 将模型转移到GPU上
    out = model(x.cuda())
    print(out.shape)



# E:\Anaconda\envs\softnetclone\python.exe E:\7Code2\SoftNet-SpotME-main22\models\soft_net.py
# E:\Anaconda\envs\softnetclone\lib\site-packages\torch\nn\modules\lazy.py:178: UserWarning: Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment.
#   warnings.warn('Lazy modules are a new feature under heavy development '
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1            [-1, 3, 42, 42]              78
#               ReLU-2            [-1, 3, 42, 42]               0
#          MaxPool2d-3            [-1, 3, 14, 14]               0
#             Conv2d-4            [-1, 5, 42, 42]             130
#               ReLU-5            [-1, 5, 42, 42]               0
#          MaxPool2d-6            [-1, 5, 14, 14]               0
#             Conv2d-7            [-1, 8, 42, 42]             208
#               ReLU-8            [-1, 8, 42, 42]               0
#          MaxPool2d-9            [-1, 8, 14, 14]               0
#         MaxPool2d-10             [-1, 16, 7, 7]               0
#           Flatten-11                  [-1, 784]               0
#            Linear-12                  [-1, 400]         314,000
#              ReLU-13                  [-1, 400]               0
#            Linear-14                    [-1, 1]             401
# ================================================================
# Total params: 314,817
# Trainable params: 314,817
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.02
# Forward/backward pass size (MB): 0.47
# Params size (MB): 1.20
# Estimated Total Size (MB): 1.69
# ---------------------------------------