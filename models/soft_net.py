from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F



class SOFTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # channel 1
        self.channel_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(5, 5), padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )

        # channel 2
        self.channel_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=5, kernel_size=(5, 5), padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 3
        self.channel_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(5, 5), padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )

        # interpretation
        self.interpretation = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.LazyLinear(out_features=400),
            nn.ReLU(),
            nn.LazyLinear(out_features=1),
        )

    def forward(self, inputs):
        inputs_1 = inputs[:, 0, :, :]
        inputs_1 = inputs_1.unsqueeze(1)
        inputs_2 = inputs[:, 1, :, :]
        inputs_2 = inputs_2.unsqueeze(1)
        inputs_3 = inputs[:, 2, :, :]
        inputs_3 = inputs_3.unsqueeze(1)
        inputs_4 = inputs[:, 3, :, :]
        inputs_4 = inputs_4.unsqueeze(1)
        inputs_5 = inputs[:, 3, :, :]
        inputs_5 = inputs_5.unsqueeze(1)
        inputs_6 = inputs[:, 3, :, :]
        inputs_6 = inputs_6.unsqueeze(1)
        # channel 1
        channel_1 = self.channel_1(inputs_1)
        # print("channel_1",channel_1.shape)
        # channel 2
        channel_2 = self.channel_2(inputs_2)
        # print("channel_2", channel_2.shape)
        # channel 3
        channel_3 = self.channel_3(inputs_3)
        # print("channel_3", channel_3.shape)
        channel_4 = self.channel_1(inputs_4)
        # print("channel_1",channel_1.shape)
        # channel 2
        channel_5 = self.channel_2(inputs_5)
        # # print("channel_2", channel_2.shape)
        # # channel 3
        channel_6 = self.channel_3(inputs_6)
        # # print("channel_3", channel_3.shape)


        # merge
        # merged = torch.cat((channel_1, channel_2, channel_3, channel_4), 1)
        merged = torch.cat((channel_1, channel_2, channel_3,channel_4,channel_5,channel_6), 1)
        # print("merged_1", merged.shape)
        # merged = merged.reshape(36,-1)
        # merged = self.channel_4(merged)
        # print("merged", merged.shape)
        # merged = merged.reshape(16, 14, 14)
        # print("111",merged.shape)
        # interpretation
        outputs = self.interpretation(merged)
        # print(outputs.shape)
        return outputs

# class MFE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.CB = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3, out_channels=3, kernel_size=(3, 3), padding="same"
#             ),
#             nn.BatchNorm2d(3),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=3, out_channels=3, kernel_size=(3, 3), padding="same"
#             ),
#             nn.BatchNorm2d(3),
#             nn.ReLU()
#         )
#         self.CL = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3, out_channels=3, kernel_size=(3, 3), padding="same"
#             ),
#             nn.BatchNorm2d(3),
#             nn.ReLU(),
#         )
#         self.Maxpooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#
#     def forward(self, x):
#         Fm1 = self.CB(x)
#         # Fm2 = self.CB(self.Maxpooling(x))
#         Fm2 = self.CL(Fm1)
#         Fm3 = self.CL(Fm2)
#         # print("Fm1,Fm2,Fm3",Fm1.shape, Fm2.shape, Fm3.shape)
#         return Fm1, Fm2, Fm3
# #
# class SOFTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # channel 1
#         self.channel_1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1, out_channels=3, kernel_size=(5, 5), padding='same'
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
#         )
#
#         # channel 2
#         self.channel_2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1, out_channels=5, kernel_size=(5, 5), padding='same'
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
#         )
#         # channel 3
#         self.channel_3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1, out_channels=8, kernel_size=(5, 5), padding='same'
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
#         )
#         # interpretation
#         self.interpretation = nn.Sequential(
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
#             nn.Flatten(),
#             nn.LazyLinear(out_features=400),
#             nn.ReLU(),
#             nn.LazyLinear(out_features=1),
#         )
#
#     def forward(self, inputs):
#         inputs_1 = inputs[:, 0, :, :]
#         inputs_1 = inputs_1.unsqueeze(1)
#         inputs_2 = inputs[:, 1, :, :]
#         inputs_2 = inputs_2.unsqueeze(1)
#         inputs_3 = inputs[:, 2, :, :]
#         inputs_3 = inputs_3.unsqueeze(1)
#         # channel 1
#         channel_1 = self.channel_1(inputs_1)
#         # print("channel_1",channel_1.shape)
#         # channel 2
#         channel_2 = self.channel_2(inputs_2)
#         # print("channel_2", channel_2.shape)
#         # channel 3
#         channel_3 = self.channel_3(inputs_3)
#         # print("channel_3", channel_3.shape)
#
#
#         # merge
#         merged = torch.cat((channel_1, channel_2, channel_3), 1)
#         # print("merged_1", merged.shape)
#         # merged = merged.reshape(36,-1)
#         # merged = self.channel_4(merged)
#         # print("merged", merged.shape)
#         # merged = merged.reshape(16, 14, 14)
#         # print("111",merged.shape)
#         # interpretation
#         outputs = self.interpretation(merged)
#         # print(outputs.shape)
#         return outputs

# input_shape = [128,128]  # 设置输入大小
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择是否使用GPU
# model = SOFTNet().to(device)  # 实例化网络
# summary(model, (6, input_shape[0], input_shape[1]))