"""DeepPhys - 2D Convolutional Attention Network.
DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks
ECCV, 2018
Weixuan Chen, Daniel McDuff
"""

import torch
import torch.nn as nn
import pytorch_wavelets as wavelets
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
import torchvision.models as models


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class Mobilenet_Lstm(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=72):
        """Definition of DeepPhys.
        Args:
          in_channels: the number of input channel. Default: 3
          img_size: height/width of each frame. Default: 36.
        Returns:
          DeepPhys model.
        """
        super(Mobilenet_Lstm, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        self.lstm = nn.LSTM(input_size=1568, hidden_size=3136,
                            num_layers=1, batch_first=True)
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # Modify the model to remove the classification head
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        # Input size is 1280 for MobileNetV2 output
        self.final_linear = nn.Linear(1280, 3136)

        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):

        raw_input = inputs[:, 3:, :, :]
        xfm = DWTForward(J=3, mode='zero', wave='db3').to("cuda")
        # ifm = DWTInverse(mode='zero', wave='db3')
        Yl, Yh = xfm(raw_input)  # Yl -> LL band
        # Yh[0]: HL subband - Contains horizontal details.
        # Yh[1]: LH subband - Contains vertical details.
        # Yh[2]: HH subband - Contains diagonal details.

        HL_subband = Yh[0]  # HL subband (horizontal details)
        LH_subband = Yh[1]  # LH subband (vertical details)

        # Compute vertical projection for HL and LH subbands
        # Sum along the vertical axis (axis=2)
        vertical_projection_HL = torch.sum(HL_subband, dim=2)
        # Sum along the vertical axis (axis=2)
        vertical_projection_LH = torch.sum(LH_subband, dim=2)

        # Sum along the horizontal axis (axis=3)
        horizontal_projection_HL = torch.sum(HL_subband, dim=3)
        # Sum along the horizontal axis (axis=3)
        horizontal_projection_LH = torch.sum(LH_subband, dim=3)

        feature_vector_HL = torch.cat(
            (vertical_projection_HL, horizontal_projection_HL), dim=2)
        # torch.Size([360, 3, 24, 21])
        feature_vector_LH = torch.cat(
            (vertical_projection_LH, horizontal_projection_LH), dim=2)

        feature_vector_HL_resized = F.interpolate(feature_vector_HL, size=(
            24, 21), mode='nearest')  # torch.Size([360, 3, 24, 21])
        final_feature_vector = torch.cat(
            (feature_vector_HL_resized, feature_vector_LH), dim=0)  # torch.Size([720, 3, 24, 21])
        # print("final",final_feature_vector.shape)

        resnet_features = self.mobilenet(final_feature_vector)
        resnet_features = resnet_features.view(
            resnet_features.size(0), -1)  # Shape: [720, 2048]

        # Apply final linear layer
        resnet_output = self.final_linear(resnet_features)  # Shape: [720, 1]
        # print("resnet_output", resnet_output.shape)

        resnet_output = resnet_output.view(1, 1, 720, 1568)

        # Apply average pooling with kernel size 2 and stride 2 along the first dimension
        resnet_output = torch.nn.functional.avg_pool2d(
            resnet_output, kernel_size=(2, 1), stride=(2, 1))

        # Reshape downsampled tensor to have a size of [360, 1]
        resnet_output = resnet_output.view(360, 1568)
        d10, _ = self.lstm(resnet_output)

        d10 = torch.tanh(self.final_dense_1(d10))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)
        # print("out", out.shape)
        downsampled_tensor = F.avg_pool1d(
            out.view(1, 1, 360), kernel_size=2).view(180, 1)
        # print("downsampled_tensor", downsampled_tensor.shape)
        return downsampled_tensor


if __name__ == "__main__":
    data = torch.rand(180, 6, 72, 72)
    model = DeepPhys()

    # Move the input data to GPU
    data = data
    print(data.shape)
    res = model(data)

    print(res)
