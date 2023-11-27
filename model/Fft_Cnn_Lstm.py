"""DeepPhys - 2D Convolutional Attention Network.
DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks
ECCV, 2018
Weixuan Chen, Daniel McDuff
"""

import numpy as np
import cv2
import torch
import torch.nn as nn




def apply_bandpass_filter(image, fmin, fmax):

    # Get the image shape
    image_shape = image.shape

    # Perform a 2D Fast Fourier Transform (FFT) of the image
    fft_image = torch.fft.fft(image, dim=0)

    # Create a frequency mask
    frequency_mask = torch.zeros(image_shape, dtype = torch.complex64)
    frequency_mask[(fmin <= torch.abs(fft_image)) & (torch.abs(fft_image) <= fmax)] = 1

    # Apply the frequency mask to the FFT of the image
    filtered_fft_image = fft_image * frequency_mask

    # Perform an inverse 2D FFT of the filtered FFT image
    filtered_image = torch.fft.ifft(filtered_fft_image, dim=0)

    return filtered_image.real.double()

def get_feature_array(raw_input):
    collection = []
    for frame in raw_input:
        frame = np.transpose(frame, (1, 2, 0))
        # print(frame.shape)
        G = frame
        for i in range(4):
            G = cv2.pyrDown(G)
        G = G.reshape(-1, 1, 3)
        # print(G.shape) #(25, 1, 3)
        collection.append(G)

    collection = np.array(collection)
    # print(collection.shape)

    S = np.empty((0, 25, 30, 3))
    ## grab batches of 30
    for i in range(0, len(collection), 30):
        batch = collection[i:min([i+30, len(collection)]), :, :, :]
        # print(batch.shape)   #(30, 25, 1, 3)
        # break
        M = np.concatenate(batch, axis=1)
        # print(M.shape)  #(25, 30, 3)

        M = torch.from_numpy(M)
        #iterate over each channel
        for c_i in range(3):
            M[:, :, c_i] = apply_bandpass_filter(M[:, :, c_i], 0.75, 4.0)
        S = np.concatenate((S, M.reshape(1, M.shape[0], M.shape[1], M.shape[2])))
    S = np.array(S)
    argumented_frames = np.empty((0, 25, 30, 3)) 
    for i in range(S.shape[0]):
        # print(S[i:i+1,:,:,:].shape)
        rep = np.repeat(S[i:i+1,:,:,:], 30, axis=0)
        # print(rep.shape)
        # argumented_frames.append(rep)
        argumented_frames = np.concatenate((argumented_frames, rep))

    argumented_frames = np.transpose(argumented_frames, (0, 3, 1, 2))
    return argumented_frames #(360, 3, 25, 30)


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
    
class Fft_Cnn_Lstm(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.2,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=72):
        """Definition of DeepPhys.
        Args:
          in_channels: the number of input channel. Default: 3
          img_size: height/width of each frame. Default: 36.
        Returns:
          DeepPhys model.
        """
        super(Fft_Cnn_Lstm, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        # self.lstm = nn.LSTM(input_size=16384, hidden_size=16384, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=16384, hidden_size=3136, num_layers=1, batch_first=True)

        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)


        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
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

        # # Dense layers original
        # if img_size == 36:
        #     self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        # elif img_size == 72:
        #     self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        # elif img_size == 96:
        #     self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        # else:
        #     raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):

        diff_input = inputs[:, :3, :, :]  
        raw_input = inputs[:, 3:, :, :]

        # n, c, h, w = raw_input.shape

        argumented_feature = np.array(raw_input.cpu())
        argumented_feature = torch.tensor(get_feature_array(argumented_feature), dtype=torch.float).to("cuda")
        # print("argumented_feature", argumented_feature.shape)

        pad_height = (72 - 25) // 2  # 23 padding rows (11 on top, 12 on bottom)
        pad_width = (72 - 30) // 2 
        argumented_feature = torch.nn.functional.pad(argumented_feature, (pad_width, pad_width, pad_height+1, pad_height))
        # print("argumented_feature", argumented_feature.shape)

        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(argumented_feature))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        # print("gated2", gated2.shape)

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)

        
        # print("d8", d8.shape)

        d9 = d8.view(d8.size(0), -1)
        # print("d9", d9.shape)

        d10, _ = self.lstm(d9)
        # print("d10", d10.shape)

        d10 = torch.tanh(self.final_dense_1(d10))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out


if __name__=="__main__":
    data = torch.rand(360, 6, 72, 72).to('cuda')
    model = DeepPhys().to('cuda')
    # print(data.shape)
    res = model(data)
    print(res.shape)