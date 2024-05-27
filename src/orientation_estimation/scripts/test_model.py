#!/usr/bin/env python3
import rospy
import cv2 as cv
#point cloud2和point cloud的区别是
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from read_data import Data
from PIL import Image
import os

#Savitzky-Golay 滤波器
from scipy.signal import savgol_filter


#卡尔曼滤波库
from pykalman import KalmanFilter

#神经网络模型可视化库
from torchviz import make_dot



fixing_range = 15
fixing_step = 5
file_list = ['1', '2', '3', '4', '5']

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_output_size, lstm_hidden_size, lstm_num_layers, output_size):
        super(CNNLSTMModel, self).__init__()
        # 定义CNN部分
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)



        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, cnn_output_size)
        
        # 定义LSTM部分
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.size()
        cnn_features = []
        
        for t in range(seq_length):
            cnn_out = self.pool(F.relu(self.conv1(x[:, t, :, :, :])))
            cnn_out = self.pool(F.relu(self.conv2(cnn_out)))
            cnn_out = cnn_out.view(batch_size, -1)
            cnn_out = F.relu(self.fc1(cnn_out))
            cnn_features.append(cnn_out)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        out = self.fc2(lstm_out[:, -1, :])
        return out

os.chdir(os.path.dirname(__file__))
model = torch.load('model.ckpt')

#可视化模型
x = torch.randn(1, 15, 1, 64, 64)
y = model(x)
g = make_dot(y)
g.view()


def prepare_data(file_name):
    train_data = []
    train_label = []
    data = Data(file_name)
    real = data.real
    esti = data.esti
    img = data.img
    rectified_data = esti[0:15]    

    for start in range(15, len(real) ):
        img_seq = []
        for i in range(fixing_range//fixing_step+1):
            img_seq.append(img[start-fixing_range + i * fixing_step])
        #翻转img_seq
        img_seq = [Image.fromarray(i) for i in img_seq]
        #convert to gray scale
        img_seq = [i.convert('L') for i in img_seq]
        img_seq = [transform(i) for i in img_seq]
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.unsqueeze(0)
        output = model(img_seq).detach().numpy()
        if(start<=30):
            temp = esti[start - fixing_range]+output[0][0]*(esti[start]-esti[start-fixing_range])
        else:
            temp = rectified_data[start - fixing_range]+output[0][0]*(esti[start]-esti[start-fixing_range])
        rectified_data.append(temp)
        #滤波

    rectified_data = savgol_filter(rectified_data, 10, 5) #10 represents the window size , 3 represents the polynomial order

    plt.plot(real, label='real')
    plt.plot(rectified_data, label='rectified')
    # plt.plot(show_data, label='show_data')
    plt.plot(esti, label='esti')
    origin_error = np.array(real) - np.array(esti)
    rectified_error = np.array(real) - np.array(rectified_data)
    plt.plot(origin_error, label='origin_error')
    plt.plot(rectified_error, label='rectified_error')

    plt.legend()
    plt.show()

for file_name in file_list:
    prepare_data(file_name)
