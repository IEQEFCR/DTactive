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


fixing_range =15
fixing_step = 5
file_list = ['1', '2', '3', '4', '5']

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

def prepare_data(file_name):
    train_data = []
    train_label = []
    data = Data(file_name)
    real = data.real
    esti = data.esti
    img = data.img
    for start in range(0, len(real) - fixing_range, 3):
        img_seq = []
        for i in range(fixing_range//fixing_step+1):
            img_seq.append(img[start + i * fixing_step])
             
        img_seq = [Image.fromarray(i) for i in img_seq]
        #convert to gray scale
        img_seq = [i.convert('L') for i in img_seq]
        img_seq = [transform(i) for i in img_seq]
        single_img_seq = torch.stack(img_seq)
        single_label =(real[start + fixing_range] - real[start]) / (esti[start + fixing_range] - esti[start])
        single_label = torch.tensor(single_label, dtype=torch.float32)
        train_data.append(single_img_seq)
        train_label.append(single_label)
    print(len(train_data))
    return train_data, train_label

train_data = []
train_label = []
for file_name in file_list:
    data, label = prepare_data(file_name)
    train_data+=data
    train_label+=label

train_data = torch.stack(train_data)
train_label = torch.stack(train_label)

# print(train_data.size())
# print(train_label.size())

# 定义LSTM模型
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

cnn_output_size = 20
lstm_hidden_size = 32
lstm_num_layers = 1
output_size = 1

model = CNNLSTMModel(cnn_output_size, lstm_hidden_size, lstm_num_layers, output_size)

#change work dir to the path of the script
os.chdir(os.path.dirname(__file__))

# model = torch.load('model.ckpt')
# model = 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100


batch_size = 20
#将数据集分成batch_size大小的数据集
train_data = train_data.view(-1, fixing_range//fixing_step+1, 1, 64, 64)
train_label = train_label.view(-1, 1)

train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

loss_list = []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# 保存模型
torch.save(model, 'model.ckpt')
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()