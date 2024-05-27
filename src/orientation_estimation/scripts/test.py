#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# 生成图像数据 (假设每个时间步对应一张图片)
def generate_image_data(seq_length, img_size=(64, 64)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    image_data = []
    for _ in range(seq_length):
        img = Image.new('RGB', img_size, (255, 0, 0))  # 生成红色图片
        img = transform(img)
        image_data.append(img)
    return torch.stack(image_data)

seq_length = 10
image_data = generate_image_data(seq_length)
image_data = image_data.unsqueeze(0)  # 添加batch_size维度，形状为 (1, seq_length, 3, 64, 64)

class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_output_size, lstm_hidden_size, lstm_num_layers, output_size):
        super(CNNLSTMModel, self).__init__()
        # 定义CNN部分
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
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

cnn_output_size = 128
lstm_hidden_size = 50
lstm_num_layers = 1
output_size = 1

model = CNNLSTMModel(cnn_output_size, lstm_hidden_size, lstm_num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

# 生成简单的目标值
target = torch.tensor([0.55])  # 示例目标值

# 转换数据为批处理格式
image_data_batch = image_data.expand(16, -1, -1, -1, -1)  # 扩展为16个样本
target_batch = target.expand(16, -1)  # 扩展为16个目标值

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(image_data_batch)
    loss = criterion(outputs, target_batch)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()

time_now = time.time()
test_output = model(image_data)
print(f'Time taken: {time.time() - time_now:.4f} seconds')

print(f'Test Output: {test_output.item():.4f}')
