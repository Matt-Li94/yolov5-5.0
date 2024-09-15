import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 数据准备
data = np.array([1.2, 2.4, 4.8, 6.4, 7.6, 10.6, 14.71, 25.4, 22, 59], dtype=np.float32)

# 定义LSTNet模型
class LSTNet(nn.Module):
    def __init__(self, input_size, hid_size, skip_size, output_size, cnn_k_size, highway_size):
        super(LSTNet, self).__init__()

        self.cnn = nn.Conv1d(1, hid_size, kernel_size=cnn_k_size)
        self.highway = nn.Linear(hid_size, highway_size)
        self.fc = nn.Linear(hid_size, output_size)

        self.rnn = nn.GRU(input_size, hid_size, batch_first=True)
        self.skip_rnn = nn.GRU(hid_size, skip_size, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加channel维度

        # CNN层
        x = self.cnn(x)
        x = torch.relu(x)

        # RNN层
        rnn_out, _ = self.rnn(x)

        # Skip-RNN层
        skip_out, _ = self.skip_rnn(rnn_out)

        # 求和
        combined_out = rnn_out + skip_out

        # 全连接层
        x = torch.mean(combined_out, dim=1)  # 求平均值
        x = self.fc(x)

        return x

# 参数设置
input_size = 1
hid_size = 64
skip_size = 10
output_size = 1
cnn_k_size = 6
highway_size = 16
lr = 0.001
epochs = 1000

# 创建模型、损失函数和优化器
model = LSTNet(input_size, hid_size, skip_size, output_size, cnn_k_size, highway_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 将数据转换为PyTorch张量
x_data = torch.from_numpy(data[:-1]).view(-1, 1,1)
y_data = torch.from_numpy(data[1:]).view(-1, 1)

# 训练模型
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_data)
    loss = criterion(output, y_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_data = torch.from_numpy(data[:-1]).view(-1, 1, 1)
    predictions = model(test_data).numpy()

# 绘制结果
plt.plot(data[:-1], label='Actual')
plt.plot(predictions, label='Predicted', linestyle='dashed')
plt.legend()
plt.show()