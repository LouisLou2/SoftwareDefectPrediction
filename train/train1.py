# 读取数据
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from const.dataset_const import now_proj_dir, proj_names, proj_versions

data_refined_dir = f'{now_proj_dir}/dataset_refined'
token_embedding_dir = f'{data_refined_dir}/token_embedding'
labeled_data_dir = f'{data_refined_dir}/labeled_and_code_except_empty'

x_list = []

# (0,1)矩阵
labels = np.empty((0, 1), int)

for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]
    token_emb_prefix = f'{token_embedding_dir}/{proj_name}/{proj_name}-'
    label_file_prefix = f'{labeled_data_dir}/{proj_name}/{proj_name}-'
    for j in range(len(version_list)):
        # 开始在项目+版本级别上处理
        version = version_list[j]
        token_emb_file = f'{token_emb_prefix}{version}.pt'
        label_file = f'{label_file_prefix}{version}.parquet'

        # read label
        label = pd.read_parquet(label_file, columns=['bug'])
        # 拼接到labels
        labels = np.vstack([labels, label.values])
        # read token embedding
        emb_matrix = torch.load(token_emb_file)
        # 放进x_list
        x_list.append(emb_matrix)

# 将x_list和labels转换为tensor
x = torch.cat(x_list, dim=0)
y = torch.tensor(labels)

# 输出大小
print(x.shape)
print(y.shape)

# 数据划分：80% 训练，20% 测试
# 将y的张量转换为一维，因为分类标签通常不需要是二维的
# y = y.view(-1)
y= y.float()
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 调整x的形状以匹配CNN的期望输入
x_train = x_train.unsqueeze(1)  # 添加通道维度
x_test = x_test.unsqueeze(1)

# 创建数据加载器
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc = nn.Linear(32 * 192, 1)  # 32*192是经过两次max pooling后的特征维度

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # 展平特征向量
        out = self.fc(out)
        return out


# 实例化模型
model = CNN()

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward(retain_graph=False)
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in test_loader:
        target = target.float().view(-1, 1)
        outputs = model(data)
        predicted = torch.sigmoid(outputs).round()
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct / total}%')
# 保存模型的参数到文件
torch.save(model.state_dict(), 'cnn_model_weights.pth')
print("cnn_model_weights.pth")
