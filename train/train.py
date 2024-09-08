import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# 自定义SupConLoss损失函数
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # 去掉自身对自身的相似度
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out对比学习的无效部分
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 计算对比损失
        loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = loss.mean()
        return loss


# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data, labels):
        super(MyDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 特征提取器（可以根据需求修改为CNN或其他复杂网络）
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 2 * 2, feature_dim)  # 假设输出的特征图大小是2x2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 每次池化后尺寸减半
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 再次池化，假设输入8x8 -> 输出2x2特征图
        x = x.view(x.size(0), -1)  # 展平操作，转换为(batch_size, 特征数量)
        x = self.fc(x)  # 输入到全连接层
        return x


# 分类器头
class Classifier(nn.Module):
    def __init__(self, feature_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feature_dim, 2)  # 二分类（0/1）

    def forward(self, x):
        return self.fc(x)


# 构建对比学习+分类器模型
class SupConModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(SupConModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return features, logits


# 模型训练
def train_supcon_model(data_loader, model, contrastive_loss_fn, optimizer, temperature=0.07):
    model.train()
    total_loss = 0
    for data, labels in data_loader:
        data, labels = data, labels
        features, _ = model(data)
        loss = contrastive_loss_fn(features, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# 分类器训练
def train_classifier(data_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, labels in data_loader:
        data, labels = data, labels
        _, logits = model(data)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy


# 测试模型
def test_model(data_loader, model, criterion):
    model.eval()  # 将模型设为评估模式
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, labels in data_loader:
            data, labels = data, labels
            _, logits = model(data)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy


# 假设我们有n个样本，每个样本是mxk的张量
n = 1000
m, k = 8, 8  # 假设每个样本是8x8的张量
num_classes = 2

# 随机生成标签
labels = torch.randint(0, num_classes, (n,))

# 根据标签生成不同分布的数据
data = torch.zeros(n, 1, m, k)
for i in range(n):
    if labels[i] == 0:
        # 标签为0的数据，均值为2，标准差为1的正态分布
        data[i] = torch.randn(1, m, k)
    else:
        data[i] = torch.randn(1, m, k)+1

# 构建数据集和数据加载器
dataset = MyDataset(data, labels)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 将数据集分为训练集和测试集
train_data, test_data = torch.utils.data.random_split(dataset, [800, 200])

# 构建训练集和测试集的数据加载器
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 模型和损失函数
feature_extractor = FeatureExtractor(input_channels=1, feature_dim=128)
classifier = Classifier(feature_dim=128)
model = SupConModel(feature_extractor, classifier)

contrastive_loss_fn = SupConLoss()
classifier_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# 第1阶段：使用SupConLoss进行对比学习
epochs = 100
for epoch in range(epochs):
    loss = train_supcon_model(train_loader, model, contrastive_loss_fn, optimizer)
    print(f"Epoch [{epoch + 1}/{epochs}], Contrastive Loss: {loss:.4f}")

# 第2阶段：使用交叉熵损失训练分类器
epochs = 100
for epoch in range(epochs):
    loss, accuracy = train_classifier(train_loader, model, classifier_loss_fn, optimizer)
    print(f"Epoch [{epoch + 1}/{epochs}], Classification Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")

# 测试阶段
test_loss, test_accuracy = test_model(test_loader, model, classifier_loss_fn)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")