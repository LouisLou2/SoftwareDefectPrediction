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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# 将 PyTorch 张量转换为 NumPy 数组
x_np = x.detach().numpy()
y_np = y.detach().numpy().ravel()  # 转换为一维数组

# 数据划分：80% 训练，20% 测试
x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.2, random_state=42)

# 实例化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(x_train, y_train)

# 预测
y_pred = clf.predict(x_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 打印分类报告
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
