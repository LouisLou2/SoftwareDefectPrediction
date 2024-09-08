# 读取数据
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, roc_auc_score, recall_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


now_proj_dir = '/root/sdp'

proj_names = ['ant', 'camel', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces']

proj_versions = [
    ['1.3', '1.4', '1.5', '1.6', '1.7'],
    ['1.0', '1.2', '1.4', '1.6'],
    ['3.2', '4.0', '4.1', '4.2', '4.3'],
    ['1.0', '1.1', '1.2'],
    ['2.0', '2.2', '2.4'],
    ['1.5', '2.0', '2.5', '3.0'],
    ['1.0', '1.1', '1.2'],
    ['1.4', '1.5', '1.6'],
    ['2.4', '2.5', '2.6', '2.7'],
    ['1.1', '1.2', '1.3', '1.4']
]

def print_scores(clf_name, y_val, y_val_pred, y_test, y_test_pred):
    print(clf_name)
    print()
    print("-- Validation Set --")
    print("Accuracy: ", accuracy_score(y_val, y_val_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_val, y_val_pred))
    print("Precision: ", precision_score(y_val, y_val_pred))
    print("AUC: ", roc_auc_score(y_val, y_val_pred))
    print("Recall: ", recall_score(y_val, y_val_pred))
    print()
    print("-- Test Score --")
    print("Accuracy: ", accuracy_score(y_test, y_test_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_test_pred))
    print("Precision: ", precision_score(y_test, y_test_pred))
    print("AUC: ", roc_auc_score(y_test, y_test_pred))
    print("Recall: ", recall_score(y_test, y_test_pred))

    print()
    print()

# random forest classifer
def random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1234)
    clf.fit(X_train, y_train)
    return clf



data_refined_dir = f'{now_proj_dir}/dataset_refined'
token_embedding_dir = f'{data_refined_dir}/token_embedding'
labeled_data_dir = f'{data_refined_dir}/labeled_and_code_except_empty'

emb_vec_tmp_list=[]
hf_vec_tmp_list=[]

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

        hf = pd.read_parquet(label_file)
        hf_vec=hf.drop(columns=['name','src'])
        hf_vec_tmp_list.append(hf_vec)
        # read token embedding
        emb_matrix = torch.load(token_emb_file)
        # 放进x_list
        emb_vec_tmp_list.append(pd.DataFrame(emb_matrix.detach().numpy()))

# total hf_vec including bug
hf_vec = pd.concat(hf_vec_tmp_list,ignore_index=True)
# get labels
labels = hf_vec[['bug']]
labels.reset_index(drop=True)
# get handcut features
hf_vec = hf_vec.drop(columns=['bug'])
# total emb_vec
emb_vec = pd.concat(emb_vec_tmp_list,ignore_index=True)

# 将hf_vec和emb_vec拼接
# x=pd.concat([hf_vec,emb_vec],axis=1)
x = hf_vec
x.columns = x.columns.astype(str)
# check the shape of emb_vec and hf_vec
print(emb_vec.shape)
print(hf_vec.shape)
print(labels.shape)
print(x.shape)

del emb_vec_tmp_list
del hf_vec_tmp_list
del emb_vec
del hf_vec
# 划分
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.3, random_state=1234)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=1234)
#Random Forest classifier
clf = random_forest(X_train, y_train)
y_val_pred = clf.predict(X_validation)
y_test_pred = clf.predict(X_test)
print_scores("RANDOM FOREST", y_validation, y_val_pred, y_test, y_test_pred)