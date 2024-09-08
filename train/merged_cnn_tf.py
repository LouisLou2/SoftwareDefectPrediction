# 读取数据
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.src.layers import MaxPooling2D, BatchNormalization
from keras.src.optimizers import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, roc_auc_score, recall_score, \
    f1_score
from const.dataset_const import now_proj_dir, proj_names, proj_versions

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def print_scores(clf_name, y_val, y_val_pred, y_test, y_test_pred):
    print(clf_name)
    print()
    print("-- Validation Set --")
    print("Accuracy: ", accuracy_score(y_val, y_val_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_val, y_val_pred))
    print("Precision: ", precision_score(y_val, y_val_pred))
    print("AUC: ", roc_auc_score(y_val, y_val_pred))
    print("Recall: ", recall_score(y_val, y_val_pred))
    # f1_score
    print("F1: ", f1_score(y_val, y_val_pred))
    print()
    print("-- Test Score --")
    print("Accuracy: ", accuracy_score(y_test, y_test_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_test_pred))
    print("Precision: ", precision_score(y_test, y_test_pred))
    print("AUC: ", roc_auc_score(y_test, y_test_pred))
    print("Recall: ", recall_score(y_test, y_test_pred))
    # f1_score
    print("F1: ", f1_score(y_test, y_test_pred))

    print()
    print()
# # convolutional neural network classifier
# def CNN(x_col, X_train, X_validation, y_train, y_validation):
#     X_train_matrix = X_train.values
#     X_validation_matrix = X_validation.values
#     y_train_matrix = y_train.values
#     y_validation_matrix = y_validation.values
#
#     img_rows = 1
#     img_cols = x_col
#
#     X_train_f = X_train_matrix.reshape(X_train_matrix.shape[0], img_rows, img_cols, 1)
#     X_validation_f = X_validation_matrix.reshape(X_validation_matrix.shape[0], img_rows, img_cols, 1)
#
#     input_shape = (img_rows, img_cols, 1)
#
#
#     model = Sequential()
#     model.add(Conv2D(64, kernel_size=1, activation='relu', input_shape=input_shape))
#     model.add(Conv2D(32, kernel_size=1, activation='relu'))
#     model.add(Conv2D(16, kernel_size=1, activation='relu'))
#     model.add(Conv2D(16, kernel_size=1, activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(loss='binary_crossentropy',
#                     optimizer='adam',
#                     metrics=['accuracy'])
#
#     model.fit(X_train_f, y_train_matrix, epochs=100, batch_size=35, validation_data=(X_validation_f, y_validation_matrix), verbose=0)
#     return model


def CNN(x_col, X_train, X_validation, y_train, y_validation):
    X_train_matrix = X_train.values
    X_validation_matrix = X_validation.values
    y_train_matrix = y_train.values
    y_validation_matrix = y_validation.values

    img_rows = 1
    img_cols = x_col

    X_train_f = X_train_matrix.reshape(X_train_matrix.shape[0], img_rows, img_cols, 1)
    X_validation_f = X_validation_matrix.reshape(X_validation_matrix.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(X_train_f, y_train_matrix, epochs=100, batch_size=32, validation_data=(X_validation_f, y_validation_matrix), verbose=1)
    return model

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
# CNN
clf = CNN(X_train.shape[1], X_train, X_validation, y_train, y_validation)
X_validation_matrix = X_validation.values
X_validation1 = X_validation_matrix.reshape(X_validation_matrix.shape[0], 1, len(X_validation.columns), 1)
y_val_pred = clf.predict(X_validation1) > 0.5
X_test_matrix = X_test.values
X_test1 = X_test_matrix.reshape(X_test_matrix.shape[0], 1, len(X_test.columns), 1)
y_test_pred = clf.predict(X_test1) > 0.5
print_scores("CNN", y_validation, y_val_pred, y_test, y_test_pred)




