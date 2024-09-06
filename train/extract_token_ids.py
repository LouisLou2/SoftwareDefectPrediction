# 将tokens转化为ids,存储起来
# 注意由于tokens的长度不一样，
# 所以形成的ids也不一样，最终存储下来的是规整化后的ids(添加pad以对齐),可知ids的统一长度就是最长的tokens的长度
# (我也不知道存储规整化后的ids是否更好)
# 存储在dataset_refined/token_ids

import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from const.dataset_const import now_proj_dir, proj_names, proj_versions
from util.list_serial import store_list_of_list, read_list_of_list

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

exceedlevels = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000,
                16000, 17000, 18000, 19000]
exceeds = [0] * len(exceedlevels)

#
# def getTorch2D(token_llist):
#     id_torch1D_list = []
#     for token_list in token_llist:
#         # 构建最终token sequence
#         tokens = [tokenizer.cls_token] + token_list + [tokenizer.eos_token]
#         # 将 token 序列转化为 ID
#         tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
#         id_torch1D_list.append(torch.tensor(tokens_ids))
#     return pad_sequence(id_torch1D_list, batch_first=True, padding_value=tokenizer.pad_token_id)
#
#
dataset_refined_dir = f'{now_proj_dir}/dataset_refined'
ast_token_dir = f'{dataset_refined_dir}/ast_token/'
token_ids_dir = f'{dataset_refined_dir}/token_ids/'
#
# data = pd.DataFrame()
#
# # 读取数据
# ids_torch2D_lis = []
# max_col = 0
# for i in range(len(proj_names)):
#     proj_name = proj_names[i]
#     version_list = proj_versions[i]
#
#     token_file_prefix = f'{ast_token_dir}{proj_name}/{proj_name}-'
#
#     for j in range(len(version_list)):
#         version = version_list[j]
#         token_file = f'{token_file_prefix}{version}.txt'
#         llis = read_list_of_list(token_file)
#         ids_torch2D = getTorch2D(llis)
#         rows, cols = ids_torch2D.shape
#         if cols > max_col:
#             max_col = cols
#         ids_torch2D_lis.append(ids_torch2D)
#
# # max
# print(f'max_col: {max_col}')
#
# # 处理最终的pad填充
# for ids_torch2D in ids_torch2D_lis:
#     rows, cols = ids_torch2D.shape
#     if cols < max_col:
#         pad = torch.full((rows, max_col - cols), tokenizer.pad_token_id)
#         ids_torch2D = torch.cat((ids_torch2D, pad), dim=1)
#
# # 存储
# count = 0
# for i in range(len(proj_names)):
#     proj_name = proj_names[i]
#     version_list = proj_versions[i]
#     token_ids_dir_proj = f'{token_ids_dir}{proj_name}'
#     if not os.path.exists(token_ids_dir_proj):
#         os.makedirs(token_ids_dir_proj)
#
#     for j in range(len(version_list)):
#         version = version_list[j]
#         target_file = f'{token_ids_dir_proj}/{proj_name}-{version}.pt'
#         print(f'storing {target_file}')
#         torch.save(ids_torch2D_lis[count], target_file)
#         count += 1

for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]

    token_file_prefix = f'{ast_token_dir}{proj_name}/{proj_name}-'

    for j in range(len(version_list)):
        version = version_list[j]
        token_file = f'{token_file_prefix}{version}.txt'
        llis = read_list_of_list(token_file)
        for lis in llis:
            for ind in range(len(exceedlevels)):
                if len(lis) >= exceedlevels[ind]:
                    exceeds[ind] += 1
                else:
                    break
# 打印结果
print('exceeds:')
for i in range(len(exceedlevels)):
    print(f'{exceedlevels[i]}: {exceeds[i]}')
