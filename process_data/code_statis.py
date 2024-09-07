# 统计每个项目每个版本的ast_token文件中，token数量超过一定阈值的数量
import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from const.dataset_const import now_proj_dir, proj_names, proj_versions
from util.list_serial import store_list_of_list, read_list_of_list

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

exceedlevels = [200, 250, 300, 350, 400, 450, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000,
                16000, 17000, 18000, 19000]

exceeds = [0] * len(exceedlevels)

dataset_refined_dir = f'{now_proj_dir}/dataset_refined'
ast_token_dir = f'{dataset_refined_dir}/ast_token/'
token_ids_dir = f'{dataset_refined_dir}/token_ids/'

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