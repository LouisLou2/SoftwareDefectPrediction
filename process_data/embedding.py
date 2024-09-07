import gc

import os
import time

print(os.getcwd())

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

now_proj_dir = 'D:/SourceCode/sdp'

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

data_refined_dir = f'{now_proj_dir}/dataset_refined'
labeled_code_dir = f'{data_refined_dir}/labeled_and_code_except_empty'
token_ids_dir = f'{data_refined_dir}/token_ids'
token_embedding_dir = f'{data_refined_dir}/token_embedding'

# 加载 CodeBERT 的 tokenizer 和模型
# tokenizer = AutoTokenizer.from_pretrained("D:/Tools/codebert")
model = AutoModel.from_pretrained("D:/Tools/codebert")


# 将数据读取
# 将synapse, velocity, xalan,xerces的最新一个版本作为测试集

def get_emb_matrix(data):
    emb_vector_list = []
    for ind in range(data.shape[0]):
        emb = model(data[ind:ind + 1])[0]
        pooled_output = F.adaptive_avg_pool1d(emb.permute(0, 2, 1), 1).squeeze(2)
        emb_vector_list.append(pooled_output)
    return torch.cat(emb_vector_list, dim=0), len(emb_vector_list)


count = 0
for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]
    token_ids_prefix = f'{token_ids_dir}/{proj_name}/{proj_name}-'
    token_emb_dir_proj = f'{token_embedding_dir}/{proj_name}'
    if not os.path.exists(token_emb_dir_proj):
        os.makedirs(token_emb_dir_proj)
    for j in range(len(version_list)):
        # 开始在项目+版本级别上处理
        version = version_list[j]
        token_ids_file = f'{token_ids_prefix}{version}.pt'
        token_emb_file = f'{token_emb_dir_proj}/{proj_name}-{version}.pt'
        data = torch.load(token_ids_file)
        print(f'begin: {proj_name}-{version}: {data.shape}')
        emb_matrix, count = get_emb_matrix(data)
        print(f'{proj_name}-{version}: {emb_matrix.shape}')
        torch.save(emb_matrix, token_emb_file)
