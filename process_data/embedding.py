import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from const.dataset_const import now_proj_dir, proj_names, proj_versions

data_refined_dir = f'{now_proj_dir}/dataset_refined'
labeled_code_dir = f'{data_refined_dir}/labeled_and_code_except_empty'
token_ids_dir = f'{data_refined_dir}/token_ids'

# 加载 CodeBERT 的 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# 将数据读取
# 将synapse, velocity, xalan,xerces的最新一个版本作为测试集


for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]
    token_ids_prefix = f'{token_ids_dir}/{proj_name}/{proj_name}-'
    for j in range(len(version_list)):
        # 开始在项目+版本级别上处理
        version = version_list[j]
        token_ids_file = f'{token_ids_prefix}{version}.pt'
        data = torch.load(token_ids_file)

        emb = model(data[0:1])[0]
        tmp1 = emb.permute(0, 2, 1)
        tmp2 = F.adaptive_avg_pool1d(tmp1, 1)
        pooled_output = F.adaptive_avg_pool1d(emb.permute(0, 2, 1), 1).squeeze(2)
        print(emb)
