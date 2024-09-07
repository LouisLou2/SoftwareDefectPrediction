import os
import torch
import torch.nn.functional as F
from transformers import AutoModel
import sys

# 获取命令行参数
args = sys.argv
if len(args) != 4:
    print("Usage: python emb_small.py proj_ind proj_ver spilit_ind")
    sys.exit()

proj_name = args[1]
version = args[2]
spilit_ind = int(args[3])

# 获取当前工作目录
print(os.getcwd())

# 项目名称及版本列表
now_proj_dir = '/root/sdp'


codebert_dir = '/root/codebert'

data_refined_dir = f'{now_proj_dir}/dataset_refined'
token_ids_dir = f'{data_refined_dir}/token_ids_small'
token_embedding_dir = f'{data_refined_dir}/token_embedding_small'

# 判断是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 加载 CodeBERT 的 tokenizer 和模型，并将模型移动到 GPU 上
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained(codebert_dir).to(device)


def get_emb_matrix(data):
    emb_vector_list = []
    for ind in range(data.shape[0]):
        emb = model(data[ind:ind + 1])[0]
        pooled_output = F.adaptive_avg_pool1d(emb.permute(0, 2, 1), 1).squeeze(2)
        emb_vector_list.append(pooled_output)
    return torch.cat(emb_vector_list, dim=0).to('cpu'), len(emb_vector_list)


token_ids_file = f'{token_ids_dir}/{proj_name}/{proj_name}-{version}-{spilit_ind}.pt'
token_emb_dir_proj = f'{token_embedding_dir}/{proj_name}'
if not os.path.exists(token_emb_dir_proj):
    os.makedirs(token_emb_dir_proj)
token_emb_file = f'{token_emb_dir_proj}/{proj_name}-{version}-{spilit_ind}.pt'

# 将数据读取
count = 0
# 将 token_ids 数据加载到 GPU 上
data = torch.load(token_ids_file).to(device)
print(f'begin resolving for {token_emb_file}')
emb_matrix, count = get_emb_matrix(data)
print(f'{token_emb_file}: {emb_matrix.shape}')
torch.save(emb_matrix, token_emb_file)

# split_res = [2, 3, 5, 6, 12, 4, 7, 9, 10, 4, 5, 5, 5, 7, 3, 2, 4, 3, 4, 5, 4, 5, 6, 7, 3, 4, 5, 4, 4, 4, 10, 12, 13, 13, 2, 8, 8, 7]