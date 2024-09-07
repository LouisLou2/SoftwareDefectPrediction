import torch
import os

# 定义项目名称和版本列表
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

# 拆分文件数量
split_res = [2, 3, 5, 6, 12, 4, 7, 9, 10, 4, 5, 5, 5, 7, 3, 2, 4, 3, 4, 5, 4, 5, 6, 7, 3, 4, 5, 4, 4, 4, 10, 12, 13, 13, 2, 8, 8, 7]

# 文件夹路径
split_token_emb_dir = '/root/sdp/dataset_refined/token_embedding_small'  # 拆分文件夹路径
merged_token_emb_dir = '/root/sdp/dataset_refined/token_embedding'        # 合并后的文件夹路径

# 创建合并后的文件夹结构
if not os.path.exists(merged_token_emb_dir):
    os.makedirs(merged_token_emb_dir)

# 合并文件
count = 0
index = 0
for i, proj_name in enumerate(proj_names):
    proj_dir = os.path.join(merged_token_emb_dir, proj_name)
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

    for j, version in enumerate(proj_versions[i]):
        num_splits = split_res[index]
        index += 1

        # 合并数据
        merged_data = []
        for k in range(1, num_splits + 1):
            split_file = os.path.join(split_token_emb_dir, proj_name, f'{proj_name}-{version}-{k}.pt')
            if os.path.exists(split_file):
                data = torch.load(split_file)
                merged_data.append(data)
            else:
                print(f'{split_file} does not exist')

        if merged_data:
            # 合并所有数据
            merged_data = torch.cat(merged_data, dim=0)
            count += merged_data.shape[0]
            # 保存合并后的数据
            merged_file = os.path.join(merged_token_emb_dir, proj_name, f'{proj_name}-{version}.pt')
            torch.save(merged_data, merged_file)
            print(f'Saved merged file: {merged_file}')
        else:
            print(f'No data found to merge for {proj_name}-{version}')

print(f'Total number of data: {count}')