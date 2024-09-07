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

# 定义文件夹路径
token_ids_dir = 'D:/SourceCode/sdp/dataset_refined/token_ids'  # 替换为实际路径
split_token_ids_dir = '/dataset_refined/token_ids_small'  # 新的拆分后文件夹路径

# 创建拆分后的文件夹结构
if not os.path.exists(split_token_ids_dir):
    os.makedirs(split_token_ids_dir)

for proj_name in proj_names:
    proj_dir = os.path.join(split_token_ids_dir, proj_name)
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

# 最大拆分大小
split_size = 50

# 拆分并保存文件
for i, proj_name in enumerate(proj_names):
    version_list = proj_versions[i]

    for version in version_list:
        # 拼接文件路径
        pt_file = os.path.join(token_ids_dir, proj_name, f'{proj_name}-{version}.pt')

        # 检查文件是否存在
        if os.path.exists(pt_file):
            # 加载.pt文件
            data = torch.load(pt_file)
            total_samples = data.size(0)

            # 按50个为一组拆分数据
            for idx in range(0, total_samples, split_size):
                split_data = data[idx:idx + split_size]
                split_file = os.path.join(split_token_ids_dir, proj_name,
                                          f'{proj_name}-{version}-{idx // split_size + 1}.pt')

                # 保存拆分后的文件
                torch.save(split_data, split_file)
                print(f'Saved {split_file}')
        else:
            print(f'{pt_file} does not exist')