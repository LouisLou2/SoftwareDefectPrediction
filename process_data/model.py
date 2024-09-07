import pandas as pd

from const.dataset_const import proj_names, proj_versions, now_proj_dir
from process_data.preprocess import prep, ast_parse
from process_data.code_tokenize import get_embeddings

labeled_code_dir = f'{now_proj_dir}/dataset/PROMISE/labeled_src_code/'

# 读取数据
# 使用每个项目的最新版本作为测试集，其余版本作为训练集
train_data = pd.DataFrame()
test_data = pd.DataFrame()

for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]
    file_prefix = f'{labeled_code_dir}{proj_name}/{proj_name}-'
    for j in range(len(version_list)):
        version = version_list[j]
        data = pd.read_excel(f'{file_prefix}{version}.xlsx')
        data = prep(data)
        if j == len(version_list) - 1:
            test_data = pd.concat([test_data, data], ignore_index=True)
        else:
            train_data = pd.concat([train_data, data], ignore_index=True)

# now the train_data and test_data are ready.

train_data_tokenlist_x = train_data['src'].apply(ast_parse)
train_y = train_data['buggy']
test_data_tokenlist_x = test_data['src'].apply(ast_parse)
test_y = test_data['buggy']

it = get_embeddings(train_data_tokenlist_x)
print(it)
