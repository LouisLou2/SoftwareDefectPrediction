import os

import pandas as pd

from const.dataset_const import now_proj_dir, proj_names, proj_versions
from train.preprocess import prep

dataset_refined_dir = f'{now_proj_dir}/dataset_refined'

# 将原来的labeled data和source code整合到一起,但是如果该行的code是空串，则整个行舍弃
# 将buggy的数量转化为是否有bug
# 将数据整合到一起还有另一个目的，就是将其以parquet格式存储，可以独立读取列，之后的处理会更加方便
labeled_and_code_except_empty_dir = f'{dataset_refined_dir}/labeled_and_code_except_empty'

ori_dataset_dir = f'{now_proj_dir}/dataset/PROMISE'
labeled_data_dir = f'{ori_dataset_dir}/labeled_data/'
labeled_code_dir = f'{ori_dataset_dir}/labeled_src_code_parquet/'

data = pd.DataFrame()
count=0
# 读取数据
for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]
    file_prefix_l = f'{labeled_data_dir}{proj_name}/{proj_name}-'
    file_prefix_c = f'{labeled_code_dir}{proj_name}/{proj_name}-'

    expected_projname_dir=f'{labeled_and_code_except_empty_dir}/{proj_name}'
    if not os.path.exists(expected_projname_dir):
        os.makedirs(expected_projname_dir)

    for j in range(len(version_list)):
        version = version_list[j]
        labeled_csv = f'{file_prefix_l}{version}.csv' #对应项目对应版本的 不含源码的标记数据
        code_parquet = f'{file_prefix_c}{version}.parquet' #对应项目对应版本的 含源码的标记数据

        expected_projv_parquet = f'{expected_projname_dir}/{proj_name}-{version}.parquet'

        data = pd.read_csv(labeled_csv) #先将标记数据读取进来
        src_col= pd.read_parquet(code_parquet,columns=['src'])
        # 拼接
        data['src'] = src_col['src']
        data = prep(data)
        count += data.shape[0]
        data.to_parquet(expected_projv_parquet)

print(f'total data lines after refined: {count}') # 13298
# now the train_data and test_data are ready.

# train_data_tokenlist_x = train_data['src'].apply(ast_parse)
# train_y = train_data['buggy']
# test_data_tokenlist_x = test_data['src'].apply(ast_parse)
# test_y = test_data['buggy']
#
# it = get_embeddings(train_data_tokenlist_x)
# print(it)

