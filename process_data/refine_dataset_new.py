import os

import pandas as pd

from const.dataset_const import now_proj_dir, proj_names, proj_versions
from process_data.ast_node import ast_parse
from process_data.preprocess import bug_to_binary
from util.list_serial import store_list_of_list

dataset_refined_dir = f'{now_proj_dir}/dataset_refined'
# 将原来的labeled data和source code整合到一起,但是如果该行的code是空串，则整个行舍弃
# 将buggy的数量转化为是否有bug
# 将数据整合到一起还有另一个目的，就是将其以parquet格式存储，可以独立读取列，之后的处理会更加方便
labeled_and_code_except_empty_dir = f'{dataset_refined_dir}/labeled_and_code_except_empty'

ori_dataset_dir = f'{now_proj_dir}/dataset/PROMISE'
labeled_data_dir = f'{ori_dataset_dir}/labeled_data/'
labeled_code_dir = f'{ori_dataset_dir}/labeled_src_code_parquet/'
ast_token_dir = f'{dataset_refined_dir}/ast_token/'

max_token_constraint = 450
data = pd.DataFrame()

count = 0
# 读取数据
for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]
    file_prefix_l = f'{labeled_data_dir}{proj_name}/{proj_name}-'
    file_prefix_c = f'{labeled_code_dir}{proj_name}/{proj_name}-'

    expected_projname_dir = f'{labeled_and_code_except_empty_dir}/{proj_name}'
    ast_token_proj_dir = f'{ast_token_dir}{proj_name}'

    if not os.path.exists(ast_token_proj_dir):
        os.makedirs(ast_token_proj_dir)

    if not os.path.exists(expected_projname_dir):
        os.makedirs(expected_projname_dir)

    for j in range(len(version_list)):
        # 开始在项目+版本级别上处理
        version = version_list[j]
        labeled_csv = f'{file_prefix_l}{version}.csv'  # 对应项目对应版本的 不含源码的标记数据
        code_parquet = f'{file_prefix_c}{version}.parquet'  # 对应项目对应版本的 含源码的标记数据

        expected_projv_parquet = f'{expected_projname_dir}/{proj_name}-{version}.parquet'
        ast_token_projv_txt = f'{ast_token_proj_dir}/{proj_name}-{version}.txt'

        data = pd.read_csv(labeled_csv)  # 先将标记数据读取进来
        src_col = pd.read_parquet(code_parquet, columns=['src'])['src']  # 读取源码列
        # 确保数据行数一致
        assert data.shape[0] == src_col.shape[0]
        drop_list = []
        token_llis = []
        # 遍历每一行，如果src为空串，则整个行舍弃
        for k in range(src_col.shape[0]):
            if src_col[k] == '':
                drop_list.append(k)
                continue
            token_list = ast_parse(src_col[k])
            if len(token_list) > max_token_constraint:
                drop_list.append(k)
                continue
            token_llis.append(token_list)
        # 删除不符合条件的行
        data.drop(drop_list, inplace=True)
        src_col.drop(drop_list, inplace=True)

        assert data.shape[0] == src_col.shape[0]
        assert data.shape[0] == len(token_llis)
        # 拼接
        data['src'] = src_col
        data.reset_index(drop=True, inplace=True)
        # 将buggy的数量转化为是否有bug
        data = bug_to_binary(data)

        # 计数
        count += data.shape[0]

        print(f'now processing:{ast_token_projv_txt}')
        # 写入parquet
        data.to_parquet(expected_projv_parquet)

        # 将处理过程中产生的token list写入txt
        store_list_of_list(token_llis, ast_token_projv_txt)

print(f'total data lines after refined: {count}')  #
