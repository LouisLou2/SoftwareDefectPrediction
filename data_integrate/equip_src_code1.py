# 将promise数据库的类源码标记buggy与否

'''
被标记的数据：dataset/PROMISE/labeled_src_code
对应标记数据的源码：dataset/PROMISE/source_code
运行此脚本，输出某一段代码是否有bug(实际上是有bug的数量)
将存储在parquet文件中：dataset/PROMISE/labeled_src_code_parquet

每个parquet文件的格式是：
name, buggy, src
name: 类名
buggy: 有bug的数量
src: 源代码
'''

import os
import re
import pandas as pd

bug_proj_name = ['ant', 'camel', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces']
# bug_data={
#     'ant': ['1.3', '1.4', '1.5', '1.6', '1.7'],
#     'camel': ['1.0', '1.2', '1.4', '1.6'],
#     # 'ivy': ['1.0', '1.1', '1.2'], ivy暂时找不到低版本的源码
#     'jedit': ['3.2', '4.0', '4.1', '4.2', '4.3'],
#     'log4j': ['1.0', '1.1', '1.2'],
#     'lucene': ['2.0', '2.2', '2.4'],
#     'poi': ['1.5', '2.0', '2.5', '3.0'],
#     'synapse': ['1.0', '1.1', '1.2'],
#     'velocity': ['1.4', '1.5', '1.6'],
#     'xalan': ['2.4', '2.5', '2.6', '2.7'],
#     'xerces': ['1.1', '1.2', '1.3', '1.4']
# }
bug_proj_versions = [
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
now_proj_dir = 'D:/SourceCode/sdp'

process_record_path = f'{now_proj_dir}/process_record/data_integrate/data_ig1.txt'
labeled_data_dir = f'{now_proj_dir}/dataset/PROMISE/labeled_data/'
source_dir = f'{now_proj_dir}/dataset/PROMISE/source_code/'
target_dir = f'{now_proj_dir}/dataset/PROMISE/labeled_src_code_parquet/'

source_dir_prefixes = [
    '/src/main/',
    '/camel-core/src/main/java/',
    '/',
    '/src/java/',
    '/src/java/',
    '/src/java/',
    '/modules/core/src/main/java/',
    '/src/java/',
    '/src/',
    '/src/'
]


# 定义一个函数来移除非法字符
def remove_illegal_characters(text):
    if isinstance(text, str):
        # 只保留有效的可打印字符
        return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text


def getSourceFilePath(proj_index, versionfile, class_name):
    slashClassPath = class_name.replace('.', '/') + '.java'
    return f'{source_dir}{bug_proj_name[proj_index]}/{bug_proj_name[proj_index]}-{versionfile}{source_dir_prefixes[proj_index]}{slashClassPath}'


# 处理过程中的记录
process_records = []
strr = ''
for i in range(len(bug_proj_name)):
    proj_name = bug_proj_name[i]  # 项目名
    process_records.append(f'{proj_name} Now Processing\n\n')
    labeled_proj_dir = labeled_data_dir + proj_name + '/'  # 项目标记数据目录
    # 在target_dir创建项目文件夹
    if not os.path.exists(target_dir + proj_name):
        os.makedirs(target_dir + proj_name)

    for version in bug_proj_versions[i]:
        # 期待的parquet文件建立
        parquet_file = f'{target_dir}{proj_name}/{proj_name}-{version}.parquet'
        csv_file = f'{labeled_proj_dir}{proj_name}-{version}.csv'
        process_records.append(f'{csv_file} Now Reading\n')
        # 读取该csv
        data = pd.read_csv(csv_file)
        # 拿到第一列，使用顺序索引
        class_names = data.iloc[:, 0]
        # 建立结果df
        res = pd.DataFrame(class_names)
        res['buggy'] = data.iloc[:, -1]
        res['src'] = pd.NA
        # 遍历类名
        for j in range(len(class_names)):
            class_name = class_names[j]
            # 获取源码路径
            source_file = getSourceFilePath(i, version, class_name)
            # 判断文件是否存在
            if not os.path.exists(source_file):
                # 报告
                res.loc[j, 'src'] = ''  # 这可能不是最好的方法
                process_records.append(f'{source_file} not exists\n')
            else:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    res.iloc[j, 'src'] = f.read()
                # print(res.loc[j, 'src'])
        # 移除非法字符
        # res = res.map(remove_illegal_characters)
        # 写入parquet
        res.to_parquet(parquet_file, index=False, )

# 写入处理记录
with open(process_record_path, 'w', encoding='utf-8') as f:
    f.writelines(process_records)
