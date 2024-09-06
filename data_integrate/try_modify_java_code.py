# 由于历史原因，部分源代码使用的java语法太古老
# 例如某些源代码使用了enum关键字，而enum是从java 1.5开始引入的，比如assert关键字...
# 为了解决这个问题，我们需要对源代码进行修改
# 运行此脚本，会对labeled_data涉及到的java文件的源代码进行更改(更改在labeled_src_code中)

import os

import javalang
import pandas

#  ivy暂时找不到低版本的源码
bug_proj_name = ['ant', 'camel', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces']
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

process_record_path = f'{now_proj_dir}/process_record/data_integrate/try_modify_java_code.txt'
labeled_data_dir = f'{now_proj_dir}/dataset/PROMISE/labeled_data/'
source_dir = f'{now_proj_dir}/dataset/PROMISE/source_code/'

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


# 实际上这个函数也只是一个简易的版本，相当一部分源代码还是需要人工修改
def modify_if_syntax_error(source_file, code, records):
    try:
        tree = javalang.parse.parse(code)
    except javalang.parser.JavaSyntaxError as e:
        records.append(f'JavaSyntaxError: {source_file}\n')
        records.append(f'{e.description} at{e.at}\n')
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(code.replace('enum', 'enu').replace('assert', 'theAssert'))
    except javalang.tokenizer.LexerError as e:
        records.append(f'LexerError: {source_file}\n')


def getSourceFilePath(proj_index, versionfile, class_name):
    slashClassPath = class_name.replace('.', '/') + '.java'
    return f'{source_dir}{bug_proj_name[proj_index]}/{bug_proj_name[proj_index]}-{versionfile}{source_dir_prefixes[proj_index]}{slashClassPath}'


# 处理过程中的记录
process_records = []
for i in range(len(bug_proj_name)):
    proj_name = bug_proj_name[i]  # 项目名
    process_records.append(f'{proj_name} Now Processing\n\n')
    labeled_proj_dir = labeled_data_dir + proj_name + '/'  # 项目标记数据目录

    for version in bug_proj_versions[i]:
        csv_file = f'{labeled_proj_dir}{proj_name}-{version}.csv'
        process_records.append(f'{csv_file} Now Reading\n')
        # 读取该csv
        data = pandas.read_csv(csv_file)
        # 拿到第一列，使用顺序索引
        class_names = data.iloc[:, 0]
        # 遍历类名
        for j in range(len(class_names)):
            class_name = class_names[j]
            # 获取源码路径
            source_file = getSourceFilePath(i, version, class_name)
            # 判断文件是否存在
            if not os.path.exists(source_file):
                # 报告
                process_records.append(f'{source_file} not exists\n')
                continue
            else:
                # print(f'noe reading {source_file}')
                code_txt = ''
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code_txt = f.read()
                modify_if_syntax_error(source_file, code_txt, process_records)

# 写入处理记录
with open(process_record_path, 'w', encoding='utf-8') as f:
    f.writelines(process_records)
