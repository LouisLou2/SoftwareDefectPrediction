import gc
import os
import subprocess
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 项目名称及版本列表
now_proj_dir = '/root/sdp'

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

split_res = [2, 3, 5, 6, 12, 4, 7, 9, 10, 4, 5, 5, 5, 7, 3, 2, 4, 3, 4, 5, 4, 5, 6, 7, 3, 4, 5, 4, 4, 4, 10, 12, 13, 13,
             2, 8, 8, 7]
# sum
sum_split_res = sum(split_res)
print(f"sum_split_res: {sum_split_res}")

# 要调用的脚本路径
script_path = 'D:/SourceCode/sdp/vault/args.py'

# 循环调用脚本
file_count = 0
for i in range(len(proj_names)):
    proj_name = proj_names[i]
    version_list = proj_versions[i]
    for j in range(len(version_list)):
        version = version_list[j]
        spilit_num = split_res[file_count]
        file_count += 1

        for k in range(spilit_num):
            spilit_ind = k + 1
            args = [proj_name, version, str(spilit_ind)]
            # 构造命令行
            command = ['python', script_path] + args
            print(f'Executing: {" ".join(command)}')

            # 调用脚本并等待完成
            result = subprocess.run(command, capture_output=True, text=True)

            # 打印输出和错误信息（如果有）
            print('Output:', result.stdout)
            if result.stderr:
                print('Error:', result.stderr)
            # 可选：添加延时以确保资源释放
            time.sleep(1)  # 根据需要调整时间