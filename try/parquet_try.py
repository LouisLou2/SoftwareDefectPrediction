# 尝试使用parquet格式存储数据

import pandas as pd
import torch

# data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
# data.to_parquet('data.parquet')
#
# # 读取数据
# data = pd.read_parquet('D:/SourceCode/sdp/dataset/PROMISE/labeled_src_code_parquet/xalan/xalan-2.4.parquet')
# print(data)

data = torch.load('D:/SourceCode/sdp/dataset_refined/token_ids/ant/ant-1.3.pt')
print(data)