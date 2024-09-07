# 尝试使用parquet格式存储数据

import pandas as pd
data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
data.to_parquet('data.parquet')

# 读取数据
data = pd.read_parquet('D:/SourceCode/sdp/dataset_refined/labeled_and_code_except_empty/ant/ant-1.3.parquet')
print(data)