# 预处理
'''
主要目的就是将已经标记的源代码
将bug数量转为二分类
将代码为空的行去掉
'''
import javalang.parse
import pandas as pd
from javalang.tree import Literal, PackageDeclaration
from javalang.tree import CompilationUnit, MethodDeclaration, FormalParameter


def prep(df):
    # 去掉代码为空的行
    df = df[df['src'].notna() & (df['src'] != '')]
    # 重置索引并丢弃旧索引
    df = df.reset_index(drop=True)
    # 将bug数量转为二分类
    df['bug'] = df['bug'].apply(lambda x: 1 if x > 0 else 0)
    return df


def bug_to_binary(df):
    df['bug'] = df['bug'].apply(lambda x: 1 if x > 0 else 0)
    return df
