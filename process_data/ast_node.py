# 此脚本用于提取源码中的ast node作为token
import os

import javalang
import pandas as pd
from javalang.tree import CompilationUnit, MethodDeclaration, FormalParameter, Literal, PackageDeclaration

from const.dataset_const import now_proj_dir, proj_names, proj_versions
from util.list_serial import store_list_of_list


def appendNodeStrToken(lis, node):
    # 这两种类型的节点不需要处理
    if isinstance(node, CompilationUnit) or isinstance(node, Literal) or isinstance(node, PackageDeclaration):
        return
    # main方法不需要处理
    if isinstance(node, MethodDeclaration) and node.name == 'main':
        return
    if isinstance(node, FormalParameter) and node.name == 'args':
        return
    # 以下是需要处理的节点
    # 有的节点有类型也有名字，有的只有类型
    lis.append(type(node).__name__)
    if hasattr(node, 'name'):
        lis.append(node.name)


def preorder_traversal_ast(node, lis, depth=0):
    """
    先序遍历AST树，打印节点类型和属性。
    :param node: 当前遍历的节点
    :param depth: 当前节点的深度（用于缩进显示）
    """
    appendNodeStrToken(lis, node)
    # 递归遍历子节点
    if isinstance(node, javalang.ast.Node):  # 检查是否为AST节点
        for child in node.children:
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, javalang.ast.Node):
                        preorder_traversal_ast(item, lis, depth + 1)
            elif isinstance(child, javalang.ast.Node):
                preorder_traversal_ast(child, lis, depth + 1)
    return lis


def ast_parse(src):
    lis = []
    tree = javalang.parse.parse(src)
    preorder_traversal_ast(tree, lis)
    return lis

#
# # 此脚本在dataset_refined目录下生成ast node token文件-存储在ast_token下
# # 以dataset_refined/labeled_and_code_except_empty数据为基础
#
#
# dataset_refined_dir = f'{now_proj_dir}/dataset_refined'
#
# # 将原来的labeled data和source code整合到一起,但是如果该行的code是空串，则整个行舍弃
# # 将buggy的数量转化为是否有bug
# # 将数据整合到一起还有另一个目的，就是将其以parquet格式存储，可以独立读取列，之后的处理会更加方便
# labeled_and_code_except_empty_dir = f'{dataset_refined_dir}/labeled_and_code_except_empty/'
# ast_token_dir = f'{dataset_refined_dir}/ast_token/'
#
# data = pd.DataFrame()
# count = 0
# # 读取数据
# for i in range(len(proj_names)):
#     proj_name = proj_names[i]
#     version_list = proj_versions[i]
#
#     labeled_prefix = f'{labeled_and_code_except_empty_dir}{proj_name}/{proj_name}-'
#     ast_token_dir_proj = f'{ast_token_dir}{proj_name}'
#
#     if not os.path.exists(ast_token_dir_proj):
#         os.makedirs(ast_token_dir_proj)
#
#     for j in range(len(version_list)):
#         version = version_list[j]
#         labeled_par = f'{labeled_prefix}{version}.parquet'
#         target_txt = f'{ast_token_dir_proj}/{proj_name}-{version}.txt'
#         data = pd.read_parquet(labeled_par, columns=['src'])
#         token_llis = []
#         for src in data['src']:
#             token_llis.append(ast_parse(src))
#         print(f'now storing {target_txt}')
#         store_list_of_list(token_llis, target_txt)
