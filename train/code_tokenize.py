# 使用CodeBERT进行tokenize,以张量的形式返回

from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.utils.rnn import pad_sequence


# train_code_series=输入为已经遍历好的ast node列表，因为是所有测试集的ast node列表，所以是一个二维列表
def get_embeddings(train_code_series, test_code_series):
    # 加载 CodeBERT 的 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    train_token_ids_list = []  # 以每个文件的token_ids为元素的列表
    test_token_ids_list = []

    for code_series in train_code_series:
        # 构建最终token sequence
        tokens = [tokenizer.cls_token] + code_series + [tokenizer.eos_token]
        # 将 token 序列转化为 ID
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        train_token_ids_list.append(torch.tensor(tokens_ids))

    for code_series in test_code_series:
        # 构建最终token sequence
        tokens = [tokenizer.cls_token] + code_series + [tokenizer.eos_token]
        # 将 token 序列转化为 ID
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        test_token_ids_list.append(torch.tensor(tokens_ids))

    # 将train_token_ids转化为张量，因为每个文件的token_ids长度不一样，所以需要填充
    train_token_ids_matrix = pad_sequence(train_token_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    test_token_ids_matrix = pad_sequence(test_token_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)

    # 将token_ids输入到CodeBERT模型中，生成嵌入向量
    train_context_embeddings = model(train_token_ids_matrix)[0]
    test_context_embeddings = model(test_token_ids_matrix)[0]
    return train_context_embeddings, test_context_embeddings
