import sys
import torch
import numpy as np
from gru_seq2seq import GRUModel
from gru_attention import GRUAttentionModel
from transformer import TransformerModel


def get_model(args, zh_tokenizer, en_tokenizer, device):
    if args.model == 'gru_seq2seq':
        model = GRUModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index).to(device)
    elif args.model == 'gru_attention':
        model = GRUAttentionModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index).to(device)
    elif args.model == 'transformer':
        model = TransformerModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index).to(device)
    else:
        print('the model name you have entered is not supported yet')
        sys.exit()

    return model


def load_pretrained_embedding(vocab_list, embedding_path, embedding_dim):
    """
    加载预训练词向量并初始化嵌入层

    :param vocab_list: 词表
    :param embedding_path: 预训练向量文件路径
    :param embedding_dim: 嵌入维度
    """
    # 初始化矩阵
    embedding_matrix = torch.zeros(len(vocab_list), embedding_dim)

    # 加载预训练词向量
    word_to_vec = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == embedding_dim + 1:
                word = parts[0]
                vec = np.array(parts[1:], dtype='float32')
                word_to_vec[word] = vec

    # 填充已知词
    for idx, word in enumerate(vocab_list):
        if word in word_to_vec:
            embedding_matrix[idx] = torch.tensor(word_to_vec[word], dtype=torch.float)

    # 处理OOV
    oov_mask = embedding_matrix.sum(dim=1) == 0
    if oov_mask.any():
        oov_count = oov_mask.sum().item()
        # 用均匀分布初始化OOV词 (-0.25~0.25)
        embedding_matrix[oov_mask] = torch.empty(oov_count, embedding_dim).uniform_(-0.25, 0.25)

    return embedding_matrix
