import os
import torch
import numpy as np
import config


def load_pretrained_embedding(vocab_list, embedding_path, embedding_dim, cache_path):
    """
    加载预训练词向量并初始化嵌入层，缓存机制

    :param vocab_list: 词表
    :param embedding_path: 预训练向量文件路径
    :param embedding_dim: 嵌入维度
    """
    # 检查是否存在缓存文件
    if os.path.exists(cache_path):
        print(f"从缓存加载预训练词向量: {cache_path}")
        return torch.load(cache_path)

    print(f"首次加载预训练词向量，将创建缓存文件: {cache_path}")

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

    # 保存缓存
    torch.save(embedding_matrix, cache_path)
    print(f"预训练词向量已缓存到: {cache_path}")

    return embedding_matrix
