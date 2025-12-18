import config
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_index, pretrained_vectors=None):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

        # 用预训练向量初始化
        if pretrained_vectors is not None:
            # 确保嵌入维度匹配
            if pretrained_vectors.size(1) != config.EMBEDDING_DIM:
                raise ValueError(f"预训练向量维度不匹配! 期望: ({config.EMBEDDING_DIM}), 实际: {pretrained_vectors.size(1)}")

            # 替换权重
            self.embedding.weight.data.copy_(pretrained_vectors)
            # 开放微调
            self.embedding.weight.requires_grad = True

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          num_layers=2,
                          batch_first=True)

    def forward(self, x, src_lengths):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]

        # 打包序列，剔除padding
        packed = pack_padded_sequence(embed, src_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, hidden = self.gru(packed)
        # hidden.shape: [num_layers, batch_size, hidden_size]

        return hidden


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_index, pretrained_vectors=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

        # 用预训练向量初始化
        if pretrained_vectors is not None:
            # 确保维度匹配
            if pretrained_vectors.size(0) != vocab_size or pretrained_vectors.size(1) != config.EMBEDDING_DIM:
                raise ValueError(
                    f"预训练向量维度不匹配! 期望: ({vocab_size}, {config.EMBEDDING_DIM}), 实际: {pretrained_vectors.shape}")

            # 替换权重
            self.embedding.weight.data.copy_(pretrained_vectors)
            # 开放微调
            self.embedding.weight.requires_grad = True

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          num_layers=2,
                          batch_first=True)

        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,
                                out_features=vocab_size)

    def forward(self, x, hidden_0):
        # x.shape: [batch_size, 1]
        # hidden_0.shape: [num_layers, batch_size, hidden_size]
        embed = self.embedding(x)
        # embed.shape: [batch_size, 1, embedding_dim]
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape: [batch_size, 1, hidden_size]
        output = self.linear(output)
        # output.shape: [batch_size, 1, vocab_size]
        return output, hidden_n


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_list, zh_vocab_size, en_vocab_list, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        # 加载中文预训练词向量
        zh_pretrained = load_pretrained_embedding(
            vocab_list=zh_vocab_list,
            embedding_path= config.VOCAB_DIR / 'tencent-ailab-embedding-zh-d100-v0.2.0.txt',
            embedding_dim=config.EMBEDDING_DIM
        )
        # 加载英文预训练词向量
        en_pretrained = load_pretrained_embedding(
            vocab_list=en_vocab_list,
            embedding_path= config.VOCAB_DIR / 'glove_2024_wikigiga_100d.txt',
            embedding_dim=config.EMBEDDING_DIM
        )

        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size, padding_index=zh_padding_index, pretrained_vectors=zh_pretrained)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size, padding_index=en_padding_index, pretrained_vectors=en_pretrained)


if __name__ == '__main__':
    import torch
    import config
    from dataset import get_dataloader
    from tokenizer import ChineseTokenizer, EnglishTokenizer

    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    train_dataloader = get_dataloader("train")
    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
    # 4. 模型
    model = TranslationModel(zh_tokenizer.vocab_list,
                             zh_tokenizer.vocab_size,
                             en_tokenizer.vocab_list,
                             en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)

    for inputs, targets, src_lengths in train_dataloader:
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]

        print("测试TranslationEncoder...")
        encoder_hidden = model.encoder(encoder_inputs, src_lengths)
        print(f"Encoder输出hidden状态形状: {encoder_hidden.shape}")

        print("\n测试TranslationDecoder...")
        decoder_hidden = encoder_hidden
        seq_len = decoder_inputs.shape[1]
        for i in range(seq_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size, 1, vocab_size]
            print(f"Decoder输出output形状: {decoder_output.shape}")  # [batch_size, 1, en_vocab_size]
            print(f"Decoder输出hidden状态形状: {decoder_hidden.shape}")
            break
        break
    print("\n测试完成！前向传播正常工作。")


