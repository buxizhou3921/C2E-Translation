import config
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import load_pretrained_embedding


class Attention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs):
        # TODO: alignment functions: multiplicative, and additive
        attention_scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2)) # dot-product
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.bmm(attention_weights, encoder_outputs)


class GRUAttentionEncoder(nn.Module):
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

        # 将打包的输出还原为填充序列
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        # outputs.shape: [batch_size, seq_len, hidden_size]

        return outputs, hidden


class GRUAttentionDecoder(nn.Module):
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

        self.attention = Attention()

        self.linear = nn.Linear(in_features=2 * config.HIDDEN_SIZE,
                                out_features=vocab_size)

    def forward(self, x, hidden_0, encoder_outputs):
        # x.shape: [batch_size, 1]
        # hidden_0.shape: [num_layers, batch_size, hidden_size]
        embed = self.embedding(x)
        # embed.shape: [batch_size, 1, embedding_dim]
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape: [batch_size, 1, hidden_size]

        # 应用注意力机制
        context_vector = self.attention(output, encoder_outputs)
        # context_vector.shape: [batch_size, 1, hidden_size]

        # 融合信息
        combined = torch.cat([output, context_vector], dim=-1)
        # combined.shape: [batch_size, 1, hidden_size * 2]

        output = self.linear(combined)
        # output.shape: [batch_size, 1, vocab_size]
        return output, hidden_n


class GRUAttentionModel(nn.Module):
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

        self.encoder = GRUAttentionEncoder(vocab_size=zh_vocab_size, padding_index=zh_padding_index, pretrained_vectors=zh_pretrained)
        self.decoder = GRUAttentionDecoder(vocab_size=en_vocab_size, padding_index=en_padding_index, pretrained_vectors=en_pretrained)




if __name__ == '__main__':
    pass
