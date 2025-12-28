import config
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tools import load_pretrained_embedding


class Attention(nn.Module):
    def __init__(self, align_func, hidden_size=256):
        super().__init__()
        self.align = align_func
        self.hidden_size = hidden_size

        if self.align == 'mul':
            self.W_mul = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.align == 'add':
            self.W_add_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_add_encoder = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_add = nn.Parameter(torch.randn(hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.align == 'dot':
            attention_scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        elif self.align == 'mul':
            decoder_trans = self.W_mul(decoder_hidden)  # (batch, 1, hidden)
            attention_scores = torch.bmm(decoder_trans, encoder_outputs.transpose(1, 2))
        elif self.align == 'add':
            # Additive attention (Bahdanau)
            _, seq_len, _ = encoder_outputs.size()
            decoder_trans = self.W_add_decoder(decoder_hidden)  # (batch, 1, hidden)
            encoder_trans = self.W_add_encoder(encoder_outputs)  # (batch, seq_len, hidden)
            # Expand decoder to match encoder dimensions for element-wise addition
            decoder_expanded = decoder_trans.expand(-1, seq_len, -1)  # (batch, seq_len, hidden)
            # add and tanh
            sum_trans = torch.tanh(decoder_expanded + encoder_trans)  # (batch, seq_len, hidden)
            # Compute scores using v_add parameter
            # (batch, seq_len, hidden) * (hidden) -> (batch, seq_len)
            attention_scores = torch.sum(sum_trans * self.v_add, dim=2)  # (batch, seq_len)
            attention_scores = attention_scores.unsqueeze(1)  # (batch, 1, seq_len)
        else:
            raise ValueError(f"无效的注意力函数: {self.align}")

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
    def __init__(self, vocab_size, padding_index, args, pretrained_vectors=None):
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

        self.attention = Attention(args.align, config.HIDDEN_SIZE)

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
    def __init__(self, zh_vocab_list, zh_vocab_size, en_vocab_list, en_vocab_size, zh_padding_index, en_padding_index, args):
        super().__init__()
        # 加载中文预训练词向量
        zh_pretrained = load_pretrained_embedding(
            vocab_list=zh_vocab_list,
            embedding_path=config.VOCAB_DIR / 'tencent-ailab-embedding-zh-d100-v0.2.0.txt',
            embedding_dim=config.EMBEDDING_DIM,
            cache_path=config.VOCAB_DIR / 'zh_pretrained_vectors.pt'
        )
        # 加载英文预训练词向量
        en_pretrained = load_pretrained_embedding(
            vocab_list=en_vocab_list,
            embedding_path=config.VOCAB_DIR / 'glove_2024_wikigiga_100d.txt',
            embedding_dim=config.EMBEDDING_DIM,
            cache_path=config.VOCAB_DIR / 'en_pretrained_vectors.pt'
        )

        self.encoder = GRUAttentionEncoder(vocab_size=zh_vocab_size, padding_index=zh_padding_index, pretrained_vectors=zh_pretrained)
        self.decoder = GRUAttentionDecoder(vocab_size=en_vocab_size, padding_index=en_padding_index, args=args, pretrained_vectors=en_pretrained)




if __name__ == '__main__':
    pass
