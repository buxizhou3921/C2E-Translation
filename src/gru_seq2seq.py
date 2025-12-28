import config
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tools import load_pretrained_embedding


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, padding_index, pretrained_vectors=None):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

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
        embed = self.embedding(x)
        packed = pack_padded_sequence(embed, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        return outputs, hidden


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, padding_index, pretrained_vectors=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

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

    def forward(self, x, hidden_0, encoder_outputs=None):
        embed = self.embedding(x)
        output, hidden_n = self.gru(embed, hidden_0)
        output = self.linear(output)
        return output, hidden_n


class GRUModel(nn.Module):
    def __init__(self, zh_vocab_list, zh_vocab_size, en_vocab_list, en_vocab_size, zh_padding_index, en_padding_index, args):
        super().__init__()
        # 加载中文预训练词向量
        zh_pretrained = load_pretrained_embedding(
            vocab_list=zh_vocab_list,
            embedding_path= config.VOCAB_DIR / 'tencent-ailab-embedding-zh-d100-v0.2.0.txt',
            embedding_dim=config.EMBEDDING_DIM,
            cache_path = config.VOCAB_DIR / 'zh_pretrained_vectors.pt'
        )
        # 加载英文预训练词向量
        en_pretrained = load_pretrained_embedding(
            vocab_list=en_vocab_list,
            embedding_path= config.VOCAB_DIR / 'glove_2024_wikigiga_100d.txt',
            embedding_dim=config.EMBEDDING_DIM,
            cache_path=config.VOCAB_DIR / 'en_pretrained_vectors.pt'
        )

        self.encoder = GRUEncoder(vocab_size=zh_vocab_size, padding_index=zh_padding_index, pretrained_vectors=zh_pretrained)
        self.decoder = GRUDecoder(vocab_size=en_vocab_size, padding_index=en_padding_index, pretrained_vectors=en_pretrained)


if __name__ == '__main__':
    pass
