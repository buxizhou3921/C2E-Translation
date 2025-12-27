import config
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tools import load_pretrained_embedding


class GRUEncoder(nn.Module):
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


class GRUDecoder(nn.Module):
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

    def forward(self, x, hidden_0, encoder_outputs=None):
        # x.shape: [batch_size, 1]
        # hidden_0.shape: [num_layers, batch_size, hidden_size]
        embed = self.embedding(x)
        # embed.shape: [batch_size, 1, embedding_dim]
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape: [batch_size, 1, hidden_size]
        output = self.linear(output)
        # output.shape: [batch_size, 1, vocab_size]
        return output, hidden_n


class GRUModel(nn.Module):
    def __init__(self, zh_vocab_list, zh_vocab_size, en_vocab_list, en_vocab_size, zh_padding_index, en_padding_index):
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
    from src.dataset import get_dataloader
    from src.tokenizer import ChineseTokenizer, EnglishTokenizer

    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    train_dataloader = get_dataloader("train")
    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
    # 4. 模型
    model = GRUModel(zh_tokenizer.vocab_list,
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
            print(f"Decoder单次输出output形状: {decoder_output.shape}")  # [batch_size, 1, en_vocab_size]
            print(f"Decoder单次输出hidden状态形状: {decoder_hidden.shape}")
            break
        break
