from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        # TODO: pretrained word vectors, fine-tuned
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

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
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        # TODO: pretrained word vectors, fine-tuned
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

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
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size, padding_index=zh_padding_index)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size, padding_index=en_padding_index)


if __name__ == '__main__':
    import torch
    import config
    from dataset import get_dataloader
    from tokenizer import ChineseTokenizer, EnglishTokenizer

    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    dataloader = get_dataloader()
    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.CHECKPOINTS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.CHECKPOINTS_DIR / 'en_vocab.txt')
    # 4. 模型
    model = TranslationModel(zh_tokenizer.vocab_size,
                             en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)

    for inputs, targets, src_lengths in dataloader:
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
