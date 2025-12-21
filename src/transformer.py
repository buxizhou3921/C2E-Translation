import math
import torch
from torch import nn
import config
from utils import load_pretrained_embedding


# class PositionEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#
#         pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # pos.shape: (max_len, 1)
#         _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float)  # _2i.shape: (d_model/2,)
#         div_term = torch.pow(10000, _2i / self.d_model)
#
#         sins = torch.sin(pos / div_term)  # sins.shape: (max_len, d_model/2)
#         coss = torch.cos(pos / div_term)  # coss.shape: (max_len, d_model/2)
#
#         pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)  # pe.shape: (max_len, d_model)
#
#         pe[:, 0::2] = sins
#         pe[:, 1::2] = coss
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.pe[:seq_len]

# TODO: absolute vs. relative
class PositionEncoding(nn.Module):
    def __init__(self, max_len, dim_model):
        super().__init__()
        pe = torch.zeros([max_len, dim_model], dtype=torch.float)
        for pos in range(max_len):
            for _2i in range(0, dim_model, 2):
                pe[pos, _2i] = math.sin(pos / (10000 ** (_2i / dim_model)))
                pe[pos, _2i + 1] = math.cos(pos / (10000 ** (_2i / dim_model)))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape: [batch_size, seq_len, dim_model]
        seq_len = x.shape[1]
        part_pe = self.pe[0:seq_len]
        # part_pe.shape: [seq_len, dim_model]
        return x + part_pe


class TransformerModel(nn.Module):
    def __init__(self, zh_vocab_list, zh_vocab_size, en_vocab_list, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()

        # 加载中文预训练词向量
        zh_pretrained = load_pretrained_embedding(
            vocab_list=zh_vocab_list,
            embedding_path=config.VOCAB_DIR / 'tencent-ailab-embedding-zh-d100-v0.2.0.txt',
            embedding_dim=config.DIM_MODEL
        )
        # 加载英文预训练词向量
        en_pretrained = load_pretrained_embedding(
            vocab_list=en_vocab_list,
            embedding_path=config.VOCAB_DIR / 'glove_2024_wikigiga_100d.txt',
            embedding_dim=config.DIM_MODEL
        )

        self.zh_embedding = nn.Embedding(num_embeddings=zh_vocab_size,
                                         embedding_dim=config.DIM_MODEL,
                                         padding_idx=zh_padding_index)

        self.en_embedding = nn.Embedding(num_embeddings=en_vocab_size,
                                         embedding_dim=config.DIM_MODEL,
                                         padding_idx=en_padding_index)

        # 用预训练向量初始化
        if zh_pretrained.size(1) != config.DIM_MODEL:
            raise ValueError(
                f"中文预训练向量维度不匹配! 期望: ({config.DIM_MODEL}), 实际: {zh_pretrained.size(1)}")

        if en_pretrained.size(1) != config.DIM_MODEL:
            raise ValueError(
                f"英文预训练向量维度不匹配! 期望: ({config.DIM_MODEL}), 实际: {en_pretrained.size(1)}")

        # 替换权重
        self.zh_embedding.weight.data.copy_(zh_pretrained)
        self.en_embedding.weight.data.copy_(en_pretrained)
        # 开放微调
        self.zh_embedding.weight.requires_grad = True
        self.en_embedding.weight.requires_grad = True

        # 位置编码
        self.position_encoding = PositionEncoding(config.MAX_SEQ_LENGTH, config.DIM_MODEL)

        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                          nhead=config.NUM_HEADS,
                                          num_encoder_layers=config.NUM_ENCODER_LAYERS,
                                          num_decoder_layers=config.NUM_DECODER_LAYERS,
                                          batch_first=True)

        self.linear = nn.Linear(in_features=config.DIM_MODEL, out_features=en_vocab_size)

    def forward(self, src, tgt, src_pad_mask, tgt_mask):
        memory = self.encode(src, src_pad_mask)
        return self.decode(tgt, memory, tgt_mask, src_pad_mask)

    def encode(self, src, src_pad_mask):
        # src.shape = [batch_size, src_len]
        # src_pad_mask.shape = [batch_size, src_len]
        embed = self.zh_embedding(src)
        # embed.shape = [batch_size, src_len, dim_model]
        embed = self.position_encoding(embed)

        memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
        # memory.shape: [batch_size, src_len, dim_model]

        return memory

    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        # tgt.shape: [batch_size, tgt_len]
        embed = self.en_embedding(tgt)
        embed = self.position_encoding(embed)
        # embed.shape: [batch_size, tgt_len, dim_model]

        output = self.transformer.decoder(tgt=embed, memory=memory,
                                          tgt_mask=tgt_mask, memory_key_padding_mask=memory_pad_mask)
        # output.shape: [batch_size, tgt_len, dim_model]

        outputs = self.linear(output)
        # outputs.shape: [batch_size, tgt_len, en_vocab_size]
        return outputs
