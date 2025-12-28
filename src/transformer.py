import math
import torch
from torch import nn
import config
from tools import load_pretrained_embedding


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AbsolutePositionEncoding(nn.Module):
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


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, dim_model):
        super().__init__()
        pe = torch.zeros([max_len, max_len, dim_model], dtype=torch.float)
        for i in range(max_len):
            for j in range(max_len):
                # 计算相对距离（i - j）
                rel_pos = i - j
                for _2i in range(0, dim_model, 2):
                    pe[i, j, _2i] = math.sin(rel_pos / (10000 ** (_2i / dim_model)))
                    pe[i, j, _2i + 1] = math.cos(rel_pos / (10000 ** (_2i / dim_model)))

        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        rel_pe = self.pe[:seq_len, :seq_len, :]  # [seq_len, seq_len, dim_model]
        rel_pe_sum = rel_pe.sum(dim=1)  # [seq_len, dim_model]
        rel_pe_sum = rel_pe_sum.unsqueeze(0)  # [1, seq_len, dim_model]
        rel_pe_sum = rel_pe_sum.expand(batch_size, -1, -1)  # [batch_size, seq_len, dim_model]

        return x + rel_pe_sum


class TransformerModel(nn.Module):
    def __init__(self, zh_vocab_list, zh_vocab_size, en_vocab_list, en_vocab_size, zh_padding_index, en_padding_index, args):
        super().__init__()

        # 加载中文预训练词向量
        zh_pretrained = load_pretrained_embedding(
            vocab_list=zh_vocab_list,
            embedding_path=config.VOCAB_DIR / 'tencent-ailab-embedding-zh-d100-v0.2.0.txt',
            embedding_dim=config.DIM_MODEL,
            cache_path=config.VOCAB_DIR / 'zh_pretrained_vectors.pt'
        )
        # 加载英文预训练词向量
        en_pretrained = load_pretrained_embedding(
            vocab_list=en_vocab_list,
            embedding_path=config.VOCAB_DIR / 'glove_2024_wikigiga_100d.txt',
            embedding_dim=config.DIM_MODEL,
            cache_path=config.VOCAB_DIR / 'en_pretrained_vectors.pt'
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
        if args.position == 'absolute':
            self.position_encoding = AbsolutePositionEncoding(config.MAX_SEQ_LENGTH, config.DIM_MODEL)
        elif args.position == 'relative':
            self.position_encoding = RelativePositionEncoding(config.MAX_SEQ_LENGTH, config.DIM_MODEL)
        else:
            raise ValueError(f"未知的位置编码类型: {args.position}")

        if args.norm == 'rms':
            torch.nn.LayerNorm = RMSNorm

        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                          nhead=args.heads,
                                          num_encoder_layers=args.layers,
                                          num_decoder_layers=args.layers,
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
