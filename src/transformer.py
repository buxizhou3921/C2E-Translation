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
    """
    相对位置编码（Relative Position Encoding）
    基于 Shaw et al. (2018) "Self-Attention with Relative Position Representations"

    使用正弦/余弦函数计算相对位置编码，基于位置之间的相对距离而非绝对位置。
    相对距离被裁剪到 [-max_relative_position, max_relative_position] 范围内。
    """

    def __init__(self, max_len, dim_model, max_relative_position=32):
        """
        初始化相对位置编码

        Args:
            max_len: 最大序列长度
            dim_model: 模型维度
            max_relative_position: 最大相对位置距离（超出此范围的位置将被裁剪）
        """
        super().__init__()
        self.max_relative_position = max_relative_position
        self.dim_model = dim_model

        # 计算编码表的大小：从 -max_relative_position 到 +max_relative_position
        # 总共 2 * max_relative_position + 1 个位置
        vocab_size = 2 * max_relative_position + 1

        # 创建相对位置编码表
        pe = torch.zeros(vocab_size, dim_model, dtype=torch.float)

        # 为每个相对位置计算编码
        for pos in range(vocab_size):
            # 将索引转换为相对位置：[0, vocab_size) -> [-max_relative_position, +max_relative_position]
            relative_pos = pos - max_relative_position

            for _2i in range(0, dim_model, 2):
                # 正弦编码
                pe[pos, _2i] = math.sin(relative_pos / (10000 ** (_2i / dim_model)))
                # 余弦编码
                if _2i + 1 < dim_model:
                    pe[pos, _2i + 1] = math.cos(relative_pos / (10000 ** (_2i / dim_model)))

        # 注册为 buffer（不参与梯度更新，但会被保存）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, dim_model]

        Returns:
            添加了相对位置编码的张量 [batch_size, seq_len, dim_model]
        """
        batch_size, seq_len, _ = x.shape

        # 创建相对位置矩阵：每个位置 i 到位置 j 的相对距离
        # positions[i, j] = i - j
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        relative_positions = positions.T - positions  # [seq_len, seq_len]

        # 裁剪相对位置到 [-max_relative_position, max_relative_position]
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )

        # 转换为索引：[-max_relative_position, +max_relative_position] -> [0, 2*max_relative_position]
        relative_positions = relative_positions + self.max_relative_position

        # 获取相对位置编码 [seq_len, seq_len, dim_model]
        rel_pe = self.pe[relative_positions]

        # 对每个查询位置，聚合所有键位置的相对编码
        # 使用平均池化来聚合：[seq_len, seq_len, dim_model] -> [seq_len, dim_model]
        rel_pe_aggregated = rel_pe.mean(dim=1)  # 平均所有相对位置

        # 扩展到批次维度 [1, seq_len, dim_model] -> [batch_size, seq_len, dim_model]
        rel_pe_aggregated = rel_pe_aggregated.unsqueeze(0).expand(batch_size, -1, -1)

        # 添加到输入
        return x + rel_pe_aggregated


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

        if args.position == 'absolute':
            self.position_encoding = AbsolutePositionEncoding(config.MAX_SEQ_LENGTH, config.DIM_MODEL)
        elif args.position == 'relative':
            self.position_encoding = RelativePositionEncoding(
                max_len=config.MAX_SEQ_LENGTH,
                dim_model=config.DIM_MODEL,
                max_relative_position=config.MAX_RELATIVE_POSITION
            )
        else:
            raise ValueError(f"未知的位置编码类型: {args.position}. 支持的类型: 'absolute', 'relative'")

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
        embed = self.zh_embedding(src)
        embed = self.position_encoding(embed)
        memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        embed = self.en_embedding(tgt)
        embed = self.position_encoding(embed)
        output = self.transformer.decoder(tgt=embed, memory=memory,
                                          tgt_mask=tgt_mask, memory_key_padding_mask=memory_pad_mask)
        outputs = self.linear(output)
        return outputs
