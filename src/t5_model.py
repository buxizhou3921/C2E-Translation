"""
T5 模型封装类
 Hugging Face 的 T5 模型
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import config
from tools import load_pretrained_embedding


class T5Model(nn.Module):
    """
    T5 翻译模型封装类
    封装 Hugging Face 的 T5ForConditionalGeneration 模型
    """

    def __init__(self, model_name_or_path=None, args=None, zh_vocab_list=None, en_vocab_list=None):
        """
        初始化 T5 模型

        Args:
            model_name_or_path: 预训练模型路径或名称
                              默认使用 checkpoints/T5/t5-base
            args: 命令行参数对象，包含模型配置
            zh_vocab_list: 中文词表列表（用于加载预训练向量）
            en_vocab_list: 英文词表列表（用于加载预训练向量）
        """
        super(T5Model, self).__init__()

        # 设置模型路径，优先使用配置的路径
        if model_name_or_path is None:
            # 默认使用本地下载好的 t5-base 模型
            model_name_or_path = config.T5_MODEL_PATH

        # 加载 T5 模型和分词器
        print(f" 正在从 {model_name_or_path} 加载 T5 模型...")
        self.tokenizer = T5Tokenizer.from_pretrained(str(model_name_or_path))
        self.model = T5ForConditionalGeneration.from_pretrained(str(model_name_or_path))

      
        if zh_vocab_list is not None and en_vocab_list is not None:
            print("使用缓存的预训练词向量初始化 T5 embedding 层...")

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

            # 获取 T5 模型的 embedding 层
            # T5 使用共享的 embedding 层（encoder 和 decoder 共享）
            shared_embedding = self.model.get_input_embeddings()


            t5_vocab_size = shared_embedding.weight.size(0)
            t5_embedding_dim = shared_embedding.weight.size(1)
            if config.EMBEDDING_DIM != t5_embedding_dim:
                print(f" 警告: 预训练向量维度({config.EMBEDDING_DIM})与T5 embedding维度({t5_embedding_dim})不匹配")
                print(f"   跳过预训练向量初始化，使用T5原始embedding")
            else:

                combined_pretrained = torch.cat([zh_pretrained, en_pretrained], dim=0)

                # 只初始化前 N 个词的 embedding
                init_size = min(t5_vocab_size, combined_pretrained.size(0))
                with torch.no_grad():
                    shared_embedding.weight[:init_size] = combined_pretrained[:init_size]

                print(f" 已使用缓存的预训练向量初始化前 {init_size} 个词的 embedding")
                print(f"   中文词表大小: {len(zh_vocab_list)}, 英文词表大小: {len(en_vocab_list)}")
                print(f"   T5词表大小: {t5_vocab_size}")

        # 保存配置
        self.args = args
        self.model_name_or_path = model_name_or_path

        print(" T5 模型加载成功！")

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """
        前向传播

        Args:
            input_ids: 编码器输入 ID [batch_size, src_seq_len]
            attention_mask: 编码器注意力掩码 [batch_size, src_seq_len]
            decoder_input_ids: 解码器输入 ID [batch_size, tgt_seq_len]
            decoder_attention_mask: 解码器注意力掩码 [batch_size, tgt_seq_len]
            labels: 目标标签 [batch_size, tgt_seq_len]

        Returns:
            模型输出，包含 loss 和 logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask=None, max_length=None,
                 num_beams=4, early_stopping=True, **kwargs):
        """
        生成翻译结果

        Args:
            input_ids: 输入 ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            max_length: 生成的最大长度
            num_beams: beam search 宽度
            early_stopping: 是否提前停止
            **kwargs: 其他生成参数

        Returns:
            生成的 token IDs [batch_size, generated_seq_len]
        """
        if max_length is None:
            max_length = config.T5_MAX_TARGET_LENGTH

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            **kwargs
        )
        return outputs

    def save_pretrained(self, save_directory):
        """
        保存模型和分词器

        Args:
            save_directory: 保存目录路径
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存模型和分词器
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f" T5 模型已保存到 {save_path}")

    @classmethod
    def from_pretrained(cls, model_path, args=None, zh_vocab_list=None, en_vocab_list=None):
        """
        从预训练模型加载

        Args:
            model_path: 模型路径
            args: 命令行参数
            zh_vocab_list: 中文词表列表（用于加载预训练向量）
            en_vocab_list: 英文词表列表（用于加载预训练向量）

        Returns:
            T5Model 实例
        """
        return cls(model_name_or_path=model_path, args=args,
                   zh_vocab_list=zh_vocab_list, en_vocab_list=en_vocab_list)


class T5TranslationWrapper:
    """
    T5 翻译任务包装器
    提供更高层次的翻译接口
    """

    def __init__(self, model, tokenizer, device, task_prefix="translate Chinese to English: "):
        """
        初始化翻译包装器

        Args:
            model: T5Model 实例
            tokenizer: T5Tokenizer 实例
            device: 运行设备 (cpu/cuda)
            task_prefix: 任务前缀，告诉模型要执行什么任务
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.task_prefix = task_prefix

    def translate(self, text, max_length=None, num_beams=4):
        """
        翻译单个文本

        Args:
            text: 输入文本（中文）
            max_length: 最大生成长度
            num_beams: beam search 宽度

        Returns:
            翻译后的文本（英文）
        """
        # 添加任务前缀
        input_text = self.task_prefix + text

        # 编码输入
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.T5_MAX_SOURCE_LENGTH
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成翻译
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams
            )

        # 解码输出
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def batch_translate(self, texts, max_length=None, num_beams=4, batch_size=8):
        """
        批量翻译

        Args:
            texts: 文本列表
            max_length: 最大生成长度
            num_beams: beam search 宽度
            batch_size: 批处理大小

        Returns:
            翻译结果列表
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 添加任务前缀
            input_texts = [self.task_prefix + text for text in batch]

            # 编码批次
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.T5_MAX_SOURCE_LENGTH
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成翻译
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=num_beams
                )

            # 解码输出
            batch_results = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            results.extend(batch_results)

        return results
