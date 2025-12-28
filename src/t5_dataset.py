"""
T5 模型数据集类
处理原始文本数据，使用 T5Tokenizer 进行编码
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class T5TranslationDataset(Dataset):
    """
    T5 翻译数据集
    从原始数据文件加载中英文本对，使用 T5Tokenizer 进行编码
    """

    def __init__(self, path, tokenizer, task_prefix="translate Chinese to English: ",
                 max_source_length=None, max_target_length=None):
        """
        初始化数据集
 
        """
        # 读取原始数据
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')
        self.tokenizer = tokenizer
        self.task_prefix = task_prefix

        # 设置最大长度
        self.max_source_length = max_source_length or config.T5_MAX_SOURCE_LENGTH
        self.max_target_length = max_target_length or config.T5_MAX_TARGET_LENGTH

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, index):
        """
        获取单个数据样本

        Args:
            index: 样本索引

        Returns:
            包含编码后的输入和目标的字典
        """
        # 获取中英文本
        zh_text = self.data[index]['zh']  # 中文（源语言）
        en_text = self.data[index]['en']  # 英文（目标语言）

        # 添加任务前缀到源文本
        # 例如: "translate Chinese to English: 你好世界"
        source_text = self.task_prefix + zh_text

        # 编码源文本
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码目标文本
        target_encoding = self.tokenizer(
            en_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 准备标签
        # 将 padding token 替换为 -100，使其在损失计算中被忽略
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),  # [seq_len]
            'attention_mask': source_encoding['attention_mask'].squeeze(),  # [seq_len]
            'labels': labels  # [seq_len]
        }


def t5_collate_fn(batch):
    """
    T5 数据集的批处理函数
    """
    # 提取所有样本的各个字段
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def get_t5_dataloader(args, mode='train', tokenizer=None):
    """
    获取 T5 数据加载器
    """
    if tokenizer is None:
        raise ValueError("T5Tokenizer must be provided!")

    if mode == 'train':
     
        path = config.RAW_DATA_DIR / 'train_10k.jsonl'
        shuffle = True
    elif mode == 'valid':
        path = config.RAW_DATA_DIR / 'valid.jsonl'
        shuffle = False
    elif mode == 'test':
        path = config.RAW_DATA_DIR / 'test.jsonl'
        shuffle = False
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'train', 'valid', or 'test'!")

    # 创建数据集
    dataset = T5TranslationDataset(
        path=path,
        tokenizer=tokenizer,
        task_prefix="translate Chinese to English: "
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        collate_fn=t5_collate_fn,
        num_workers=0  
    )

    return dataloader


class T5EvaluationDataset(Dataset):
    """
    T5 评估专用数据集
    返回未编码的原始文本，用于评估时的灵活处理
    """

    def __init__(self, path):
        """
        初始化评估数据集

        Args:
            path: 数据文件路径
        """
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        返回原始文本对

        Returns:
            包含 zh_text 和 en_text 的字典
        """
        return {
            'zh_text': self.data[index]['zh'],
            'en_text': self.data[index]['en']
        }


if __name__ == '__main__':
    """测试数据集加载"""
    from transformers import T5Tokenizer


    class Args:
        bs = 4

    args = Args()

    # 加载 tokenizer
    print("加载 T5 Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(str(config.T5_MODEL_PATH))

    # 测试数据加载器
    print("创建数据加载器...")
    train_loader = get_t5_dataloader(args, mode='train', tokenizer=tokenizer)

    # 获取一个批次
    print("获取第一个批次...")
    batch = next(iter(train_loader))

    print(f"批次大小: {batch['input_ids'].shape[0]}")
    print(f"输入形状: {batch['input_ids'].shape}")
    print(f"注意力掩码形状: {batch['attention_mask'].shape}")
    print(f"标签形状: {batch['labels'].shape}")

    # 解码查看内容
    print("\n第一个样本:")
    print(f"输入: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
    # 标签中的 -100 需要替换回 pad_token_id 才能解码
    labels = batch['labels'][0].clone()
    labels[labels == -100] = tokenizer.pad_token_id
    print(f"目标: {tokenizer.decode(labels, skip_special_tokens=True)}")
