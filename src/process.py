import config
import pandas as pd
from tokenizer import EnglishTokenizer, ChineseTokenizer


def process_dataset(df, zh_tokenizer, en_tokenizer, dataset_name):
    """
    处理单个数据集（训练集、验证集或测试集）

    Args:
        df: 数据集
        zh_tokenizer: 中文分词器
        en_tokenizer: 英文分词器
        dataset_name: 数据集名称（保存）

    Returns:
        处理后的数据集
    """

    df['zh'] = df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    df['en'] = df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    df.to_json(config.PROCESSED_DATA_DIR / f'{dataset_name}.jsonl', orient='records', lines=True)
    return df


def process():
    print("开始处理数据")

    # 读取文件
    train_df = pd.read_json(config.RAW_DATA_DIR / 'train_10k.jsonl', lines=True,
                            orient='records', encoding='utf-8').dropna()
    valid_df = pd.read_json(config.RAW_DATA_DIR / 'valid.jsonl', lines=True,
                            orient='records', encoding='utf-8').dropna()
    test_df = pd.read_json(config.RAW_DATA_DIR / 'test.jsonl', lines=True,
                            orient='records', encoding='utf-8').dropna()

    # TODO: Remove illegal characters and filter out rare words; filter or truncate excessively long sentences

    # 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.CHECKPOINTS_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.CHECKPOINTS_DIR / 'en_vocab.txt')

    # 构建Tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.CHECKPOINTS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.CHECKPOINTS_DIR / 'en_vocab.txt')

    # 处理各个数据集
    process_dataset(train_df, zh_tokenizer, en_tokenizer, "train")
    process_dataset(valid_df, zh_tokenizer, en_tokenizer, "valid")
    process_dataset(test_df, zh_tokenizer, en_tokenizer, "test")

    print("处理数据完成")


if __name__ == '__main__':
    process()
