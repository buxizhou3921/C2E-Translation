import re
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tokenizer import EnglishTokenizer, ChineseTokenizer

def analyze_seq_len(df):
    # 计算长度
    df['zh_len'] = df['zh'].apply(len)  # 字符数
    df['en_len'] = df['en'].apply(lambda x: len(x.split()))  # 单词数

    # 统计分位数
    for q in [0.95, 0.98, 0.99]:
        print(f"中文长度{q}分位数: {np.percentile(df['zh_len'], q * 100)}")
        print(f"英文长度{q}分位数: {np.percentile(df['en_len'], q * 100)}")

    # 绘制直方图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df['zh_len'], bins=50)
    axes[0].set_title('Chinese Text Length')
    axes[1].hist(df['en_len'], bins=50)
    axes[1].set_title('English Text Length')
    plt.show()


def clean_text(text, language):
    """ 文本清洗 """
    if language == 'zh':
        cleaned = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff01-\uff1f\s]+', '', str(text))
    elif language == 'en':
        cleaned = re.sub(r'[^a-zA-Z!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~\s]+', '', str(text)).lower()
    else:
        raise ValueError(f"Invalid language: {language}. Use 'zh' or 'en'!")

    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def preprocess_df(df, filter_long=True):
    """
    预处理数据

    Args:
        df (pd.DataFrame): 原始数据
        filter_long (bool): 是否过滤过长句子

    Returns:
        pd.DataFrame: 预处理后的数据
    """

    # 文本清洗
    df['zh'] = df['zh'].apply(clean_text, language='zh')
    df['en'] = df['en'].apply(clean_text, language='en')

    # 过滤空行
    df = df[(df['zh'].str.len() > 0) & (df['en'].str.len() > 0)]

    # 截断过长句子
    if filter_long:
        zh_lengths = df['zh'].str.len()
        en_word_counts = df['en'].str.split().str.len()
        df = df[(zh_lengths <= config.MAX_ZH_LENGTH) & (en_word_counts <= config.MAX_EN_LENGTH)]

    return df


def tokenize_dataset(df, zh_tokenizer, en_tokenizer, dataset_name):
    """
    tokenize单个数据集（训练集、验证集或测试集）

    Args:
        df: 数据集
        zh_tokenizer: 中文分词器
        en_tokenizer: 英文分词器
        dataset_name: 数据集名称（保存）

    Returns:
        tokenize后的数据集
    """

    df['zh'] = df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    df['en'] = df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    df.to_json(config.PROCESSED_DATA_DIR / f'{dataset_name}.jsonl', orient='records', lines=True)
    return df


def process():
    print("开始处理数据")

    # 读取文件
    train_df = pd.read_json(config.RAW_DATA_DIR / 'train_100k.jsonl', lines=True,
                            orient='records', encoding='utf-8').dropna()
    valid_df = pd.read_json(config.RAW_DATA_DIR / 'valid.jsonl', lines=True,
                            orient='records', encoding='utf-8').dropna()
    test_df = pd.read_json(config.RAW_DATA_DIR / 'test.jsonl', lines=True,
                            orient='records', encoding='utf-8').dropna()

    # 数据预处理
    train_df = preprocess_df(train_df, filter_long=True)
    valid_df = preprocess_df(valid_df, filter_long=False)
    test_df = preprocess_df(test_df, filter_long=False)

    # 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.VOCAB_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.VOCAB_DIR / 'en_vocab.txt')

    # 构建Tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')

    # 处理各个数据集
    tokenize_dataset(train_df, zh_tokenizer, en_tokenizer, "train")
    tokenize_dataset(valid_df, zh_tokenizer, en_tokenizer, "valid")
    tokenize_dataset(test_df, zh_tokenizer, en_tokenizer, "test")

    print("处理数据完成")


if __name__ == '__main__':
    # train_df = pd.read_json(config.RAW_DATA_DIR / 'train_10k.jsonl', lines=True,
    #                         orient='records', encoding='utf-8').dropna()
    # valid_df = pd.read_json(config.RAW_DATA_DIR / 'valid.jsonl', lines=True,
    #                         orient='records', encoding='utf-8').dropna()
    # test_df = pd.read_json(config.RAW_DATA_DIR / 'test.jsonl', lines=True,
    #                        orient='records', encoding='utf-8').dropna()
    # analyze_seq_len(train_df)
    # analyze_seq_len(valid_df)
    # analyze_seq_len(test_df)
    process()
