import os
import jieba
from tqdm import tqdm
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer


class BaseTokenizer:
    pad_token = '<pad>'
    unk_token = '<unk>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        self.pad_token_index = self.word2index[self.pad_token]
        self.unk_token_index = self.word2index[self.unk_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def tokenize(cls, text) -> list[str]:
        pass

    def encode(self, text, add_sos_eos=False):
        tokens = self.tokenize(text)

        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]

        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):

        # 统计词频
        vocab_counter = {}
        for sentence in tqdm(sentences, desc="构建词表"):
            tokens = cls.tokenize(sentence)
            for token in tokens:
                vocab_counter[token] = vocab_counter.get(token, 0) + 1

        # 过滤低频词并构建词表
        min_freq = 2
        special_tokens = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token]
        vocab_list = special_tokens + [
            token for token, freq in vocab_counter.items()
            if freq >= min_freq and token.strip() != '' ]

        print(f'词表大小:{len(vocab_list)}')

        # 5.保存词表
        if os.path.exists(vocab_path):
            os.remove(vocab_path)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return jieba.lcut(text)


class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    def decode(self, indexes):
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)


if __name__ == '__main__':
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    word_list = tokenizer.tokenize(
        'On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.')
    print(word_list)
    print(detokenizer.detokenize(word_list))
