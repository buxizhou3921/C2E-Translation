import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import config
import argparse
from utils import load_model
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer
# 导入 T5 相关模块
from transformers import T5Tokenizer


def evaluate(model, dataloader, device, tokenizer_for_predict, args):
    """
    评估模型性能

    Args:
        model: 模型实例
        dataloader: 数据加载器
        device: 设备
        tokenizer_for_predict: 用于预测的分词器（T5使用T5Tokenizer，其他使用EnglishTokenizer）
        args: 命令行参数

    Returns:
        BLEU 分数
    """
    predictions = []
    # predictions: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]
    references = []
    # references: [[[*,*,*,*,*]],[[*,*,*,*]],[[*,*,*]]]

    # 所有模型使用相同的评估逻辑
    for inputs, targets, src_lengths in dataloader:
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.tolist()
        # targets: [[sos,*,*,*,*,*,eos],[sos,*,*,*,*,eos,pad],[sos,*,*,*,eos,pad,pad]]
        batch_result = predict_batch(model, inputs, tokenizer_for_predict, src_lengths, args)
        # batch_result: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]

        batch_references = []
        for target in targets:
            # Find EOS index; if not found, use full length
            # 对于T5，使用EnglishTokenizer的eos_token_index来找参考答案
            # 因为target是用EnglishTokenizer编码的
            try:
                if args.model == 'T5':
                    from tokenizer import EnglishTokenizer
                    en_tok_temp = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
                    eos_idx = target.index(en_tok_temp.eos_token_index)
                else:
                    eos_idx = target.index(tokenizer_for_predict.eos_token_index)
            except ValueError:
                eos_idx = len(target)
            ref_tokens = target[1:eos_idx]  # remove <sos>
            batch_references.append([ref_tokens])

        predictions.extend(batch_result)
        references.extend(batch_references)

    smooth = SmoothingFunction().method4
    return corpus_bleu(references, predictions, smoothing_function=smooth)


def run_evaluate():
    """运行评估"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='Model type')
    parser.add_argument('-align', type=str, default='dot',
                        help='Attention alignment function (dot/mul/add)')
    parser.add_argument('-train', type=str, default='teacher', help='Training policy (teacher/free)')
    parser.add_argument('-decode', type=str, default='greedy', help='Decoding policy (greedy/beam-search)')
    parser.add_argument('-position', type=str, default='absolute',
                        help='Position embedding scheme (absolute/relative)')
    parser.add_argument('-norm', type=str, default='layer', help='Normalization method (layer/rms)')

    parser.add_argument('-bs', type=int, default=128, help='batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-heads', type=int, default=2, help='number of attention heads')
    parser.add_argument('-layers', type=int, default=4, help='number of layers')
    args = parser.parse_args()

    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 分词器（根据模型类型选择）
    if args.model == 'T5':
        # T5 tokenizer 预测
        print(" 加载 T5 Tokenizer...")
        t5_tokenizer = T5Tokenizer.from_pretrained(str(config.T5_MODEL_PATH))

        # 加载中英文分词器以使用预训练向量（用于模型初始化）
        zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
        en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')

        # 使用传统的数据加载器（与其他模型保持一致）
        test_dataloader = get_dataloader(args, 'test')

        # 用于预测的tokenizer
        predict_tokenizer = t5_tokenizer
    else:
        # 其他模型使用传统分词器和数据加载器
        zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
        en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
        test_dataloader = get_dataloader(args, 'test')

        # 用于预测的tokenizer
        predict_tokenizer = en_tokenizer

    # 3. 模型
    model = load_model(args, zh_tokenizer, en_tokenizer, device)

    # 4. 评估逻辑
    bleu = evaluate(model, test_dataloader, device, predict_tokenizer, args)
    print("评估结果")
    print(f"BLEU Score: {bleu:.4f}")


if __name__ == '__main__':
    run_evaluate()
