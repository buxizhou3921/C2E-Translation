import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import config
import argparse
from utils import load_model
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer


def evaluate(model, dataloader, device, en_tokenizer, args):
    predictions = []
    # predictions: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]
    references = []
    # references: [[[*,*,*,*,*]],[[*,*,*,*]],[[*,*,*]]]
    for inputs, targets, src_lengths in dataloader:
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.tolist()
        # targets: [[sos,*,*,*,*,*,eos],[sos,*,*,*,*,eos,pad],[sos,*,*,*,eos,pad,pad]]
        batch_result = predict_batch(model, inputs, en_tokenizer, src_lengths, args)
        # batch_result: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]

        batch_references = []
        for target in targets:
            # Find EOS index; if not found, use full length
            try:
                eos_idx = target.index(en_tokenizer.eos_token_index)
            except ValueError:
                eos_idx = len(target)
            ref_tokens = target[1:eos_idx]  # remove <sos>
            batch_references.append([ref_tokens])

        predictions.extend(batch_result)
        references.extend(batch_references)
    smooth = SmoothingFunction().method4
    return corpus_bleu(references, predictions, smoothing_function=smooth)


def run_evaluate():
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
    parser.add_argument('-heads', type=int, default=2, help='learning rate')
    parser.add_argument('-layers', type=int, default=4, help='learning rate')
    args = parser.parse_args()

    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')

    # 3. 模型
    model = load_model(args, zh_tokenizer, en_tokenizer, device)

    # 4. 数据集
    test_dataloader = get_dataloader(args, 'test')

    # 5.评估逻辑
    bleu = evaluate(model, test_dataloader, device, en_tokenizer, args)
    print("评估结果")
    print(f"bleu: {bleu}")


if __name__ == '__main__':
    run_evaluate()
