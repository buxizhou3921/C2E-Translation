import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import config
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer


def evaluate(model, dataloader, device, en_tokenizer):
    predictions = []
    # predictions: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]
    references = []
    # references: [[[*,*,*,*,*]],[[*,*,*,*]],[[*,*,*]]]
    for inputs, targets, src_lengths in dataloader:
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.tolist()
        # targets: [[sos,*,*,*,*,*,eos],[sos,*,*,*,*,eos,pad],[sos,*,*,*,eos,pad,pad]]
        batch_result = predict_batch(model, inputs, en_tokenizer, src_lengths)
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
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
    print("词表加载成功")

    # 3. 模型
    print("模型加载较缓慢，请耐心等待...")
    model = TranslationModel(zh_tokenizer.vocab_list,
                             zh_tokenizer.vocab_size,
                             en_tokenizer.vocab_list,
                             en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINTS_GRU_DIR / 'best.pth'))
    print("模型加载成功")

    # 4. 数据集
    test_dataloader = get_dataloader('test')

    # 5.评估逻辑
    bleu = evaluate(model, test_dataloader, device, en_tokenizer)
    print("评估结果")
    print(f"bleu: {bleu}")


if __name__ == '__main__':
    run_evaluate()
