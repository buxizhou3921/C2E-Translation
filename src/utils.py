import sys
import torch
import config
from gru_seq2seq import GRUModel
from gru_attention import GRUAttentionModel
from transformer import TransformerModel
from t5_model import T5Model  # 导入 T5 模型


def get_model(args, zh_tokenizer, en_tokenizer, device):
    if args.model == 'gru_seq2seq':
        model = GRUModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index,
                         args).to(device)
    elif args.model == 'gru_attention':
        model = GRUAttentionModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index,
                         args).to(device)
    elif args.model == 'transformer':
        model = TransformerModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index,
                         args).to(device)
    elif args.model == 'T5':
        # T5 模型初始化
        # 传入词表以使用缓存的预训练向量
        model = T5Model(
            model_name_or_path=config.T5_MODEL_PATH,
            args=args,
            zh_vocab_list=zh_tokenizer.vocab_list if zh_tokenizer is not None else None,
            en_vocab_list=en_tokenizer.vocab_list if en_tokenizer is not None else None
        ).to(device)
    else:
        print('the model name you have entered is not supported yet')
        sys.exit()

    return model


def load_model(args, zh_tokenizer, en_tokenizer, device):
    model = get_model(args, zh_tokenizer, en_tokenizer, device)
    load_path = config.CHECKPOINTS_DIR / args.model
    if args.model == 'gru_seq2seq':
        load_path = load_path / f"{args.train}_best.pth"
    elif args.model == 'gru_attention':
        load_path = load_path / f"{args.train}_{args.align}_best.pth"
    elif args.model == 'transformer':
        lr_str = f"{args.lr:.1e}".replace(".0", "")
        load_path = load_path / f"{args.position}_{args.norm}_{args.bs}_{lr_str}_{args.heads}_{args.layers}_best.pth"
    elif args.model == 'T5':
        # T5 模型加载路径
        load_path = load_path / "best_model"
    else:
        print('the model name you have entered is not supported yet')
        sys.exit()

    # T5 模型使用特殊的加载方式
    if args.model == 'T5':
        # T5 使用 from_pretrained 方法加载整个模型，并传入 vocab 以使用预训练向量
        model = T5Model.from_pretrained(
            load_path,
            args=args,
            zh_vocab_list=zh_tokenizer.vocab_list if zh_tokenizer is not None else None,
            en_vocab_list=en_tokenizer.vocab_list if en_tokenizer is not None else None
        ).to(device)
    else:
        # 其他模型使用 load_state_dict
        model.load_state_dict(torch.load(load_path, weights_only=True))

    return model
