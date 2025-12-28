import sys
import torch
import config
from gru_seq2seq import GRUModel
from gru_attention import GRUAttentionModel
from transformer import TransformerModel


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
    else:
        print('the model name you have entered is not supported yet')
        sys.exit()

    model.load_state_dict(torch.load(load_path, weights_only=True))
    return model
