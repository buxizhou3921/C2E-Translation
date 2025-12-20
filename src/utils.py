import sys
from gru_seq2seq import GRUModel
from gru_attention import GRUAttentionModel


def get_model(args, zh_tokenizer, en_tokenizer, device):
    if args.model == 'gru_seq2seq':
        model = GRUModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index).to(device)
    elif args.model == 'gru_attention':
        model = GRUAttentionModel(zh_tokenizer.vocab_list,
                         zh_tokenizer.vocab_size,
                         en_tokenizer.vocab_list,
                         en_tokenizer.vocab_size,
                         zh_tokenizer.pad_token_index,
                         en_tokenizer.pad_token_index).to(device)
    else:
        print('the model name you have entered is not supported yet')
        sys.exit()

    return model