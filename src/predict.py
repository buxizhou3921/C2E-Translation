import torch
import config
import argparse
from utils import get_model
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, en_tokenizer, src_lengths, model_type):
    model.eval()
    with torch.no_grad():
        # 编码
        if model_type in ['gru_seq2seq', 'gru_attention']:
            encoder_outputs, context_vector = model.encoder(inputs, src_lengths)
            # 隐藏状态
            decoder_hidden = context_vector
            # decoder_hidden.shape: [num_layers, batch_size, hidden_size]
        elif model_type == 'transformer':
            src_pad_mask = (inputs == model.zh_embedding.padding_idx)
            memory = model.encode(inputs, src_pad_mask)
            # memory.shape: [batch_size, src_seq_len, d_model]

        # 解码
        batch_size = inputs.shape[0]
        device = inputs.device

        decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)
        # decoder_input.shape: [batch_size, tgt_seq_len]

        # 预测结果缓存
        generated = []

        # 记录每个样本是否已经生成结束符
        is_finished = torch.full([batch_size], False, device=device)

        # 自回归生成: greedy
        # TODO: beam-search
        for i in range(config.MAX_SEQ_LENGTH):
            if model_type in ['gru_seq2seq', 'gru_attention']:
                # 解码
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # decoder_output.shape: [batch_size, 1, vocab_size]
                next_token_indexes = torch.argmax(decoder_output, dim=-1)
                # next_token_indexes.shape: [batch_size, 1]

                # 更新输入(decoder_input)
                decoder_input = next_token_indexes
            elif model_type == 'transformer':
                tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1])
                decoder_output = model.decode(decoder_input, memory, tgt_mask, src_pad_mask)
                # decoder_output.shape: [batch_size, tgt_seq_len, en_vocab_size]
                next_token_indexes = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)
                # next_token_indexes.shape: [batch_size, 1]

                # 更新输入(decoder_input)
                decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)

            # 保存预测结果
            generated.append(next_token_indexes)

            # 判断是否应该结束
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all():
                break

        # 整理预测结果形状
        # generated：[tensor([batch_size, 1])]
        generated_tensor = torch.cat(generated, dim=1)
        # generated_tensor.shape: [batch_size,seq_len]
        generated_list = generated_tensor.tolist()
        # generated_list：[[*,*,*,*,*],[*,*,*,eos,*],[*,*,eos,*,*]]

        # 去掉eos之后的token id
        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        # generated_list：[[*,*,*,*,*],[*,*,*],[*,*]]
        return generated_list


def predict(text, model, zh_tokenizer, en_tokenizer, device, model_type):
    # 1. 处理输入
    indexes = zh_tokenizer.encode(text)
    src_lengths = torch.tensor([len(indexes)], dtype=torch.long)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [1, seq_len]

    # 2.预测逻辑
    batch_result = predict_batch(model, input_tensor, en_tokenizer, src_lengths, model_type)
    return en_tokenizer.decode(batch_result[0])


def run_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='Model type')
    parser.add_argument('-align', type=str, default='dot',
                        help='Attention alignment function (dot/multiplicative/additive)')
    parser.add_argument('-decode', type=str, default='greedy', help='Decoding policy (greedy/beam-search)')
    args = parser.parse_args()

    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')

    # 3. 模型
    model = get_model(args, zh_tokenizer, en_tokenizer, device)
    model.load_state_dict(torch.load(config.CHECKPOINTS_DIR / args.model / 'best.pth', weights_only=True))

    print("欢迎使用中英翻译模型(输入q或者quit退出)")
    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device, args.model)
        print("英文：", result)


if __name__ == '__main__':
    run_predict()
