import torch
import config
import argparse
from utils import load_model
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_with_beam_search(model, inputs, en_tokenizer, encoder_outputs, initial_hidden, device):
    batch_size = inputs.size(0)
    beam_width = config.BEAM_WIDTH

    results = []
    for batch_idx in range(batch_size):
        # 获取当前样本的encoder输出
        encoder_out = encoder_outputs[batch_idx:batch_idx + 1]  # [1, seq_len, hidden_size]
        hidden = initial_hidden[:, batch_idx:batch_idx + 1, :]  # [num_layers, 1, hidden_size]

        # 初始化beams: (score, sequence, hidden_state)
        beams = [(0.0, [en_tokenizer.sos_token_index], hidden)]

        for _ in range(config.MAX_SEQ_LENGTH):
            all_candidates = []

            # 扩展每个beam
            for score, seq, hid in beams:
                if seq[-1] == en_tokenizer.eos_token_index:
                    # 如果已结束，直接添加到候选
                    all_candidates.append((score, seq, hid))
                    continue

                # 获取最后一个token
                last_token = torch.tensor([[seq[-1]]], device=device)

                # 预测下一个token
                with torch.no_grad():  # 避免不必要的梯度计算
                    output, new_hidden = model.decoder(last_token, hid.contiguous(), encoder_out)
                # output: [1, 1, vocab_size]

                # 获取log概率
                log_probs = torch.log_softmax(output[0, 0, :], dim=-1)

                # 获取top-k概率和索引
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)

                # 为当前beam生成候选
                for i in range(beam_width):
                    token_id = top_k_indices[i].item()
                    new_score = score + top_k_probs[i].item()
                    new_seq = seq + [token_id]
                    # 为每个扩展创建新的hidden state副本
                    all_candidates.append((new_score, new_seq, new_hidden.clone()))

            if not all_candidates:
                break

            # 选择top-k个候选作为新的beams
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_width]

            # 检查是否所有beam都已完成
            if all(seq[-1] == en_tokenizer.eos_token_index for _, seq, _ in beams):
                break

        # 选择得分最高的序列
        best_seq = beams[0][1]  # 获取最高分的序列
        # 去掉SOS token并移除EOS之后的内容
        if en_tokenizer.eos_token_index in best_seq:
            eos_pos = best_seq.index(en_tokenizer.eos_token_index)
            best_seq = best_seq[1:eos_pos]  # 去掉SOS，保留到EOS之前
        else:
            best_seq = best_seq[1:]  # 去掉SOS

        results.append(best_seq)

    return results


def predict_batch(model, inputs, en_tokenizer, src_lengths, args):
    model.eval()
    with torch.no_grad():
        # 编码
        if args.model in ['gru_seq2seq', 'gru_attention']:
            encoder_outputs, context_vector = model.encoder(inputs, src_lengths)
            # 隐藏状态
            decoder_hidden = context_vector
            # decoder_hidden.shape: [num_layers, batch_size, hidden_size]
        elif args.model == 'transformer':
            src_pad_mask = (inputs == model.zh_embedding.padding_idx)
            memory = model.encode(inputs, src_pad_mask)
            # memory.shape: [batch_size, src_seq_len, d_model]

        # 解码
        batch_size = inputs.shape[0]
        device = inputs.device

        if args.decode == 'beam-search' and args.model in ['gru_seq2seq', 'gru_attention']:
            return predict_with_beam_search(model, inputs, en_tokenizer, encoder_outputs, decoder_hidden, device)
        elif args.decode == 'greedy':
            decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)
            # decoder_input.shape: [batch_size, tgt_seq_len]

            # 预测结果缓存
            generated = []

            # 记录每个样本是否已经生成结束符
            is_finished = torch.full([batch_size], False, device=device)

            # 自回归生成
            for _ in range(config.MAX_SEQ_LENGTH):
                if args.model in ['gru_seq2seq', 'gru_attention']:
                    # 解码
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # decoder_output.shape: [batch_size, 1, vocab_size]
                    next_token_indexes = torch.argmax(decoder_output, dim=-1)
                    # next_token_indexes.shape: [batch_size, 1]

                    # 更新输入(decoder_input)
                    decoder_input = next_token_indexes
                elif args.model == 'transformer':
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
        else:
            raise ValueError('the decode name you have entered is not supported yet')


def predict(text, model, zh_tokenizer, en_tokenizer, device, args):
    # 1. 处理输入
    indexes = zh_tokenizer.encode(text)
    src_lengths = torch.tensor([len(indexes)], dtype=torch.long)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [1, seq_len]

    # 2.预测逻辑
    batch_result = predict_batch(model, input_tensor, en_tokenizer, src_lengths, args)
    return en_tokenizer.decode(batch_result[0])


def run_predict():
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

    # 2.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')

    # 3. 模型
    model = load_model(args, zh_tokenizer, en_tokenizer, device)

    print("欢迎使用中英翻译模型(输入q或者quit退出)")
    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device, args)
        print("英文：", result)


if __name__ == '__main__':
    run_predict()
