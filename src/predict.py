import torch
import config
import argparse
from utils import load_model
from tokenizer import ChineseTokenizer, EnglishTokenizer
# 导入 T5 相关模块
from transformers import T5Tokenizer


def predict_with_beam_search(model, inputs, en_tokenizer, encoder_outputs, initial_hidden, device, beam_width=config.BEAM_WIDTH):
    """
    使用Beam Search进行解码

    :param model: 模型实例
    :param inputs: 输入张量 [batch_size, src_seq_len]
    :param en_tokenizer: 英文分词器
    :param encoder_outputs: 编码器输出 [batch_size, src_seq_len, hidden_size]
    :param initial_hidden: 初始隐藏状态 [num_layers, batch_size, hidden_size]
    :param device: 设备
    :param beam_width: beam宽度
    :return: 批次中每个样本的预测序列列表
    """
    batch_size = inputs.shape[0]
    batch_results = []

    # 对批次中的每个样本分别进行beam search
    for batch_idx in range(batch_size):
        # 提取当前样本的编码器输出和隐藏状态
        sample_encoder_outputs = encoder_outputs[batch_idx:batch_idx+1]  # [1, src_seq_len, hidden_size]
        sample_hidden = initial_hidden[:, batch_idx:batch_idx+1, :].contiguous()  # [num_layers, 1, hidden_size]

        # 初始化beam candidates
        # 每个候选: (token_ids, log_prob, hidden_state)
        beams = [(
            [en_tokenizer.sos_token_index],  # 初始序列
            0.0,  # 初始对数概率
            sample_hidden  # 初始隐藏状态
        )]

        completed_beams = []  # 存储已完成的候选序列

        # 自回归生成
        for step in range(config.MAX_SEQ_LENGTH):
            all_candidates = []

            # 对每个beam候选进行扩展
            for token_ids, log_prob, hidden in beams:
                # 如果已经生成了EOS，直接加入完成列表
                if token_ids[-1] == en_tokenizer.eos_token_index:
                    completed_beams.append((token_ids, log_prob))
                    continue

                # 准备解码器输入
                decoder_input = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=device)
                # decoder_input.shape: [1, 1]

                # 解码
                decoder_output, new_hidden = model.decoder(decoder_input, hidden, sample_encoder_outputs)
                # decoder_output.shape: [1, 1, vocab_size]

                # 计算对数概率
                log_probs = torch.log_softmax(decoder_output.squeeze(0).squeeze(0), dim=-1)
                # log_probs.shape: [vocab_size]

                # 获取top-k个最可能的token
                top_k = min(beam_width, log_probs.shape[0])
                top_log_probs, top_indices = torch.topk(log_probs, top_k)

                # 生成所有候选扩展
                for i in range(top_k):
                    new_token = top_indices[i].item()
                    new_log_prob = log_prob + top_log_probs[i].item()
                    new_token_ids = token_ids + [new_token]

                    all_candidates.append((new_token_ids, new_log_prob, new_hidden))

            # 如果没有候选了
            if not all_candidates:
                break

            # 按对数概率排序，选择top beam_width个候选
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # 如果所有beam都生成了EOS，提前结束
            if all(token_ids[-1] == en_tokenizer.eos_token_index for token_ids, _, _ in beams):
                for token_ids, log_prob, _ in beams:
                    if (token_ids, log_prob) not in completed_beams:
                        completed_beams.append((token_ids, log_prob))
                break

        # 将未完成的beam也加入完成列表
        for token_ids, log_prob, _ in beams:
            if token_ids[-1] != en_tokenizer.eos_token_index:
                completed_beams.append((token_ids, log_prob))

        # 选择最佳序列
        if completed_beams:
            best_sequence, _ = max(completed_beams, key=lambda x: x[1])
        else:
            # 如果没有完成的beam，选择当前最佳
            best_sequence = beams[0][0] if beams else [en_tokenizer.sos_token_index]

        # 移除BOS token，如果存在EOS则截断
        result = best_sequence[1:] if best_sequence[0] == en_tokenizer.sos_token_index else best_sequence
        if en_tokenizer.eos_token_index in result:
            eos_pos = result.index(en_tokenizer.eos_token_index)
            result = result[:eos_pos]

        batch_results.append(result)

    return batch_results


def predict_batch(model, inputs, en_tokenizer, src_lengths, args):
    """
    批量预测
    """
    model.eval()
    with torch.no_grad():
        # T5 模型的预测逻辑
        if args.model == 'T5':
            # inputs 是原始中文文本列表
            batch_size = len(inputs)
            device = next(model.parameters()).device

            # 添加任务前缀
            input_texts = [f"translate Chinese to English: {text}" for text in inputs]

            # 编码输入
            encoded_inputs = en_tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.T5_MAX_SOURCE_LENGTH
            )

            # 移动到设备
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

            # 生成翻译
            outputs = model.generate(
                input_ids=encoded_inputs['input_ids'],
                attention_mask=encoded_inputs['attention_mask'],
                max_length=config.T5_MAX_TARGET_LENGTH,
                num_beams=config.T5_NUM_BEAMS,
                early_stopping=True
            )

            # 解码输出到 token ID 列表
            generated_list = []
            for output in outputs:
                # 移除 padding 和特殊 token
                tokens = output.tolist()
                # 移除 pad_token_id
                tokens = [t for t in tokens if t != en_tokenizer.pad_token_id]
                # 移除 eos_token 之后的内容
                if en_tokenizer.eos_token_id in tokens:
                    eos_pos = tokens.index(en_tokenizer.eos_token_id)
                    tokens = tokens[:eos_pos]
                generated_list.append(tokens)

            return generated_list

        # 原有模型的预测逻辑
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
    """
    单个文本预测

    """
    # T5 模型的预测逻辑
    if args.model == 'T5':
        # 直接使用文本列表调用 predict_batch
        batch_result = predict_batch(model, [text], en_tokenizer, None, args)
        # 解码第一个结果
        return en_tokenizer.decode(batch_result[0], skip_special_tokens=True)

    # 原有模型的预测逻辑
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
    """运行交互式预测"""
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

    # 2. 分词器
    if args.model == 'T5':
     
        print("加载 T5 Tokenizer...")
        t5_tokenizer = T5Tokenizer.from_pretrained(str(config.T5_MODEL_PATH))
        zh_tokenizer = None
        en_tokenizer = t5_tokenizer  
    else:
       
        zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
        en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')

    # 3. 模型
    model = load_model(args, zh_tokenizer, en_tokenizer, device)

  
    print("欢迎使用中英翻译模型")
    if args.model == 'T5':
        print(" 当前模型: T5")
    print("输入 q 或 quit 退出")

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
