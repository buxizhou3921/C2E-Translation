import time
import torch
import argparse
from tensorboardX import SummaryWriter
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import config
from dataset import get_dataloader
from utils import get_model
from tokenizer import ChineseTokenizer, EnglishTokenizer
from evaluate import evaluate
# 导入 T5 相关模块
from t5_dataset import get_t5_dataloader
from transformers import T5Tokenizer


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, args):
    """
    训练一个 epoch

    Args:
        model: 模型实例
        dataloader: 数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        device: 设备
        args: 命令行参数

    Returns:
        平均损失值
    """
    total_loss = 0
    model.train()

    # T5 模型使用不同的训练循环
    if args.model == 'T5':
        # T5 模型的训练循环
        gradient_accumulation_steps = config.T5_GRADIENT_ACCUMULATION_STEPS
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc='训练')):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播（T5 模型内部会计算损失）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 获取损失
            loss = outputs.loss

            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # 每 gradient_accumulation_steps 步更新一次参数
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

        # 处理最后不足一个累积步数的情况
        if (step + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        return total_loss / len(dataloader)

    # 其他模型的训练循环
    for inputs, targets, src_lengths in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]

        # 前向传播
        if args.model in ['gru_seq2seq', 'gru_attention']:
            # 编码阶段
            encoder_outputs, context_vector = model.encoder(encoder_inputs, src_lengths)
            # context_vector.shape: [num_layers, batch_size, hidden_size]

            # 解码阶段
            decoder_hidden = context_vector
            decoder_outputs = []
            seq_len = decoder_inputs.shape[1]

            if args.train == 'teacher': # Teacher Forcing
                for i in range(seq_len):
                    decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # decoder_output.shape: [batch_size, 1, vocab_size]
                    decoder_outputs.append(decoder_output)
            elif args.train == 'free': # Free Running
                decoder_input = decoder_inputs[:, 0].unsqueeze(1)  # 第一个输入 <sos>
                for _ in range(seq_len):
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # decoder_output.shape: [batch_size, 1, vocab_size]
                    decoder_outputs.append(decoder_output)
                    # 使用预测结果概率最高的词作为下一个时间步的输入
                    next_input_indices = torch.argmax(decoder_output, dim=-1)  # [batch_size, 1]
                    decoder_input = next_input_indices

            # decoder_outputs：[tensor([batch_size,1,vocab_size])] -> [batch_size * seq_len, vocab_size]
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            # decoder_outputs.shape: [batch_size ,seq_len, vocab_size]

        elif args.model == 'transformer':
            src_pad_mask = (encoder_inputs == model.zh_embedding.padding_idx)
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1])
            decoder_outputs = model(encoder_inputs, decoder_inputs, src_pad_mask, tgt_mask)
            # decoder_outputs.shape: [batch_size, seq_len, en_vocab_size]
        else:
            raise ValueError(f"未知的模型类型: {args.model}")

        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_outputs.shape: [batch_size * seq_len, vocab_size]

        # decoder_targets：[batch_size, seq_len] -> [batch_size * seq_len]
        decoder_targets = decoder_targets.reshape(-1)

        loss = loss_fn(decoder_outputs, decoder_targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
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

    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 数据加载器和分词器
    if args.model == 'T5':
        # T5 使用自己的 tokenizer 和数据加载器
        print("加载 T5 Tokenizer 和数据集...")
        t5_tokenizer = T5Tokenizer.from_pretrained(str(config.T5_MODEL_PATH))

        # 覆盖批次大小为 T5 推荐值
        if args.bs == 128:  # 如果是默认值，使用 T5 配置
            args.bs = config.T5_BATCH_SIZE

        train_dataloader = get_t5_dataloader(args, 'train', tokenizer=t5_tokenizer)
        test_dataloader = get_t5_dataloader(args, 'valid', tokenizer=t5_tokenizer)

        # T5 也需要加载中英文分词器以获取 vocab_list 用于预训练向量
        zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
        en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
    else:
        # 其他模型使用传统数据加载器
        train_dataloader = get_dataloader(args, 'train')
        test_dataloader = get_dataloader(args, 'valid')

        # 加载中英文分词器
        zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
        en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')

    # 3. 模型
    model = get_model(args, zh_tokenizer, en_tokenizer, device)

    # 4. 损失函数（T5 模型内部计算损失，不需要外部损失函数）
    if args.model == 'T5':
        loss_fn = None  # T5 不需要外部损失函数
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

    # 5. 优化器和调度器
    if args.model == 'T5':
        # T5 使用 AdamW 优化器和推荐的学习率
        if args.lr == 1e-3:  # 如果是默认值，使用 T5 配置
            args.lr = config.T5_LEARNING_RATE
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    # 6. TensorBoard Writer
    log_dir = config.LOGS_DIR / args.model
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    lr_str = f"{args.lr:.1e}".replace(".0", "")
    if args.model == 'gru_seq2seq':
        log_dir = log_dir / f"{args.train}_{timestamp}"
    elif args.model == 'gru_attention':
        log_dir = log_dir / f"{args.train}_{args.align}_{timestamp}"
    elif args.model == 'transformer':
        log_dir = log_dir / f"{args.position}_{args.norm}_{args.bs}_{lr_str}_{args.heads}_{args.layers}_{timestamp}"
    elif args.model == 'T5':
        log_dir = log_dir / f"T5_bs{args.bs}_{lr_str}_{timestamp}"
    else:
        raise ValueError(f"未知的模型类型: {args.model}")
    writer = SummaryWriter(log_dir=log_dir)

    # 7. 训练循环
    best_bleu = 0
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device, args)

        # 评估（根据模型类型选择评估方式）
        if args.model == 'T5':
            # T5 需要传入 tokenizer
            bleu = evaluate(model, test_dataloader, device, t5_tokenizer, args)
        else:
            bleu = evaluate(model, test_dataloader, device, en_tokenizer, args)

        # 更新学习率
        scheduler.step()

        # 记录过程量
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('bleu', bleu, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        print(f"Loss: {loss:.4f} | bleu: {bleu:.4f} | lr: {current_lr}")

        # 保存模型
        if bleu > best_bleu and epoch % 2 == 0:
            best_bleu = bleu
            save_path = config.CHECKPOINTS_DIR / args.model
            if args.model == 'gru_seq2seq':
                save_path = save_path / f"{args.train}_best.pth"
            elif args.model == 'gru_attention':
                save_path = save_path / f"{args.train}_{args.align}_best.pth"
            elif args.model == 'transformer':
                lr_str = f"{args.lr:.1e}".replace(".0", "")
                save_path = save_path / f"{args.position}_{args.norm}_{args.bs}_{lr_str}_{args.heads}_{args.layers}_best.pth"
            elif args.model == 'T5':
                # T5 模型保存整个模型和 tokenizer
                save_path = save_path / "best_model"
                model.save_pretrained(save_path)
                print(f'T5 模型已保存到 {save_path}')
                continue  # 跳过 torch.save，因为已经保存了
            else:
                raise ValueError(f"未知的模型类型: {args.model}")

            # 其他模型使用 torch.save
            if args.model != 'T5':
                torch.save(model.state_dict(), save_path)
                print('保存模型')

    writer.close()


if __name__ == '__main__':
    train()
