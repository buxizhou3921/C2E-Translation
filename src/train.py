import time
import torch
import argparse
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import config
from dataset import get_dataloader
from utils import get_model
from tokenizer import ChineseTokenizer, EnglishTokenizer
from evaluate import evaluate


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, model_type):
    total_loss = 0
    model.train()
    for inputs, targets, src_lengths in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]

        # 前向传播
        if model_type in ['gru_seq2seq', 'gru_attention']:
            # 编码阶段
            encoder_outputs, context_vector = model.encoder(encoder_inputs, src_lengths)
            # context_vector.shape: [num_layers, batch_size, hidden_size]

            # 解码阶段: Teacher Forcing
            # TODO: Free Running
            decoder_hidden = context_vector
            decoder_outputs = []
            seq_len = decoder_inputs.shape[1]
            for i in range(seq_len):
                decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # decoder_output.shape: [batch_size, 1, vocab_size]
                decoder_outputs.append(decoder_output)

            # decoder_outputs：[tensor([batch_size,1,vocab_size])] -> [batch_size * seq_len, vocab_size]
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            # decoder_outputs.shape: [batch_size ,seq_len, vocab_size]
        elif model_type == 'transformer':
            src_pad_mask = (encoder_inputs == model.zh_embedding.padding_idx)
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1])
            decoder_outputs = model(encoder_inputs, decoder_inputs, src_pad_mask, tgt_mask)
            # decoder_outputs.shape: [batch_size, seq_len, en_vocab_size]
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

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
                        help='Attention alignment function (dot/multiplicative/additive)')
    parser.add_argument('-train', type=str, default='teacher', help='Training policy (teacher/free)')
    parser.add_argument('-decode', type=str, default='greedy', help='Decoding policy (greedy/beam-search)')
    args = parser.parse_args()

    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    train_dataloader = get_dataloader('train')
    # valid_dataloader = get_dataloader('valid')
    test_dataloader = get_dataloader('test')
    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
    # 4. 模型
    print("模型加载较缓慢，请耐心等待...")
    model = get_model(args, zh_tokenizer, en_tokenizer, device)
    print("模型加载成功")
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    # 6. 优化器 and 调度器
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    # 7. TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / args.model / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # best_loss = float('inf')
    best_bleu = 0
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device, args.model)
        bleu = evaluate(model, test_dataloader, device, en_tokenizer, args.model)
        # 更新学习率
        scheduler.step()
        # 记录过程量
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('bleu', bleu, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        print(f"Loss: {loss:.4f} | bleu: {bleu:.4f} | lr: {current_lr}")

        # 保存模型
        # if loss < best_loss:
        #     best_loss = loss
        if bleu > best_bleu and epoch % 2 == 0:
            best_bleu = bleu
            torch.save(model.state_dict(), config.CHECKPOINTS_DIR / args.model / 'best.pth')
            print('保存模型')

    writer.close()


if __name__ == '__main__':
    train()
