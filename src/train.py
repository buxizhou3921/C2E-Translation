import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
from model import TranslationModel
from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
from evaluate import evaluate


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    epoch_start_time = time.time()
    for inputs, targets, src_lengths in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]

        # 编码阶段
        context_vector = model.encoder(encoder_inputs, src_lengths)
        # context_vector.shape: [num_layers, batch_size, hidden_size]

        # 解码阶段
        decoder_hidden = context_vector
        decoder_outputs = []
        seq_len = decoder_inputs.shape[1]
        for i in range(seq_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size, 1, vocab_size]
            decoder_outputs.append(decoder_output)

        # decoder_outputs：[tensor([batch_size,1,vocab_size])] -> [batch_size * seq_len, vocab_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs.shape: [batch_size ,seq_len, vocab_size]
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

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    return total_loss / len(dataloader), epoch_duration


def train():
    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    train_dataloader = get_dataloader('train')
    valid_dataloader = get_dataloader('valid')
    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.VOCAB_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.VOCAB_DIR / 'en_vocab.txt')
    # 4. 模型
    model = TranslationModel(zh_tokenizer.vocab_list,
                             zh_tokenizer.vocab_size,
                             en_tokenizer.vocab_list,
                             en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 记录总耗时
    total_start_time = time.time()
    # best_loss = float('inf')
    best_bleu = 0
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        loss, epoch_duration = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
        writer.add_scalar('Loss', loss, epoch)

        bleu = evaluate(model, valid_dataloader, device, en_tokenizer)
        writer.add_scalar('bleu', bleu, epoch)
        print(f"Loss: {loss:.4f} | bleu: {bleu:.4f} | time: {int(epoch_duration // 60):.2f} min {epoch_duration % 60:.2f} s")

        # 保存模型
        # if loss < best_loss:
        #     best_loss = loss
        if bleu > best_bleu and epoch % 2 == 0:
            best_bleu = bleu
            torch.save(model.state_dict(), config.CHECKPOINTS_DIR / 'best.pth')
            print('保存模型')

    writer.close()
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'总耗时: {int(total_duration // 60):.2f} min {total_duration % 60:.2f} s')


if __name__ == '__main__':
    train()
