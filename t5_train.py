"""
T5 模型训练脚本
训练一轮并进行评估
"""
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import T5Tokenizer
from torch.optim import AdamW
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import argparse

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config
from t5_model import T5Model, T5TranslationWrapper
from t5_dataset import get_t5_dataloader, t5_collate_fn
from torch.utils.data import Subset, DataLoader


class T5Trainer:
    """T5 训练器"""

    def __init__(self, model, tokenizer, device, args):
        """
        初始化训练器


        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate
        )

        self.wrapper = T5TranslationWrapper(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            task_prefix="translate Chinese to English: "
        )

    def train_epoch(self, train_loader, epoch):

        self.model.train()
        total_loss = 0
        num_batches = 0

        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # 梯度累积
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()

            # 每隔若干步更新一次参数
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 统计
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item() * self.args.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, valid_loader):
        """
        在验证集上评估模型

        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        predictions = []
        references = []

        print("\n评估模型...")
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="评估进度"):
                # 将数据移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 计算损失
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                num_batches += 1

                # 生成翻译结果（用于BLEU评分）
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.args.max_target_length,
                    num_beams=self.args.num_beams
                )

                # 解码预测结果
                for i in range(len(generated_ids)):
                    # 解码预测
                    pred_text = self.tokenizer.decode(
                        generated_ids[i],
                        skip_special_tokens=True
                    )
                    pred_tokens = pred_text.split()
                    predictions.append(pred_tokens)

                    # 解码参考答案
                    # 将 -100 替换回 pad_token_id
                    ref_ids = labels[i].clone()
                    ref_ids[ref_ids == -100] = self.tokenizer.pad_token_id
                    ref_text = self.tokenizer.decode(
                        ref_ids,
                        skip_special_tokens=True
                    )
                    ref_tokens = ref_text.split()
                    references.append([ref_tokens])  # BLEU需要列表的列表

        # 计算平均损失
        avg_loss = total_loss / num_batches

        # 计算 BLEU 分数
        smooth = SmoothingFunction().method4
        bleu_score = corpus_bleu(references, predictions, smoothing_function=smooth)

        return bleu_score, avg_loss

    def save_model(self, save_path):
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存模型和分词器
        self.model.save_pretrained(save_path)
        print(f" 模型已保存到 {save_path}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="T5 模型训练脚本")

    # 模型参数
    parser.add_argument(
        '-model_path', '--model_path',
        type=str,
        default=None,
        help='T5 预训练模型路径（默认使用 config 中的路径）'
    )

    # 训练参数
    parser.add_argument(
        '-bs', '--batch_size',
        type=int,
        default=config.T5_BATCH_SIZE,
        help=f'批次大小（默认: {config.T5_BATCH_SIZE}）'
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=config.T5_LEARNING_RATE,
        help=f'学习率（默认: {config.T5_LEARNING_RATE}）'
    )
    parser.add_argument(
        '-accumulation', '--gradient_accumulation_steps',
        type=int,
        default=config.T5_GRADIENT_ACCUMULATION_STEPS,
        help=f'梯度累积步数（默认: {config.T5_GRADIENT_ACCUMULATION_STEPS}）'
    )
    parser.add_argument(
        '-max_source', '--max_source_length',
        type=int,
        default=config.T5_MAX_SOURCE_LENGTH,
        help=f'源文本最大长度（默认: {config.T5_MAX_SOURCE_LENGTH}）'
    )
    parser.add_argument(
        '-max_target', '--max_target_length',
        type=int,
        default=config.T5_MAX_TARGET_LENGTH,
        help=f'目标文本最大长度（默认: {config.T5_MAX_TARGET_LENGTH}）'
    )
    parser.add_argument(
        '-beams', '--num_beams',
        type=int,
        default=config.T5_NUM_BEAMS,
        help=f'Beam search 宽度（默认: {config.T5_NUM_BEAMS}）'
    )

    # 保存参数
    parser.add_argument(
        '-save_dir', '--save_dir',
        type=str,
        default=str(config.CHECKPOINTS_DIR / "T5" / "trained_model"),
        help='模型保存目录'
    )

    # 设备参数
    parser.add_argument(
        '-device', '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='训练设备 自动检测'
    )

    # 数据参数
    parser.add_argument(
        '-data_fraction', '--data_fraction',
        type=float,
        default=1,
        help='使用数据的比例（0.0-1.0，默认: 0.1表示使用10%%的数据）'
    )

    # 训练轮次
    parser.add_argument(
        '-epochs', '--epochs',
        type=int,
        default=config.EPOCHS,
        help=f'训练轮次（默认: {config.EPOCHS}）'
    )

    args = parser.parse_args()

    # 打印配置
    print("T5 模型训练")
    print(f"训练轮次: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"源文本最大长度: {args.max_source_length}")
    print(f"目标文本最大长度: {args.max_target_length}")
    print(f"Beam search 宽度: {args.num_beams}")
    print(f"数据使用比例: {args.data_fraction*100:.0f}%")
 

    # 确定设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\n使用设备: {device}")

    # 加载 T5 模型和分词器
    print("\n加载 T5 模型和分词器...")
    model_path = args.model_path if args.model_path else config.T5_MODEL_PATH

    try:
        tokenizer = T5Tokenizer.from_pretrained(str(model_path))
        model = T5Model(model_name_or_path=str(model_path))
        model.to(device)
        print(" T5 模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print(f"   请确保模型路径 {model_path} 存在")
        return

    # 创建数据加载器
    print("\n创建数据加载器...")

    
    class DataLoaderArgs:
        bs = args.batch_size

    dl_args = DataLoaderArgs()

    try:
        train_loader = get_t5_dataloader(dl_args, mode='train', tokenizer=tokenizer)
        valid_loader = get_t5_dataloader(dl_args, mode='valid', tokenizer=tokenizer)


        if args.data_fraction < 1.0:
            # 计算要使用的数据量
            train_size = int(len(train_loader.dataset) * args.data_fraction)
            valid_size = int(len(valid_loader.dataset) * args.data_fraction)

            # 创建子数据集
            train_subset = Subset(train_loader.dataset, range(train_size))
            valid_subset = Subset(valid_loader.dataset, range(valid_size))

            # 重新创建数据加载器
            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=t5_collate_fn,
                num_workers=0
            )
            valid_loader = DataLoader(
                valid_subset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=t5_collate_fn,
                num_workers=0
            )

            print(f"使用 {args.data_fraction*100:.0f}% 的数据进行训练")

        print(f"训练集: {len(train_loader.dataset)} 样本")
        print(f"验证集: {len(valid_loader.dataset)} 样本")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("   请确保数据文件存在:")
        print(f"   - {config.RAW_DATA_DIR / 'train_10k.jsonl'}")
        print(f"   - {config.RAW_DATA_DIR / 'valid.jsonl'}")
        return

    # 创建训练器
    trainer = T5Trainer(model, tokenizer, device, args)

    print("开始训练")

    best_bleu = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"\nEpoch {epoch} 训练完成")
        print(f"   平均训练损失: {train_loss:.4f}")

        # 评估模型
        print("\n开始评估...")
        bleu_score, valid_loss = trainer.evaluate(valid_loader)

        print(f"\nEpoch {epoch} 评估结果:")
        print(f"   验证损失: {valid_loss:.4f}")
        print(f"   BLEU 分数: {bleu_score:.4f}")

        # 保存最佳模型
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            best_epoch = epoch
            best_model_path = f"{args.save_dir}_best"
            print(f"\n新的最佳模型！BLEU: {bleu_score:.4f}")
            print(f"保存最佳模型到 {best_model_path}...")
            trainer.save_model(best_model_path)

    # 训练结束，保存最终模型
    print("训练完成！")
    print(f"最佳 BLEU 分数: {best_bleu:.4f} (Epoch {best_epoch})")

    print(f"\n保存最终模型到 {args.save_dir}...")
    trainer.save_model(args.save_dir)

    print("\n所有训练完成！")


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) == 1:
    
        print("T5 模型训练脚本")


    main()
