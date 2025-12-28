"""
T5 模型推理脚本
使用训练好的 T5 模型进行翻译
"""
import os
import sys
import torch
import argparse
from pathlib import Path
from transformers import T5Tokenizer
from tqdm import tqdm

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config
from t5_model import T5Model, T5TranslationWrapper


class T5Translator:
    """
    T5 翻译器
    封装 T5 模型的加载和推理功能
    """

    def __init__(self, model_path=None, device=None):
        """
        初始化 T5 翻译器

        Args:
            model_path: 模型路径，默认使用 checkpoints/T5/trained_model_best
            device: 设备类型 ('cuda', 'cpu' 或 None 自动检测)
        """
        # 设置模型路径
        if model_path is None:
            # 优先使用训练好的模型，如果不存在则使用预训练模型
            trained_model_path = config.CHECKPOINTS_DIR / "T5" / "trained_model_best"
            if trained_model_path.exists():
                model_path = trained_model_path
                print(f"使用训练好的模型: {model_path}")
            else:
                model_path = config.T5_MODEL_PATH
                print(f" 训练模型未找到，使用预训练模型: {model_path}")
        else:
            model_path = Path(model_path)

        # 自动检测设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f" 使用设备: {self.device}")

        # 加载模型和分词器
        print("正在加载 T5 模型和分词器...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(str(model_path))
            self.model = T5Model.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()
            print("T5 模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

        # 创建翻译包装器
        self.wrapper = T5TranslationWrapper(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            task_prefix="translate Chinese to English: "
        )

    def translate(self, text, num_beams=4, max_length=128):
        """
        翻译单个文本
        """
        return self.wrapper.translate(text, max_length=max_length, num_beams=num_beams)

    def batch_translate(self, texts, num_beams=4, max_length=128, batch_size=8):
        """
        批量翻译
        """
        return self.wrapper.batch_translate(
            texts,
            max_length=max_length,
            num_beams=num_beams,
            batch_size=batch_size
        )

    def translate_file(self, input_file, output_file, num_beams=4, batch_size=8):
        """
        翻译文件中的所有行

        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        # 读取所有行
        print(f" 读取文件: {input_file}")
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"共 {len(lines)} 行待翻译")

        # 批量翻译
        print("开始翻译...")
        results = []
        for i in tqdm(range(0, len(lines), batch_size), desc="翻译进度"):
            batch = lines[i:i + batch_size]
            batch_results = self.batch_translate(batch, num_beams=num_beams, batch_size=batch_size)
            results.extend(batch_results)

        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')

        print(f" 翻译完成！结果已保存到: {output_file}")

    def interactive_mode(self):
        """交互式翻译模式"""
  
        print(" T5 中英翻译系统 - 交互模式")

        print("提示:")
        print("  - 输入中文文本进行翻译")
        print("  - 输入 'q' 或 'quit' 退出")
        print("  - 输入 'file' 进行文件翻译")


        while True:
            try:
                user_input = input("\n中文: ").strip()

                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("再见！")
                    break

                if user_input.lower() == 'file':
                    input_file = input("输入文件路径: ").strip()
                    output_file = input("输出文件路径: ").strip()
                    try:
                        self.translate_file(input_file, output_file)
                    except Exception as e:
                        print(f" 文件翻译失败: {e}")
                    continue

                if not user_input:
                    print(" 请输入内容")
                    continue

                # 翻译
                result = self.translate(user_input, num_beams=config.T5_NUM_BEAMS)
                print(f"英文: {result}")

            except KeyboardInterrupt:
                print("\n 再见！")
                break
            except Exception as e:
                print(f"翻译出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="T5 中英翻译推理工具")

    # 模型参数
    parser.add_argument(
        '-model_path', '--model_path',
        type=str,
        default=None,
        help='T5 模型路径（默认: checkpoints/T5/best_model）'
    )
    parser.add_argument(
        '-device', '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='运行设备（默认: 自动检测）'
    )

    # 运行模式
    parser.add_argument(
        '-mode', '--mode',
        type=str,
        default='interactive',
        choices=['interactive', 'file', 'single'],
        help='运行模式: interactive(交互), file(文件), single(单句)'
    )

    # 单句翻译参数
    parser.add_argument(
        '-text', '--text',
        type=str,
        default=None,
        help='要翻译的中文文本（单句模式）'
    )

    # 文件翻译参数
    parser.add_argument(
        '-input', '--input_file',
        type=str,
        default=None,
        help='输入文件路径（文件模式）'
    )
    parser.add_argument(
        '-output', '--output_file',
        type=str,
        default=None,
        help='输出文件路径（文件模式）'
    )

    # 翻译参数
    parser.add_argument(
        '-beams', '--num_beams',
        type=int,
        default=4,
        help='Beam search 宽度（默认: 4）'
    )
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int,
        default=8,
        help='批处理大小（默认: 8）'
    )

    args = parser.parse_args()

    # 创建翻译器
    translator = T5Translator(model_path=args.model_path, device=args.device)

    # 根据模式执行
    if args.mode == 'interactive':
        # 交互模式
        translator.interactive_mode()

    elif args.mode == 'single':
        # 单句翻译模式
        if args.text is None:
            print("请使用 -text 参数指定要翻译的文本")
            return

        print(f"\n中文: {args.text}")
        result = translator.translate(args.text, num_beams=args.num_beams)
        print(f"英文: {result}\n")

    elif args.mode == 'file':
        # 文件翻译模式
        if args.input_file is None or args.output_file is None:
            print("请使用 -input 和 -output 参数指定输入输出文件")
            return

        translator.translate_file(
            args.input_file,
            args.output_file,
            num_beams=args.num_beams,
            batch_size=args.batch_size
        )


if __name__ == "__main__":

    if len(sys.argv) == 1:

        print("T5 中英翻译推理")


    main()
