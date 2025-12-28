"""
加载 T5 模型进行翻译推理
"""
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 获取项目根目录
ROOT_DIR = Path(__file__).parent
# 默认使用项目本地 models/t5-base 目录
MODEL_DIR = ROOT_DIR / "models" / "t5-base"


class TranslationModel:
    """翻译模型加载和推理类"""

    def __init__(self, model_name_or_path=None, device=None):
        """
        初始化翻译模型

       
        """
        if model_name_or_path is None:
            model_name_or_path = MODEL_DIR

        # 自动检测设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"模型: {model_name_or_path}")
        print(f"使用设备: {self.device}")

        # 加载模型和分词器
        print("正在加载 T5 模型...")
        try:
            # 加载 tokenizer 和模型
            self.tokenizer = T5Tokenizer.from_pretrained(str(model_name_or_path))
            self.model = T5ForConditionalGeneration.from_pretrained(str(model_name_or_path))
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            print(" T5 模型加载成功！")
          
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def translate(self, text, max_length=128, num_beams=4, temperature=1.0):
        """
        执行翻译
        """
        # 编码输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # 将输入移到设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成翻译
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True
            )

        # 解码输出
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def batch_translate(self, texts, max_length=128, num_beams=4, batch_size=8):
        """
        批量翻译

        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 编码批次
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # 将输入移到设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成翻译
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )

            # 解码输出
            batch_results = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            results.extend(batch_results)

        return results


def main():
   
    print("T5-base 模型加载与翻译测试")

    # 加载模型
    translator = TranslationModel()
    print("开始翻译测试...")


   
    test_cases = [
        "translate English to German: Hello, how are you?",
        "translate English to German: I love machine learning.",
        "translate English to French: The weather is nice today.",
        "translate English to French: Good morning, everyone.",
        "translate English to Romanian: Thank you very much.",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n[测试 {i}]")
        print(f"输入: {text}")
        result = translator.translate(text, num_beams=4)
        print(f"输出: {result}")



if __name__ == "__main__":
    main()
