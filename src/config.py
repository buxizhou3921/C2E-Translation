from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# 路径
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
VOCAB_DIR = ROOT_DIR / "vocab"
LOGS_DIR = ROOT_DIR / "logs"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

# 句子长度
MAX_SEQ_LENGTH = 180
MAX_ZH_LENGTH = 110
MAX_EN_LENGTH = 50

# GRU 模型结构
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256

# Transformer 模型结构
DIM_MODEL = 100
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

# 训练参数
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
