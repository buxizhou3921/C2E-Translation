from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_GRU_DIR = ROOT_DIR / "logs" / "GRU"
CHECKPOINTS_GRU_DIR = ROOT_DIR / "checkpoints" / "GRU"
VOCAB_DIR = ROOT_DIR / "vocab"

MAX_SEQ_LENGTH = 180
MAX_ZH_LENGTH = 110
MAX_EN_LENGTH = 50
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
