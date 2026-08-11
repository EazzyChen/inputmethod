# 超参数配置

import pathlib

# 标准文件路径获取写法
RAW_DATA_DIR = pathlib.Path(__file__).parent.parent / 'data' / 'raw'
PROCESSED_DATA_DIR = pathlib.Path(__file__).parent.parent / 'data' / 'processed'
MODELS_DIR = pathlib.Path(__file__).parent.parent / 'models'
LOGS_DIR = pathlib.Path(__file__).parent.parent / 'logs'

SEQ_LEN = 5
BATCH_SIZE = 64

EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 10