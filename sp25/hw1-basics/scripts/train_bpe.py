import os
import sys
import pickle
import pathlib
from tests.adapters import run_train_bpe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 数据集路径
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")

# Tokenizer保存路径
TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

# 训练参数
vocab_size = 10_000
special_tokens = ["<|endoftext|>"]

# 训练
vocab, merges = run_train_bpe(
    input_path=INPUT_PATH,
    vocab_size=vocab_size,
    special_tokens=special_tokens
)

# 序列化到磁盘
os.makedirs(TOKENIZER_DIR, exist_ok=True)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)
with open(MERGES_PATH, "wb") as f:
    pickle.dump(merges, f)

# 统计最长 token
longest_token = max(vocab.values(), key=len)
print("最长token:", longest_token, "长度:", len(longest_token))

