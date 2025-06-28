import os
import sys
import json
import torch
import pickle
import pathlib
import argparse
from utils import _to_device_and_compile
from model import BasicsTransformerLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import Tokenizer


# 1. 设定路径
TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")
special_tokens = ["<|endoftext|>"]

# 2. 加载词表和 merges
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
with open(MERGES_PATH, 'rb') as f:
    merges = pickle.load(f)

# 3. 构造 tokenizer
tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens
)

CKPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints/ckpt_iter5000.pt"
CONFIG_PATH = "scripts/config.json"
def main():
    parser = argparse.ArgumentParser(description="Text Generation Inference")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a pretty girl", help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k sampling")
    args = parser.parse_args()

    # ==== 加载模型结构 ====
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = BasicsTransformerLM(**config["model"])
    model, device = _to_device_and_compile(model)
    model.eval()

    # ==== 导入模型权重 ====
    with open(CKPT_PATH, 'rb') as f:
        checkpoint = torch.load(f, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    # ==== 生成 ====
    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    output_tokens = model.generate(
        input_tensor, 
        max_new_tokens=args.max_new_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k,
        eos_token_id=256
    )
    output_ids = output_tokens[0].cpu().numpy().tolist()
    
    # ==== 合并原prompt和生成内容 ====
    full_ids = input_ids + output_ids
    text = tokenizer.decode(full_ids)
    print("输入：", args.prompt)
    print("生成结果：", text)


if __name__ == "__main__":
    main()