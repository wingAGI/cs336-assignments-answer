import os
import sys
import json
import torch
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from utils import _to_device_and_compile
from model import BasicsTransformerLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import *

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.dat")
VAL_DATA_PATH = os.path.join(DATA_DIR, "valid.dat")
CONFIG_PATH = "scripts/config.json"

def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode="r")   # 单列token id序列
    return arr

def get_batch(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    ix = np.random.randint(0, N-context_length-1, size=(batch_size,))
    x = np.stack([memmap_arr[i:i+context_length] for i in ix])
    y = np.stack([memmap_arr[i+1:i+context_length+1] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def memmap_val_iterator(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    nb = (N-context_length-1)//batch_size
    for bi in range(nb):
        base = bi*batch_size
        x = np.stack([memmap_arr[i:i+context_length] for i in range(base, base+batch_size)])
        y = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(base, base+batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
def main():
    # 1. 导入模型和配置
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = BasicsTransformerLM(**config["model"])  

    params = {}
    for group in config.values():
        params.update(group)

    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = DotDict(params)

    model, device = _to_device_and_compile(model)

    os.makedirs(args.save_path, exist_ok=True)

    # 2. 加载数据集
    train_data = get_memmap_dataset(TRAIN_DATA_PATH)
    val_data = get_memmap_dataset(VAL_DATA_PATH)

    # 3. 构建优化器
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. 恢复断点
    start_iter = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint {args.resume_checkpoint}")
        resume_ckpt_path = pathlib.Path(__file__).resolve().parent.parent / f"checkpoints/ckpt_iter{args.resume_checkpoint}.pt"
        start_iter = run_load_checkpoint(resume_ckpt_path, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    # 5. 训练loop
    for iteration in tqdm(range(start_iter, args.train_steps), desc="Training"):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length)
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = run_cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), args.clip_grad_norm)
        
        # 更新学习率
        lr = run_get_lr_cosine_schedule(
            iteration, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # 验证
        if (iteration+1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in memmap_val_iterator(val_data, args.batch_size, args.context_length):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    val_loss = run_cross_entropy(
                        val_logits.reshape(-1, val_logits.shape[-1]),
                        y_val.reshape(-1)
                    )
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= args.val_batches:
                        break
                val_loss_mean = np.mean(val_losses)
                print(f"iter {iteration:05d}: VALID loss = {val_loss_mean:.4f}")

        # 保存
        if (iteration+1) % args.save_interval == 0:
            ckpt_name = os.path.join(args.save_path, f"ckpt_iter{iteration+1}.pt")
            run_save_checkpoint(model, optimizer, iteration+1, ckpt_name)
            print(f"Checkpoint saved to {ckpt_name}")

if __name__ == "__main__":
    main()
