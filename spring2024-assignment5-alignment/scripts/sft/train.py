from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizerBase
import os
import json
import mlflow
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="SFT")

parser.add_argument("pretrain_model_size", type=str, default="0.6B", help="Pretrained model size")
args = parser.parse_args()


class SFTDataset(Dataset):
    
    def __init__(self,
                tokenizer: PreTrainedTokenizerBase,
                dataset_path: str | os.PathLike,
                seq_length: int,
                shuffle: bool,):
        super(SFTDataset, self).__init__()

        data_str_list = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt, response = data['prompt'], data['response']
                data_str = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response}"""
                
                data_str += '<|end_of_text|>'
                data_str_list.append(data_str)
                
        if shuffle:
            import random
            random.shuffle(data_str_list)
            
        # data_str_list = data_str_list[:100]
            
        data_ids_list = []
        for x in tqdm(data_str_list, desc="Tokenizer encoding..."):
            data_ids_list.append(tokenizer.encode(x))
        
        data_ids = []
        for l in data_ids_list:
            data_ids.extend(l)

        input_ids, labels = data_ids[:-1], data_ids[1:]
        
        n = len(input_ids)
        n = n - n % seq_length      # 丢弃最后不足 seq_length 的部分
        starts = np.arange(n, step=seq_length)
        
        input_ids_seq_len = []
        labels_seq_len = []
        
        for start in starts:
            input_ids_seq_len.append(input_ids[start : start + seq_length])
            labels_seq_len.append(labels[start : start + seq_length])
        
        self.X, self.y = torch.tensor(input_ids_seq_len, dtype=torch.long), torch.tensor(labels_seq_len, dtype=torch.long)
        
    def __getitem__(self, i):
        return {'input_ids': self.X[i], 'labels': self.y[i]}

    def __len__(self):
        return len(self.y)
        
        

def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    return SFTDataset(tokenizer,
                      dataset_path,
                      seq_length,
                      shuffle)

def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    # raise NotImplementedError

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


output_dir = f"/home/hex/spring2024-assignment5-alignment/results/sft/checkpoints_{args.pretrain_model_size}/"
os.makedirs(output_dir, exist_ok=True)
model_name_or_path = f"/home/hex/spring2024-assignment5-alignment/models/Qwen/Qwen3-{args.pretrain_model_size}-Base"
dataset_path = "/home/hex/spring2024-assignment5-alignment/data/sft/train.jsonl"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

total_steps = 6200
warmup_steps = int(0.03 * total_steps)
gradient_accumulation_steps = 8
def linear_warmup(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    else:
        return 1.0

cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)

dataset = get_packed_sft_dataset(tokenizer, dataset_path, seq_length=512, shuffle=False)

# dataset = dataset[:100]

valid_ratio = 0.1
valid_size = int(len(dataset) * valid_ratio)
train_dataset, valid_dataset = random_split(dataset, [len(dataset) - valid_size, valid_size])
train_loader = run_iterate_batches(train_dataset, batch_size=4, shuffle=True)
val_loader = run_iterate_batches(valid_dataset, batch_size=2, shuffle=False)

print("Training...")
print(f"Total epochs around {total_steps // (len(train_loader) // 4)}")
mlflow.set_experiment(f'sft-1-epoch-{args.pretrain_model_size}')
mlflow.start_run()
step = 0
log_inteval = 1
val_inteval = 10
min_val_loss = float('inf')
while step < total_steps:

    for idx, train_batch in enumerate(train_loader):
        input_ids = train_batch["input_ids"].to(device)
        labels = train_batch["labels"].to(device)
        
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)) / gradient_accumulation_steps
        
        loss.backward()
        
        if (idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            if step < warmup_steps:
                scheduler.step()
            else:
                cosine_scheduler.step()
            
            # 打印当前学习率和损失
            if step % log_inteval == 0:
                print(f'Step [{step}/{total_steps}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
                mlflow.log_metric('loss', loss.item(), step)
                mlflow.log_metric('lr', optimizer.param_groups[0]["lr"], step)
            
            # 计算验证损失
            if step % val_inteval == 0:
                model.eval()
                loss_list = []
                for valid_batch in val_loader:
                    input_ids = valid_batch["input_ids"].to(device)
                    labels = valid_batch["labels"].to(device)
                    
                    logits = model(input_ids).logits
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    loss_list.append(loss.item())
                model.train()
                
                val_loss = np.mean(loss_list)
                print(f'Step [{step}/{total_steps}], Val Loss: {val_loss:.4f}')
                mlflow.log_metric('val_loss', val_loss, step)
                
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    model.save_pretrained(save_directory=output_dir)
                    tokenizer.save_pretrained(save_directory=output_dir)
                    
mlflow.end_run()