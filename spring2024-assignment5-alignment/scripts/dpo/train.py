from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
# from tests.adapters import run_iterate_batches, get_packed_sft_dataset
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

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    chosen_input_ids: torch.Tensor,
    rejected_input_ids: torch.Tensor
) -> torch.Tensor:
    chosen_input_ids, rejected_input_ids = chosen_input_ids.to(lm_ref.device), rejected_input_ids.to(lm_ref.device)
    # 计算参考模型的 log-probabilities
    with torch.no_grad():
        chosen_ref_logits = lm_ref(chosen_input_ids).logits
        rejected_ref_logits = lm_ref(rejected_input_ids).logits
        
    chosen_input_ids, rejected_input_ids = chosen_input_ids.to(lm.device), rejected_input_ids.to(lm.device)
    chosen_ref_logits, rejected_ref_logits = chosen_ref_logits.to(lm.device), rejected_ref_logits.to(lm.device)
    
    # 计算训练模型的 log-probabilities
    # 需要求导
    chosen_logits = lm(chosen_input_ids).logits
    rejected_logits = lm(rejected_input_ids).logits

    # 计算 log-probability 差值
    chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
    chosen_ref_log_probs = torch.log_softmax(chosen_ref_logits, dim=-1)
    rejected_ref_log_probs = torch.log_softmax(rejected_ref_logits, dim=-1)
    
    chosen_log_prob = chosen_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_log_prob = rejected_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    chosen_ref_log_prob = chosen_ref_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_ref_log_prob = rejected_ref_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    # import pdb; pdb.set_trace()
    # 计算 DPO 损失
    chosen_ratios, rejected_ratios = chosen_log_prob - chosen_ref_log_prob, rejected_log_prob - rejected_ref_log_prob
    
    dpo_loss = -F.logsigmoid(beta * (chosen_ratios - rejected_ratios) )
    # import pdb; pdb.set_trace()
    return dpo_loss, chosen_ratios, rejected_ratios

output_dir = f"/home/hex/spring2024-assignment5-alignment/results/dpo/checkpoints_{args.pretrain_model_size}/"
os.makedirs(output_dir, exist_ok=True)
model_name_or_path = f"/home/hex/spring2024-assignment5-alignment/results/sft/checkpoints_{args.pretrain_model_size}/"
dataset_path = "/home/hex/spring2024-assignment5-alignment/results/dpo/pair_data.json"
device = "cuda:1"
device_ref = "cuda:2"


## 导入模型
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.to(device)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
ref_model.to(device_ref)
ref_model.eval()  # 将参考模型设置为评估模式，避免其参数更新

## 导入数据

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

chosen_input_ids_list = []
rejected_input_ids_list = []

# data = data[:1000]

pbar = tqdm(total=len(data), desc="Processing", ncols=100, unit="item")

for item in data:
    prompt = item['prompt']
    response_chosen = item['chosen']
    response_rejected = item['rejected']
    
    chosen_sequence   = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_chosen}"""
    rejected_sequence = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_rejected}"""
    chosen_sequence   += tokenizer.eos_token    
    rejected_sequence += tokenizer.eos_token

    chosen_input_ids = tokenizer.encode(chosen_sequence, return_tensors="pt")
    rejected_input_ids = tokenizer.encode(rejected_sequence, return_tensors="pt")

    chosen_input_ids_list.append(chosen_input_ids)
    rejected_input_ids_list.append(rejected_input_ids)
    
    pbar.update(1)

pbar.close()

## 划分训练、验证
valid_size = 200
train_chosen_input_ids_list = chosen_input_ids_list[valid_size:]
train_rejected_input_ids_list = rejected_input_ids_list[valid_size:]
valid_chosen_input_ids_list = chosen_input_ids_list[:valid_size]
valid_rejected_input_ids_list = rejected_input_ids_list[:valid_size]

## 定义训练参数
beta = 0.1
optimizer = optim.RMSprop(model.parameters(), lr=1e-6)

n_epochs = 1
gradient_accumulation_steps = 64
epoch_steps = len(train_chosen_input_ids_list) // gradient_accumulation_steps + 1
total_steps = epoch_steps * n_epochs
warmup_steps = int(0.03 * total_steps)

def linear_warmup(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    else:
        return 1.0

cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)

## 训练循环
print("Training...")
mlflow.set_experiment('dpo-1-epoch')
mlflow.start_run()
step = 0
log_inteval = 1
val_inteval = 10
min_accuracy = -float('inf')

model.train()
pbar = tqdm(total=n_epochs * len(chosen_input_ids_list), desc="Training", ncols=100, unit="item")
for epoch in range(n_epochs):
    for idx, (chosen_input_ids, rejected_input_ids) in enumerate(zip(chosen_input_ids_list, rejected_input_ids_list)):
        loss, chosen_reward, rejected_reward = compute_per_instance_dpo_loss(lm=model,
                                             lm_ref=ref_model,
                                             tokenizer=tokenizer,
                                             beta=beta,
                                             chosen_input_ids=chosen_input_ids,
                                             rejected_input_ids=rejected_input_ids)

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
                mlflow.log_metric('chosen reward', chosen_reward.item(), step)
                mlflow.log_metric('rejected reward', rejected_reward.item(), step)
            
            # 计算验证损失
            if step % val_inteval == 0:
                model.eval()
                loss_list, chosen_reward_list, rejected_reward_list = [], [], []
                for idx, (chosen_input_ids, rejected_input_ids) in enumerate(zip(valid_chosen_input_ids_list, valid_rejected_input_ids_list)):
                    loss, chosen_reward, rejected_reward = compute_per_instance_dpo_loss(lm=model,
                                                        lm_ref=ref_model,
                                                        tokenizer=tokenizer,
                                                        beta=beta,
                                                        chosen_input_ids=chosen_input_ids,
                                                        rejected_input_ids=rejected_input_ids)
                    loss_list.append(loss.item())
                    chosen_reward_list.append(chosen_reward.item())
                    rejected_reward_list.append(rejected_reward.item())
                    
                model.train()
                
                val_loss = np.mean(loss_list)
                val_chosen_reward = np.mean(chosen_reward_list)
                val_rejected_reward = np.mean(rejected_reward_list)
                accuracy = np.mean(np.array(chosen_reward_list) > np.array(rejected_reward_list))

                print(f'Step [{step}/{total_steps}], Val Loss: {val_loss:.4f}, Val Chosen Reward: {val_chosen_reward:.4f}, Val Rejected Reward: {val_rejected_reward:.4f}')
                mlflow.log_metric('val_loss', val_loss, step)
                mlflow.log_metric('val_chosen_reward', val_chosen_reward, step)
                mlflow.log_metric('val_rejected_reward', val_rejected_reward, step)
                mlflow.log_metric('accuracy', accuracy, step)

                
                if accuracy > min_accuracy:
                    min_accuracy = accuracy
                    model.save_pretrained(save_directory=output_dir)
                    tokenizer.save_pretrained(save_directory=output_dir)
                    print(f"Model saved at step {step} with accuracy {accuracy:.4f}")
              
        pbar.update(1)
        
pbar.close()
      
mlflow.end_run()