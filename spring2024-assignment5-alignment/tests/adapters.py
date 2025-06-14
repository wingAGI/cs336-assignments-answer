#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
import gzip
import json
import numpy as np
import torch.nn.functional as F

class SFTDataset(Dataset):
    
    def __init__(self,
                tokenizer: PreTrainedTokenizerBase,
                dataset_path: str | os.PathLike,
                seq_length: int,
                shuffle: bool,):
        super(SFTDataset, self).__init__()

        # print(dataset_path)   # /home/hex/spring2024-assignment5-alignment/tests/fixtures/sft_sample.jsonl
        data_str_list = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt, response = data['prompt'], data['response']
                data_str = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response}"""
                # data_str += tokenizer.eos_token
                data_str += '<|end_of_text|>'
                
                data_str_list.append(data_str)
                
        if shuffle:
            import random
            random.shuffle(data_str_list)

        data_ids_list = [tokenizer.encode(x) for x in data_str_list]    # 会添加 <|begin_of_text|>
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
    # raise NotImplementedError
    
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

def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    legal_options = ["A", "B", "C", "D"]
    
    for option in legal_options:
        if option in model_output:
            return option
    
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # raise NotImplementedError
    import re
    numbers = re.findall(r'-?\d+\.?\d*', model_output)
    
    # 如果没有找到任何数字，返回 None
    if not numbers:
        return None
    
    # 返回最后一个数字
    return numbers[-1]


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # raise NotImplementedError
    chosen_sequence   = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_chosen}"""
    rejected_sequence = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_rejected}"""
    chosen_sequence   += tokenizer.eos_token    
    rejected_sequence += tokenizer.eos_token


    # 对格式化后的序列进行编码
    chosen_input_ids = tokenizer.encode(chosen_sequence, return_tensors="pt")
    rejected_input_ids = tokenizer.encode(rejected_sequence, return_tensors="pt")

    # 计算训练模型的 log-probabilities
    with torch.no_grad():
        chosen_logits = lm(chosen_input_ids).logits
        rejected_logits = lm(rejected_input_ids).logits

    # 计算参考模型的 log-probabilities

    chosen_ref_logits = lm_ref(chosen_input_ids).logits
    rejected_ref_logits = lm_ref(rejected_input_ids).logits

    # 计算 log-probability 差值
    chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
    chosen_ref_log_probs = torch.log_softmax(chosen_ref_logits, dim=-1)
    rejected_ref_log_probs = torch.log_softmax(rejected_ref_logits, dim=-1)
    
    """
    (Pdb) chosen_log_probs.shape
    torch.Size([1, 48, 50257])
    (Pdb) chosen_input_ids.shape
    torch.Size([1, 48])
    """
    chosen_log_prob = chosen_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_log_prob = rejected_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    chosen_ref_log_prob = chosen_ref_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_ref_log_prob = rejected_ref_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    import pdb; pdb.set_trace()
    # 计算 DPO 损失
    chosen_ratios, rejected_ratios = chosen_log_prob - chosen_ref_log_prob, rejected_log_prob - rejected_ref_log_prob
    
    dpo_loss = -F.logsigmoid(beta * (chosen_ratios - rejected_ratios) )

    return dpo_loss