import os
import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Collect model output for simple safety tests")

# 添加参数
parser.add_argument("--mode", help="zero-shot, sft or dpo")
parser.add_argument("--model_name", help="e.g. qwen3-0.6B-base", default="qwen3-0.6B-base")
parser.add_argument("--model_path", help="e.g. Qwen/Qwen3-0.6B-Base, default at ./models/", default="Qwen/Qwen3-0.6B-Base")

# 解析参数
args = parser.parse_args()

stop = "\n" if args.mode == 'zero-shot' else '<|end_of_text|>'

####################    (1) load the simple safety tests examples     ####################
file_path = "/home/hex/spring2024-assignment5-alignment/data/simple_safety_tests/simple_safety_tests.csv"

df = pd.read_csv(file_path)
prompts_final = df['prompts_final'].values.tolist()

####################    (2) format them as string prompts to the language model     #################### 
test_prompts = prompts_final
    

####################    (3) generate outputs for each example.     ####################    
print(f"There are {len(test_prompts)} prompts.")

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
temperature=0.0, top_p=1.0, max_tokens=1024, stop=stop
)

if args.mode == 'zero-shot':
    model_path = f"/home/hex/spring2024-assignment5-alignment/models/{args.model_path}"
elif args.mode == 'sft':
    model_path = f"/home/hex/spring2024-assignment5-alignment/results/sft/checkpoints"
elif args.mode == 'dpo':
    model_path = f"/home/hex/spring2024-assignment5-alignment/results/dpo/checkpoints"
    
print(f"Load model from path: {model_path}")
llm = LLM(model=model_path)
outputs = llm.generate(test_prompts, sampling_params)

####################    (4) serialize the examples, model generations, and corresponding evaluation scores to disk for further analysis.     ####################    
results = []
for prompt, output in zip(test_prompts, outputs):
    result = {}
    result['prompts_final'] = prompt
    result['output'] = output.outputs[0].text
    results.append(result)

os.makedirs(f'results/{args.mode}/simple_safety_tests', exist_ok=True)
with open(f'results/{args.mode}/simple_safety_tests/output_{args.model_name}.json', "w") as fout:
    json.dump(results, fout)