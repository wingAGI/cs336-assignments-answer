import os
import json
import pandas as pd



import argparse
parser = argparse.ArgumentParser(description="Collect model output for alpaca eval")

# 添加参数
parser.add_argument("--mode", help="zero-shot, sft or dpo")
parser.add_argument("--model_name", help="e.g. qwen3-0.6B-base", default="qwen3-0.6B-base")
parser.add_argument("--model_path", help="e.g. Qwen/Qwen3-0.6B-Base, default at ./models/", default="Qwen/Qwen3-0.6B-Base")

# 解析参数
args = parser.parse_args()

stop = "\n" if args.mode == 'zero-shot' else '<|end_of_text|>'

####################    (1) load the alpaca_eval examples     ####################
file_path = "/home/hex/spring2024-assignment5-alignment/data/alpaca_eval/alpaca_eval.jsonl"

eval_set = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        eval_set.append(data)

####################    (2) format them as string prompts to the language model     #################### 
instructions = [example["instruction"] for example in eval_set]

test_prompts = []
for instruction in instructions:
    prompt = f"""# Instruction
    Below is a list of conversations between a human and an AI assistant (you).
    Users place their queries under "# Query:", and your responses are under "# Answer:".
    You are a helpful, respectful, and honest assistant.
    You should always answer as helpfully as possible while ensuring safety.
    Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
    Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
    Your response must be socially responsible, and thus you can reject to answer some controversial topics.

    # Query:
    ```{instruction}```

    # Answer:
    ```"""
    test_prompts.append(prompt)
    

####################    (3) generate outputs for each example.     ####################    
print(f"There are {len(test_prompts)} prompts.")

from vllm import LLM, SamplingParams

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
temperature=0.0, top_p=1.0, max_tokens=256, stop=stop      
)

# Create an LLM.
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

for example, output in zip(eval_set, outputs):
    generated_text = output.outputs[0].text
    # generate here is a placeholder for your models generations
    example["output"] = generated_text
    example["generator"] = args.model_name # name of your model

os.makedirs(f'results/{args.mode}/alpaca_eval', exist_ok=True)
with open(f'results/{args.mode}/alpaca_eval/output_{args.model_name}.json', "w") as fout:
    json.dump(eval_set, fout)