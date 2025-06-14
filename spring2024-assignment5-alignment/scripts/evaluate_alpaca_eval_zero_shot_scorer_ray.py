import ray
import tqdm
from openai import OpenAI
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Compare two models")

parser.add_argument("model_name1")
parser.add_argument("model_name2")
parser.add_argument("mode1")
parser.add_argument("mode2")

args = parser.parse_args()


def get_model_output(mode, model_name):
    file_path = f"/home/hex/spring2024-assignment5-alignment/results/{mode}/alpaca_eval/"
    with open(os.path.join(file_path, f"output_{model_name}.json"), 'r', encoding='utf-8') as file:
        eval_set = json.load(file)
            
    return eval_set

model_outputs1, model_outputs2 = get_model_output(args.mode1, args.model_name1), get_model_output(args.mode2, args.model_name2)

# import pdb; pdb.set_trace()
print(f"There are {len(model_outputs1)} examples")

# 初始化 Ray
ray.init()

# 假设 client 是你的 DeepSeek 客户端实例
your_api_key = "sk-d2fb431427654384a6411c57011d80c5"    # "<DeepSeek API Key>"



def parse(response):
    """
    Return the rank of model_1, e.g. the first model
    """
    try:
        eval_obj = eval(response)
        if isinstance(eval_obj, dict):
            ranking_list = ['ranking']
        elif isinstance(eval_obj, list):
            ranking_list = eval_obj
        for x in ranking_list:
            if x['model'] == 'model_1':
                return x['rank']
    except:
        return None

    return None
    
@ray.remote
def process_example(example1, example2):
    """
    Return rank of model1, e.g. 1 or 2
    """
    instruction = example1["instruction"]
    output_1 = example1["output"]
    output_2 = example2["output"]
    
    content = f"""I want you to create a leaderboard of different large language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be Python dictionaries.

Here is the prompt:
{{
    "instruction": {instruction},
}}

Here are the outputs of the models:
[
    {{
        "model": "model_1",
        "answer": {output_1}
    }},
    {{
        "model": "model_2",
        "answer": {output_2}
    }}
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{'model': <model-name>, 'rank': <model-rank>}},
    {{'model': <model-name>, 'rank': <model-rank>}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give. Do not need to wrap your output by ```Python ```.
"""
    client = OpenAI(api_key=your_api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": content},
        ],
        stream=False
    )
    preference_response = response.choices[0].message.content
    
    model_1_rank = parse(preference_response)
    
    result = {}
    result["instruction"] = instruction
    result["output1"] = example1["output"]
    result["output2"] = example2["output"]
    result["model_1_rank"] = model_1_rank
    
    return result



# 创建任务列表
tasks = [process_example.remote(example1, example2) for example1, example2 in zip(model_outputs1, model_outputs2)]

# 使用 tqdm 显示进度条
results = []
for task in tqdm(tasks, total=len(tasks)):
    result = ray.get(task)
    results.append(result)

# 关闭 Ray
ray.shutdown()

# 保存所有结果
os.makedirs(f'results/{args.mode1}/alpaca_eval', exist_ok=True)
with open(f'results/{args.mode1}/alpaca_eval/model_rank_{args.model_name1}_VS_{args.model_name2}.json', "w") as fout:
    json.dump(results, fout)
    
#保存10个 model1 不如 model2 的样例
disprefer_results = [x for x in results if x['model_1_rank'] == 2]
disprefer_results = disprefer_results[:10]
with open(f'results/{args.mode1}/alpaca_eval/model_rank_{args.model_name1}_VS_{args.model_name2}_10_disprefer.json', "w") as fout:
    json.dump(disprefer_results, fout)

# 打印结果
win_1, win_2, parse_fail = 0, 0, 0
for result in results:
    model_1_rank = result["model_1_rank"]
    if model_1_rank == 1:
        win_1 += 1
    elif model_1_rank == 2:
        win_2 += 1
    elif model_1_rank == None:
        parse_fail += 1
        
print(f"Win rate of model 1: {win_1}/{len(model_outputs1)}")
print(f"Win rate of model 2: {win_2}/{len(model_outputs1)}")
print(f"Fail to parse: {parse_fail}/{len(model_outputs1)}") 
    