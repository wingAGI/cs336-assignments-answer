# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Compare two models")

parser.add_argument("model_name1")
parser.add_argument("model_name2")

args = parser.parse_args()

file_path = "/home/hex/spring2024-assignment5-alignment/results/zero-shot/alpaca_eval/"

def get_model_output(model_name):
    with open(os.path.join(file_path, f"output_{model_name}.json"), 'r', encoding='utf-8') as file:
        eval_set = json.load(file)
            
    return eval_set


your_api_key = "<DeepSeek API Key>"    # "<DeepSeek API Key>"
client = OpenAI(api_key=your_api_key, base_url="https://api.deepseek.com")



model_outputs1, model_outputs2 = get_model_output(args.model_name1), get_model_output(args.model_name2)


result_df = pd.DataFrame(columns=['Instruction', 'Model1', 'Model2', 'Output1', 'Output2', 'Preference_Response', 'Parsed_Preference'])

model_1_win, model_2_win, fail_parse = 0, 0, 0
for example1, example2 in tqdm(zip(model_outputs1, model_outputs2)):
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


    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": content},
    ],
    stream=False
    )
    preference_response = response.choices[0].message.content
    
    def parse(response):
        """
        Return the rank of model_1, e.g. the first model
        """
        try:
            eval_obj = eval(preference_response)
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
    
    
    model_1_rank = parse(preference_response)
    
    print("Instruction:")
    print(instruction)
    
    print("output of model 1:")
    print(output_1)
    
    print("output of model 2:")
    print(output_2)
    
    print("Response from deepseek:")
    print(preference_response)
    
    print("Rank of model 1:")
    print(model_1_rank)
    
    # break
    
    if model_1_rank == 1:
        model_1_win += 1
    elif model_1_rank == 2:
        model_2_win += 1
    elif model_1_rank == None:
        fail_parse += 1
        
        
        
print(f"Win rate of model 1: {model_1_win}/{len(model_outputs1)}")
print(f"Win rate of model 2: {model_2_win}/{len(model_outputs1)}")
print(f"Fail to parse: {fail_parse}/{len(model_outputs1)}")    