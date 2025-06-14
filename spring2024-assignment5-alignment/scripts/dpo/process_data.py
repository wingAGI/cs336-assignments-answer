import gzip
import json
import os

# 
divisions = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']     
split = 'train'


pair_data_list = []
harmless_samples = []
helpful_samples = []
for division in divisions:
    file_path = f"/home/hex/spring2024-assignment5-alignment/data/dpo/Anthropic/hh-rlhf/{division}/{split}.jsonl.gz"
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chosen_text, rejected_text = data['chosen'], data['rejected']
            
            turns = chosen_text.count('\n\nHuman')           # count `\n\nHuman` 出现的次数，作为对话轮数
            if turns > 1:
                continue
            
            # 提取 prompt   
            prompt_start = "\n\nHuman: "
            prompt_end = "\n\nAssistant: "
            prompt = chosen_text[chosen_text.find(prompt_start) + len(prompt_start):chosen_text.find(prompt_end)]
            
            # 提取 chosen response
            chosen_response_start = chosen_text.find(prompt_end) + len(prompt_end)
            chosen_response = chosen_text[chosen_response_start:]
            
            # 提取 rejected response
            rejected_response_start = rejected_text.find(prompt_end) + len(prompt_end)
            rejected_response = rejected_text[rejected_response_start:]
            

            pair_data =  {'prompt': prompt, 
                          'chosen': chosen_response, 
                          'rejected': rejected_response,
                          'division': division}
            
            pair_data_list.append(pair_data)
            
            if division == 'harmless-base':
                harmless_samples.append(pair_data)
            if division == 'helpful-base':
                helpful_samples.append(pair_data)
            
os.makedirs('results/dpo/', exist_ok=True)
with open(f'results/dpo/pair_data.json', "w") as fout:
    json.dump(pair_data_list, fout)
    
with open(f'results/dpo/harmless_few_samples.json', "w") as fout:
    json.dump(harmless_samples, fout)
    
with open(f'results/dpo/helpful_few_samples.json', "w") as fout:
    json.dump(helpful_samples, fout)