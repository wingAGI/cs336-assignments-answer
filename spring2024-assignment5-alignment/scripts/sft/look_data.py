import gzip
import json
import os

ten_samples = []
with gzip.open('/home/hex/spring2024-assignment5-alignment/data/sft/train.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        ten_samples.append(data)
        
        if len(ten_samples) == 10:
            break


#保存10个样例
os.makedirs('results/sft/', exist_ok=True)
with open(f'results/sft/look_data.json', "w") as fout:
    json.dump(ten_samples, fout)
        
        
# 打印数据
# print(data)
# import pdb; pdb.set_trace()