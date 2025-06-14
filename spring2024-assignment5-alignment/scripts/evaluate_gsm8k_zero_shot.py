import os
import json
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description="Evaluate gsm8k")

parser.add_argument("mode", help="Can be `zero-shot`, `sft` or `dpo`")
parser.add_argument("--model_size", help="If not specified, will use the default model size of 0.6B", default="0.6B")


args = parser.parse_args()

stop = "\n" if args.mode == 'zero-shot' else '<|end_of_text|>'
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

####################    (1) load the GSM8K examples     ####################
path_to_gsm8k = "/home/hex/spring2024-assignment5-alignment/data/gsm8k/"
split = 'test'      # train, test
file_path = os.path.join(path_to_gsm8k, f'{split}.jsonl')

questions, answers = [], []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 将每一行解析为JSON对象
        data = json.loads(line)
        questions.append(data['question'])
        answers.append(data['answer'])
            
####################    (2) format them as string prompts to the language model     ####################     
test_prompts = []
test_answers = []
for question, answer in zip(questions, answers):
    
    instruction = f"""{question}
 Answer:"""

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
    test_answers.append(answer)


####################    (3) generate outputs for each example.     ####################    
print(f"There are {len(test_prompts)} prompts.")

from vllm import LLM, SamplingParams

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
temperature=0.0, top_p=1.0, max_tokens=64, stop=stop
)

# Create an LLM.
if args.mode == 'zero-shot':
    model_path = f"/home/hex/spring2024-assignment5-alignment/models/Qwen/Qwen3-{args.model_size}-Base"
elif args.mode == 'sft':
    model_path = f"/home/hex/spring2024-assignment5-alignment/results/sft/checkpoints_{args.model_size}"
elif args.mode == 'dpo':
    model_path = f"/home/hex/spring2024-assignment5-alignment/results/dpo/checkpoints_{args.model_size}"
    
llm = LLM(model=model_path)
outputs = llm.generate(test_prompts, sampling_params)
    

####################    (4)calculate evaluation metrics     ####################    
####################    (5) serialize the examples, model generations, and corresponding evaluation scores to disk for further analysis.     ####################    

result_df = pd.DataFrame(columns=['Prompt', 'Generated_Text', 'Correct_Answer', 'Parsed_Answer', 'Parsed_Correct_Answer', 'Evaluation_Score', 'ParseFail'])
correct, parse_fail_cnt = 0, 0

for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    correct_answer = test_answers[i]
    
    parsed_answer = run_parse_gsm8k_response(generated_text)
    parsed_correct_answer = run_parse_gsm8k_response(correct_answer)
    
    parse_fail = parsed_answer == None
    evaluation_score = 1 if parsed_correct_answer == parsed_answer else 0
    if evaluation_score == 1:
        correct += 1
    if parse_fail:
        parse_fail_cnt += 1
    
    
    temp_df = pd.DataFrame({
        'Prompt': [prompt],
        'Generated_Text': [generated_text],
        'Correct_Answer': [correct_answer],
        'Parsed_Answer': [parsed_answer],
        'Parsed_Correct_Answer': [parsed_correct_answer],
        'Evaluation_Score': [evaluation_score],
        'ParseFail': [parse_fail]
    })
    
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

# # 保存结果到 CSV 文件
# os.makedirs('results/zero-shot/gsm8k', exist_ok=True)
# result_df.to_csv('results/zero-shot/gsm8k/model_evaluation_results.csv', index=False)
# print(f"Parse fail {parse_fail_cnt}/{len(outputs)}")
# print(f"Correct {correct}/{len(outputs)} Accuracy is {round(correct/len(outputs)*100, 2)}%")

# 保存结果到 CSV 文件
os.makedirs(f'results/{args.mode}/gsm8k/', exist_ok=True)
result_df.to_csv(f'results/{args.mode}/gsm8k/model_evaluation_results_{args.model_size}.csv', index=False)
print(f"Parse fail {parse_fail_cnt}/{len(outputs)}")
print(f"Correct {correct}/{len(outputs)} Accuracy is {round(correct/len(outputs)*100, 2)}%")
