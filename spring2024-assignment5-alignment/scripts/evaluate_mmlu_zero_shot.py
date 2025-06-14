import os
import pandas as pd
from typing import Any

import argparse
parser = argparse.ArgumentParser(description="Evaluate MMLU")

parser.add_argument("mode", help="Can be `zero-shot`, `sft` or `dpo`")
parser.add_argument("--model_size", help="If not specified, will use the default model size of 0.6B", default="0.6B")

args = parser.parse_args()


stop = "\n" if args.mode == 'zero-shot' else '<|end_of_text|>'

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
    # raise NotImplementedError
    legal_options = ["A", "B", "C", "D"]
    
    # model_option = 
    for option in legal_options:
        if option in model_output:
            return option
    
    # If no option is found, return None
    return None

####################    (1) load the MMLU examples     ####################
path_to_mmlu = "/home/hex/spring2024-assignment5-alignment/data/mmlu/"

df_mmlu = pd.DataFrame(columns=['Question', 'Option_A', 'Option_B', 'Option_C', 'Option_D', 'Answer', 'tag', 'split'])

for split in ['dev', 'val', 'test']:
    split_path = os.path.join(path_to_mmlu, split)
    for file_name in os.listdir(split_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(split_path, file_name)
            df = pd.read_csv(file_path, header=None)
            df.columns = ['Question', 'Option_A', 'Option_B', 'Option_C', 'Option_D', 'Answer']

            df['tag'] = file_name.replace(f'_{split}.csv', '')
            df['split'] = split
            
            df_mmlu = pd.concat([df_mmlu, df], ignore_index=True)
            
####################    (2) format them as string prompts to the language model     ####################     
test_prompts = []
test_answers = []
for index, row in df_mmlu.iterrows():
    if not row['split'] == 'test':
        continue
    
    question = row['Question']
    options = [row['Option_A'], row['Option_B'], row['Option_C'], row['Option_D']]
    answer = row['Answer']
    subject = row['tag']
    
    instruction = f"""Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).
                    Question: {question}
                    A. {options[0]}
                    B. {options[1]}
                    C. {options[2]}
                    D. {options[3]}
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
temperature=0.0, top_p=1.0, max_tokens=100, stop=stop
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

result_df = pd.DataFrame(columns=['Prompt', 'Generated_Text', 'Correct_Answer', 'Parsed_Answer', 'Evaluation_Score', 'ParseFail'])
correct, parse_fail_cnt = 0, 0

for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    parsed_answer = run_parse_mmlu_response(None, generated_text)
    correct_answer = test_answers[i]
    
    parse_fail = parsed_answer == None
    evaluation_score = 1 if correct_answer == parsed_answer else 0
    if evaluation_score == 1:
        correct += 1
    if parse_fail:
        parse_fail_cnt += 1
    
    
    temp_df = pd.DataFrame({
        'Prompt': [prompt],
        'Generated_Text': [generated_text],
        'Correct_Answer': [correct_answer],
        'Parsed_Answer': [parsed_answer],
        'Evaluation_Score': [evaluation_score],
        'ParseFail': [parse_fail]
    })
    
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

# 保存结果到 CSV 文件
os.makedirs(f'results/{args.mode}/mmlu/', exist_ok=True)
result_df.to_csv(f'results/{args.mode}/mmlu/model_evaluation_results_{args.model_size}.csv', index=False)
print(f"Parse fail {parse_fail_cnt}/{len(outputs)}")
print(f"Correct {correct}/{len(outputs)} Accuracy is {round(correct/len(outputs)*100, 2)}%")
