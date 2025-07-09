import os
import pathlib
import fasttext
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from adapters import run_extract_text_from_html_bytes
from adapters import run_identify_language
from adapters import run_mask_emails, run_mask_ips, run_mask_phone_numbers
from adapters import run_gopher_quality_filter

PIPELINE_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "pipeline_data"
WET_PATH = os.path.join(PIPELINE_PATH, "wet")
DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "cs336_data"

# fasttext models
LANG_MODEL_PATH = os.path.join(DATA_PATH, "lid.176.bin")
NSFW_MODEL_PATH = os.path.join(DATA_PATH, 'jigsaw_fasttext_bigrams_nsfw_final.bin')
TOXIC_MODEL_PATH = os.path.join(DATA_PATH, 'jigsaw_fasttext_bigrams_hatespeech_final.bin')
QUALITY_MODEL_PATH = os.path.join(DATA_PATH, 'quality_classifier.ftz')

nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)
quality_model = fasttext.load_model(QUALITY_MODEL_PATH)


def filter_pipeline(log_harmful=False):
    """
    1. 过滤掉非英文
    2. 过滤掉有害内容
    """
    FIEL_AFTER_LANG = "tmp/after_lang.txt"
    FIEL_AFTER_HARM = "tmp/after_harm.txt"
    
    SCORE_THRESHOLD = 0.5
    CHUNK_SIZE = 100            # batch size
    LINES_PER_TEXT = 1
    MIN_CHARS_PER_LINE = 150

    model = fasttext.load_model(str(LANG_MODEL_PATH))
    nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
    toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)

    def predict_and_filter(chunk_lines, log=False):
        chunk_lines = [line.strip() for line in chunk_lines]
        if not chunk_lines:
            return []
        labels, scores = model.predict(chunk_lines)
        # 仅保留英文且置信度大于0.3的行
        results = [
            line for line, label, score in zip(chunk_lines, labels, scores)
            if label[0].replace('__label__', '') == 'en' and score[0] > SCORE_THRESHOLD
        ]
        if log and not results:
            print('No results')
        return results
        
    # 保证tmp目录存在
    os.makedirs(os.path.dirname(FIEL_AFTER_LANG), exist_ok=True)
    total_lines = 0
    total_origin_lines = 0  # 去掉50字符以下的行后
    total_en_lines = 0
    total_no_harm_lines = 0
    total_gopher_lines = 0
    total_high_quality_lines = 0

    
    def read_file_in_blocks(file_path, n=5):
        result = []
        with open(file_path, "r", encoding='utf8') as f:
            block = []
            for line in f:
                block.append(line.strip('\n'))  # 去除行尾的换行符
                if len(block) == n:
                    result.append(' '.join(block))
                    block = []
            if block:  # 如果最后不足n行
                result.append(' '.join(block))
        return result

    with open(FIEL_AFTER_LANG, "w", encoding='utf8') as out_f:
        files = [file for file in os.listdir(WET_PATH)]
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(WET_PATH, file)
            with open(file_path, "r", encoding='utf8') as f:
                texts = read_file_in_blocks(file_path, LINES_PER_TEXT)
            
            total_lines += len(texts)
            # 过滤掉字符数特别少的行
            texts = [text for text in texts if len(text) > MIN_CHARS_PER_LINE]

            total_origin_lines += len(texts)

            # 按chunk划分
            chunks = [texts[i:i+CHUNK_SIZE] for i in range(0, len(texts), CHUNK_SIZE)]

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(predict_and_filter, chunk) for chunk in chunks]
                for future in tqdm(as_completed(futures), total=len(chunks), leave=False, desc=f"File: {file}"):
                    filtered_lines = future.result()
                    if len(filtered_lines) == 0:
                        continue

                    total_en_lines += len(filtered_lines)

                    # Filter harmful
                    # nsfw_labels, _ = nsfw_model.predict(filtered_lines)
                    # toxic_labels, _ = toxic_model.predict(filtered_lines)
                    # nsfw_labels = [label[0].replace('__label__', '') for label in nsfw_labels]
                    # toxic_labels = [label[0].replace('__label__', '') for label in toxic_labels]
                    # nsfw_labels, toxic_labels = np.array(nsfw_labels), np.array(toxic_labels)
                    # no_harm_mask = (nsfw_labels == "non-nsfw") & (toxic_labels == "non-toxic")
                    # if len(no_harm_mask) == 0:
                    #     continue
                    # filtered_lines = np.array(filtered_lines)[no_harm_mask]
                    # total_no_harm_lines += len(filtered_lines)

                    # 质量规则
                    filtered_lines = np.array(filtered_lines)
                    gopher_mask = np.array([run_gopher_quality_filter(line) for line in filtered_lines])
                    if len(gopher_mask) == 0:
                        continue
                    filtered_lines = list(filtered_lines[gopher_mask])
                    total_gopher_lines += len(filtered_lines)

                    # 质量分类器
                    # quality_labels, confidences = quality_model.predict(filtered_lines)
                    # quality_labels = [label[0].replace('__label__', '') for label in quality_labels]
                    # quality_labels, filtered_lines = np.array(quality_labels), np.array(filtered_lines)
                    # high_quality_mask = quality_labels == "high"
                    # filtered_lines = filtered_lines[high_quality_mask]
                    # total_high_quality_lines += len(filtered_lines)

                    # 写入文件
                    out_f.writelines([line + '\n' for line in filtered_lines])
                    
    print(f"总行数: {total_lines}")
    print(f"单行字符数大于{MIN_CHARS_PER_LINE}的行数: {total_origin_lines}")
    print(f"保留比率: {total_origin_lines / total_lines:.2f}")

    print(f"英文且置信度大于{SCORE_THRESHOLD}的行数: {total_en_lines}")
    print(f"保留比率: {total_en_lines / total_origin_lines:.2f}")

    print(f"gopher质量行数：{total_gopher_lines}")
    print(f"保留比率: {total_gopher_lines / total_en_lines:.2f}")


    # print(f"无害内容行数：{total_no_harm_lines}")
    # print(f"保留比率: {total_no_harm_lines / total_en_lines:.2f}")

    # print(f"gopher质量行数：{total_gopher_lines}")
    # print(f"保留比率: {total_gopher_lines / total_no_harm_lines:.2f}")

    # print(f"高质量行数：{total_high_quality_lines}")
    # print(f"保留比率: {total_high_quality_lines / total_gopher_lines:.2f}")




def filter_lang_en():
    FIEL_AFTER_LANG = "tmp/after_lang.txt"
    SCORE_THRESHOLD = 0.5
    CHUNK_SIZE = 100            # batch size
    LINES_PER_TEXT = 10

    model = fasttext.load_model(str(LANG_MODEL_PATH))

    def predict_and_filter(chunk_lines, log=False):
        chunk_lines = [line.strip() for line in chunk_lines]
        if not chunk_lines:
            return []
        labels, scores = model.predict(chunk_lines)
        # 仅保留英文且置信度大于0.3的行
        results = [
            line for line, label, score in zip(chunk_lines, labels, scores)
            if label[0].replace('__label__', '') == 'en' and score[0] > SCORE_THRESHOLD
        ]
        if log and not results:
            print('No results')
        return results
        
    # 保证tmp目录存在
    os.makedirs(os.path.dirname(FIEL_AFTER_LANG), exist_ok=True)
    total_origin_lines = 0
    total_en_lines = 0

    
    def read_file_in_blocks(file_path, n=5):
        result = []
        with open(file_path, "r", encoding='utf8') as f:
            block = []
            for line in f:
                block.append(line.strip('\n'))  # 去除行尾的换行符
                if len(block) == n:
                    result.append(' '.join(block))
                    block = []
            if block:  # 如果最后不足n行
                result.append(' '.join(block))
        return result

    with open(FIEL_AFTER_LANG, "w", encoding='utf8') as out_f:
        files = [file for file in os.listdir(WET_PATH)]
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(WET_PATH, file)
            with open(file_path, "r", encoding='utf8') as f:
                # lines = f.readlines()
                texts = read_file_in_blocks(file_path, LINES_PER_TEXT)
            # print(f"len texts: {len(texts)}")
            
            total_origin_lines += len(texts)

            # 按chunk划分
            chunks = [texts[i:i+CHUNK_SIZE] for i in range(0, len(texts), CHUNK_SIZE)]

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(predict_and_filter, chunk) for chunk in chunks]
                for future in tqdm(as_completed(futures), total=len(chunks), leave=False, desc=f"File: {file}"):
                    filtered_lines = future.result()
                    total_en_lines += len(filtered_lines)
                    for line in filtered_lines:
                        out_f.write(line + '\n')

    # 输出统计信息
    if total_origin_lines > 0:
        ratio = total_en_lines / total_origin_lines
    else:
        ratio = 0
    print(f"总行数: {total_origin_lines}")
    print(f"英文且置信度大于{SCORE_THRESHOLD}的行数: {total_en_lines}")
    print(f"保留比率: {ratio:.4f}")


def filter_harmful():
    FIEL_AFTER_LANG = "tmp/after_lang.txt"
    FIEL_AFTER_HARM = "tmp/after_harm.txt"
    HARM = "tmp/detected_harm.txt"
    

    nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
    toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)

    with open(FIEL_AFTER_LANG, "r", encoding='utf8') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines]

    for line in tqdm(lines):
        # print("#### line: ", line)
        labels, probs = nsfw_model.predict(line)
        label = labels[0].replace('__label__', '')  # e.g. "nsfw" 或 "non-nsfw"
        confidence = float(probs[0])
        # print(f"Predicted label: {label}, Confidence: {confidence}")
        # if 'non' not in label:
        #     print("#### line: ", line)
        #     print("##### Harmful")

        labels, probs = toxic_model.predict(line)
        label = labels[0].replace('__label__', '')  # e.g. "nsfw" 或 "non-nsfw"
        confidence = float(probs[0])
        # print(f"Predicted label: {label}, Confidence: {confidence}")

        # if 'non' not in label:
        #     print("#### line: ", line)
        #     print("##### Harmful")





if __name__ == "__main__":
    # filter_lang_en()
    # filter_harmful()
    filter_pipeline()