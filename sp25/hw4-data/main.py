import os
import pathlib
from tqdm import tqdm

from adapters import run_extract_text_from_html_bytes
from adapters import run_identify_language
from adapters import run_mask_emails, run_mask_ips, run_mask_phone_numbers
from adapters import run_gopher_quality_filter

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "cs336_data"
HTML_FOLDER = os.path.join(DATA_PATH, "downloaded_pages")
QUALITY_MODEL_PATH = os.path.join(DATA_PATH, 'quality_classifier.ftz')

def test_extract_text_from_warc_file():
    filepath = os.path.join(DATA_PATH, "CC-MAIN-20250417135010-20250417165010-00065.warc")
    with open(filepath, "rb") as f:
        warc_bytes = f.read()
    # subset the warc_bytes
    warc_bytes = warc_bytes[:1000000]
    text = run_extract_text_from_html_bytes(warc_bytes)
    with open(os.path.join(DATA_PATH, "warc_extracted.txt"), "w") as f:
        f.write(text)


def test_identify_language():
    with open(os.path.join(DATA_PATH, "manually_selected.txt"), "r") as f:
        lines = f.readlines()

    for text in lines:
        predicted_language, score = run_identify_language(text)
        print(f"Predicted language: {predicted_language}, Score: {score}")


def test_mask_pii_from_warc():
    import random
    filepath = os.path.join(DATA_PATH, "warc_extracted.txt")
    with open(filepath, "r") as f:
        text = f.read()

    sample_lines = text.replace('\n', ' ')

    for mask_pii in [run_mask_emails, run_mask_ips, run_mask_phone_numbers]:
        n_replacements = 0
        for idx, line in enumerate(sample_lines):
            masked, masked_cnt = mask_pii(line)
            if masked != line:
                n_replacements += 1
                print(f"Example #{n_replacements}")
                print(f"Original: {line}")
                print(f"Masked  : {masked}\n")
        
        print(f"Total replacements made: {n_replacements} out of {len(sample_lines)} samples")


def run_text_preprocess(text: str) -> str | None:
    """
    1. run_gopher_quality_filter
    2. mask_pii
    """
    if not run_gopher_quality_filter(text):
        return None

    n_replacements = 0
    text = text.replace('\n', ' ')

    masked_cnt_total = 0
    masked = text
    # 依次应用三个mask
    for mask_pii in [run_mask_emails, run_mask_ips, run_mask_phone_numbers]:
        masked, masked_cnt = mask_pii(masked)
        masked_cnt_total += masked_cnt
    
    # 输出仅当有替换发生时
    if masked != text:
        n_replacements += 1

    return masked


def test_form_dataset_quality_classifier():
    """
    处理、保存一体化，防止内存溢出
    """
    html_folder = HTML_FOLDER
    num_examples = 0
    # 遍历下载好的网页
    files = os.listdir(html_folder)[:5000]
    for file_path in tqdm(files, desc="Processing wiki htmls", unit="file"):
        if file_path.endswith(".html"):
            with open(os.path.join(html_folder, file_path), "rb") as f:
                html_bytes = f.read()
            text = run_extract_text_from_html_bytes(html_bytes)
            if text:
                text = run_text_preprocess(text)
                if text:
                    with open(os.path.join(DATA_PATH, "train.txt"), 'a', encoding='utf-8') as f:
                        f.write(f"__label__wiki {text}\n")
                    num_examples += 1
    
    
    num_bytes_per_example = 100000
    filepath = os.path.join(DATA_PATH, "CC-MAIN-20250417135010-20250417165010-00065.warc")
    with open(filepath, "rb") as f:
        warc_bytes = f.read()
    i = 0

    num_examples_cc = 0
    with tqdm(total=num_examples, desc="extracting examples") as pbar:
        while num_examples_cc < num_examples:
            warc_bytes_example = warc_bytes[i * num_bytes_per_example : (i + 1) * num_bytes_per_example]
            text = run_extract_text_from_html_bytes(warc_bytes_example)
            if text:
                text = run_text_preprocess(text)
                if text:
                    with open(os.path.join(DATA_PATH, "train.txt"), 'a', encoding='utf-8') as f:
                        f.write(f"__label__cc {text}\n")
                    pbar.update(1)
                    num_examples_cc += 1
            
            i += 1

def test_train_fasttest_model():
    import fasttext

    # 训练
    model = fasttext.train_supervised(
        input=os.path.join(DATA_PATH, "train.txt"), 
        lr=0.5, 
        epoch=5, 
        wordNgrams=2, 
        dim=100,
        verbose=2,
    )

    model.save_model(QUALITY_MODEL_PATH)

    print("训练完成, （训练集上）测试结果如下:")
    result = model.test(os.path.join(DATA_PATH, "train.txt"))
    print(f"Precision: {result[1]}")
    print(f"Recall:    {result[2]}")
    print(f"Tested samples: {result[0]}")



if __name__ == "__main__":
    ### Problem (extract_text) (b)
    # test_extract_text_from_warc_file()

    ### Problem (language_identification) (c)
    # test_identify_language()

    ### Problem (mask_pii) 5.
    # test_mask_pii_from_warc()

    ### Problem (quality_classifier) (a)
    # test_form_dataset_quality_classifier()
    # test_train_fasttest_model()

    pass

    