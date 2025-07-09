from __future__ import annotations

import os
import re
import fasttext
import pathlib
import hashlib
import unicodedata
import random
from typing import Any
from collections import defaultdict
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

random.seed(2)


DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "cs336_data"
LANG_MODEL_PATH = os.path.join(DATA_PATH, "lid.176.bin")
NSFW_MODEL_PATH = os.path.join(DATA_PATH, 'jigsaw_fasttext_bigrams_nsfw_final.bin')
TOXIC_MODEL_PATH = os.path.join(DATA_PATH, 'jigsaw_fasttext_bigrams_hatespeech_final.bin')
QUALITY_MODEL_PATH = os.path.join(DATA_PATH, 'quality_classifier.ftz')

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        # 尝试用 UTF-8 解码
        html_string = html_bytes.decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        # 如果 UTF-8 解码失败，检测编码并尝试解码
        encoding = detect_encoding(html_bytes)
        if encoding:
            try:
                html_string = html_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                # 如果检测到的编码也无法解码，则返回 None
                print("无法识别编码，encoding:", encoding)
                # import pdb; pdb.set_trace()
                return None
        else:
            # 如果无法检测到编码，则返回 None
            print("无法检测到编码")
            return None

    # 使用 resiliparse 提取纯文本
    extracted_text = extract_plain_text(html_string)

    return extracted_text

def run_identify_language(text: str) -> tuple[Any, float]: 
    model = fasttext.load_model(str(LANG_MODEL_PATH))         
    def identify_lang_sentence(sentence: str) -> tuple[Any, float]:
        predictions = model.predict(sentence, k=1)  

        language_label = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]  
        
        if language_label == "en" or language_label == "zh":
            return (language_label, confidence)
        
        return (language_label, confidence)

    return identify_lang_sentence(text.replace("\n", " "))
    
def run_mask_emails(text: str) -> tuple[str, int]:
    # 定义匹配电子邮件的正则表达式模式
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    
    # 替换所有匹配的邮件地址为 "|||EMAIL_ADDRESS|||"
    masked_text, count = re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)
    
    # 返回处理后的文本和替换次数
    return (masked_text, count)

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    # 这里正则覆盖多种常见美国电话号码格式
    pattern = re.compile(
        r'''
        (?:(?:\+?1[\s\-.]*)?)         # 可选国家码
        (?:\(?\d{3}\)?)[\s\-\.]*      # 区号，可以有括号，后可跟空格、横线、点
        \d{3}[\s\-\.]*\d{4}           # 主体7位数字（3位+4位），中间分隔符可有
        ''', re.VERBOSE
    )

    replaced_str, count = pattern.subn("|||PHONE_NUMBER|||", text)
    return replaced_str, count

def run_mask_ips(text: str) -> tuple[str, int]:
    # IPv4 pattern, numbers from 0 to 255
    octet = r'(?:25[0-5]|2[0-4][0-9]|1?\d\d?)'
    ip_pattern = rf'\b{octet}\.{octet}\.{octet}\.{octet}\b'

    # Find all matches for counting
    matches = re.findall(ip_pattern, text)
    count = len(matches)

    # Replace matches
    new_text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)

    return new_text, count

def run_classify_nsfw(text: str) -> tuple[Any, float]:
    nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
    labels, probs = nsfw_model.predict(text.strip().replace('\n', ' '), k=1)
    label = labels[0].replace('__label__', '')  # e.g. "nsfw" 或 "non-nsfw"
    confidence = float(probs[0])
    return label, confidence

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)
    processed_text = text.strip().replace('\n', ' ')
    labels, scores = toxic_model.predict(processed_text, k=1)
    label = labels[0].replace('__label__', '')
    score = float(scores[0])
    return label, score

def run_gopher_quality_filter(text: str) -> bool:
    # 分词
    words = word_tokenize(text)

    # 单词数量在 [50, 100000]
    num_words = len(words)
    if num_words < 50 or num_words > 100000:
        return False

    # 平均单词长度在 [3, 10]
    word_lens = [len(w) for w in words]
    mean_word_len = sum(word_lens) / num_words if num_words > 0 else 0
    if mean_word_len < 3 or mean_word_len > 10:
        return False

    # 超过30%的行以省略号结尾要过滤
    lines = text.splitlines()
    if lines:
        lines_with_ellipsis = sum(1 for line in lines if line.rstrip().endswith("..."))
        if (lines_with_ellipsis / len(lines)) > 0.3:
            return False

    # 至少80%单词含有字母
    words_with_alpha = sum(1 for w in words if re.search('[a-zA-Z]', w))
    if num_words == 0 or (words_with_alpha / num_words) < 0.8:
        return False

    return True

def run_classify_quality(text: str) -> tuple[Any, float]:
    text = text.replace('\n', ' ')  # remove '\n'
    model = fasttext.load_model(QUALITY_MODEL_PATH)
    labels, probs = model.predict([text], k=1)  # top1
    # labels: e.g. [ ['__label__high'] ]
    # probs: e.g. [ [0.85] ]
    pred_label = labels[0][0].replace("__label__", "")
    pred_prob = probs[0][0]

    return (pred_label, float(pred_prob))

def hash_line(line):
    # 使用SHA256确保不同内容出现hash碰撞概率极低，且保持固定长度
    return hashlib.sha256(line.encode('utf-8')).hexdigest()

def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    # 第一步：遍历统计所有行出现的次数（用hash作key，减少内存）
    line_count = defaultdict(int)
    line_content = dict()  # 哈希值到真正行内容的映射（第一个遇到hash值时存原文），避免后续写入
    for path in input_files:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                h = hash_line(line)
                line_count[h] += 1
                if h not in line_content:
                    line_content[h] = line

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 第二步：重写每个文件，只保留unique的（只出现过一次的）行
    for path in input_files:
        filename = os.path.basename(path)
        output_path = os.path.join(output_directory, filename)
        with open(path, 'r', encoding='utf-8') as fr, \
             open(output_path, 'w', encoding='utf-8') as fw:
            for line in fr:
                line = line.rstrip('\n')
                h = hash_line(line)
                if line_count[h] == 1:
                    fw.write(line + '\n')

def normalize_text(text):
    # Lowercase
    text = text.lower()
    # Unicode NFD
    text = unicodedata.normalize("NFD", text)
    # Remove accents
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_ngrams(words, n):
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

def minhash_signature(ngrams, hash_funcs):
    # ngrams: set of ngram tuples
    sig = []
    for hf in hash_funcs:
        sig.append(min(hf(ng) for ng in ngrams) if ngrams else 0)
    return tuple(sig)

def jaccard(set1, set2):
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def stable_hash_func(seed):
    # Returns a stable hash function for strings using built-in hash + salt
    def _hash(x):
        # For tuples: join to a single str
        s = " ".join(x) if isinstance(x, (tuple, list)) else str(x)
        return hash((seed, s))
    return _hash

def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    # 1. preprocess all input docs
    os.makedirs(output_directory, exist_ok=True)
    docs = []
    filenames = []
    for path in input_files:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        norm_txt = normalize_text(txt)
        words = norm_txt.split()
        ngram_set = get_ngrams(words, ngrams)
        docs.append({
            "text": txt,          # original
            "norm": norm_txt,     # normalized
            "ngrams": ngram_set,  # set of tuple ngrams
        })
        filenames.append(os.path.basename(path))
    
    # 2. minhash signature for each doc
    hash_funcs = [stable_hash_func(seed) for seed in range(num_hashes)]
    for d in docs:
        d["minhash"] = minhash_signature(d["ngrams"], hash_funcs)

    # 3. LSH: bucket by band
    band_size = num_hashes // num_bands
    buckets = defaultdict(list)  # (band#，band tuple值): [doc idx]
    for idx, d in enumerate(docs):
        sig = d["minhash"]
        for band in range(num_bands):
            lo = band * band_size
            hi = lo + band_size
            band_val = tuple(sig[lo:hi])
            buckets[(band, band_val)].append(idx)
    
    # 4. Candidate duplicate pairs (by bucket, undirected, distinct)
    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i+1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    if a != b:
                        candidate_pairs.add(tuple(sorted((a, b))))
    
    # 5. For each candidate pair, compute "real" jaccard over ngrams of normalized
    clusters = []  # list of sets of doc indices
    parent = list(range(len(docs)))  # Union-Find DSU

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, j in candidate_pairs:
        jac = jaccard(docs[i]["ngrams"], docs[j]["ngrams"])
        if jac >= jaccard_threshold:
            union(i, j)

    # Build clusters
    clusters_dict = defaultdict(list)
    for i in range(len(docs)):
        root = find(i)
        clusters_dict[root].append(i)
    
    # For each cluster, randomly keep one doc, others are dropped
    keep_idx = set()
    for cluster in clusters_dict.values():
        idx_to_keep = random.choice(cluster)
        keep_idx.add(idx_to_keep)

    # 6. Write output files (keep only not duplicates, or in a dup cluster,只保留被选中的)
    for idx, doc in enumerate(docs):
        filename = filenames[idx]
        output_path = os.path.join(output_directory, filename)
        if find(idx) == idx or idx in keep_idx:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc["text"])
        else:
            # skip: this is a duplicate, removed
            continue
