import os
import re
import random
import hashlib
import unicodedata
from tqdm import tqdm
from collections import defaultdict
# from adapters import normalize_text, get_ngrams, minhash_signature, jaccard, stable_hash_func
from multiprocessing import Pool, cpu_count

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

# def stable_hash_func(seed):
#     # Returns a stable hash function for strings using built-in hash + salt
#     def _hash(x):
#         # For tuples: join to a single str
#         s = " ".join(x) if isinstance(x, (tuple, list)) else str(x)
#         return hash((seed, s))
#     return _hash

def stable_hash_func(seed):
    def _hash(x):
        s = " ".join(x) if isinstance(x, (tuple, list)) else str(x)
        v = f"{seed}-{s}".encode('utf-8')
        # 使用稳定的哈希算法，比如SHA1，输出int
        return int(hashlib.sha1(v).hexdigest(), 16)
    return _hash

# import mmh3
# def stable_hash_func(seed):
#     def _hash(x):
#         s = " ".join(x) if isinstance(x, (tuple, list)) else str(x)
#         return mmh3.hash(s, seed, signed=False)
#     return _hash


# def minhash_worker(args):
#     ngram_set, seeds = args
#     hash_funcs = [stable_hash_func(seed) for seed in seeds]
#     return minhash_signature(ngram_set, hash_funcs)

def minhash_worker(args):
    ngram_set, seeds = args
    hash_funcs = [stable_hash_func(seed) for seed in seeds]
    return minhash_signature(ngram_set, hash_funcs)



def run_minhash_deduplication(
    input_file: os.PathLike,        # 每行作为一个doc
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_path: os.PathLike,
    chunk_size: int = 100000 # 100000
):
    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()

    # min_length = min(len(line) for line in lines)
    # import pdb; pdb.set_trace()     # 188
    print(f"Total chunks = {len(lines) // chunk_size + 1}")
    for chunk_idx, strat in tqdm(enumerate(range(0, len(lines), chunk_size)), desc="Processing chunks", leave=True):
        chunk = lines[strat:strat+chunk_size]

        # 1. preprocess all input docs
        docs = []
        cnt = 0
        for txt in tqdm(chunk, desc=f"Reading lines of chunk {chunk_idx}"):
            cnt += 1

            norm_txt = normalize_text(txt)
            words = norm_txt.split()
            ngram_set = get_ngrams(words, ngrams)
            docs.append({
                "text": txt,          # original
                "norm": norm_txt,     # normalized
                "ngrams": ngram_set,  # set of tuple ngrams
            })
        
        # 2. minhash signature for each doc
        # hash_funcs = [stable_hash_func(seed) for seed in range(num_hashes)]
        # for d in tqdm(docs, desc="Minhash signatures"):
        #     d["minhash"] = minhash_signature(d["ngrams"], hash_funcs)

        hash_func_seeds = list(range(num_hashes))
        tasks = [(d["ngrams"], hash_func_seeds) for d in docs]
        # tasks = [(d["ngrams"], hash_func_seeds, idx) for idx, d in enumerate(docs)]

        
        # tasks = [(d["ngrams"], hash_funcs) for d in docs]
        with Pool() as pool:
            signatures = list(tqdm(pool.imap(minhash_worker, tasks), total=len(tasks), desc="Minhash signatures"))

        for d, sig in zip(docs, signatures):
            d["minhash"] = sig


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

        # 计算pairwise jaccard
        for i, j in tqdm(candidate_pairs, desc="Jaccard"):
            jac = jaccard(docs[i]["ngrams"], docs[j]["ngrams"])
            if jac >= jaccard_threshold:
                union(i, j)


        # Build clusters
        clusters_dict = defaultdict(list)
        for i in range(len(docs)):
            root = find(i)
            clusters_dict[root].append(i)


        # 6. Write output: cluster代表写一个即可
        saved_cnt = 0
        with open(output_path, "a", encoding="utf-8") as f:
            for cluster in clusters_dict.values():
                idx_to_keep = random.choice(cluster)
                f.write(docs[idx_to_keep]["text"])
                saved_cnt += 1


        print(f"Saved {saved_cnt} docs, ratio = {saved_cnt / len(docs):.2f}")




if __name__ == "__main__":
    # filter_lang_en()
    # filter_harmful()
    # filter_pipeline()
    # run_minhash_deduplication("tmp/after_lang.txt", 128, 4, 3, 0.5, "tmp/dedup.txt")

    run_minhash_deduplication(input_file="tmp/after_lang.txt", 
                              num_hashes=128,
                              num_bands=16,
                              ngrams=2, 
                              jaccard_threshold=0.5, 
                              output_path="tmp/dedup.txt",
                              chunk_size=100000)