# Stanford CS336 assignments answer

[中文](./README.md) | English

This is my implementation of [Stanford CS336 assignments](https://github.com/stanford-cs336). I also share my notes while doing these assignments on [小红书](https://www.xiaohongshu.com/user/profile/5a9409554eacab27ee3c50b0). 

# News
I have a comprehensive guide for assignment 1 in my other project [clean-llm](https://github.com/wingAGI/clean-llm). It includes a complete tutorial for reproduction, and you can finish training the tokenizer and the model in half an hour.

# Progress

## Pass `pytest`

- [x] Homework 1 of spring 2025
- [ ] Homework 2 of spring 2025
- [ ] Homework 3 of spring 2025
- [x] Homework 4 of spring 2025
- [x] Homework 5 of spring 2025
- [x] Homework 5 of spring 2024

## Experiments

- [x] Train your own tokenizer
- [x] Train and test your own LLM


# How to test
### HW1
To pass tests, you can:

(1) Replace file `tests/adapters.py` in the original reppo with `sp25/hw1-basics/tests/adapters.py`.

(2) Run `uv run pytest`.

After your pass tests, you can use extra scripts in `sp25/hw1-basics/scripts` to run some experiments, see the following instructions:

1. Download the text data in `txt` format to the `data/` directory.
2. Run `uv run python scripts/train_bpe.py`. Train the tokenizer. 
3. Run `uv run python scripts/tokenize.py`. Use the trained tokenizer to encode the original training data in `txt` format into integer ids and save it as a one-dimensional array in dat format for easy access during training.
4. Run `uv run python scripts/train.py`. Train the model. The default configuration is to train for 5000 steps with a context length of 256 and a batch size of 32, which is a total of 256 x 32 x 5000 = 40M tokens.
5. Run `uv run python scripts/generate.py`. Test the model's ability to generate stories.

### HW4
(1) Replace file `tests/adapters.py` in the original reppo with `sp25/hw4-data/tests/adapters.py`.

(2) Download these three models to the `cs336_data` directory as instructed in the assignment.

    - lid.176.bin
    - jigsaw_fasttext_bigrams_nsfw_final.bin
    - jigsaw_fasttext_bigrams_hatespeech_final.bin

(3) Train your quality classifier and save as `cs336_data/quality_classifier.ftz`.

(4) Run `uv run pytest`.


In fact, `sp25/hw4-data/main.py` contains a script for building a quality classifier, on the condition that you already have high-quality pages downloaded in the `cs336_data/download_pages` directory, feel free to try it.

### HW4 Guide: Collect data, filtering and deduplication
(1) Run `bash get_wet_parallel.sh` in `sp25/hw4-data/pipeline_data/` to download data.
(2) Run `uv run python tests/pipeline.py` to filter data.
(3) Run `uv run python tests/pipeline_dedup.py` to do deduplication.

### HW5
(1) Replace file `tests/adapters.py` in the original reppo with `sp25/hw5-alignment/adapters.py`.

(2) Modify the model_id on line 213 of `tests/conftest.py` to the location of your own model.

(3) [Optional] If you are using a Mac and can't install `flash-attn`, which prevents you from developing, you can delete the lines related to `flash-attn` in `pyproject.toml`.

(4) Run `uv run pytest`.

(5) You will see 7 failed and 22 passed. The 7 failed ones are optional in spring 2025. Note: The assignment 5 for the year 2024 was DPO, and it was changed to GRPO in 2025. If you still want to learn about DPO, you can refer to the assignment 5 from 2024.


### HW5 of spring2024
See `spring2024-assignment5-alignment/README.md`.

