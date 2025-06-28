# Stanford CS336 assignments answer

This is my implementation of [Stanford CS336 assignments](https://github.com/stanford-cs336). I also share my notes while doing these assignments on [小红书](https://www.xiaohongshu.com/user/profile/5a9409554eacab27ee3c50b0). 

## Progress

### Pass `pytest`

- [x] Homework 1 of spring 2025
- [ ] Homework 2 of spring 2025
- [ ] Homework 3 of spring 2025
- [x] Homework 4 of spring 2025
- [x] Homework 5 of spring 2025
- [x] Homework 5 of spring 2024



## How to test
### HW1
(1) Replace file `tests/adapters.py` in the original reppo with `sp25/hw1-basics/adapters.py`.

(2) Run `uv run pytest`.

### HW2
(1) Replace file `tests/adapters.py` in the original reppo with `sp25/hw4-data/adapters.py`.

(2) Download these three models to the `cs336_data` directory as instructed in the assignment.

    - lid.176.bin
    - jigsaw_fasttext_bigrams_nsfw_final.bin
    - jigsaw_fasttext_bigrams_hatespeech_final.bin

(3) Train your quality classifier and save as `cs336_data/quality_classifier.ftz`.

(4) Run `uv run pytest`.


In fact, `sp25/hw4-data/main.py` contains a script for building a quality classifier, on the condition that you already have high-quality pages downloaded in the `cs336_data/download_pages` directory, feel free to try it.

### HW5
(1) Replace file `tests/adapters.py` in the original reppo with `sp25/hw5-alignment/adapters.py`.

(2) Modify the model_id on line 213 of `tests/conftest.py` to the location of your own model.

(3) [Optional] If you are using a Mac and can't install `flash-attn`, which prevents you from developing, you can delete the lines related to `flash-attn` in `pyproject.toml`.

(4) Run `uv run pytest`.

(5) You will see 7 failed and 22 passed. The 7 failed ones are optional in spring 2025. Note: The assignment 5 for the year 2024 was DPO, and it was changed to GRPO in 2025. If you still want to learn about DPO, you can refer to the assignment 5 from 2024.


### HW5 of spring2024
See `spring2024-assignment5-alignment/README.md`.