# Stanford CS336 assignments answer

This is my implementation of [Stanford CS336 assignments](https://github.com/stanford-cs336). I also share my notes while doing these assignments on [小红书](https://www.xiaohongshu.com/user/profile/5a9409554eacab27ee3c50b0). 

## Progress

### Pass `pytest`

- [x] Homework 1 of spring 2025
- [ ] Homework 2 of spring 2025
- [ ] Homework 3 of spring 2025
- [x] Homework 4 of spring 2025
- [ ] Homework 5 of spring 2025
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

