# 斯坦福 CS336 作业答案

中文 | [English](./README_en.md)

这是我完成的 [斯坦福 CS336 作业](https://github.com/stanford-cs336) 的实现。同时，我在 [小红书](https://www.xiaohongshu.com/user/profile/5a9409554eacab27ee3c50b0) 上分享了完成这些作业时的笔记。

# ⚠️注意
在我的另一个项目 [clean-llm](https://github.com/wingAGI/clean-llm) 中，我为作业 1 提供了一份全面的指南。它包含了完整的复现教程，你可以在半小时内完成训练分词器和模型。

# 进度

## 通过 `pytest`

- [x] 2025 年春季作业 1
- [ ] 2025 年春季作业 2
- [ ] 2025 年春季作业 3
- [x] 2025 年春季作业 4
- [x] 2025 年春季作业 5
- [x] 2024 年春季作业 5

## 实验

- [x] 训练自己的分词器
- [x] 训练并测试自己的 LLM

# 如何测试
### HW1
要通过测试，你可以：

1. 将原始仓库中的文件 `tests/adapters.py` 替换为 `sp25/hw1-basics/tests/adapters.py`。
2. 运行 `uv run pytest`。

通过测试后，你可以使用 `sp25/hw1-basics/scripts` 中的额外脚本来运行一些实验，具体步骤如下：

1. 将文本数据以 `txt` 格式下载到 `data/` 目录中。
2. 运行 `uv run python scripts/train_bpe.py`。训练分词器。
3. 运行 `uv run python scripts/tokenize.py`。使用训练好的分词器将原始训练数据从 `txt` 格式编码为整数 ID，并将其保存为一维数组，以便在训练期间轻松访问。
4. 运行 `uv run python scripts/train.py`。训练模型。默认配置是训练 5000 步，上下文长度为 256，批量大小为 32，总共 256 x 32 x 5000 = 4000 万 tokens。
5. 运行 `uv run python scripts/generate.py`。测试模型生成故事的能力。

### HW4
1. 将原始仓库中的文件 `tests/adapters.py` 替换为 `sp25/hw4-data/tests/adapters.py`。
2. 按照作业说明，将以下三个模型下载到 `cs336_data` 目录中：
    - lid.176.bin
    - jigsaw_fasttext_bigrams_nsfw_final.bin
    - jigsaw_fasttext_bigrams_hatespeech_final.bin
3. 训练你的质量分类器并保存为 `cs336_data/quality_classifier.ftz`。
4. 运行 `uv run pytest`。

实际上，`sp25/hw4-data/main.py` 包含了一个用于构建质量分类器的脚本，前提是您已经在 `cs336_data/download_pages` 目录中下载了高质量页面，可以尝试运行。

### HW4 指南：收集数据、过滤和去重
1. 在 `sp25/hw4-data/pipeline_data/` 中运行 `bash get_wet_parallel.sh` 来下载数据。
2. 运行 `uv run python tests/pipeline.py` 来过滤数据。
3. 运行 `uv run python tests/pipeline_dedup.py` 来进行去重。

### HW5
1. 将原始仓库中的文件 `tests/adapters.py` 替换为 `sp25/hw5-alignment/adapters.py`。
2. 修改 `tests/conftest.py` 中第 213 行的 `model_id`，将其更改为你的模型路径。
3. [可选] 如果你使用的是 Mac 并且无法安装 `flash-attn`（这会阻止你进行开发），你可以删除 `pyproject.toml` 中与 `flash-attn` 相关的行。
4. 运行 `uv run pytest`。
5. 你会看到 7 个失败和 22 个通过。2025 年春季的 7 个失败项是可选的。注意：2024 年的作业 5 是 DPO，2025 年改为了 GRPO。如果你仍然想了解 DPO，可以参考 2024 年的作业 5。

### 2024 年春季 HW5
请查看 `spring2024-assignment5-alignment/README.md`。