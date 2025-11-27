
# MedQA_LLM_Evaluation
评测大模型在 MedQA 基准上的得分（包括 USMLE 基准和 MCMLE 基准）, Evaluate LLMs’ scores on the MedQA benchmark (including USMLE benchmark and MCMLE benchmark).

**English Introduction:**

This project provides a convenient evaluation script that can be used to compute the scores of LLMs on the MedQA benchmark, supporting LoRA-fine-tuned models and MoE models. Users only need to provide the model name or a local model path to directly run the evaluation script and obtain the results. The MedQA dataset is already included in this project (located in `dataset/`), so no additional downloads are required. The commonly used USMLE benchmark and MCMLE benchmark are subsets of MedQA and can therefore be evaluated directly with this project.

The MedQA dataset comes from the [official MedQA repository](https://github.com/jind11/MedQA). It includes question sets from Mainland China medical exams (MCMLE), United States medical exams (USMLE), and Taiwan medical exams. All questions have been converted into single-choice format; MCMLE and USMLE each provide both 4-option and 5-option versions, while the Taiwan set only includes a 5-option version.

If you also need to evaluate LLM performance on PubMedQA, you may refer to [PubMedQA_LLM_Evaluation](https://github.com/liaoyanqing666/PubMedQA_LLM_Evaluation). The structure of the two projects is completely identical, and they can be combined; functions with the same names in corresponding files are fully consistent.

Below you will find detailed usage instructions, file descriptions, and common issues.

[Click here to jump directly to the English usage guide.](#how-to-use)

**Chinese Introdution:**

本项目提供了一个便捷的评测代码，可用于计算 LLM 在 MedQA 基准上的得分，支持 LoRA 微调后的模型与 MoE 模型。用户只需提供模型名称或本地路径，即可直接运行评测脚本并获得结果。MedQA 数据集已包含于本项目（位于 `dataset/`），无需额外下载。常见的 USMLE 数据集与 MCMLE 数据集均属于 MedQA 的子集，因此均可直接使用本项目进行评测。

MedQA 数据集来源于 [MedQA 官方仓库](https://github.com/jind11/MedQA)。其中包含大陆医学考试题集（MCMLE）、美国医学考试题集（USMLE）以及台湾医学考试题集。所有题目均被处理为单选题；MCMLE 和 USMLE 数据集均提供 4 选项与 5 选项版本，台湾医学考试题集仅有 5 选项版本。

如果你也需要评测 LLM 在 PubMedQA 上的表现，可参考 [PubMedQA_LLM_Evaluation](https://github.com/liaoyanqing666/PubMedQA_LLM_Evaluation)。两个项目结构完全一致，可合并使用，同名文件中的同名函数保持完全一致。

下面我会详细介绍使用方法、文件说明以及常见问题和现象。

[点击此处直接跳转至中文使用说明。](#如何使用)

---

### How to Use:

1. **Clone this project locally**

   ```bash
   git clone https://github.com/liaoyanqing666/MedQA_LLM_Evaluation.git
    ```

2. **Install dependencies**

   You need to manually install `vllm`, `transformers`, `torch`, and other dependencies. If you want to use `flash-attn` acceleration, please install it yourself and ensure it is available.

3. **Modify the evaluation model and runtime parameters**

   Modify the parameters under `if __name__ == "__main__"` in either `bench_eval.py` or `bench_eval_vllm.py` (recommended). For example, in `bench_eval_vllm.py`:

   ```python
   if __name__ == "__main__":
       eval_medqa(
           model_path="/nfsdata4/Qwen/Qwen3-32B",  # Replace with your local model path/online model name
           # lora_path="models/sft_Qwen3",  # Optional: path to LoRA weights
           data_paths=[
               "dataset/MedQA/questions/US/test.jsonl",  # USMLE-5options
               "dataset/MedQA/questions/Mainland/test.jsonl",  # MCMLE-5options
               "dataset/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl",  # USMLE-4options
               "dataset/MedQA/questions/Mainland/4_options/test.jsonl",  # MCMLE-4options
           ],
           visible_gpus="2",
           max_tokens=32768,
           print_errors=False,
           record_file=False,
       )
   ```

   The project does not currently provide an `argparse`-based command-line interface; this may be added later if demand is high. PRs are welcome.

4. **(Optional) View the full model outputs**

   If you set `record_file=True` in the previous step, the script will generate a file named `{original_filename}_eval_{model_name}.jsonl` in the same directory, containing the model's complete outputs and parsed results.

   If you want to recompute accuracy or other metrics, you can run the `jsonl_eval.py` script, set its `path` to the generated file, and run the script to obtain evaluation results.
=

### File Descriptions:

* `bench_eval.py`: A script that evaluates models using `transformers`. It has higher compatibility but is extremely slow. (Reference speed: 32B full model, bf16, dual A800 GPUs, roughly one sample per minute)

* `bench_eval_vllm.py`: A script that evaluates models using `vllm`. This is the recommended script. Currently, only this script supports evaluating LoRA-fine-tuned models. It is compatible with most popular base and fine-tuned models (such as Baichuan-m2). See the [vllm official support documentation](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md). It is extremely fast. (Reference speed: 32B full model, bf16, dual A800 GPUs, under one second per sample)

* `bench_eval_messages.py`: Contains the message format used when the model is executed, i.e., the prompt templates. Also includes functions for parsing model outputs.

* `jsonl_eval.py`: A script for evaluating recorded model outputs. It can compute accuracy, valid accuracy, longest valid text length, shortest invalid text length, and other metrics. The longest-valid and shortest-invalid lengths help users tune the `max_tokens` parameter. (Note that the lengths are based on string length, not token length.)

* `dataset/`: Folder containing the original, unprocessed MedQA dataset. No additional downloads are required.

* `README.md`: This file, the project documentation.

* `LICENSE`: The project uses the Apache-2.0 license.

* `.gitignore`: Git ignore configuration.


### Common Issues and Phenomena:

1. **Illegal model output**

   If the model output cannot be correctly parsed into an option, it is treated as an invalid answer. This often occurs when the model falls into repetitive looping and fails to produce an answer, or when `max_tokens` is set too low, causing truncation, non-option outputs, or multiple options. If the model’s content is correct but the formatting prevents parsing, you may modify the `parse_model_response` function in `bench_eval_messages.py`, or manually correct the saved output and evaluate accuracy using `jsonl_eval.py`.

2. **Extremely slow progression at the end of vllm evaluation**

   Due to vllm's continuous batching mechanism, samples with extremely long outputs may slow down the final part of evaluation. These are typically illegal looping outputs and can be truncated by setting an appropriate `max_tokens` value.

3. **vllm outputs are not strictly reproducible**

   Due to vllm’s internal mechanisms, the same model may not produce identical outputs across runs (even when randomness is disabled), especially when evaluated on multiple GPUs. This behavior is inherent to vllm. Turning off multiprocessing may help, but the current code does not provide such options. Future updates may explore solutions; PRs are welcome.

---

### 如何使用：

1. **克隆本项目到本地**

   ```bash
   git clone https://github.com/liaoyanqing666/MedQA_LLM_Evaluation.git
    ```

2. **安装依赖**

   需自行安装 `vllm`、`transformers`、`torch` 等依赖。如果希望使用 `flash-attn` 加速，请自行安装并确保可用。

3. **修改评测模型和运行参数**

   在 `bench_eval.py` 或 `bench_eval_vllm.py`（推荐）中修改 `if __name__ == "__main__"` 下的参数。例如在 `bench_eval_vllm.py` 中：

   ```python
   if __name__ == "__main__":
       eval_medqa(
           model_path="/nfsdata4/Qwen/Qwen3-32B",  # Replace with your local model path/online model name
           # lora_path="models/sft_Qwen3",  # Optional: path to LoRA weights
           data_paths=[
               "dataset/MedQA/questions/US/test.jsonl",  # USMLE-5options
               "dataset/MedQA/questions/Mainland/test.jsonl",  # MCMLE-5options
               "dataset/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl",  # USMLE-4options
               "dataset/MedQA/questions/Mainland/4_options/test.jsonl",  # MCMLE-4options
           ],
           visible_gpus="2",
           max_tokens=32768,
           print_errors=False,
           record_file=False,
       )
   ```

   当前未提供 `argparse` 命令行接口，后续若有需求的人多将会添加，也欢迎提交 PR。

4. **（可选）查看模型全部完整输出**

   在上一步将 `record_file=True` 后，会在相同目录下生成 `{原文件名}_eval_{模型名}.jsonl`，其中包含模型的完整生成内容及解析结果。

   在此基础上，如果你想重新查看模型的正确率等信息，可以运行 `jasonl_eval.py` 脚本，修改其中的 `path` 为对应文件路径，然后运行脚本即可得到评测结果。


### 文件说明：

* `bench_eval.py`：使用 `transformers` 加载模型进行评测的脚本。兼容性更强，但速度极慢。（参考速度：32B 全量模型，bf16，A800 双卡，一分钟余一条）

* `bench_eval_vllm.py`：使用 `vllm` 加载模型进行评测的脚本，推荐优先使用此脚本。暂时仅此脚本支持 LoRA 微调模型的评测。兼容绝大部分常用模型及其基础上的微调模型（如 Baichuan-m2），详见 [vllm 官方支持文档](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md)。速度极快。（参考速度：32B 全量模型，bf16，A800 双卡，不到一秒一条）

* `bench_eval_messages.py`：包含模型实际运行时的 message 格式，即 prompt 模板，也包含解析模型输出结果的函数。

* `jsonl_eval.py`：对记录的模型生成结果进行评测的脚本，可以计算模型正确率、有效正确率、最长有效文本长度、最短无效文本长度等指标。最长有效文本长度和最短无效文本长度可以帮助用户调试模型 `max_tokens` 参数。（但注意这两个长度是通过 string 长度计算的，并非 token 长度。）

* `dataset/`：包含原始完全未处理的 MedQA 数据集的文件夹，无需额外下载。

* `README.md`：本文件，项目说明文件。

* `LICENSE`：本项目采用 Apache-2.0 许可证。

* `.gitignore`：git 忽略文件。


### 常见问题和现象：

1. **模型输出非法**

   如果模型输出的内容无法被正确解析为选项，则会被视为非法输出，计入无效回答。通常是由于模型陷入循环输出，未能给出答案。也可能是 `max_tokens` 设置过小，导致模型输出被截断、模型输出非候选选项 / 多选 等。如果是模型输出内容正确，但格式出错导致无法解析，可以尝试修改 `bench_eval_messages.py` 中的 `parse_model_response` 函数以适应模型的输出格式，或者打开保存的文件查看具体输出内容，手动修正解析结果后使用 `jsonl_eval.py` 计算正确率。

2. **vllm 评测时最后的进度极慢**

   由于 vllm 的 continuous batching 机制，一些生成内容极长的样本会导致最后的速度非常慢。通常而言，这些样本是陷入循环输出的非法样本，可以通过设置 `max_tokens` 参数来限制最大生成长度，从而直接截断这些样本。

3. **vllm 输出结果不固定（无严格可复现性）**

   由于 vllm 的自身机制，可能会导致同一模型在不同运行中输出结果 / 评测结果不完全相同（即使设置了各种去随机方法），尤其是在使用多卡进行评测时更为明显。这是 vllm 的设计机制决定的，可能可以通过关闭多进程等方法缓解，但本代码暂未提供相关选项，后续可能会测试，也欢迎 PR。

---

### If you have any question of the code, please contact me or leave an issue. My email: 1793706453@qq.com
