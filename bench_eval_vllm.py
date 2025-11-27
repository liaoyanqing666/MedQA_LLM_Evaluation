import os
import json
from re import L
import traceback
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from bench_eval_messages import *

torch.backends.cuda.enable_flash_sdp(True)

def load_model_and_tokenizer(model_path: str,
                             visible_gpus: str,
                             dtype=torch.bfloat16,
                             enable_lora: bool = False,
                             max_tokens: int | None = None):
    """
    Load model and tokenizer.
    """
    gpu_list = [g.strip() for g in str(visible_gpus).split(",") if g.strip()]
    clean_visible = ",".join(gpu_list) if gpu_list else ""
    if clean_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = clean_visible

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='left',
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == torch.bfloat16:
        vllm_dtype = "bfloat16"
    elif dtype == torch.float16:
        vllm_dtype = "float16"
    else:
        vllm_dtype = "auto"

    tp_size = max(len(gpu_list), 1)

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype=vllm_dtype,
        enable_lora=enable_lora,
        gpu_memory_utilization=0.95,
        max_model_len=max_tokens,
    )

    llm.name_or_path = model_path
    return llm, tokenizer


def generate_batch(model,
                   tokenizer,
                   conversations,
                   max_tokens: int = None,
                   lora_path: str | None = None):
    """
    Generate responses for all conversations.
    """
    texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
        )
        for conv in conversations
    ]

    if max_tokens is None:
        engine_cfg = getattr(getattr(model, "llm_engine", None), "model_config", None)
        if engine_cfg is not None and hasattr(engine_cfg, "max_model_len"):
            max_tokens = engine_cfg.max_model_len
        else:
            max_tokens = 30602

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id]
        if tokenizer.eos_token_id is not None else None,
        skip_special_tokens=True,
    )

    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest(
            lora_name="default_lora",
            lora_int_id=1,
            lora_path=lora_path,
        )

    outputs = model.generate(
        texts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )

    responses = []
    for out in outputs:
        if out.outputs:
            text = out.outputs[0].text
        else:
            text = ""
        responses.append(text.strip())
    return responses


def eval_single_medqa_jsonl(path: str,
                      model,
                      tokenizer,
                      build_prompt_fn,
                      max_tokens: int = None,
                      print_errors: bool = True,
                      record_file: bool = False,
                      lora_path: str | None = None):
    data = []
    """
    Evaluate the model on a single MedQA JSONL file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lstrip().startswith("//"):
                continue
            data.append(json.loads(line))

    total = len(data)

    invalid_cnt = 0
    valid_cnt = 0
    correct_cnt = 0
    error_list = []
    error_details = []

    conversations = []
    for item in data:
        q = item["question"]
        opts = item["options"]
        conversations.append(build_prompt_fn(q, opts))

    try:
        # Generate model responses. For vllm, all data are processed in "one" batch.
        responses = generate_batch(
            model=model,
            tokenizer=tokenizer,
            conversations=conversations,
            max_tokens=max_tokens,
            lora_path=lora_path,
        )
    except Exception:
        tb_str = traceback.format_exc()
        for idx, item in enumerate(data):
            error_list.append(idx)
            error_details.append({
                "idx": idx,
                "question": item.get("question", "") + str(item.get("options", "")),
                "response": None,
                "traceback": tb_str,
            })
            data[idx]["model_response"] = None
            data[idx]["model_answer_idx"] = None

        invalid_cnt = total
        valid_cnt = 0
    else:
        # Process model responses.
        for idx, item in enumerate(data):
            response = responses[idx]
            data[idx]["model_response"] = response

            try:
                pred_idx = parse_medqa_answer(response, item["options"])
                data[idx]["model_answer_idx"] = pred_idx
                valid_cnt += 1
                if str(pred_idx) == str(item["answer_idx"]):
                    correct_cnt += 1
            except Exception:
                invalid_cnt += 1
                error_list.append(idx)
                tb_str = traceback.format_exc()
                error_details.append({
                    "idx": idx,
                    "question": item.get("question", "") + str(item.get("options", "")),
                    "response": response,
                    "traceback": tb_str,
                })
                data[idx]["model_answer_idx"] = None

    # Save all model answers and decisions. (same directory as input file)
    if record_file:
        if lora_path is not None:
            raw_model_name = lora_path.rstrip("/")
        else:
            raw_model_name = getattr(model, "name_or_path", "")
            raw_model_name = str(raw_model_name).rstrip("/")
        if "/" in raw_model_name:
            model_name = raw_model_name.split("/")[-1]
        else:
            model_name = raw_model_name or "model"

        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path)
        root, ext = os.path.splitext(base_name)
        out_name = f"{root}_eval_{model_name}.jsonl"
        out_path = os.path.join(base_dir, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved eval results to {out_path}")

    # Print error details.
    if error_details and print_errors:
        for err in error_details:
            print(f"id:\n{err['idx']}\n")
            print(f"Question:\n{err['question']}\n")
            print(f"Model response:\n{err['response']}\n")
            print(err["traceback"])

    # Print evaluation summary.
    acc = (correct_cnt / valid_cnt) if valid_cnt > 0 else 0.0
    print(
        f"file: {path}\n"
        f"total: {total}, invalid: {invalid_cnt}, valid: {valid_cnt}, "
        f"correct: {correct_cnt}, valid accuracy: {acc:.4f}, overall accuracy: {(correct_cnt / total):.4f}"
    )
    print(f"error_list = {error_list}")


def eval_medqa(model_path: str,
               data_paths,
               visible_gpus: str,
               max_tokens: int = None,
               print_errors: bool = True,
               record_file: bool = False,
               lora_path: str | None = None):
    """
    Entrance for evaluating MedQA dataset.
    
    model_path: Path to the base model. Also can be an online model name.
    data_paths: List of paths to MedQA JSONL files.
    visible_gpus: Comma-separated GPU device ids to use.
    max_tokens: Maximum tokens for generation. If None, use model's max length.
    print_errors: Whether to print detailed error information of every invalid case.
    record_file: Whether to save all model answers and decisions to a file.
    lora_path: Optional path to LoRA weights to apply during evaluation.
    """

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        visible_gpus=visible_gpus,
        dtype=torch.bfloat16,
        enable_lora=lora_path is not None,
        max_tokens=max_tokens,
    )

    build_prompt_fn = build_medqa_messages

    for path in data_paths:
        eval_single_medqa_jsonl(
            path=path,
            model=model,
            tokenizer=tokenizer,
            build_prompt_fn=build_prompt_fn,
            max_tokens=max_tokens,
            print_errors=print_errors,
            record_file=record_file,
            lora_path=lora_path,
        )


if __name__ == "__main__":
    eval_medqa(
        model_path="/nfsdata4/Qwen/Qwen3-32B", # Replace with your local model path/online model name
        # lora_path="models/sft_Qwen3", # Optional: path to LoRA weights
        data_paths=[
            "dataset/MedQA/questions/US/test.jsonl", # USMLE-5options
            "dataset/MedQA/questions/Mainland/test.jsonl", # MCMLE-5options
            "dataset/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl", # USMLE-4options
            "dataset/MedQA/questions/Mainland/4_options/test.jsonl", # MCMLE-4options
        ],
        visible_gpus="2",
        max_tokens=32768,
        print_errors=False,
        record_file=False,
    )