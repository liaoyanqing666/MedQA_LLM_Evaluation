# Evaluation script for model predictions stored in JSONL files.
import json

def eval_jsonl(
    path: str | list[str],
    ground_truth_key: str = "answer_idx",
    pred_key: str = "model_answer_idx",
    pred_content_key: str = "model_response",
    print_content: bool = False,
):
    """
    Evaluate model predictions stored in a JSONL file.
    
    path: Path to the JSONL file or a list of file paths.
    ground_truth_key: Key for ground truth answers in JSON objects.
    pred_key: Key for model predictions in JSON objects.
    pred_content_key: Key for model response content in JSON objects.
    print_content: Whether to print the longest valid and shortest invalid responses.
    """
    path_list = None
    if isinstance(path, list):
        path_list = path
        path = path[0]
    
    total = 0
    correct = 0
    invalid = 0

    longest_valid_len = 0
    longest_valid_text = ""

    shortest_invalid_len = None
    shortest_invalid_text = ""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if ground_truth_key not in obj or pred_key not in obj:
                continue

            total += 1
            pred = obj[pred_key]
            resp = obj.get(pred_content_key, "")

            if pred is None:
                invalid += 1
                length = len(resp)
                if shortest_invalid_len is None or length < shortest_invalid_len:
                    shortest_invalid_len = length
                    shortest_invalid_text = resp
                continue

            length = len(resp)
            if length > longest_valid_len:
                longest_valid_len = length
                longest_valid_text = resp

            if str(pred) == str(obj[ground_truth_key]):
                correct += 1

    valid = total - invalid
    overall_acc = correct / total if total > 0 else 0.0
    valid_acc = correct / valid if valid > 0 else 0.0

    if print_content:
        print("Longest valid model_response content:")
        print(longest_valid_text)
        print("Shortest invalid model_response content:")
        print(shortest_invalid_text)

    print("Longest valid model_response length:", longest_valid_len)
    print("Shortest invalid model_response length:", shortest_invalid_len)
    
    print(f"file: {path}")
    print(f"total: {total}, invalid: {invalid}, valid: {valid}, "
          f"correct: {correct}, valid acc: {valid_acc:.4f}, overall acc: {overall_acc:.4f}")
    print()
    
    if path_list is not None and len(path_list) > 1:
        eval_jsonl(
            path=path_list[1:],
            ground_truth_key=ground_truth_key,
            pred_key=pred_key,
            pred_content_key=pred_content_key,
            print_content=print_content,
        )
        

if __name__ == "__main__":
    model = "Qwen3-32B"
    eval_jsonl(
        [
            f"dataset/MedQA/questions/US/test_eval_{model}.jsonl",
            f"dataset/MedQA/questions/Mainland/test_eval_{model}.jsonl",
            f"dataset/MedQA/questions/US/4_options/phrases_no_exclude_test_eval_{model}.jsonl",
            f"dataset/MedQA/questions/Mainland/4_options/test_eval_{model}.jsonl",
        ], # File paths to evaluate
        ground_truth_key="answer_idx",
        pred_key="model_answer_idx",
        pred_content_key="model_response",
        print_content=False, # Whether to print the longest valid and shortest invalid responses
    )