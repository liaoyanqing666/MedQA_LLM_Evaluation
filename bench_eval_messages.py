import re

def build_medqa_messages(question: str, options: dict):
    """
    Build messages for MedQA task. (prompt engineering)
    """
    option_lines = []
    for key in sorted(options.keys()):
        option_lines.append(f"{key}. {options[key]}")
    options_str = "\n".join(option_lines)

    # Chinese prompt:
    
    sys_content = (
        "你是一名专业的医生，下面是一道医学相关的单项选择题，只允许输出一个最终选项。\n"
        "请使用思考模式进行一步一步分析与推理，最后再给出答案。"
        "在输出的最后一行必须给出唯一一个选项，并严格使用如下格式：\n"
        "Final answer: X\n"
        "其中 X 为大写字母选项（例如 A、B、C、D、E等）。"
        "请确保最后一行只有这一行内容，不要包含多余文字。\n"
    )
    user_content = (
        f"题目：{question}\n"
        f"选项：\n{options_str}\n"
    )
    
    
    # English prompt: 
    
    # sys_content = (
    #     "You are a professional doctor. Below is a medical multiple-choice question, and you are only allowed to output one final option.\n"
    #     "Please use a step-by-step reasoning process, then provide the final answer."
    #     "The last line of your output must contain only one option, strictly in the following format:\n"
    #     "Final answer: X\n"
    #     "where X is the uppercase letter of the chosen option (such as A, B, C, D, E, etc.)."
    #     "Make sure the last line contains only this line and no additional text.\n"
    # )
    # user_content = (
    #     f"Question: {question}\n"
    #     f"Options:\n{options_str}\n"
    # )


    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def parse_medqa_answer(response: str, options: dict) -> str:
    """
    Parse the model response to extract the selected answer option.
    Raises ValueError if parsing fails.
    """
    if response is None:
        raise ValueError("The model answers as empty.")

    text = response.strip()
    if not text:
        raise ValueError("The model answers as empty.")

    valid_keys = set(str(k) for k in options.keys())
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = []

    strict_pattern = re.compile(
        r"^[#>*\s\*]*"
        r"(?:最终答案|答案|正确选项|选择|故选择|所以选|Final answer|Answer|answer is|Assistant: Final answer|assistant: Final answer)"
        r"\s*[:：]?\s*([A-Z])\s*"
        r"[*\s]*$",
        flags=re.IGNORECASE
    )

    for line in reversed(lines):
        m = strict_pattern.search(line)
        if m:
            candidates.append(m.group(1).upper())
            break

    if not candidates:
        raise ValueError(f"No candidate options can be parsed from the model's responses.")

    for c in candidates:
        if c in valid_keys:
            return c

    raise ValueError(
        f"The options parsed from the model responses are not in the given option set. Candidate ={candidates}, valid set ={sorted(valid_keys)}"
    )