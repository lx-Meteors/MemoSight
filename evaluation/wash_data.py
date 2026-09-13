import json
from pathlib import Path


# ===============================
# 配置区（自己改这里即可）
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/train/distill.jsonl"        # 原始数据
OUTPUT_PATH = PROJECT_ROOT / "data/train/new_data.jsonl"      # 输出数据

SPLIT_FLAG = "Now, try to solve the following question through the above guidelines:"
BEGIN_TOKENS = ("<｜begin▁of▁sentence｜>", "<|im_start|>system\n")
USER_TOKENS = ("<｜User｜>", "<|im_end|>\n<|im_start|>user\n")


# ===============================
# 核心处理函数
# ===============================
def process_example(example: dict):
    """
    将原 prompt:
        - 拆成 system_prompt 和 question
        - 清洗特殊token
    """
    if "prompt" not in example:
        return example

    text = example["prompt"]

    # 1️⃣ 去掉 begin token
    for token in BEGIN_TOKENS:
        text = text.replace(token, "")
    text = text.strip()

    # 2️⃣ 按关键句切分
    if SPLIT_FLAG not in text:
        print("⚠️ 未找到 split_flag，跳过样本")
        return None

    system_part, question_part = text.split(SPLIT_FLAG, 1)
    system_part = system_part + SPLIT_FLAG
    # 3️⃣ 清洗 question
    for token in USER_TOKENS:
        question_part = question_part.replace(token, "")
    question_part = question_part.strip()

    # 4️⃣ 写入新字段
    example["system_prompt"] = system_part.strip()
    example["question"] = question_part.strip()

    result = {
        "system_prompt": example["system_prompt"],
        "question": example["question"],
        "gt_output": example.get("output", "")
    }

    return result


# ===============================
# 主函数
# ===============================
def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    total = 0
    success = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            example = json.loads(line)

            new_ex = process_example(example)

            if new_ex is None:
                continue

            fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
            success += 1

    print("=================================")
    print(f"Total samples : {total}")
    print(f"Processed     : {success}")
    print(f"Saved to      : {OUTPUT_PATH}")
    print("=================================")


# ===============================
if __name__ == "__main__":
    main()
