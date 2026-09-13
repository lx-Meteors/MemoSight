import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/train/distill.jsonl"        # 原始数据
OUTPUT_PATH = PROJECT_ROOT / "data/train/new_data.jsonl"      # 输出数据


# =====================================================
# ⭐ 通用安全切句（不会破坏latex和特殊token）
# =====================================================
def split_sentences_safe(text: str):
    """
    用于 system_prompt / question
    """

    text = text.strip()

    # 先把换行统一
    text = text.replace("\r", "")

    # 按句号 + 换行切
    parts = re.split(r'(?<=[.:])\s+', text)

    # 清洗空字符串
    parts = [p.strip() for p in parts if p.strip()]

    return parts


# =====================================================
# ⭐ question 专用切分
# =====================================================
def build_question_list(question: str):

    # 先让选项换行
    question = question.replace("\\text{(A)}", "\n$\\text{(A)}")
    question = question.replace("\\text{(B)}", "\n$\\text{(B)}")
    question = question.replace("\\text{(C)}", "\n$\\text{(C)}")
    question = question.replace("\\text{(D)}", "\n$\\text{(D)}")
    question = question.replace("\\text{(E)}", "\n$\\text{(E)}")

    lines = [l.strip() for l in question.split("\n") if l.strip()]

    return lines


# =====================================================
# ⭐ thoughts_list（直接按真实换行拆）
# =====================================================
def build_thoughts_list(gt_output: str):

    gt_output = gt_output.replace("\r", "")

    # 关键：按真实空行切（R1标准）
    blocks = re.split(r'\n\s*\n', gt_output)

    blocks = [b.strip() for b in blocks if b.strip()]

    return blocks


# =====================================================
# 主流程
# =====================================================
def main():

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:

            data = json.loads(line)

            # ===== system_list =====
            if "system_prompt" in data:
                data["system_list"] = split_sentences_safe(
                    data["system_prompt"]
                )

            # ===== question_list =====
            if "question" in data:
                data["question_list"] = build_question_list(
                    data["question"]
                )

            # ===== thoughts_list =====
            if "gt_output" in data:
                data["thoughts_list"] = build_thoughts_list(
                    data["gt_output"]
                )

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("✅ Done! 已生成 system_list / question_list / thoughts_list")


if __name__ == "__main__":
    main()
