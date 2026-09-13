import json
from pathlib import Path

# 读取 JSONL
PROJECT_ROOT = Path(__file__).resolve().parents[1]
jsonl_file = PROJECT_ROOT / "data/train/train.jsonl"
json_file = PROJECT_ROOT / "data/eval/distill.json"

data_list = []

with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        # 构造新结构
        new_item = {
            "meta_data": item.get("question_list", []),
            "question": " ".join(item.get("question_list", [])),
            "answer": item.get("gt_output", "").split("\\boxed{")[-1].split("}")[0] if "\\boxed{" in item.get("gt_output", "") else "",
            "choices_list": [c.split(" ")[-1] for c in item.get("question_list", []) if c.startswith("$\\text{(")],
            "domain": "distill",
            "question_list": item.get("question_list", [])
        }
        data_list.append(new_item)

# 保存成 JSON
output = {"distill": data_list}
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
