# The optional values for the method argument are 'anchor-token', 'normal', 'kvcache', and 'anchor-thought'.
method="normal"
tokenizer_path="Qwen/Qwen3-8B"
comp_config="configs/LightThinker/qwen/distillr1.json"
model_type="qwen"
dataset="bbh"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size=1024
folder="1.5_wo_pretrain"
ckpt=5220
file1="${1:-inference_results/${folder}/${dataset}/${ckpt}/1-4distillr1_qwen3_8b.jsonl}"
python evaluation/eval_file.py \
  --method $method \
  --tokenizer_path $tokenizer_path \
  --comp_config $comp_config \
  --model_type $model_type \
  --dataset $dataset \
  --files "$file1" \
  --cache_size $cache_size \
  --bos_token $bos_token \
  --eos_token $eos_token 
  # --interaction
