
model_path="/mnt/jinbo/RLRM/model/Qwen/Qwen3-8B"
datasets="gpqa"
batch_size=8
output_dir="./sglang_inference_results"
extend_name="qwen3_8b"

root_dir="./LightThinker"


  python "${root_dir}/sglang_inference.py" \
    --model_path $model_path \
    --datasets $datasets \
    --output_dir $output_dir \
    --batch_size $batch_size \
    --extend_name $extend_name
