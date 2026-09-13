
## zrs运行操作
1. 在our_train_lighthinker.sh、our_train_lighthinker_epl.sh、our_train_lighthinker_epl_,tp.sh、our_train_baseline.sh中修改如下绝对路径
```
# ===== zrs项目根路径 =======
root_dir="/zhaorunsong/RRcot"
# ===== zrs项目根路径 =======
```
2. 在auto_training_inference_evaluate.sh的48行和56行修改成自己的conda位置
```
source /mnt/zhaorunsong/anaconda3/etc/profile.d/conda.sh
```
3. 运行 bash auto_training_inference_evaluate.sh

```
/*
 *                                                     __----~~~~~~~~~~~------___
 *                                    .  .   ~~//====......          __--~ ~~
 *                    -.            \_|//     |||\\  ~~~~~~::::... /~
 *                 ___-==_       _-~o~  \/    |||  \\            _/~~-
 *         __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
 *     _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
 *   .~       .~       |   \\ -_    /  /-   /   ||      \   /
 *  /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
 *  |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
 *           '         ~-|      /|    |-~\~~       __--~~
 *                       |-~~-_/ |    |   ~\_   _-~            /\
 *                            /  \     \__   \/~                \__
 *                        _--~ _/ | .-~~____--~-/                  ~~==.
 *                       ((->/~   '.|||' -_|    ~~-/ ,              . _||
 *                                  -_     ~\      ~~---l__i__i__i--~~_/
 *                                  _-~-__   ~)  \--______________--~~
 *                                //.-~~~-~_--~- |-------~~~~~~~~
 *                                       //.-~~~--\
 *                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *                               神兽保佑            永无BUG
 */
```

## Table of Contents

- 🔧[Installation](#installation)
- 🏃[Quick Start](#quick-start)
- 🎁[Acknowledgement](#acknowledgement)
- 🚩[Citation](#citation)


## 🔧Installation

This branch targets `Qwen/Qwen3-8B`. Use the model and tokenizer from the same
Qwen3 checkpoint; Qwen2/Qwen2.5 tokenizers are rejected at startup. SepLLM's
`SepCache` is included in this repository, so there is no need to patch the
installed Transformers package manually.

```bash
git clone -b SepLLM-Qwen3 https://github.com/lx-Meteors/MemoSight.git
cd MemoSight
conda create -n lightthinker python=3.9 -y
conda activate lightthinker
pip install -r requirements.txt
cd data && unzip data.zip && cd ..
```


## 🏃Quick Start

> First, we train the model to learn how to compress (step 1). Then, we perform inference on the test set to obtain output results (step 2). Finally, we evaluate the output results (step 3).

### Step 1. Training

To execute the training, run the following command:

```bash
bash our_train_baseline.sh
bash our_train_lighthinker.sh
bash our_train_lighthinker_epl.sh
bash our_train_lighthinker_mtp.sh (need more exp to make it useful.)
```

Currently, the script's parameters are set to run on a machine with 8 A800 GPUs. If you encounter OOM (Out Of Memory) issues, please reduce the `micro_batch_size` and `max_length`. For other parameters in the script, please refer to the [documentation](./ARGS.md).

### Step 2. Inference

To execute the inference, run the following command for models exclude baseline:

```bash
bash our_inference_repe.sh
```

Here, you need to modify the script file's `model_tag`, `model_short_tag`, `ckpt`, `output_tag`, and `split_size`. For details regarding the script's parameters, please refer to the [documentation](./ARGS.md).


### Step 3. Evaluation

If this is your **first time** conducting an evaluation, please execute the following code first:
```bash
python evaluation/init.py
```

To execute the evaluation, run the following command:

```bash
method=""
tokenizer_path=""
comp_config=""
model_type=""
dataset=""
bos_token=""
eos_token=""
cache_size=1024
file1=""
file2=""
file3=""
file4=""
python evaluation/eval_file.py \
  --method $method \
  --tokenizer_path $tokenizer_path \
  --comp_config $comp_config \
  --model_type $model_type \
  --dataset $dataset \
  --files $file1 $file2 $file3 $file4 \
  --cache_size $cache_size \
  --bos_token $bos_token \
  --eos_token $eos_token \
  --interaction 
```

Please note that if you set `split_size>1` in the second step, the number of file i here should match the value of `split_size`. It should be noted that manual evaluation was conducted during the assessment. Use the `--interaction` flag to enable manual evaluation. The `cache_size` parameter is used for `H2O` and `SepLLM`, but not for `LightThinker` or `AnLLM`.

<details> 
<summary><b>Evaluation Script Example</b></summary>

```bash
# The optional values for the method argument are 'anchor-token', 'normal', 'kvcache', and 'anchor-thought'.
method="anchor-thought"
tokenizer_path="Qwen/Qwen3-8B"
comp_config="configs/LightThinker/qwen/v1.json"
model_type="qwen"
dataset="gpqa"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size=1024
folder=""
ckpt=1045
file1="inference_results/${folder}/${dataset}/${ckpt}/1-4qwen_7b.jsonl"
file2="inference_results/${folder}/${dataset}/${ckpt}/2-4qwen_7b.jsonl"
file3="inference_results/${folder}/${dataset}/${ckpt}/3-4qwen_7b.jsonl"
file4="inference_results/${folder}/${dataset}/${ckpt}/4-4qwen_7b.jsonl"
python evaluation/eval_file.py \
  --method $method \
  --tokenizer_path $tokenizer_path \
  --comp_config $comp_config \
  --model_type $model_type \
  --dataset $dataset \
  --files $file1 $file2 $file3 $file4 \
  --cache_size $cache_size \
  --bos_token $bos_token \
  --eos_token $eos_token \
  --interaction 
```
</details>

<details> 
<summary><b>Manual Evaluation Instructions</b></summary>

When string matching fails, the output will be displayed in the format "Model Answer" <=> "Standard Answer". At this point, you can input "y" or "n" to evaluate this case. If you believe the model's answer extraction is incorrect, you can input "e" to print the model's complete output, and then input "y" or "n" to evaluate this case.
</details>
