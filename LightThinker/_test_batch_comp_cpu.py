"""CPU self-consistency test for the COMPRESSION branch (no CUDA / no trained ckpt needed).

Forces split tokens on a deterministic, STAGGERED per-sample schedule, then asserts each
sample's output is identical whether run alone (B=1) or in a mixed batch (B=3). Correct
batching (masks / positions / gather-rebuild compaction) => per-sample equality despite
divergent compression timing. This exercises the exact index math that argmax-driven runs
can't reach with an untrained model.
"""
import sys, torch
sys.path.insert(0, '.')
from config import Config
from model_qwen import Qwen3ForCausalLM
from tokenizer import Tokenizer
from inference_batched import batched_generate

MODEL_PATH = "/mnt/lxy/hf_models/Qwen3-0.6B"
CONFIG = "/mnt/lxy/MemoSight/configs/LightThinker/qwen/v1.json"
MAX_NEW = 30
DEVICE, DTYPE = "cpu", torch.float32

comp_config = Config.from_file(CONFIG)
tokenizer = Tokenizer(tokenizer_path=MODEL_PATH, bos_token="<|im_start|>", eos_token="<|im_end|>",
                      special_token_list=None, add_prefix_space=False)
special = [t for t in comp_config.special_token_name_list if tokenizer.convert_tokens_to_ids(t) is None]
if special:
    tokenizer.add_special_token(special)
model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE, attn_implementation="sdpa").eval()
comp_config.convert2id(tokenizer)

system_prompt = "You are a helpful assistant."
questions = ["What is 2+2?", "Capital of France?", "Say hello."]
# staggered split schedules (by step_count) -> divergent compression timing
schedules = [{3, 7}, {5}, {2, 6, 10}]

def run(qs, sched):
    return batched_generate(model=model, tokenizer=tokenizer, comp_config=comp_config,
                            system_prompt=system_prompt, questions=qs, max_new_tokens=MAX_NEW,
                            max_length=MAX_NEW + 256, device=DEVICE, dtype=DTYPE,
                            update_attention_method="local", use_EPL=False, spec_decode=False,
                            force_split_schedule=sched)

batch_out = run(questions, schedules)
ok = True
for i, q in enumerate(questions):
    single_out = run([q], [schedules[i]])[0]
    m = (single_out == batch_out[i]); ok = ok and m
    print(f"[{i}] match={m}")
    if not m:
        print("  single:", repr(single_out[:180]))
        print("  batch :", repr(batch_out[i][:180]))
print("COMPRESSION SELF-CONSISTENCY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
