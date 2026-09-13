"""CPU consistency test (no CUDA needed): batched_generate vs a plain bs=1 greedy reference.

Valid for the NO-COMPRESSION path (an untrained base model never emits the split token),
so it exercises prefill left-padding, per-row positions, batched argmax, and active-row
compaction. Compression-branch + speedup benchmark must run on a healthy GPU separately.
"""
import sys, types, torch
sys.path.insert(0, '.')

from config import Config
from model_qwen import Qwen3ForCausalLM
from tokenizer import Tokenizer
from transformers import DynamicCache
from inference_batched import batched_generate

MODEL_PATH = "/mnt/lxy/hf_models/Qwen3-0.6B"
CONFIG = "/mnt/lxy/MemoSight/configs/LightThinker/qwen/v1.json"
MAX_NEW = 24
DEVICE = "cpu"
DTYPE = torch.float32

comp_config = Config.from_file(CONFIG)
tokenizer = Tokenizer(tokenizer_path=MODEL_PATH, bos_token="<|im_start|>", eos_token="<|im_end|>",
                      special_token_list=None, add_prefix_space=False)
special = [t for t in comp_config.special_token_name_list if tokenizer.convert_tokens_to_ids(t) is None]
if special:
    tokenizer.add_special_token(special)
model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE, attn_implementation="sdpa")
model.eval()
comp_config.convert2id(tokenizer)
print("attn_impl:", model.config._attn_implementation, "| split_id:", comp_config.split_token_id)

system_prompt = "You are a helpful assistant."
questions = [
    "What is 2+2?",
    "Capital of France?",
    "Say hello.",
]

@torch.no_grad()
def ref_greedy(q):
    prompt = tokenizer.bos_token + comp_config.template_cfg['complete'].format(system=system_prompt, question=q)
    ids = tokenizer.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
    input_ids = torch.tensor([ids], device=DEVICE)
    cache = DynamicCache()
    out = []
    pos = torch.arange(len(ids), device=DEVICE).unsqueeze(0)
    mo = model(input_ids=input_ids, position_ids=pos, past_key_values=cache, use_cache=True, return_dict=True)
    nxt = int(mo.logits[0, -1, :].argmax())
    steps = 0
    while steps < MAX_NEW:
        out.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        cur = torch.tensor([[nxt]], device=DEVICE)
        p = torch.tensor([[len(ids) + steps]], device=DEVICE)
        mo = model(input_ids=cur, position_ids=p, past_key_values=cache, use_cache=True, return_dict=True)
        nxt = int(mo.logits[0, -1, :].argmax())
        steps += 1
    return out

def check(qs, tag):
    print(f"\n=== {tag} B={len(qs)} ===")
    ref = [ref_greedy(q) for q in qs]
    bat = batched_generate(model=model, tokenizer=tokenizer, comp_config=comp_config,
                           system_prompt=system_prompt, questions=qs, max_new_tokens=MAX_NEW,
                           max_length=MAX_NEW + 256, device=DEVICE, dtype=DTYPE,
                           update_attention_method="local", use_EPL=False, spec_decode=False)
    ok = True
    for i in range(len(qs)):
        rt = tokenizer.decode(ref[i]); bt = bat[i]
        m = (rt == bt); ok = ok and m
        print(f"[{i}] match={m}")
        if not m:
            print("  ref:", repr(rt[:160]))
            print("  bat:", repr(bt[:160]))
    print(tag, "PASS" if ok else "FAIL")
    return ok

r1 = check([questions[0]] * 3, "identical x3")
r2 = check(questions, "mixed 3")
print("\nALL:", "PASS" if (r1 and r2) else "FAIL")
sys.exit(0 if (r1 and r2) else 1)
