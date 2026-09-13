"""Ad-hoc: assert batched_generate == bs=1 oracle generate, token-for-token (greedy)."""
import sys, types, torch
sys.path.insert(0, '.')

from config import Config
from inference import get_model_and_tokenizer, generate as oracle_generate, AttentionUtils, TokenUtils
from inference_batched import batched_generate

MODEL_PATH = "/mnt/lxy/hf_models/Qwen3-0.6B"
CONFIG = "/mnt/lxy/MemoSight/configs/LightThinker/qwen/v1.json"
MAX_NEW = 48
MAX_PROMPT = 256
DEVICE = "cuda"
DTYPE = torch.bfloat16

args = types.SimpleNamespace(
    model_path=MODEL_PATH, model_tag=None, ckpt=None, tokenizer_path=MODEL_PATH,
    model_type="qwen", bos_token="<|im_start|>", eos_token="<|im_end|>",
)
comp_config = Config.from_file(CONFIG)
attention_config = dict(diagonal=False, bi_attention=False, see_current=False,
                        prefill_compress=False, exclude_continue=False)

model, tokenizer = get_model_and_tokenizer(args, comp_config)
print("attn_impl:", model.config._attn_implementation)

system_prompt = "You are a helpful assistant."
questions = [
    "What is 2+2? Answer briefly.",
    "Name the capital of France.",
    "List three prime numbers.",
    "What color is the sky on a clear day?",
]

def run_oracle(q):
    attn = AttentionUtils(max_length=MAX_NEW + MAX_PROMPT, device=DEVICE, dtype=DTYPE,
                          attention_config=attention_config, prefill_compress=False,
                          max_comp_size=300, n_inst=0, n_continue=1)
    tok = TokenUtils(max_length=MAX_NEW + MAX_PROMPT, device=DEVICE, rolling_rope=False)
    prompt, output = oracle_generate(
        model=model, tokenizer=tokenizer, comp_config=comp_config,
        question=q, question_list=None, system_prompt=system_prompt, system_prompt_list=None,
        max_new_tokens=MAX_NEW, attention_config=attention_config, prefill_compress=False,
        exclude_continue=False, compress_prompt=False, attn_utils=attn, token_utils=tok,
        update_attention_method="local", use_EPL=False, repetition_penalty=1.0, spec_decode=False,
    )
    return tok.show_output_input_ids  # committed output ids

def check(qs, tag):
    print(f"\n=== {tag}: B={len(qs)} ===")
    oracle_ids = [run_oracle(q) for q in qs]
    batched_out = batched_generate(
        model=model, tokenizer=tokenizer, comp_config=comp_config,
        system_prompt=system_prompt, questions=qs, max_new_tokens=MAX_NEW,
        max_length=MAX_NEW + MAX_PROMPT, device=DEVICE, dtype=DTYPE,
        update_attention_method="local", use_EPL=False, spec_decode=False,
    )
    # re-decode oracle for readable compare; primary check is on ids
    ok = True
    for i, q in enumerate(qs):
        o_ids = oracle_ids[i]
        b_text = batched_out[i]
        o_text = tokenizer.decode(o_ids)
        match = (o_text == b_text)
        ok = ok and match
        print(f"[{i}] match={match}")
        if not match:
            print("   oracle:", repr(o_text[:200]))
            print("   batch :", repr(b_text[:200]))
    print(f"{tag}: {'PASS' if ok else 'FAIL'}")
    return ok

r1 = check([questions[0]] * 4, "identical-prompt x4")
r2 = check(questions, "mixed 4 prompts")
print("\nALL:", "PASS" if (r1 and r2) else "FAIL")
sys.exit(0 if (r1 and r2) else 1)
