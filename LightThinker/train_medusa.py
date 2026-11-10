"""Train Medusa-1 heads on a frozen LightThinker Qwen3-8B checkpoint."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Trainer, TrainerCallback, TrainingArguments, set_seed

from config import Config
from dataset import MyDataCollator, MyDataset
from medusa import MedusaHeads
from model_qwen import Qwen3ForCausalLM
from tokenizer import Tokenizer


class MedusaConfigCallback(TrainerCallback):
    def __init__(self, heads, base_model_path):
        self.heads = heads
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        self.heads.save_config(
            os.path.join(args.output_dir, f"checkpoint-{state.global_step}"),
            self.base_model_path,
        )


class MedusaTrainer(Trainer):
    def __init__(self, *args, backbone_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_model = backbone_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        inputs.pop("system_prompt_length", None)
        inputs.pop("register_token_index", None)
        with torch.no_grad():
            outputs = self.backbone_model(
                **inputs,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1].detach()

        total_loss = hidden_states.sum() * 0.0
        transformed_states = model(hidden_states)
        for head_index, shifted_hidden in enumerate(transformed_states):
            offset = head_index + 2
            if hidden_states.shape[1] <= offset:
                continue
            logits = self.backbone_model.lm_head(
                shifted_hidden[:, :-offset, :]
            ).float()
            targets = labels[:, offset:].to(logits.device)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            valid = flat_targets.ne(-100)
            if valid.any():
                # The official Medusa-1 objective sums the per-head losses.
                total_loss = total_loss + F.cross_entropy(
                    flat_logits[valid], flat_targets[valid]
                )
        loss = total_loss
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--compress_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    comp_config = Config.from_file(args.compress_config)
    tokenizer = Tokenizer(
        tokenizer_path=args.tokenizer_path,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        special_token_list=None,
        add_prefix_space=False,
    )
    tokenizer.validate_qwen3()
    comp_config.convert2id(tokenizer)

    model_config = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True)
    backbone = Qwen3ForCausalLM.from_pretrained(
        args.base_model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    backbone.requires_grad_(False)
    backbone.eval()
    backbone.to(torch.device("cuda", max(args.local_rank, 0)))

    heads = MedusaHeads(
        hidden_size=model_config.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    dataset = MyDataset(
        file_path=args.train_path,
        config=comp_config,
        tokenizer=tokenizer,
        padding_config=dict(
            padding_side="right",
            label_padding_id=-100,
            input_padding_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            position_ids_padding_id=0,
        ),
        train_on_input=False,
        change_rope=False,
        output_compress_instruction="",
        use_EPL=False,
        mode="aug-wo-pc",
    )
    collator = MyDataCollator(
        dataset=dataset,
        attention_config=dict(
            diagonal=False,
            bi_directional=False,
            see_current=False,
            prefill_compress=False,
        ),
        exclude_continue=False,
        sample_config=dict(mode="aug-wo-pc", hybrid=False),
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed,
        bf16=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="tensorboard",
        remove_unused_columns=False,
        seed=args.seed,
        local_rank=args.local_rank,
    )
    trainer = MedusaTrainer(
        model=heads,
        backbone_model=backbone,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[MedusaConfigCallback(heads, args.base_model_path)],
    )
    resume = None
    if os.path.isdir(args.output_dir):
        checkpoints = [name for name in os.listdir(args.output_dir) if name.startswith("checkpoint-")]
        if checkpoints:
            resume = True
    trainer.train(resume_from_checkpoint=resume)
    heads.save_config(args.output_dir, args.base_model_path)


if __name__ == "__main__":
    main()
