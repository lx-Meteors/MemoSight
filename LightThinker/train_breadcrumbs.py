"""Distill a Breadcrumbs Reasoning policy from the trained Vanilla teacher.

This intentionally replaces the paper's concurrently RL-trained teacher with
the user's existing Vanilla Qwen3-8B checkpoint.  The student still learns the
paper's beacon representation with token-level KL and the block-sparse mask.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from config import Config
from model_qwen import Qwen3ForCausalLM


BEACON_TOKEN = "<|breadcrumb|>"


class BreadcrumbDataset(Dataset):
    def __init__(self, path, tokenizer, template, max_length, compression_ratio, beacon_id):
        self.rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        self.beacon_id = beacon_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = self.tokenizer.bos_token + self.template["complete"].format(
            system=row["system_prompt"], question=row["question"]
        )
        answer = row["gt_output"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        teacher_ids = (prompt_ids + answer_ids)[: self.max_length]
        answer_ids = teacher_ids[len(prompt_ids):]
        if not prompt_ids or not answer_ids:
            raise ValueError(f"example {index} has no prompt or answer tokens after truncation")

        student_ids = list(prompt_ids)
        student_positions = list(range(len(prompt_ids)))
        student_predictors = []
        teacher_predictors = []
        block_ids = [-1] * len(prompt_ids)
        beacon_positions = []

        for answer_index, token_id in enumerate(answer_ids):
            if answer_index == 0:
                student_predictors.append(len(prompt_ids) - 1)
            elif answer_index % self.compression_ratio == 0:
                student_predictors.append(len(student_ids) - 1)  # preceding beacon
            else:
                student_predictors.append(len(student_ids) - 1)
            teacher_predictors.append(len(prompt_ids) + answer_index - 1)

            student_ids.append(token_id)
            student_positions.append(len(prompt_ids) + answer_index)
            block_ids.append(answer_index // self.compression_ratio)

            has_next = answer_index + 1 < len(answer_ids)
            if has_next and (answer_index + 1) % self.compression_ratio == 0:
                student_ids.append(self.beacon_id)
                # The beacon replaces the completed block at its logical tail.
                student_positions.append(len(prompt_ids) + answer_index)
                block_ids.append(answer_index // self.compression_ratio)
                beacon_positions.append(len(student_ids) - 1)

        length = len(student_ids)
        prompt_length = len(prompt_ids)
        mask = np.zeros((length, length), dtype=np.bool_)
        for query in range(length):
            for key in range(query + 1):
                if key < prompt_length:
                    mask[query, key] = True
                elif key in beacon_positions:
                    mask[query, key] = True
                elif query >= prompt_length and block_ids[key] == block_ids[query]:
                    mask[query, key] = True
        return dict(
            teacher_input_ids=teacher_ids,
            student_input_ids=student_ids,
            student_position_ids=student_positions,
            student_attention_mask=mask,
            teacher_predictor_indices=teacher_predictors,
            student_predictor_indices=student_predictors,
        )


@dataclass
class BreadcrumbCollator:
    teacher_pad_id: int

    def __call__(self, features: List[Dict]):
        if len(features) != 1:
            raise ValueError("Breadcrumbs distillation currently requires micro_batch_size=1")
        item = features[0]
        allowed = torch.as_tensor(item["student_attention_mask"], dtype=torch.bool)
        additive = torch.zeros_like(allowed, dtype=torch.bfloat16)
        additive.masked_fill_(~allowed, torch.finfo(torch.bfloat16).min)
        return dict(
            teacher_input_ids=torch.as_tensor([item["teacher_input_ids"]], dtype=torch.long),
            student_input_ids=torch.as_tensor([item["student_input_ids"]], dtype=torch.long),
            student_position_ids=torch.as_tensor([item["student_position_ids"]], dtype=torch.long),
            student_attention_mask=additive.unsqueeze(0).unsqueeze(0),
            teacher_predictor_indices=torch.as_tensor(item["teacher_predictor_indices"], dtype=torch.long),
            student_predictor_indices=torch.as_tensor(item["student_predictor_indices"], dtype=torch.long),
        )


class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        self.tokenizer.save_pretrained(
            os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        )


class BreadcrumbTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, kl_chunk_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_chunk_size = kl_chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_ids = inputs.pop("teacher_input_ids")
        student_ids = inputs.pop("student_input_ids")
        teacher_positions = inputs.pop("teacher_predictor_indices")
        student_positions = inputs.pop("student_predictor_indices")
        student_mask = inputs.pop("student_attention_mask")
        student_position_ids = inputs.pop("student_position_ids")

        teacher_device = next(self.teacher_model.parameters()).device
        teacher_ids = teacher_ids.to(teacher_device)
        teacher_positions = teacher_positions.to(teacher_device)
        with torch.no_grad():
            teacher_logits = self.teacher_model(
                input_ids=teacher_ids,
                use_cache=False,
                return_dict=True,
                logits_to_keep=teacher_positions,
            ).logits.float()

        student_positions = student_positions.to(student_ids.device)
        outputs = model(
            input_ids=student_ids,
            attention_mask=student_mask,
            position_ids=student_position_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=student_positions,
        )
        student_logits = outputs.logits.float()
        if teacher_logits.shape[:2] != student_logits.shape[:2]:
            raise RuntimeError(
                f"teacher/student logit mismatch: {teacher_logits.shape} vs {student_logits.shape}"
            )
        if student_logits.shape[-1] != teacher_logits.shape[-1] + 1:
            raise RuntimeError(
                "Breadcrumbs student must add exactly one beacon vocabulary entry: "
                f"teacher={teacher_logits.shape[-1]}, student={student_logits.shape[-1]}"
            )

        loss_sum = student_logits.sum() * 0.0
        token_count = student_logits.shape[1]
        for start in range(0, token_count, self.kl_chunk_size):
            end = min(start + self.kl_chunk_size, token_count)
            teacher_probs = torch.softmax(teacher_logits[:, start:end].to(student_logits.device), dim=-1)
            # Normalize over the *expanded* student vocabulary, then compare
            # the teacher's base-vocabulary support.  This also penalizes the
            # student if it leaks probability mass onto <|breadcrumb|> when a
            # normal content token should be emitted.
            student_log_probs = torch.log_softmax(
                student_logits[:, start:end], dim=-1
            )[..., :teacher_logits.shape[-1]]
            loss_sum = loss_sum + F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="sum",
            )
        loss = loss_sum / max(token_count, 1)
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--compress_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--compression_ratio", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.micro_batch_size != 1:
        raise ValueError("Breadcrumbs currently requires micro_batch_size=1")
    if args.compression_ratio < 2:
        raise ValueError("compression_ratio must be at least 2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": [BEACON_TOKEN]}
    )
    if added_tokens != 1:
        raise ValueError(
            f"The Vanilla teacher tokenizer must not already contain {BEACON_TOKEN!r}"
        )
    beacon_id = tokenizer.convert_tokens_to_ids(BEACON_TOKEN)
    if beacon_id != len(tokenizer) - 1:
        raise ValueError("Breadcrumb beacon must be the final vocabulary entry")
    comp_config = Config.from_file(args.compress_config)

    student_config = AutoConfig.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    student_config.breadcrumbs_compression_ratio = args.compression_ratio
    student_config.breadcrumbs_beacon_token_id = beacon_id
    student = Qwen3ForCausalLM.from_pretrained(
        args.teacher_model_path,
        config=student_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    student.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        mean_embedding = student.model.embed_tokens.weight[:-1].mean(dim=0)
        student.model.embed_tokens.weight[beacon_id].copy_(mean_embedding)
        if student.lm_head.weight.shape[0] == len(tokenizer):
            student.lm_head.weight[beacon_id].copy_(student.lm_head.weight[:-1].mean(dim=0))

    teacher_config = AutoConfig.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    teacher = Qwen3ForCausalLM.from_pretrained(
        args.teacher_model_path,
        config=teacher_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    teacher.requires_grad_(False)
    teacher.eval()
    local_device = torch.device("cuda", max(args.local_rank, 0))
    teacher.to(local_device)

    dataset = BreadcrumbDataset(
        args.train_path,
        tokenizer,
        comp_config.template_cfg,
        args.max_length,
        args.compression_ratio,
        beacon_id,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed,
        bf16=True,
        gradient_checkpointing=True,
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
    trainer = BreadcrumbTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=dataset,
        data_collator=BreadcrumbCollator(tokenizer.pad_token_id or tokenizer.eos_token_id),
        callbacks=[SaveTokenizerCallback(tokenizer)],
    )
    trainer.train(resume_from_checkpoint=True if os.path.isdir(args.output_dir) and any(
        name.startswith("checkpoint-") for name in os.listdir(args.output_dir)
    ) else None)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
