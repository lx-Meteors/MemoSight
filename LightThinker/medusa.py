"""Medusa-1 residual prediction heads for the local Qwen3 implementation."""

import json
import os
from typing import List

import torch
import torch.nn as nn


# Official Medusa greedy sparse tree (mc_sim_7b_63).  A path [a, b, ...]
# selects rank-a from head 0, rank-b from head 1, and so on.  Reusing the
# published tree keeps this adapter comparable to Medusa-1 while the target
# backbone is changed from Vicuna to the user's LightThinker Qwen3-8B.
MEDUSA_CHOICES_63 = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3],
    [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7],
    [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9],
    [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4],
    [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0],
    [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4],
    [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0],
    [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1],
    [1, 6], [0, 7, 0],
]


class MedusaResidualBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        return hidden_states + self.activation(self.linear(hidden_states))


class MedusaHeads(nn.Module):
    """Future-token heads that share the frozen backbone LM projection."""

    def __init__(self, hidden_size: int, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        if num_heads <= 0 or num_layers <= 0:
            raise ValueError("Medusa num_heads and num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    *[MedusaResidualBlock(hidden_size) for _ in range(num_layers)]
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, hidden_states) -> List[torch.Tensor]:
        """Return transformed hidden states for every future-token head.

        Keeping this as the module's real forward path is important under
        DeepSpeed ZeRO-3: it lets DeepSpeed gather/shard head parameters at
        the correct module boundary while the frozen backbone stays outside
        the optimizer.
        """
        return [head(hidden_states) for head in self.heads]

    def future_logits(self, hidden_states, lm_head) -> List[torch.Tensor]:
        return [lm_head(transformed) for transformed in self(hidden_states)]

    def save_config(self, directory: str, base_model_path: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "medusa_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                dict(
                    architecture="medusa_1_shared_lm_head",
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    base_model_path=os.path.realpath(base_model_path),
                    target_offsets=list(range(2, self.num_heads + 2)),
                ),
                handle,
                indent=2,
            )


def load_medusa_heads(path: str, hidden_size: int, device, dtype) -> MedusaHeads:
    config_path = os.path.join(path, "medusa_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"missing Medusa config: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config["hidden_size"]) != int(hidden_size):
        raise ValueError(
            f"Medusa hidden size {config['hidden_size']} does not match backbone {hidden_size}"
        )
    heads = MedusaHeads(
        hidden_size=hidden_size,
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
    )
    safetensor_path = os.path.join(path, "model.safetensors")
    torch_path = os.path.join(path, "pytorch_model.bin")
    if os.path.isfile(safetensor_path):
        from safetensors.torch import load_file

        state = load_file(safetensor_path, device="cpu")
    elif os.path.isfile(torch_path):
        state = torch.load(torch_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no Medusa head weights found under {path}")
    heads.load_state_dict(state, strict=True)
    heads.to(device=device, dtype=dtype)
    heads.eval()
    return heads
