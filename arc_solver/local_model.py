"""Offline Qwen program proposer for use with attached Kaggle model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .operations import Operation
from .proposer import ProgramProposal, build_program_prompt, parse_program_proposal


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MIN_TRANSFORMERS_VERSION = (4, 51)


def find_local_qwen_model(input_root: Path) -> Path:
    candidates: set[Path] = set()
    for config_path in input_root.rglob("config.json"):
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("model_type") != "qwen3_moe":
            continue
        if not any(
            list(config_path.parent.glob(pattern))
            for pattern in ("*.safetensors", "*.bin", "*.safetensors.index.json")
        ):
            continue
        candidates.add(config_path.parent.resolve())
    if not candidates:
        raise FileNotFoundError(
            f"No local Qwen3-MoE model with weights was found under {input_root}. "
            f"Attach {MODEL_NAME} as a Kaggle Model input; runtime downloads are disabled."
        )
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise ValueError(f"Expected one attached Qwen3-MoE model, found: {paths}")
    return next(iter(candidates))


class LocalQwenProgramProposer:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        max_new_tokens: int = 384,
        seed: int = 2026,
    ) -> None:
        if max_new_tokens < 64:
            raise ValueError("max_new_tokens must be at least 64.")
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    @classmethod
    def from_pretrained(cls, model_path: Path) -> LocalQwenProgramProposer:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        version_parts = tuple(
            int(part) for part in transformers.__version__.split(".")[:2]
        )
        if version_parts < MIN_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "Qwen3-MoE requires transformers>=4.51.0; "
                f"the Kaggle image has {transformers.__version__}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The attached Qwen model requires a CUDA GPU.")

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.eval()
        return cls(tokenizer, model)

    def propose(
        self,
        task: dict[str, Any],
        operations: list[Operation],
        count: int,
    ) -> list[ProgramProposal]:
        if count < 1:
            raise ValueError("At least one model proposal is required.")
        operation_names = {operation.name for operation in operations}
        prompt = build_program_prompt(task, operations)
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(device)
        proposals: list[ProgramProposal] = []

        import torch

        for index in range(count):
            torch.manual_seed(self.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed + index)
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=(0.35, 0.6, 0.85)[index % 3],
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            proposals.append(parse_program_proposal(response, operation_names))
        return proposals
