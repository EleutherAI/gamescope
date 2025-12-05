"""Fine-tune Qwen on Diplomacy SFT dataset (orders and negotiation prediction).

Uses a standard Hugging Face Trainer setup with LoRA. For base models like
`Qwen/Qwen3-8B-Base`, we construct a plain completion prompt. For chat/instruct
models, we build the chat template text and mask the prompt portion.

The dataset format is JSONL with keys:
  - game_id, phase, power, type (orders/negotiation)
  - prompt: Full game context prompt (SYSTEM:\n...\n\nUSER:\n...)
  - completion: JSON string with reasoning and orders/messages

Example:
    python scripts/train_diplomacy_sft.py \
        --model_name_or_path Qwen/Qwen3-8B \
        --dataset_path results/diplomacy/sft/merged_dataset.jsonl \
        --output_dir results/diplomacy/sft/qwen_diplomacy_sft
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure repo root is importable
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from gamescope.libs.run_utils import capture_metadata, start_run, write_config_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0) for distributed training."""
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    return str(rank) == "0"


@dataclass
class DiplomacySample:
    """Single training sample from the Diplomacy SFT dataset."""
    game_id: str
    phase: str
    power: str
    sample_type: str  # "orders" or "negotiation"
    prompt: str  # Full prompt including SYSTEM: and USER: markers
    completion: str  # JSON completion string


class DiplomacySFTDataset(Dataset):
    """Dataset for Diplomacy SFT training from JSONL file."""

    def __init__(
        self,
        dataset_path: Path,
        max_records: Optional[int] = None,
        filter_type: Optional[str] = None,
    ) -> None:
        """Load dataset from JSONL file.

        Args:
            dataset_path: Path to JSONL file
            max_records: Maximum number of records to load (None for all)
            filter_type: Filter to specific type ("orders" or "negotiation")
        """
        items: List[DiplomacySample] = []

        with open(dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                sample_type = record.get("type", "orders")

                # Apply filter if specified
                if filter_type and sample_type != filter_type:
                    continue

                items.append(
                    DiplomacySample(
                        game_id=record.get("game_id", "unknown"),
                        phase=record.get("phase", ""),
                        power=record.get("power", ""),
                        sample_type=sample_type,
                        prompt=record.get("prompt", ""),
                        completion=record.get("completion", ""),
                    )
                )

                if max_records is not None and len(items) >= max_records:
                    break

        self._items = items
        logger.info(f"Loaded {len(items)} samples from {dataset_path}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> DiplomacySample:
        return self._items[idx]


def parse_prompt_parts(prompt: str) -> tuple[str, str]:
    """Parse prompt into system and user parts.

    The prompts have format: "SYSTEM:\n...\n\nUSER:\n..."
    """
    if "USER:\n" in prompt:
        parts = prompt.split("USER:\n", 1)
        system_text = parts[0].replace("SYSTEM:\n", "").strip()
        user_text = parts[1].strip()
    else:
        system_text = ""
        user_text = prompt
    return system_text, user_text


class DiplomacyCollator:
    """Tokenize and collate Diplomacy samples, masking prompt tokens from loss.

    Produces:
      - input_ids: LongTensor [B, T]
      - attention_mask: LongTensor [B, T]
      - labels: LongTensor [B, T] with -100 for prompt/pad positions
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        use_chat_template: bool,
        max_length: int = 4096,
    ) -> None:
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.max_length = max_length

    def _build_text(self, sample: DiplomacySample) -> tuple[str, str]:
        """Build prompt text and target text for a sample."""
        system_text, user_text = parse_prompt_parts(sample.prompt)

        if self.use_chat_template:
            # Build chat template
            messages = []
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": user_text})

            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # For chat models, completion directly follows
            target_text = sample.completion
        else:
            # Plain completion format - reconstruct the original prompt format
            prompt_text = sample.prompt
            # Add newline before completion for base models
            target_text = "\n" + sample.completion

        return prompt_text, target_text

    def __call__(self, batch: List[DiplomacySample]) -> Dict[str, torch.Tensor]:
        tokenized_inputs: List[List[int]] = []
        tokenized_labels: List[List[int]] = []

        for sample in batch:
            prompt_text, target_text = self._build_text(sample)

            prompt_ids = self.tokenizer(
                prompt_text, add_special_tokens=False, return_tensors=None
            )["input_ids"]
            target_ids = self.tokenizer(
                target_text, add_special_tokens=False, return_tensors=None
            )["input_ids"]

            # Truncate if needed (keep end of prompt + full target if possible)
            total_len = len(prompt_ids) + len(target_ids)
            if total_len > self.max_length:
                # Truncate prompt, keep target intact
                max_prompt_len = self.max_length - len(target_ids)
                if max_prompt_len > 100:  # Keep at least some context
                    prompt_ids = prompt_ids[-max_prompt_len:]
                else:
                    # Both need truncation
                    prompt_ids = prompt_ids[-(self.max_length // 2):]
                    target_ids = target_ids[: self.max_length // 2]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            tokenized_inputs.append(input_ids)
            tokenized_labels.append(labels)

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        max_len = max(len(x) for x in tokenized_inputs) if tokenized_inputs else 0
        batch_size = len(batch)

        batch_input_ids = np.full((batch_size, max_len), pad_id, dtype=np.int64)
        batch_attention = np.zeros((batch_size, max_len), dtype=np.int64)
        batch_labels = np.full((batch_size, max_len), -100, dtype=np.int64)

        for i, (inp, lab) in enumerate(zip(tokenized_inputs, tokenized_labels)):
            L = len(inp)
            batch_input_ids[i, :L] = np.asarray(inp, dtype=np.int64)
            batch_attention[i, :L] = 1
            batch_labels[i, :L] = np.asarray(lab, dtype=np.int64)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _infer_prompt_mode(model_name_or_path: str, tokenizer: AutoTokenizer) -> bool:
    """Return True if chat template should be used; False for plain completion."""
    lower = model_name_or_path.lower()
    if "base" in lower:
        return False
    has_chat = getattr(tokenizer, "chat_template", None) is not None
    return bool(has_chat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B"),
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(_REPO_ROOT / "results" / "diplomacy" / "sft" / "merged_dataset.jsonl"),
        help="Path to Diplomacy SFT JSONL dataset.",
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=None,
        help="Number of training examples to use (None for all).",
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        default=None,
        choices=["orders", "negotiation", None],
        help="Filter to specific sample type.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_REPO_ROOT / "results" / "diplomacy" / "sft" / "qwen_diplomacy_sft"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (default: 2*r)")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and model but don't train (for testing).",
    )

    args = parser.parse_args()

    # Standardized run scaffolding - only on main process
    # For distributed training, we need to coordinate the run directory
    # Use a fixed coordination file location based on original output_dir
    original_output_base = Path(args.output_dir).parent
    run_dir_file = original_output_base / ".run_dir"

    if _is_main_process():
        run_dir = start_run(base_dir=original_output_base, run_prefix="diplomacy_sft_train")
        training_output_dir = Path(run_dir) / "artifacts"
        training_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(training_output_dir)
        write_config_yaml(run_dir, f"{sys.executable} " + " ".join(sys.argv), vars(args))
        capture_metadata(run_dir)
        # Write run_dir to coordination file for other ranks
        run_dir_file.write_text(str(run_dir))
    else:
        # Wait for main process to create run directory
        import time
        original_output_base.mkdir(parents=True, exist_ok=True)
        for _ in range(120):  # Wait up to 120 seconds (model loading can be slow)
            if run_dir_file.exists():
                run_dir = Path(run_dir_file.read_text().strip())
                args.output_dir = str(run_dir / "artifacts")
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Timeout waiting for main process to create run directory at {run_dir_file}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_chat_template = _infer_prompt_mode(args.model_name_or_path, tokenizer)
    logger.info(f"Using chat template: {use_chat_template}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Gradient checkpointing for memory savings
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Apply LoRA adapters
    lora_alpha = args.lora_alpha if args.lora_alpha else int(max(2 * args.lora_r, 32))
    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = DiplomacySFTDataset(
        dataset_path=Path(args.dataset_path),
        max_records=args.num_train_data,
        filter_type=args.filter_type,
    )

    collator = DiplomacyCollator(
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        max_length=args.max_length,
    )

    if args.dry_run:
        logger.info("Dry run - testing data loading and model setup")
        # Test a batch
        test_batch = [dataset[i] for i in range(min(2, len(dataset)))]
        test_output = collator(test_batch)
        logger.info(f"Test batch shapes: {test_output['input_ids'].shape}")
        logger.info("Dry run complete - exiting without training")
        return

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Save final artifacts
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
