import argparse
import json
import logging
import shutil
import pickle
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DATA_BASE_URL = "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data"
DATA_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}
DATA_DIR = Path(__file__).resolve().parent / "data"


def _ensure_dataset_files() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in DATA_FILES.items():
        dest = DATA_DIR / filename
        paths[split] = dest
        if dest.exists():
            continue
        url = f"{DATA_BASE_URL}/{filename}"
        logging.info("Downloading %s", url)
        try:
            with urlopen(url) as response, open(dest, "wb") as outfile:
                shutil.copyfileobj(response, outfile)
        except URLError as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return paths


def _normalize_truth_label(raw_label: object) -> Optional[bool]:
    """Normalize various truth/deception label formats to bool.

    Returns True for truthful, False for deceptive, None for unknown.
    """
    if isinstance(raw_label, bool):
        return raw_label
    if raw_label is None:
        return None
    if isinstance(raw_label, str):
        normalized = raw_label.strip().lower()
        if normalized in {"true", "truth", "truthful", "honest"}:
            return True
        if normalized in {"false", "lie", "lying", "deceptive", "deception"}:
            return False
    return None


def _iter_dataset_examples(paths: dict[str, Path], splits: Iterable[str]) -> Iterable[tuple[str, int]]:
    for split in splits:
        file_path = paths.get(split)
        if file_path is None or not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages") or []
                sender_labels = record.get("sender_labels") or []
                for message, label in zip(messages, sender_labels):
                    text = (message or "").strip()
                    truth_value = _normalize_truth_label(label)
                    if not text or truth_value is None:
                        continue
                    yield text, 0 if truth_value else 1


CONFIDENCE_LEVELS = {"high": 3, "medium": 2, "low": 1}


@dataclass
class ProbeExample:
    """An example for probe training with optional span marking."""
    context: str  # Full text to feed to model
    probe_span: Optional[str] = None  # Text span to extract representations from (if None, use full context)
    label: int = 0  # 0 = truthful, 1 = deceptive


def load_diplomacy_dataset(
    dataset_path: Optional[str] = None,
    min_confidence: Optional[str] = None,
) -> Tuple[List[ProbeExample], np.ndarray]:
    """Load diplomacy deception dataset.

    If dataset_path is provided, loads from that JSONL file.
    Otherwise, downloads and uses the original ACL 2020 diplomacy dataset.

    Args:
        dataset_path: Path to custom JSONL dataset
        min_confidence: Minimum confidence level to include ("high", "medium", "low").
                       If None, includes all examples.

    Returns:
        List of ProbeExample objects and numpy array of labels.

    Supports multiple formats:
    - wes_deception_dataset: {context, label, confidence, wes_final_statements, ...}
    - ACL 2020 format: {messages, sender_labels}
    - Simple format: {text, label}
    """
    min_conf_val = CONFIDENCE_LEVELS.get(min_confidence, 0) if min_confidence else 0

    if dataset_path is not None:
        # Load from custom JSONL file
        examples: List[ProbeExample] = []
        labels: List[int] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Format 1: wes_deception_dataset (game-level with context + label)
                if "context" in record and "label" in record:
                    # Check confidence filter
                    conf = record.get("confidence", "high")
                    if CONFIDENCE_LEVELS.get(conf, 0) < min_conf_val:
                        continue

                    truth_value = _normalize_truth_label(record["label"])
                    if truth_value is None:
                        continue

                    context = record["context"].strip()
                    if not context:
                        continue

                    # Extract missive bodies for span-based probing
                    statements = record.get("wes_final_statements", [])
                    if statements:
                        missive_texts = [s.get("body", "") for s in statements if s.get("body")]
                        probe_span = " ".join(missive_texts).strip() if missive_texts else None
                    else:
                        probe_span = None

                    label_int = 0 if truth_value else 1
                    examples.append(ProbeExample(context=context, probe_span=probe_span, label=label_int))
                    labels.append(label_int)

                # Format 2: Simple text/label pairs
                elif "text" in record and "label" in record:
                    text = record["text"].strip()
                    truth_value = _normalize_truth_label(record["label"])
                    if text and truth_value is not None:
                        label_int = 0 if truth_value else 1
                        examples.append(ProbeExample(context=text, probe_span=None, label=label_int))
                        labels.append(label_int)

                # Format 3: ACL 2020 style with messages/sender_labels arrays
                elif "messages" in record:
                    messages = record.get("messages") or []
                    sender_labels = record.get("sender_labels") or []
                    for message, lbl in zip(messages, sender_labels):
                        text = (message or "").strip()
                        truth_value = _normalize_truth_label(lbl)
                        if text and truth_value is not None:
                            label_int = 0 if truth_value else 1
                            examples.append(ProbeExample(context=text, probe_span=None, label=label_int))
                            labels.append(label_int)

        return examples, np.array(labels, dtype=np.int64)

    # Default: use original ACL 2020 dataset
    paths = _ensure_dataset_files()
    examples = []
    labels = []
    for text, label in _iter_dataset_examples(paths, ("train", "validation")):
        examples.append(ProbeExample(context=text, probe_span=None, label=label))
        labels.append(label)
    return examples, np.array(labels, dtype=np.int64)


def _find_span_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    probe_spans: List[Optional[str]],
    tokenizer: PreTrainedTokenizer,
) -> torch.Tensor:
    """Create a mask for tokens corresponding to probe_span text.

    For each example in the batch, finds tokens that correspond to the probe_span
    text and creates a mask. If probe_span is None, uses the full attention_mask.

    Returns:
        Tensor of shape (batch, seq_len) with 1s for probe span tokens.
    """
    batch_size, seq_len = input_ids.shape
    span_mask = torch.zeros_like(attention_mask)

    for i, probe_span in enumerate(probe_spans):
        if probe_span is None:
            # No span specified, use full sequence
            span_mask[i] = attention_mask[i]
        else:
            # Tokenize the probe span to find its tokens
            span_tokens = tokenizer.encode(probe_span, add_special_tokens=False)
            if not span_tokens:
                # Fallback to full sequence if span tokenizes to nothing
                span_mask[i] = attention_mask[i]
                continue

            # Search for the span tokens in the input_ids
            input_list = input_ids[i].tolist()
            span_len = len(span_tokens)

            # Find all occurrences and use the last one (most likely the actual output)
            last_match_start = -1
            for j in range(len(input_list) - span_len + 1):
                if input_list[j:j + span_len] == span_tokens:
                    last_match_start = j

            if last_match_start >= 0:
                span_mask[i, last_match_start:last_match_start + span_len] = 1
            else:
                # Span not found exactly - try partial match or fallback
                # This can happen due to tokenization differences
                # Fallback: use full sequence
                span_mask[i] = attention_mask[i]

    return span_mask


@torch.inference_mode()
def compute_representations(
    examples: List[ProbeExample],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Compute representations for probe examples.

    If an example has a probe_span, extracts representations only from those tokens.
    Otherwise, uses mean pooling over the full sequence.
    """
    features: List[np.ndarray] = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        end = min(start + batch_size, len(examples))
        batch_examples = examples[start:end]
        batch_texts = [ex.context for ex in batch_examples]
        batch_spans = [ex.probe_span for ex in batch_examples]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(**encoded, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-2]

        # Create span mask for selective pooling
        span_mask = _find_span_token_mask(
            encoded["input_ids"],
            encoded["attention_mask"],
            batch_spans,
            tokenizer,
        ).unsqueeze(-1).to(device)

        masked_hidden = hidden_states * span_mask
        token_counts = span_mask.sum(dim=1).clamp(min=1)
        pooled = (masked_hidden.sum(dim=1) / token_counts).float()
        features.append(pooled.cpu().numpy())

    return np.concatenate(features, axis=0)


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


 


def evaluate(
    clf: LogisticRegression, X: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    return acc, f1


def main(
    batch_size: int,
    max_length: int,
    test_size: float,
    seed: int,
    device_arg: Optional[str],
    model_name: str,
    output_path: str,
    lora_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    test_dataset_path: Optional[str] = None,
    min_confidence: Optional[str] = None,
):
    if device_arg is not None:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO, format="%(message)s"
    )
    logging.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use AutoModelForCausalLM when loading LoRA (trained on causal LM),
    # otherwise use AutoModel for base representations
    if lora_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            device_map="auto",
        )
        logging.info("Loading LoRA adapter from %s", lora_path)
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            device_map="auto",
        )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    logging.info("Loading training dataset")
    train_examples, train_labels = load_diplomacy_dataset(dataset_path, min_confidence)
    logging.info("Loaded %d labelled training examples", len(train_examples))

    # Log class balance
    n_truthful = (train_labels == 0).sum()
    n_deceptive = (train_labels == 1).sum()
    logging.info("Train class balance: %d truthful, %d deceptive", n_truthful, n_deceptive)

    # Log span coverage
    n_with_span = sum(1 for ex in train_examples if ex.probe_span is not None)
    logging.info("Train examples with probe spans: %d / %d", n_with_span, len(train_examples))

    if test_dataset_path is not None:
        # Use separate test dataset
        logging.info("Loading test dataset from %s", test_dataset_path)
        test_examples, test_labels = load_diplomacy_dataset(test_dataset_path, min_confidence)
        logging.info("Loaded %d labelled test examples", len(test_examples))

        n_truthful_test = (test_labels == 0).sum()
        n_deceptive_test = (test_labels == 1).sum()
        logging.info("Test class balance: %d truthful, %d deceptive", n_truthful_test, n_deceptive_test)

        n_with_span_test = sum(1 for ex in test_examples if ex.probe_span is not None)
        logging.info("Test examples with probe spans: %d / %d", n_with_span_test, len(test_examples))

        X_train_examples = train_examples
        y_train = train_labels
        X_test_examples = test_examples
        y_test = test_labels
    else:
        # Split training dataset
        X_train_examples, X_test_examples, y_train, y_test = train_test_split(
            train_examples,
            train_labels,
            test_size=test_size,
            random_state=seed,
            stratify=train_labels,
        )

    logging.info(
        "Encoding %d training and %d evaluation examples",
        len(X_train_examples),
        len(X_test_examples),
    )
    X_train = compute_representations(
        X_train_examples, tokenizer, model, device, batch_size, max_length
    )
    X_test = compute_representations(
        X_test_examples, tokenizer, model, device, batch_size, max_length
    )

    logging.info("Scaling features")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logging.info("Training linear (logistic regression) probe")
    clf = train_linear_probe(X_train, y_train)

    logging.info("Evaluating")
    train_acc, train_f1 = evaluate(clf, X_train, y_train)
    test_acc, test_f1 = evaluate(clf, X_test, y_test)

    logging.info("Train accuracy: %.4f | Train F1: %.4f", train_acc, train_f1)
    logging.info("Test accuracy: %.4f | Test F1: %.4f", test_acc, test_f1)

    print(
        {
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        }
    )

    # Save probe artifact
    artifact = {
        "probe_type": "logreg",
        "model_name": model_name,
        "seed": seed,
        "max_length": max_length,
        "batch_size": batch_size,
        "scaler": scaler,
        "classifier": clf,
        "metrics": {
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        },
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)
    logging.info("Saved probe to %s", str(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a deception linear probe on DipLLM"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for representation extraction",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data used for evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for input tensors (defaults to cuda:0 if available)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4b",
        help="Hugging Face model name or local path to load",
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="probe.pkl",
        help="Path to save the trained probe (pickle)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint (optional)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to training JSONL dataset (optional, defaults to ACL 2020 diplomacy)",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help="Path to separate test JSONL dataset (optional, if not provided splits --dataset)",
    )
    parser.add_argument(
        "--min-confidence",
        type=str,
        default=None,
        choices=["high", "medium", "low"],
        help="Minimum confidence level to include (high > medium > low)",
    )
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        max_length=args.max_length,
        test_size=args.test_size,
        seed=args.seed,
        device_arg=args.device,
        model_name=args.model_name,
        output_path=args.output_path,
        lora_path=args.lora_path,
        dataset_path=args.dataset,
        test_dataset_path=args.test_dataset,
        min_confidence=args.min_confidence,
    )
