import logging
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Ensure the classes used in the pickle (e.g., DiffInMeansClassifier) are importable
# from the original training module path
from gamescope.environments.diplomacy import train_probe as train_probe_module

logger = logging.getLogger(__name__)


@torch.inference_mode()
def _compute_representations(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    features: List[np.ndarray] = []
    model.eval()
    
    # Get the second-to-last layer for hooking
    # Most transformer models have layers in model.encoder.layer or model.layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = model.encoder.layer
        layer_path = "model.encoder.layer"
    elif hasattr(model, 'layers'):
        layers = model.layers
        layer_path = "model.layers"
    else:
        # Fallback: use output_hidden_states if we can't find layers
        logger.warning("Could not find model layers for hooking, falling back to output_hidden_states")
        return _compute_representations_fallback(texts, tokenizer, model, device, batch_size, max_length)
    
    target_layer = layers[-2]
    logger.info(f"Using forward hook on {layer_path}[-2] to capture hidden states")
    
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Hook to capture hidden states from target layer
        captured_hidden_states = []
        def hook_fn(module, input, output):
            # output is typically (hidden_states,) or just hidden_states
            hidden = output[0] if isinstance(output, tuple) else output
            captured_hidden_states.append(hidden)
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _ = model(**encoded)
            
            # Extract the captured hidden states
            if not captured_hidden_states:
                logger.error("Forward hook failed to capture hidden states!")
                raise RuntimeError("Hook did not capture any hidden states")
            
            hidden_states = captured_hidden_states[0]
            logger.debug(f"Hook captured hidden states with shape: {hidden_states.shape}, device: {hidden_states.device}")
            
            # Move attention_mask to same device as hidden_states (important for multi-GPU)
            # Note: with device_map="auto", layer -2 may be on a different GPU than the primary device
            attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_states.device)
            masked_hidden = hidden_states * attention_mask
            token_counts = attention_mask.sum(dim=1).clamp(min=1)
            pooled = (masked_hidden.sum(dim=1) / token_counts).float()
            features.append(pooled.cpu().numpy())
        finally:
            handle.remove()
        
        # Explicitly free GPU memory between batches
        del hidden_states, masked_hidden, encoded, captured_hidden_states
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    logger.info(f"Successfully processed {len(texts)} texts using forward hook approach (batch_size={batch_size})")
    return np.concatenate(features, axis=0) if features else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def _compute_representations_fallback(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Fallback implementation using output_hidden_states=True"""
    logger.info("Using fallback method with output_hidden_states=True")
    features: List[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
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
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = (masked_hidden.sum(dim=1) / token_counts).float()
        features.append(pooled.cpu().numpy())
        
        # Explicitly free GPU memory between batches
        del outputs, hidden_states, masked_hidden, encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    logger.info(f"Successfully processed {len(texts)} texts using fallback method (batch_size={batch_size})")
    return np.concatenate(features, axis=0) if features else np.zeros((0, model.config.hidden_size), dtype=np.float32)


@dataclass
class ProbeContext:
    model: Any
    tokenizer: Any
    device: torch.device
    scaler: Any
    classifier: Any
    probe_type: str
    max_length: int
    batch_size: int

    def classify_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        X = _compute_representations(
            texts=texts,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        X = self.scaler.transform(X)

        results: List[Dict[str, Any]] = []

        # Try to use predict_proba if available (e.g., LogisticRegression)
        proba: Optional[np.ndarray] = None
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X)  # shape (n, 2)
            preds = (proba[:, 1] >= 0.5).astype(np.int64)
        else:
            # Fallback to predict only
            preds = self.classifier.predict(X)

        for i, y in enumerate(preds):
            label = "deceptive" if int(y) == 1 else "truthful"
            entry: Dict[str, Any] = {"label": label, "y": int(y)}
            if proba is not None:
                # Probability of class 1 (deceptive)
                entry["prob_deceptive"] = float(proba[i, 1])
                entry["prob_truthful"] = float(proba[i, 0])
            results.append(entry)

        return results


def load_probe(artifact_path: str, device: Optional[str] = "auto") -> ProbeContext:
    """Load a trained probe model.
    
    Args:
        artifact_path: Path to the probe artifact pickle file
        device: Device placement strategy:
            - "auto": Use device_map="auto" for automatic multi-GPU placement (default)
            - "cuda:N": Pin to specific GPU N
            - "cpu": Use CPU only
            - None: Same as "auto"
    """
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    model_name: str = artifact["model_name"]
    scaler = artifact["scaler"]
    classifier = artifact["classifier"]
    probe_type: str = artifact.get("probe_type", "logreg")
    max_length: int = artifact.get("max_length", 512)
    batch_size: int = artifact.get("batch_size", 8)

    if device is None or device == "auto":
        # Use device_map="auto" for automatic multi-GPU placement
        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if torch.cuda.is_available() else None
    else:
        # Pin to specific device (e.g., 'cuda:1' or 'cpu')
        torch_device = torch.device(device)
        device_map = {"": device} if torch_device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Log model dtype
    if hasattr(model, 'dtype'):
        logger.info(f"Model dtype: {model.dtype}")
    
    # Log device distribution for multi-GPU verification
    if hasattr(model, 'hf_device_map'):
        logger.info(f"Model device map: {model.hf_device_map}")
        # Count parameters per device
        device_params = {}
        for name, device_id in model.hf_device_map.items():
            device_params[device_id] = device_params.get(device_id, 0) + 1
        logger.info(f"Module distribution across devices: {device_params}")
    else:
        logger.info(f"Model loaded on single device: {torch_device}")

    # Choose primary device consistent with placement
    primary_device = torch_device if device is not None else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    return ProbeContext(
        model=model,
        tokenizer=tokenizer,
        device=primary_device,
        scaler=scaler,
        classifier=classifier,
        probe_type=probe_type,
        max_length=max_length,
        batch_size=batch_size,
    )


