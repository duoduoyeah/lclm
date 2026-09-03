"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.models.gpt import GPT, GPTConfig
from nanochat.models.gpt_legacy import LegacyGPT, LegacyGPTConfig
from nanochat.models.simple_gpt import SimpleGPT, SimpleGPTConfig
from nanochat.models.multithread_lm import MultithreadLM, MultithreadLMConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Maps user_config["model"] (recorded by base_train.py) to (ModelCls, ConfigCls).
# Defaults to "gpt" for legacy checkpoints whose meta predates the --model flag.
# "gpt_legacy" is selected via model_class_override (e.g. karpathy/nanochat-d34).
_MODEL_REGISTRY = {
    "gpt": (GPT, GPTConfig),
    "gpt_legacy": (LegacyGPT, LegacyGPTConfig),
    "simple_gpt": (SimpleGPT, SimpleGPTConfig),
    "multithread_lm": (MultithreadLM, MultithreadLMConfig),
}

_REMOVED_MODEL_MESSAGES = {
    "butterfly_lm": (
        "butterfly_lm was removed from the active codebase. "
        "Recover it from git history if you need to inspect or run old "
        "butterfly checkpoints."
    ),
}

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs, model_name):
    """Add default values for new config keys missing in old checkpoints."""
    # Old GPT checkpoints predate the sliding-window flag; only GPT declares
    # window_pattern, so only patch that model family.
    if model_name == "gpt" and "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config, model_name):
    """Add default values for new parameters that may be missing in old checkpoints.
    These keys are gpt-specific; other models in _MODEL_REGISTRY (including
    gpt_legacy) don't declare them, so injecting would break strict load."""
    if model_name != "gpt":
        return
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def _prune_old_checkpoints(checkpoint_dir, keep_last):
    """Keep only the most recent `keep_last` step checkpoints. Removes
    model_<step>.pt, meta_<step>.json, and optim_<step>_rank*.pt for older
    steps. Rank-0 only (caller's responsibility)."""
    meta_paths = glob.glob(os.path.join(checkpoint_dir, "meta_*.json"))
    steps = []
    for path in meta_paths:
        m = re.match(r"meta_(\d+)\.json$", os.path.basename(path))
        if m:
            steps.append(int(m.group(1)))
    if len(steps) <= keep_last:
        return
    steps.sort()
    for step in steps[:-keep_last]:
        for name in (f"model_{step:06d}.pt", f"meta_{step:06d}.json"):
            p = os.path.join(checkpoint_dir, name)
            if os.path.exists(p):
                os.remove(p)
        for p in glob.glob(os.path.join(checkpoint_dir, f"optim_{step:06d}_rank*.pt")):
            os.remove(p)
        logger.info(f"Pruned old checkpoint step {step}")


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0, keep_last=None):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    # Prune older checkpoints on rank 0 only (safe: never touches current step).
    if rank == 0 and keep_last is not None and keep_last > 0:
        _prune_old_checkpoints(checkpoint_dir, keep_last)

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase, model_class_override=None):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    `model_class_override` (e.g. "gpt_legacy") bypasses the user_config.model
    lookup -- needed for externally-shipped checkpoints whose meta lacks our
    user_config.model field (karpathy/nanochat-d34).
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    if model_class_override is not None:
        if model_class_override not in _MODEL_REGISTRY:
            if model_class_override in _REMOVED_MODEL_MESSAGES:
                raise ValueError(_REMOVED_MODEL_MESSAGES[model_class_override])
            raise ValueError(f"Unknown model_class_override: {model_class_override!r}; "
                             f"known: {sorted(_MODEL_REGISTRY)}")
        model_name = model_class_override
        meta_model_name = meta_data.get("user_config", {}).get("model")
        log0(f"model_class_override={model_name!r} (meta user_config.model={meta_model_name!r})")
    else:
        # user_config.model is stamped by base_train.py; legacy checkpoints predate it.
        model_name = meta_data.get("user_config", {}).get("model", "gpt")
        if model_name not in _MODEL_REGISTRY:
            if model_name in _REMOVED_MODEL_MESSAGES:
                raise ValueError(_REMOVED_MODEL_MESSAGES[model_name])
            raise ValueError(f"Unknown model name in checkpoint meta: {model_name!r}; "
                             f"known: {sorted(_MODEL_REGISTRY)}")
    ModelCls, ConfigCls = _MODEL_REGISTRY[model_name]
    _patch_missing_config_keys(model_config_kwargs, model_name)
    log0(f"Building {model_name} with config: {model_config_kwargs}")
    model_config = ConfigCls(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config, model_name)
    with torch.device("meta"):
        model = ModelCls(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer. Prefer the name recorded at training time so we rehydrate
    # the exact tokenizer the model was trained with. If NANOCHAT_TOKENIZER is set
    # to a different value, fail loudly instead of silently loading a mismatched vocab.
    tokenizer_name = meta_data.get("tokenizer_name")
    env_name = os.environ.get("NANOCHAT_TOKENIZER") or None
    if tokenizer_name is not None and env_name is not None and env_name != tokenizer_name:
        raise RuntimeError(
            f"Checkpoint was trained with tokenizer '{tokenizer_name}' but "
            f"NANOCHAT_TOKENIZER={env_name}. Unset the env var or align it with the checkpoint."
        )
    tokenizer = get_tokenizer(name=tokenizer_name)
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None, model_class_override=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(
        checkpoint_dir, step, device, phase,
        model_class_override=model_class_override,
    )
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
