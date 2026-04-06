"""Checkpoint helpers for compiled and non-compiled models."""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


_COMPILED_PREFIX = "_orig_mod."
_AST_POS_EMBED_SUFFIX = "ast.embeddings.position_embeddings"


def unwrap_model(model):
    """Return the underlying module when torch.compile wraps the model."""
    return getattr(model, "_orig_mod", model)


def export_model_state_dict(model):
    """Save unwrapped weights so checkpoints are portable across runtimes."""
    return unwrap_model(model).state_dict()


def normalize_model_state_dict(state_dict):
    """Strip torch.compile prefixes from checkpoint keys when present."""
    normalized = OrderedDict()
    changed = False

    for key, value in state_dict.items():
        new_key = key
        while new_key.startswith(_COMPILED_PREFIX):
            new_key = new_key[len(_COMPILED_PREFIX):]
        changed = changed or (new_key != key)
        normalized[new_key] = value

    return normalized, changed


def _resolve_attr_path(root, path: str):
    node = root
    if not path:
        return node
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _resize_model_ast_position_embeddings(model, state_dict):
    """Match AST positional embedding shapes to the checkpoint before loading."""
    model = unwrap_model(model)
    resized_any = False

    ast_keys = [key for key in state_dict if key.endswith(_AST_POS_EMBED_SUFFIX)]
    for ast_key in ast_keys:
        prefix = ast_key[: -len("." + _AST_POS_EMBED_SUFFIX)] if ast_key != _AST_POS_EMBED_SUFFIX else ""
        try:
            owner = _resolve_attr_path(model, prefix)
        except AttributeError:
            continue
        if not hasattr(owner, "ast"):
            continue

        checkpoint_pos = state_dict[ast_key]
        ast = owner.ast
        current_pos = ast.embeddings.position_embeddings
        if checkpoint_pos.shape == current_pos.shape:
            continue

        config = ast.config
        patch = config.patch_size
        freq_stride = config.frequency_stride
        time_stride = config.time_stride
        freq_out = (config.num_mel_bins - patch) // freq_stride + 1

        patch_tokens = checkpoint_pos.shape[1] - 2
        if patch_tokens <= 0 or patch_tokens % freq_out != 0:
            raise ValueError(
                "Checkpoint AST position embedding shape is incompatible with the "
                f"current AST config: {tuple(checkpoint_pos.shape)}"
            )

        time_out = patch_tokens // freq_out
        max_length = (time_out - 1) * time_stride + patch

        ast.embeddings.position_embeddings = nn.Parameter(current_pos.new_empty(checkpoint_pos.shape))
        ast.config.max_length = max_length
        if hasattr(owner, "_ast_pos_len"):
            owner._ast_pos_len = max_length
        resized_any = True

    return resized_any


def load_model_state_dict(model, state_dict, strict: bool = True):
    """Load checkpoint weights after normalizing common wrapper differences."""
    normalized_state_dict, normalized = normalize_model_state_dict(state_dict)
    resized_ast_pos = _resize_model_ast_position_embeddings(model, normalized_state_dict)
    model.load_state_dict(normalized_state_dict, strict=strict)
    return {
        "normalized_compiled_keys": normalized,
        "resized_ast_position_embeddings": resized_ast_pos,
    }
