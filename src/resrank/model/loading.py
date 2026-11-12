"""Build a ResRankModel from the YAML config."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from resrank.model.prompts import DEFAULT_INSTRUCTION_TEMPLATE
from resrank.model.resrank import ResRankConfig, ResRankModel
from resrank.paths import Paths, checkpoint_parts
from resrank.torch_utils import DTYPES, default_device


def resrank_config_from_cfg(cfg: dict[str, Any]) -> ResRankConfig:
    loss, model = cfg.get("loss", {}), cfg.get("model", {})
    return ResRankConfig(
        fusion_mode=cfg.get("fusion", {}).get("mode", "residual"),
        inject_mode=model.get("inject_mode", "raw"),
        inject_scale=float(model.get("inject_scale", 1.0)),
        instruction_template=model.get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE),
        tau_infonce=loss.get("tau_infonce", 0.05),
        tau_ranknet=loss.get("tau_ranknet", 0.05),
        lam=loss.get("lambda_infonce", 0.1),
        pair_reduction=loss.get("ranknet_pair_reduction", "sum"),
        passage_chunk_size=model.get("passage_chunk_size"),
    )


def freeze_embeddings(model: ResRankModel) -> int:
    """Stop training both models' token-embedding tables. They hold about a quarter of the parameters
    (vocab 151936 x 1024 each at 0.6B), so this cuts gradient and optimizer memory by about that much.
    It is a departure from the paper's full-parameter training. Returns the number of frozen params."""
    frozen = 0
    for part in (model.encoder.model, model.reranker.model):
        w = part.get_input_embeddings().weight
        w.requires_grad_(False)
        frozen += w.numel()
    return frozen


def load_model(cfg: dict[str, Any], ckpt_dir: Optional[str | Path] = None, *,
               device: str = "cuda", weights_dtype: Optional[str] = None) -> ResRankModel:
    """Load from a checkpoint dir (saved by ResRankModel.save_pretrained) or from the pretrained
    Qwen3 models named in the config. `weights_dtype` is fp32/fp16/bf16 or None for the default."""
    m = cfg.get("model", {})
    common = dict(
        dtype=DTYPES[weights_dtype] if weights_dtype else None,
        attn_implementation=m.get("attn_implementation"),
        gradient_checkpointing=bool(m.get("gradient_checkpointing", False)),
    )
    enc_kwargs = dict(max_length=m.get("max_passage_len", 512), query_max_length=m.get("max_query_len", 128))
    if ckpt_dir is not None:
        enc_path, rr_path = (str(p) for p in checkpoint_parts(ckpt_dir))
    else:
        enc_path, rr_path = cfg["models"]["encoder"], cfg["models"]["reranker"]
    model = ResRankModel.from_pretrained(enc_path, rr_path, resrank_config_from_cfg(cfg),
                                         encoder_kwargs=enc_kwargs, **common)
    if m.get("freeze_embeddings", False):
        freeze_embeddings(model)
    return model.to(device)


def load_eval_model(cfg: dict[str, Any], paths: Paths, ckpt: str) -> ResRankModel:
    """Model for the evaluation stages (s06, s07): checkpoint `ckpt` under checkpoints/, or the untrained
    Qwen3 models when ckpt is 'pretrained'. fp16 weights on the GPU when there is one."""
    return load_model(cfg, None if ckpt == "pretrained" else paths.stage_ckpt(ckpt),
                      device=default_device(), weights_dtype="fp16")
