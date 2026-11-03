"""Plain-PyTorch ViT that loads HuggingFace ``ViTForImageClassification`` checkpoints unchanged.

No ``transformers`` dependency (recent releases need a newer PyTorch than the cluster's). The
module and parameter names mirror the HuggingFace ones, so ``pytorch_model.bin`` /
``model.safetensors`` load with ``strict`` key matching; tests/test_cascade.py checks the logits
against ``transformers`` when it is installed.
"""

import json
import logging
import math
import os
import re
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ViTSelfAttention(nn.Module):
    def __init__(self, d, heads, qkv_bias=True):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(d, d, bias=qkv_bias)
        self.key = nn.Linear(d, d, bias=qkv_bias)
        self.value = nn.Linear(d, d, bias=qkv_bias)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        def split(t):
            return t.view(b, n, h, d // h).transpose(1, 2)
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // h), dim=-1)
        return (att @ v).transpose(1, 2).reshape(b, n, d)


class _Dense(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.dense = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.dense(x)


class _Attention(nn.Module):
    def __init__(self, d, heads, qkv_bias):
        super().__init__()
        self.attention = ViTSelfAttention(d, heads, qkv_bias)
        self.output = _Dense(d, d)

    def forward(self, x):
        return self.output(self.attention(x))


class ViTLayer(nn.Module):
    def __init__(self, d, heads, d_mlp, eps, qkv_bias=True):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(d, eps=eps)
        self.attention = _Attention(d, heads, qkv_bias)
        self.layernorm_after = nn.LayerNorm(d, eps=eps)
        self.intermediate = _Dense(d, d_mlp)
        self.output = _Dense(d_mlp, d)

    def forward(self, x):
        x = x + self.attention(self.layernorm_before(x))
        return x + self.output(F.gelu(self.intermediate(self.layernorm_after(x))))


class _PatchEmbeddings(nn.Module):
    def __init__(self, c, d, p):
        super().__init__()
        self.projection = nn.Conv2d(c, d, kernel_size=p, stride=p)

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)


class _Embeddings(nn.Module):
    def __init__(self, c, d, p, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, d))
        self.patch_embeddings = _PatchEmbeddings(c, d, p)

    def forward(self, x):
        z = self.patch_embeddings(x)
        z = torch.cat([self.cls_token.expand(z.shape[0], -1, -1), z], dim=1)
        return z + self.position_embeddings


class _Encoder(nn.Module):
    def __init__(self, n_layers, **kw):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(**kw) for _ in range(n_layers)])

    def forward(self, x):
        for blk in self.layer:
            x = blk(x)
        return x


class _ViTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, p, size = cfg["hidden_size"], cfg["patch_size"], cfg["image_size"]
        self.embeddings = _Embeddings(cfg.get("num_channels", 3), d, p, (size // p) ** 2)
        self.encoder = _Encoder(cfg["num_hidden_layers"], d=d, heads=cfg["num_attention_heads"],
                                d_mlp=cfg["intermediate_size"], eps=cfg.get("layer_norm_eps", 1e-12),
                                qkv_bias=cfg.get("qkv_bias", True))
        self.layernorm = nn.LayerNorm(d, eps=cfg.get("layer_norm_eps", 1e-12))

    def forward(self, x):
        return self.layernorm(self.encoder(self.embeddings(x)))


class _Out:
    def __init__(self, logits):
        self.logits = logits


class ViTForImageClassification(nn.Module):
    """``model(pixel_values=x).logits`` as in HuggingFace; ``x`` already normalized and resized."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.get("hidden_act", "gelu") != "gelu":
            raise ValueError(f"unsupported activation {cfg.get('hidden_act')}")
        self.config = cfg
        self.vit = _ViTModel(cfg)
        self.classifier = nn.Linear(cfg["hidden_size"], len(cfg["id2label"]))

    def forward(self, pixel_values):
        return _Out(self.classifier(self.vit(pixel_values)[:, 0]))


_NEW_TO_LEGACY = (
    (r"^vit\.layers\.(\d+)\.attention\.q_proj\.", r"vit.encoder.layer.\1.attention.attention.query."),
    (r"^vit\.layers\.(\d+)\.attention\.k_proj\.", r"vit.encoder.layer.\1.attention.attention.key."),
    (r"^vit\.layers\.(\d+)\.attention\.v_proj\.", r"vit.encoder.layer.\1.attention.attention.value."),
    (r"^vit\.layers\.(\d+)\.attention\.o_proj\.", r"vit.encoder.layer.\1.attention.output.dense."),
    (r"^vit\.layers\.(\d+)\.mlp\.fc1\.", r"vit.encoder.layer.\1.intermediate.dense."),
    (r"^vit\.layers\.(\d+)\.mlp\.fc2\.", r"vit.encoder.layer.\1.output.dense."),
    (r"^vit\.layers\.(\d+)\.", r"vit.encoder.layer.\1."),
)


def to_legacy_keys(state):
    """Map ``transformers>=5`` ViT parameter names to the original (checkpoint) names; idempotent."""
    out = {}
    for k, v in state.items():
        for pat, rep in _NEW_TO_LEGACY:
            k2 = re.sub(pat, rep, k)
            if k2 != k:
                k = k2
                break
        out[k] = v
    return out


def _download(repo, filename, cache_dir):
    """Local path of ``filename`` of a Hub repo, downloaded once into ``cache_dir``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    if hf_hub_download is not None:
        return hf_hub_download(repo, filename, cache_dir=cache_dir)
    dst = os.path.join(cache_dir, repo.replace("/", "--"), filename)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        logger.info("downloading %s", url)
        urllib.request.urlretrieve(url, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    return dst


def load_vit_from_hub(repo, cache_dir):
    """Config + weights of a HuggingFace ViT classifier (``pytorch_model.bin`` or ``model.safetensors``)."""
    with open(_download(repo, "config.json", cache_dir), encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg.get("model_type") != "vit":
        raise ValueError(f"{repo}: model_type {cfg.get('model_type')} is not 'vit'")
    model = ViTForImageClassification(cfg)
    try:
        path = _download(repo, "model.safetensors", cache_dir)
        from safetensors.torch import load_file
        state = load_file(path)
    except Exception:                                    # no safetensors file / package
        path = _download(repo, "pytorch_model.bin", cache_dir)
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in to_legacy_keys(state).items() if not k.startswith("vit.pooler.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{repo}: missing keys {missing[:5]}..., unexpected {unexpected[:5]}...")
    logger.info("ViT %s loaded from %s", repo, path)
    return model.eval()
