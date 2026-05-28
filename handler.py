"""RunPod serverless handler for Qwen-Image-2512 (Alibaba, Apache 2.0, 20B).

Modeled on the textcortex/qwen-image-runpod pattern but built locally so the
container image lives at a registry we control.

Hardened per /root/RUNPOD-SECURITY-RULES.md:
- Rule D1: sanitize_prompt() strips bidi/zero-width, caps 2000 chars, rejects markers.
- Rule A5: model snapshot is downloaded once to the attached network volume,
  reused across cold starts (no per-cold-start re-download).
- Rule C4: prompt only passed as string to the diffuser; no shell, no eval.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import threading
import time
from typing import Any

import runpod
import torch
from diffusers import DiffusionPipeline

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2512")
STORAGE_PATH = os.environ.get(
    "MODEL_STORAGE_PATH",
    os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume/qwen-image-2512"),
)
MIN_FREE_GB = int(os.environ.get("MIN_STORAGE_FREE_GB", "40"))

# Pin all HF/Diffusers caches to the network volume so the ~22GB model is
# downloaded once and reused forever.
os.environ.setdefault("HF_HOME", STORAGE_PATH)
os.environ.setdefault("HF_HUB_CACHE", STORAGE_PATH)
os.environ.setdefault("TRANSFORMERS_CACHE", STORAGE_PATH)
os.environ.setdefault("DIFFUSERS_CACHE", STORAGE_PATH)
os.environ.setdefault("XDG_CACHE_HOME", STORAGE_PATH)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Bidi + zero-width characters per RUNPOD-SECURITY-RULES Rule D1
_BIDI = "‪‫‬‭‮⁦⁧⁨⁩"
_ZWIDTH = "​‌‍﻿"
_STRIP_TABLE = str.maketrans("", "", _BIDI + _ZWIDTH)
_INJECTION_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)
_MAX_PROMPT_CHARS = 2000

_pipe_lock = threading.Lock()
_pipe: DiffusionPipeline | None = None


def sanitize_prompt(text: str | None) -> str:
    """Strip bidi/zero-width, cap length, reject system-prompt markers."""
    if not text:
        return ""
    cleaned = str(text).translate(_STRIP_TABLE)
    lowered = cleaned.lower()
    for marker in _INJECTION_MARKERS:
        if marker.lower() in lowered:
            raise ValueError(f"prompt rejected: contains injection marker {marker!r}")
    if len(cleaned) > _MAX_PROMPT_CHARS:
        cleaned = cleaned[:_MAX_PROMPT_CHARS]
    return cleaned.strip()


def ensure_disk_space() -> None:
    target = STORAGE_PATH
    os.makedirs(target, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"insufficient free disk at {target}: {free_gb:.1f}GB free "
            f"< required {MIN_FREE_GB}GB"
        )


def get_pipeline() -> DiffusionPipeline:
    """Lazy-init the pipeline on the first request (warm worker reuse)."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        ensure_disk_space()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"loading {MODEL_ID} (cache: {STORAGE_PATH})…")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            cache_dir=STORAGE_PATH,
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        _pipe = pipe
        print("pipeline ready")
        return _pipe


def handler(event: dict[str, Any]) -> dict[str, Any]:
    inp = event.get("input", {}) or {}
    try:
        prompt = sanitize_prompt(inp.get("prompt"))
    except ValueError as exc:
        return {"error": str(exc)}
    if not prompt:
        return {"error": "prompt is required"}

    try:
        negative = sanitize_prompt(inp.get("negative_prompt") or "")
    except ValueError as exc:
        return {"error": str(exc)}

    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))
    steps = int(inp.get("num_inference_steps", 50))
    cfg = float(inp.get("true_cfg_scale", inp.get("guidance_scale", 4.0)))
    seed = inp.get("seed")

    width = max(256, min(width, 2048))
    height = max(256, min(height, 2048))
    steps = max(8, min(steps, 80))

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

    pipe = get_pipeline()
    t0 = time.time()
    out = pipe(
        prompt=prompt,
        negative_prompt=negative or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=generator,
    )
    elapsed = time.time() - t0
    image = out.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "image": b64,  # matches textcortex/arkodeepsen response shape
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "model": MODEL_ID,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
