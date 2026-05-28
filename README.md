# qwen-image-2512-runpod

RunPod serverless worker for **Qwen-Image-2512** (Alibaba, Apache 2.0, 20B params, Dec 2025 release).

State-of-the-art open-source text rendering — scores 0.99 on short-text rendering benchmarks (vs FLUX.2-klein 0.87, original Qwen-Image 0.89). Built for Nobody School text-heavy infographic + cover production.

## Hardening

Per [`/root/RUNPOD-SECURITY-RULES.md`](../RUNPOD-SECURITY-RULES.md):
- **D1** — `sanitize_prompt()` strips bidi + zero-width characters before generation, rejects system-prompt markers (`<|im_start|>` etc), caps prompts at 2000 chars.
- **A5** — Network-volume cached: ~22GB model downloaded once, reused across cold starts.
- **C4** — Prompt is only ever passed as a string argument to the diffuser; no shell, no eval.

## Endpoint sizing

| Param | Value | Source |
|---|---|---|
| GPU | A100 80GB / H100 80GB | Qwen-2512 needs 80GB VRAM at bf16 |
| Container disk | 30 GB | runtime only — weights on the volume |
| Network volume | 100 GB (shared) | persists ~22 GB of weights |
| `workersMin` | 0 | scale to zero idle |
| `workersMax` | 3 | (Rule A1 cap) |
| `idleTimeout` | 5s | (Rule A4) |
| `executionTimeoutMs` | 600000 | (Rule A3) |
| `flashboot` | true | reduces cold start |

## Request

```json
{
  "input": {
    "prompt": "Editorial magazine cover with bold headline 'BUILD THE BEE STACK', cream background, illustrated honeybee in tangerine-orange",
    "negative_prompt": "blurry, low quality, distorted text, misspelled, watermark",
    "width": 1024,
    "height": 1280,
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "seed": 42
  }
}
```

## Response

```json
{
  "image": "<base64-PNG>",
  "width": 1024, "height": 1280, "steps": 40,
  "seed": 42, "elapsed_seconds": 9.4,
  "model": "Qwen/Qwen-Image-2512"
}
```

## Cold start

First-ever worker on a fresh volume: ~5–10 min while the model downloads from HF. Subsequent cold starts (volume already populated): ~30–60s. Warm calls: ~8–12s per 1024×1280 at 40 steps on A100 80GB.

## Local build + push

```bash
docker build -t ghcr.io/omni-woman/qwen-image-2512-runpod:latest .
echo "$GH_PAT" | docker login ghcr.io -u omni-woman --password-stdin
docker push ghcr.io/omni-woman/qwen-image-2512-runpod:latest
```
