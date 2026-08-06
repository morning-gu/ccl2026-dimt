# Renderer & Eraser API Servers

Each plugin that depends on an external model repo runs as an independent
FastAPI service.  The pipeline plugins call these services via HTTP,
achieving full dependency isolation.

## Architecture

```
Pipeline (client)                        API Server (independent venv)
+-----------------------+                +-----------------------------+
| easytext.py (client)  | -- HTTP POST -> | easytext_server.py          |
| fluxtext.py (client)  | -- HTTP POST -> | fluxtext_server.py          |
| anytext2.py (client)  | -- HTTP POST -> | anytext2_server.py          |
| strokenet.py (client) | -- HTTP POST -> | strokenet_server.py         |
| pert.py (client)      | -- HTTP POST -> | pert_server.py              |
| sser.py (client)      | -- HTTP POST -> | sser_server.py              |
+-----------------------+                +-----------------------------+
         |                                          |
   httpx (already in deps)              torch, diffusers, flash_attn, ...
   No heavy ML deps needed              Each server has its own venv
```

## Quick start

### 1. Install server dependencies

```bash
pip install -r servers/requirements.txt
```

Then install each external repo's own requirements in its dedicated venv.

### 2. Start a server

Each server reads its model-path env vars (the same vars the old plugins
used).  Set them in the server's environment, then launch:

```bash
# EasyText (port 8001)
export EASYTEXT_REPO_PATH=/path/to/EasyText
export EASYTEXT_FLUX_PATH=black-forest-labs/FLUX.1-dev
export EASYTEXT_PRETRAIN_LORA=/path/to/pretrain.safetensors
export EASYTEXT_FINETUNE_LORA=/path/to/finetune.safetensors
python servers/easytext_server.py --host 0.0.0.0 --port 8001

# FluxText (port 8002)
export FLUXTEXT_REPO_PATH=/path/to/FluxText
export FLUXTEXT_MODEL_PATH=/path/to/lora.safetensors
export FLUXTEXT_CONFIG_PATH=/path/to/config.yaml
python servers/fluxtext_server.py --host 0.0.0.0 --port 8002

# AnyText2 (port 8003)
export ANYTEXT2_MODEL_PATH=/path/to/AnyText2
python servers/anytext2_server.py --host 0.0.0.0 --port 8003

# STRNet eraser (port 8011)
export STROKENET_REPO=/path/to/SceneTextRemover-pytorch
export STROKENET_CKPT=/path/to/strnet_ckpt.pth
python servers/strokenet_server.py --host 0.0.0.0 --port 8011

# PERT eraser (port 8012)
export PERT_REPO=/path/to/PERT
export PERT_CKPT=/path/to/pert_ckpt.pth
python servers/pert_server.py --host 0.0.0.0 --port 8012

# SSER eraser (port 8013)
export SSER_REPO=/path/to/Self-supervised-Text-Erasing
export SSER_CKPT=/path/to/best_net_G.pth
python servers/sser_server.py --host 0.0.0.0 --port 8013

# PowerPaint eraser (port 8014)
# Weights auto-download from HuggingFace; no manual clone needed.
export POWERPAINT_MODEL_ID=SanityZero/PowerPaint-v2  # optional override
python servers/powerpaint_server.py --host 0.0.0.0 --port 8014
```

### 3. Point the pipeline at the servers

In `.env` (or environment):

```env
EASYTEXT_API_URL=http://localhost:8001/render
FLUXTEXT_API_URL=http://localhost:8002/render
ANYTEXT2_API_URL=http://localhost:8003/render
STROKENET_API_URL=http://localhost:8011/erase
PERT_API_URL=http://localhost:8012/erase
SSER_API_URL=http://localhost:8013/erase
POWERPAINT_API_URL=http://localhost:8014/erase
```

Select the plugin in your pipeline config as before (e.g. `renderer: easytext`,
`eraser: strokenet`).  No other pipeline code changes are needed.

## API protocol

Renderer servers expose `POST /render`:

```
POST /render
Content-Type: application/json

{
  "image": "<base64 PNG>",
  "regions": [
    {
      "text": "...",
      "bbox": [x1, y1, x2, y2],
      "translated_text": "...",
      "style_info": {"color": [b, g, r], "font_weight": "normal"},
      "bbox_poly": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      ...
    }
  ],
  "style_reference": "<base64 PNG or null>"
}

Response:
{
  "image": "<base64 PNG>"
}
```

`GET /health` returns `{"status": "ok", "model_loaded": true/false}`.

Eraser servers expose `POST /erase`:

```
POST /erase
Content-Type: application/json

{
  "image": "<base64 PNG>",
  "regions": [ ... ],
  "dilate_pixels": 5
}

Response:
{
  "image": "<base64 PNG>"
}
```

## Files

| File | Purpose |
|------|---------|
| `_shared.py` | Shared image codec, region models, mask building, request/response schemas |
| `easytext_server.py` | EasyText (FLUX DiT + LoRA) API server |
| `fluxtext_server.py` | FluxText (FLUX-Text inpainting) API server |
| `anytext2_server.py` | AnyText2 API server |
| `strokenet_server.py` | STRNet stroke-level eraser API server |
| `pert_server.py` | PERT scene text removal API server |
| `sser_server.py` | SSER (STRnet2) self-supervised eraser API server |
| `powerpaint_server.py` | PowerPaint prompt-augmented diffusion eraser API server |
| `requirements.txt` | Common server dependencies (fastapi, uvicorn, etc.) |

## Notes

- Images are encoded as base64 PNG.  For 1024x1024 images this is ~1.4 MB
  per request; on localhost or a fast LAN the overhead is negligible.
- Each server lazy-loads its model on the first request.  Use `GET /health`
- Each server loads its model at startup via FastAPI's ``lifespan`` handler.
  `GET /health` confirms the server is ready.
- Timeout defaults to 120 seconds.  Override with `<NAME>_API_TIMEOUT`.
- Servers can run on different machines (e.g. a GPU server) from the
  pipeline client.
