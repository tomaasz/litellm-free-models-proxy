# litellm-free-models-proxy

Self-hosted [LiteLLM](https://github.com/BerriAI/litellm) proxy that automatically discovers and registers free/trial LLM models from multiple providers.

## What it does

- Exposes a single OpenAI-compatible API endpoint for all your LLM providers
- **Auto-discovers new free models** every 24h via `sync_models.py` — no manual config updates needed
- Load-balances across multiple API keys for the same provider
- Logs usage to Postgres (optional: Langfuse for observability)

## Providers with auto-discovery

| Provider | How free models are detected |
|---|---|
| **OpenRouter** | `pricing.prompt == "0"` in `/api/v1/models` response |
| **Groq** | All models from `/v1/models` (entire tier is free, rate-limited) |
| **Cerebras** | All models (1M tokens/day free) |
| **SambaNova** | All models from `/v1/models` |
| **Together AI** | Models with `-Free` suffix or `pricing.input == 0` |
| **Cohere** | Chat models from `/v2/models` + [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) cross-reference |
| **Gemini** | Flash and Gemma variants from `/v1beta/models` (Pro/Ultra excluded) |
| **NVIDIA NIM** | All models (40 RPM free credits) |
| **HuggingFace** | All text models from HF Router |
| **Mistral** | All models from `/v1/models` (free Experiment plan) |

The sync script also cross-references [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) — a community-maintained, auto-generated list of free LLM APIs — to catch models that providers' APIs don't mark as free explicitly.

## Quick start

```bash
cp .env.example .env
# Edit .env with your API keys

docker compose up -d
```

The proxy listens on port `4000`. Access the UI at `http://localhost:4000/ui`.

## Services

| Service | Purpose |
|---|---|
| `litellm` | The proxy itself |
| `model-sync` | Auto-discovery service, runs every 24h |
| `postgres` | Persistence for model configs and usage logs |

> `postgres` is expected as an external Docker network (`postgres_default`). Adjust `docker-compose.yml` if you run Postgres differently.

## Manual model config

`config.yaml` defines the initial model list and routing groups (`smart`, `fast`, `reasoning`, `coder`, `long`, `vision`). The auto-sync adds named model routes (e.g. `or/llama-3.3-70b`, `groq/qwen3-32b`) but never modifies routing groups — those require human judgment.

## Environment variables for sync

| Variable | Default | Description |
|---|---|---|
| `SYNC_INTERVAL_HOURS` | `24` | How often to check for new models |
| `STARTUP_DELAY_SECONDS` | `60` | Wait for LiteLLM to be ready before first sync |
| `LITELLM_BASE_URL` | `http://litellm:4000` | LiteLLM internal URL |

## Credits

- [LiteLLM](https://github.com/BerriAI/litellm) — the proxy
- [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) — community list used as cross-reference
