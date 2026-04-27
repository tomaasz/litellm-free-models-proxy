# litellm-free-models-proxy

Self-hosted [LiteLLM](https://github.com/BerriAI/litellm) proxy that automatically discovers and registers LLM models available on **free API tiers** (free tokens, no credit card required) from multiple providers.

## What it does

- Exposes a single OpenAI-compatible API endpoint for all your LLM providers
- **Auto-discovers models with free API access** every 8h via `sync_models.py` — no manual config updates needed
- Load-balances across multiple API keys for the same provider
- Logs usage to Postgres (optional: Langfuse for observability)

> **Note:** "Free" here means _free API tokens_ — providers that let you call their models via API at no cost (within rate/token limits). This is not a list of open-source or self-hostable models.

## Providers with auto-discovery

| Provider | Free tier | How detected |
|---|---|---|
| **OpenRouter** | Free tokens for selected models | `pricing.prompt == "0"` in `/api/v1/models` |
| **Groq** | Rate-limited free tier for all models | All models from `/v1/models` |
| **Cerebras** | 1M tokens/day free | All models from `/v1/models` |
| **SambaNova** | Free tier for all models | All models from `/v1/models` |
| **Together AI** | Free for models with `-Free` suffix | `-Free` suffix or `pricing.input == 0` |
| **Cohere** | Trial key with free tokens | Chat models from `/v2/models` + [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) cross-reference |
| **Gemini** | Free quota via AI Studio keys | Flash and Gemma variants from `/v1beta/models` |
| **NVIDIA NIM** | 40 RPM free credits | All models from `/v1/models` |
| **HuggingFace** | Free credits/month via HF Router | All text models from HF Router |
| **Mistral** | Free Experiment plan | All models from `/v1/models` |

The sync script also cross-references [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) — a community-maintained list of providers offering free API access — to catch models that providers' own APIs don't mark as free explicitly.

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
