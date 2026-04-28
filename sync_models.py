#!/usr/bin/env python3
"""
LiteLLM model auto-sync.

Queries each configured provider for available (free) models and registers
any new ones via LiteLLM's management API.

Sources:
  - Provider /models APIs (primary)
  - cheahjs/free-llm-api-resources README (cross-reference for providers
    whose API does not expose pricing info)

Does NOT touch routing groups (smart/fast/etc.) — only adds named routes
like or/llama-3.3-70b, groq/llama-3.3-70b-versatile, etc.
"""

import os
import re
import time
import logging
import urllib.request
import urllib.error
import json
from html.parser import HTMLParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LITELLM_BASE = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000")
LITELLM_KEY = os.environ.get("LITELLM_MASTER_KEY", "")
SYNC_INTERVAL_H = int(os.environ.get("SYNC_INTERVAL_HOURS", "24"))
STARTUP_DELAY_S = int(os.environ.get("STARTUP_DELAY_SECONDS", "60"))

# Community-maintained list of free LLM APIs (auto-generated, updated frequently).
CHEAHJS_README_URL = (
    "https://raw.githubusercontent.com/cheahjs/free-llm-api-resources"
    "/refs/heads/main/README.md"
)


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────

_HEADERS = {"User-Agent": "litellm-free-models-proxy/1.0"}


def _http_get(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers={**_HEADERS, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def _json_get(url, headers=None, timeout=20):
    return json.loads(_http_get(url, headers, timeout))


def _post_litellm(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{LITELLM_BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {LITELLM_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _get_litellm(path):
    return _json_get(
        f"{LITELLM_BASE}{path}",
        headers={"Authorization": f"Bearer {LITELLM_KEY}"},
    )


# ── LiteLLM state ─────────────────────────────────────────────────────────────

def get_existing_litellm_models():
    """Return set of litellm_params.model strings already registered."""
    try:
        data = _get_litellm("/model/info")
        return {
            entry.get("litellm_params", {}).get("model", "")
            for entry in data.get("data", [])
            if entry.get("litellm_params", {}).get("model")
        }
    except Exception as e:
        log.error(f"Failed to fetch existing models: {e}")
        return set()


def add_model(model_name, litellm_model, api_key_env, rpm=None, api_base=None):
    params = {"model": litellm_model, "api_key": f"os.environ/{api_key_env}"}
    if rpm:
        params["rpm"] = rpm
    if api_base:
        params["api_base"] = api_base
    try:
        _post_litellm("/model/new", {"model_name": model_name, "litellm_params": params})
        return True
    except Exception as e:
        log.error(f"  Failed to add {model_name} ({litellm_model}): {e}")
        return False


# ── cheahjs/free-llm-api-resources cross-reference ───────────────────────────

class _TableTextParser(HTMLParser):
    """Extracts <td> text content from an HTML table."""
    def __init__(self):
        super().__init__()
        self.in_td = False
        self.cells = []
    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.in_td = True
    def handle_endtag(self, tag):
        if tag == "td":
            self.in_td = False
    def handle_data(self, data):
        if self.in_td:
            self.cells.append(data.strip())


def _extract_section(readme, heading):
    """Return text of a markdown/HTML section starting at ### heading."""
    pattern = rf"### \[?{re.escape(heading)}"
    m = re.search(pattern, readme, re.IGNORECASE)
    if not m:
        return ""
    start = m.start()
    next_section = re.search(r"\n### ", readme[start + 1:])
    end = start + 1 + next_section.start() if next_section else len(readme)
    return readme[start:end]


def fetch_community_free_models():
    """
    Parse cheahjs/free-llm-api-resources README.
    Returns dict provider_key → set of model IDs.

    Only extracts providers where the README lists actual model IDs
    (not just display names). Currently: cohere, openrouter.
    """
    result = {"cohere": set(), "openrouter": set()}
    try:
        readme = _http_get(CHEAHJS_README_URL, timeout=20)
    except Exception as e:
        log.warning(f"[community] Could not fetch cheahjs README: {e}")
        return result

    # Cohere section lists model IDs directly as plain-text list items
    cohere_section = _extract_section(readme, "Cohere")
    for line in cohere_section.splitlines():
        line = line.strip().lstrip("- ")
        if line and not line.startswith("[") and not line.startswith("#") \
                and not line.startswith("*") and not line.startswith("<"):
            if "/" not in line and len(line) < 60:
                result["cohere"].add(line)

    # OpenRouter section has links like (https://openrouter.ai/provider/model:free)
    or_section = _extract_section(readme, "OpenRouter")
    for m in re.finditer(r"openrouter\.ai/([^)\"'\s]+:free)", or_section):
        result["openrouter"].add(m.group(1))

    log.info(
        f"[community] cheahjs: {len(result['openrouter'])} OR models, "
        f"{len(result['cohere'])} Cohere models"
    )
    return result


# ── Provider fetchers ──────────────────────────────────────────────────────────

def fetch_openrouter(api_key):
    """Free models: pricing.prompt == '0' AND pricing.completion == '0'."""
    try:
        data = _json_get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        free = [
            m["id"]
            for m in data.get("data", [])
            if str(m.get("pricing", {}).get("prompt", "1")) == "0"
            and str(m.get("pricing", {}).get("completion", "1")) == "0"
        ]
        log.info(f"[OpenRouter] {len(free)} free models from API")
        return free
    except Exception as e:
        log.error(f"[OpenRouter] {e}")
        return []


def fetch_groq(api_key):
    """All text/chat models are on free tier (rate-limited)."""
    try:
        data = _json_get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [
            m["id"]
            for m in data.get("data", [])
            if not any(x in m.get("id", "").lower() for x in ("whisper", "tts", "embed", "guard"))
        ]
        log.info(f"[Groq] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[Groq] {e}")
        return []


def fetch_cerebras(api_key):
    """All models are on free tier (1M tokens/day)."""
    try:
        data = _json_get(
            "https://api.cerebras.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [m["id"] for m in data.get("data", [])]
        log.info(f"[Cerebras] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[Cerebras] {e}")
        return []


def fetch_sambanova(api_key):
    try:
        data = _json_get(
            "https://api.sambanova.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [m["id"] for m in data.get("data", [])]
        log.info(f"[SambaNova] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[SambaNova] {e}")
        return []


def fetch_together(api_key):
    """Free models: -Free/-free suffix or pricing == 0."""
    try:
        data = _json_get(
            "https://api.together.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        items = data if isinstance(data, list) else data.get("data", [])
        free = []
        for m in items:
            mid = m.get("id", "")
            p = m.get("pricing", {})
            if (p.get("input", 1) == 0 and p.get("output", 1) == 0) \
                    or "-Free" in mid or "-free" in mid:
                free.append(mid)
        log.info(f"[Together] {len(free)} free models")
        return free
    except Exception as e:
        log.error(f"[Together] {e}")
        return []


def fetch_cohere(api_key, community_ids=None):
    """
    Chat models from the trial key, cross-referenced with cheahjs README.
    If the API fails, falls back to the community list.
    """
    try:
        data = _json_get(
            "https://api.cohere.com/v2/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [
            m["name"]
            for m in data.get("models", [])
            if "chat" in m.get("endpoints", [])
        ]
        if community_ids:
            # add any models listed in community reference that API missed
            ids = list(set(ids) | community_ids)
        log.info(f"[Cohere] {len(ids)} chat models")
        return ids
    except Exception as e:
        log.error(f"[Cohere] API error: {e}")
        if community_ids:
            log.info(f"[Cohere] Falling back to community list ({len(community_ids)} models)")
            return list(community_ids)
        return []


def fetch_gemini(api_key):
    """
    Free-tier Gemini models: generateContent-capable flash/gemma variants.
    Pro and Ultra models require billing; TTS/embedding models are excluded.
    """
    try:
        data = _json_get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key},
        )
        free = []
        for m in data.get("models", []):
            name = m.get("name", "").replace("models/", "")  # "models/gemini-2.5-flash" → "gemini-2.5-flash"
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            nl = name.lower()
            # Exclude non-free variants
            if any(x in nl for x in ("-pro", "-ultra", "embedding", "-tts", "robotics")):
                continue
            # Include flash, lite, and gemma (open) models
            if any(x in nl for x in ("flash", "gemma")):
                free.append(name)
        log.info(f"[Gemini] {len(free)} free-tier models")
        return free
    except Exception as e:
        log.error(f"[Gemini] {e}")
        return []


def fetch_nvidia(api_key):
    """All NVIDIA NIM models have 40 RPM free credits."""
    try:
        data = _json_get(
            "https://integrate.api.nvidia.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [
            m["id"]
            for m in data.get("data", [])
            if not any(x in m.get("id", "").lower() for x in ("embed", "rerank", "tts"))
        ]
        log.info(f"[NVIDIA NIM] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[NVIDIA NIM] {e}")
        return []


def fetch_huggingface(api_key):
    try:
        data = _json_get(
            "https://router.huggingface.co/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [
            m["id"]
            for m in data.get("data", [])
            if not any(x in m.get("id", "").lower() for x in ("embed", "vision", "tts", "stt"))
        ]
        log.info(f"[HuggingFace] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[HuggingFace] {e}")
        return []


def fetch_mistral(api_key):
    """
    Mistral La Plateforme — Experiment/free plan.
    API does not expose pricing, so we add all text gen models and let
    the Mistral rate-limiter handle it (free tier: 1 req/s, 1B tok/month).
    Only adds models not already in config.
    """
    try:
        data = _json_get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [
            m["id"]
            for m in data.get("data", [])
            if m.get("object") == "model"
            and not any(x in m.get("id", "").lower() for x in ("embed", "moderation"))
        ]
        log.info(f"[Mistral] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[Mistral] {e}")
        return []


def fetch_github(api_key):
    """GitHub Models — free tier (rate-limited), higher limits with Copilot."""
    try:
        data = _json_get(
            "https://models.inference.ai.azure.com/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        items = data if isinstance(data, list) else data.get("data", [])
        ids = [
            m.get("id") or m.get("name", "")
            for m in items
            if not any(x in (m.get("id") or m.get("name", "")).lower()
                       for x in ("embed", "tts", "whisper", "dall-e", "image"))
            and (m.get("id") or m.get("name", ""))
        ]
        log.info(f"[GitHub Models] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[GitHub Models] {e}")
        return []


def fetch_chutes(api_key):
    """Chutes.ai — free OpenAI-compatible inference, rate-limited."""
    try:
        data = _json_get(
            "https://llm.chutes.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [
            m["id"]
            for m in data.get("data", [])
            if not any(x in m.get("id", "").lower() for x in ("embed", "tts", "stt", "image", "vision"))
        ]
        log.info(f"[Chutes] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[Chutes] {e}")
        return []


def fetch_cloudflare(api_key):
    """Cloudflare Workers AI — 10k neurons/day free."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    if not account_id:
        log.warning("[Cloudflare] CLOUDFLARE_ACCOUNT_ID not set, skipping")
        return []
    try:
        data = _json_get(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?per_page=100",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        ids = [
            m["name"]
            for m in data.get("result", [])
            if "text" in str(m.get("task", {}).get("name", "")).lower()
            and "gen" in str(m.get("task", {}).get("name", "")).lower()
        ]
        log.info(f"[Cloudflare] {len(ids)} models")
        return ids
    except Exception as e:
        log.error(f"[Cloudflare] {e}")
        return []


# ── Slug helper ───────────────────────────────────────────────────────────────

def slug(model_id):
    return model_id.split("/")[-1].replace(":free", "").lower()


# ── Provider table ─────────────────────────────────────────────────────────────
# fetch_fn receives (api_key) or (api_key, community_data) — see sync() below.

PROVIDERS = [
    {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "fetch": fetch_openrouter,
        "litellm_fmt": lambda mid: f"openrouter/{mid}",
        "name_fmt": lambda mid: f"or/{slug(mid)}",
        "rpm": 20,
        "api_base": None,
    },
    {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "fetch": fetch_groq,
        "litellm_fmt": lambda mid: f"groq/{mid}",
        "name_fmt": lambda mid: f"groq/{mid}",
        "rpm": 30,
        "api_base": None,
    },
    {
        "name": "Cerebras",
        "env_key": "CEREBRAS_API_KEY",
        "fetch": fetch_cerebras,
        "litellm_fmt": lambda mid: f"cerebras/{mid}",
        "name_fmt": lambda mid: f"cerebras/{mid}",
        "rpm": 30,
        "api_base": None,
    },
    {
        "name": "SambaNova",
        "env_key": "SAMBANOVA_API_KEY",
        "fetch": fetch_sambanova,
        "litellm_fmt": lambda mid: f"sambanova/{mid}",
        "name_fmt": lambda mid: f"sn/{slug(mid)}",
        "rpm": 30,
        "api_base": None,
    },
    {
        "name": "Together",
        "env_key": "TOGETHER_API_KEY",
        "fetch": fetch_together,
        "litellm_fmt": lambda mid: f"together_ai/{mid}",
        "name_fmt": lambda mid: f"t/{slug(mid)}",
        "rpm": 15,
        "api_base": None,
    },
    {
        "name": "Cohere",
        "env_key": "COHERE_API_KEY",
        "fetch": None,  # handled specially with community data
        "litellm_fmt": lambda mid: f"cohere/{mid}",
        "name_fmt": lambda mid: f"co/{mid}",
        "rpm": 20,
        "api_base": None,
    },
    {
        "name": "Gemini",
        "env_key": "GEMINI_API_KEY",
        "fetch": fetch_gemini,
        "litellm_fmt": lambda mid: f"gemini/{mid}",
        "name_fmt": lambda mid: f"gemini/{mid}",
        "rpm": 15,
        "api_base": None,
    },
    {
        "name": "NVIDIA NIM",
        "env_key": "NVIDIA_NIM_API_KEY",
        "fetch": fetch_nvidia,
        "litellm_fmt": lambda mid: f"nvidia_nim/{mid}",
        "name_fmt": lambda mid: f"nv/{slug(mid)}",
        "rpm": 20,
        "api_base": None,
    },
    {
        "name": "HuggingFace",
        "env_key": "HF_TOKEN",
        "fetch": fetch_huggingface,
        "litellm_fmt": lambda mid: f"openai/{mid}",
        "name_fmt": lambda mid: f"hf/{slug(mid)}",
        "rpm": 10,
        "api_base": "https://router.huggingface.co/v1",
    },
    {
        "name": "Mistral",
        "env_key": "MISTRAL_API_KEY",
        "fetch": fetch_mistral,
        "litellm_fmt": lambda mid: f"mistral/{mid}",
        "name_fmt": lambda mid: f"mistral/{mid}",
        "rpm": 5,
        "api_base": None,
    },
    {
        "name": "GitHub Models",
        "env_key": "GH_MODELS_TOKEN",
        "fetch": fetch_github,
        "litellm_fmt": lambda mid: f"github/{mid}",
        "name_fmt": lambda mid: f"gh/{slug(mid)}",
        "rpm": 15,
        "api_base": None,
    },
    {
        "name": "Cloudflare",
        "env_key": "CLOUDFLARE_API_KEY",
        "fetch": fetch_cloudflare,
        "litellm_fmt": lambda mid: f"cloudflare/{mid}",
        "name_fmt": lambda mid: f"cf/{slug(mid)}",
        "rpm": 20,
        "api_base": None,  # constructed dynamically in sync() — includes account_id
    },
    {
        "name": "Chutes",
        "env_key": "CHUTES_API_KEY",
        "fetch": fetch_chutes,
        "litellm_fmt": lambda mid: f"openai/{mid}",
        "name_fmt": lambda mid: f"chutes/{slug(mid)}",
        "rpm": 10,
        "api_base": "https://llm.chutes.ai/v1",
    },
]


# ── Main sync ─────────────────────────────────────────────────────────────────

def sync():
    log.info("=== Model sync started ===")

    # Fetch community cross-reference first (best-effort)
    community = fetch_community_free_models()

    existing = get_existing_litellm_models()
    log.info(f"Currently {len(existing)} litellm model entries registered")

    added = skipped = errors = 0

    for provider in PROVIDERS:
        api_key = os.environ.get(provider["env_key"], "")
        if not api_key:
            continue

        # Cohere needs community data passed in
        if provider["name"] == "Cohere":
            models = fetch_cohere(api_key, community.get("cohere"))
        else:
            models = provider["fetch"](api_key)

        # Cloudflare api_base includes account_id — resolve at sync time
        cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        api_base = (
            f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai"
            if provider["name"] == "Cloudflare" and cf_account_id
            else provider.get("api_base")
        )

        for mid in models:
            litellm_model = provider["litellm_fmt"](mid)
            if litellm_model in existing:
                skipped += 1
                continue

            model_name = provider["name_fmt"](mid)
            ok = add_model(
                model_name=model_name,
                litellm_model=litellm_model,
                api_key_env=provider["env_key"],
                rpm=provider["rpm"],
                api_base=api_base,
            )
            if ok:
                log.info(f"  + {model_name}  ({litellm_model})")
                existing.add(litellm_model)
                added += 1
            else:
                errors += 1

    log.info(f"=== Done: +{added} added, {skipped} already existed, {errors} errors ===")


def wait_for_litellm():
    log.info(f"Waiting up to {STARTUP_DELAY_S}s for LiteLLM to be ready...")
    deadline = time.time() + STARTUP_DELAY_S
    while time.time() < deadline:
        try:
            _get_litellm("/health/liveliness")
            log.info("LiteLLM is up.")
            return
        except Exception:
            time.sleep(5)
    log.warning("LiteLLM did not become ready in time — proceeding anyway.")


if __name__ == "__main__":
    wait_for_litellm()
    while True:
        try:
            sync()
        except Exception as e:
            log.error(f"Sync failed unexpectedly: {e}")
        log.info(f"Next sync in {SYNC_INTERVAL_H}h")
        time.sleep(SYNC_INTERVAL_H * 3600)
