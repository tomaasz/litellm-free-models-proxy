#!/usr/bin/env python3
"""
Generates docs/index.html and docs/models.json from provider APIs.
Runs standalone (no LiteLLM needed) — used by GitHub Actions to build the site.
"""

import os
import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from html import escape
from pathlib import Path

OUT_DIR = Path(__file__).parent / "docs"
OUT_DIR.mkdir(exist_ok=True)

CHEAHJS_URL = (
    "https://raw.githubusercontent.com/cheahjs/free-llm-api-resources"
    "/refs/heads/main/README.md"
)


# ── HTTP ──────────────────────────────────────────────────────────────────────

_HEADERS = {"User-Agent": "litellm-free-models-proxy/1.0"}


def _get(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers={**_HEADERS, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            ct = r.headers.get_content_type() or ""
            return json.loads(body) if "json" in ct or body.lstrip().startswith("{") or body.lstrip().startswith("[") else body
    except Exception as e:
        raise RuntimeError(f"GET {url} → {e}")


# ── Provider fetchers (same logic as sync_models.py) ─────────────────────────

def fetch_openrouter(key):
    data = _get("https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [
        {"id": m["id"],
         "name": m.get("name", m["id"]),
         "context": m.get("context_length"),
         "limits": "20 req/min · 50 req/day"}
        for m in data.get("data", [])
        if str(m.get("pricing", {}).get("prompt", "1")) == "0"
        and str(m.get("pricing", {}).get("completion", "1")) == "0"
    ]


def fetch_groq(key):
    data = _get("https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [
        {"id": m["id"], "name": m["id"], "context": m.get("context_window"), "limits": "varies per model"}
        for m in data.get("data", [])
        if not any(x in m.get("id","").lower() for x in ("whisper","tts","embed","guard"))
    ]


def fetch_cerebras(key):
    data = _get("https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["id"], "name": m["id"],
             "context": m.get("context_length"), "limits": "1M tokens/day"}
            for m in data.get("data", [])]


def fetch_sambanova(key):
    data = _get("https://api.sambanova.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["id"], "name": m["id"],
             "context": m.get("context_length"), "limits": "free tier"}
            for m in data.get("data", [])]


def fetch_together(key):
    data = _get("https://api.together.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    items = data if isinstance(data, list) else data.get("data", [])
    out = []
    for m in items:
        mid = m.get("id", "")
        p = m.get("pricing", {})
        if (p.get("input", 1) == 0 and p.get("output", 1) == 0) or \
           "-Free" in mid or "-free" in mid:
            out.append({"id": mid, "name": m.get("display_name", mid),
                        "context": m.get("context_length"), "limits": "free"})
    return out


def fetch_cohere(key):
    data = _get("https://api.cohere.com/v2/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["name"], "name": m["name"],
             "context": m.get("context_length"), "limits": "20 req/min · 1000 req/month"}
            for m in data.get("models", [])
            if "chat" in m.get("endpoints", [])]


def fetch_gemini(key):
    data = _get("https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": key})
    out = []
    for m in data.get("models", []):
        name = m.get("name", "").replace("models/", "")
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        nl = name.lower()
        if any(x in nl for x in ("-pro", "-ultra", "embedding", "-tts", "robotics")):
            continue
        if any(x in nl for x in ("flash", "gemma")):
            out.append({"id": name,
                        "name": m.get("displayName", name),
                        "context": m.get("inputTokenLimit"),
                        "limits": "varies — see AI Studio"})
    return out


def fetch_nvidia(key):
    data = _get("https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["id"], "name": m["id"],
             "context": None, "limits": "40 req/min"}
            for m in data.get("data", [])
            if not any(x in m.get("id","").lower() for x in ("embed","rerank","tts"))]


def fetch_huggingface(key):
    data = _get("https://router.huggingface.co/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["id"], "name": m["id"],
             "context": None, "limits": "free credits/month"}
            for m in data.get("data", [])
            if not any(x in m.get("id","").lower() for x in ("embed","vision","tts","stt"))]


def fetch_github(key):
    data = _get("https://models.inference.ai.azure.com/models",
                headers={"Authorization": f"Bearer {key}"})
    items = data if isinstance(data, list) else data.get("data", [])
    return [
        {"id": m.get("id") or m.get("name", ""),
         "name": m.get("friendly_name") or m.get("display_name") or m.get("name", ""),
         "context": None,
         "limits": "rate-limited (free / higher with Copilot)"}
        for m in items
        if not any(x in (m.get("id") or m.get("name", "")).lower()
                   for x in ("embed", "tts", "whisper", "dall-e", "image"))
        and (m.get("id") or m.get("name", ""))
    ]


def fetch_cloudflare(key):
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID not set")
    data = _get(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?per_page=100",
        headers={"Authorization": f"Bearer {key}"},
    )
    return [
        {"id": m["name"], "name": m["name"], "context": None, "limits": "10k neurons/day"}
        for m in data.get("result", [])
        if "text" in str(m.get("task", {}).get("name", "")).lower()
        and "gen" in str(m.get("task", {}).get("name", "")).lower()
    ]


def fetch_mistral(key):
    data = _get("https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["id"], "name": m.get("name", m["id"]),
             "context": None, "limits": "1 req/s · 1B tok/month"}
            for m in data.get("data", [])
            if m.get("object") == "model"
            and not any(x in m.get("id","").lower() for x in ("embed","moderation"))]


def fetch_pollinations(key):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = _get("https://gen.pollinations.ai/v1/models", headers=headers)
    out = []
    for m in data.get("data", []):
        out_mods = m.get("output_modalities") or []
        if "text" not in out_mods:
            continue
        if "/v1/chat/completions" not in (m.get("supported_endpoints") or []):
            continue
        out.append({
            "id": m["id"],
            "name": m["id"],
            "context": m.get("context_length"),
            "limits": "free tier — token from enter.pollinations.ai",
        })
    return out


def fetch_kluster(key):
    data = _get("https://api.kluster.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["id"], "name": m.get("name", m["id"]),
             "context": m.get("context_length"),
             "limits": "free tier"}
            for m in data.get("data", [])
            if not any(x in m.get("id","").lower()
                       for x in ("embed","bge","rerank","tts","whisper"))]


def fetch_llm7(key):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = _get("https://api.llm7.io/v1/models", headers=headers)
    items = data if isinstance(data, list) else data.get("data", [])
    out = []
    for m in items:
        mid = m.get("id", "")
        if not mid:
            continue
        if any(x in mid.lower() for x in ("embed","tts","audio","whisper","image")):
            continue
        ctx_field = m.get("context_window") or {}
        ctx = ctx_field.get("tokens") if isinstance(ctx_field, dict) else None
        out.append({
            "id": mid, "name": mid, "context": ctx,
            "limits": "anonymous · 30 req/min with token",
        })
    return out


def fetch_zai(key):
    # Z.ai/BigModel /v4/models often returns nothing useful with trial keys.
    # Hardcode the documented free Flash tier; augment with anything Flash-ish
    # the API does expose.
    free_flash = {
        "glm-4-flash": "GLM-4-Flash",
        "glm-4-flash-250414": "GLM-4-Flash (250414)",
        "glm-4v-flash": "GLM-4V-Flash (vision)",
        "glm-z1-flash": "GLM-Z1-Flash (reasoning)",
        "glm-4.5-flash": "GLM-4.5-Flash",
        "cogvideox-flash": "CogVideoX-Flash (video)",
    }
    out = [{"id": mid, "name": name, "context": None, "limits": "free Flash tier"}
           for mid, name in free_flash.items()]

    try:
        data = _get("https://open.bigmodel.cn/api/paas/v4/models",
                    headers={"Authorization": f"Bearer {key}"})
        items = data.get("data", []) if isinstance(data, dict) else data
        seen = set(free_flash)
        for m in items:
            mid = m.get("id") or m.get("modelCode") or ""
            if not mid or mid in seen or "flash" not in mid.lower():
                continue
            if any(x in mid.lower() for x in
                   ("embed", "rerank", "tts", "stt", "audio", "image")):
                continue
            out.append({
                "id": mid,
                "name": m.get("name") or m.get("displayName") or mid,
                "context": m.get("context_length") or m.get("maxInputTokens"),
                "limits": "free Flash tier",
            })
    except Exception:
        pass
    return out


# ── LiteLLM model metadata enrichment ────────────────────────────────────────
# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

LITELLM_DB_URL = ("https://raw.githubusercontent.com/BerriAI/litellm/main/"
                  "model_prices_and_context_window.json")

# Map our provider key → litellm "litellm_provider" field
LITELLM_PROVIDER_MAP = {
    "openrouter": "openrouter",
    "groq": "groq",
    "cerebras": "cerebras",
    "sambanova": "sambanova",
    "together": "together_ai",
    "cohere": "cohere",
    "gemini": "gemini",
    "nvidia": "nvidia_nim",
    "huggingface": "huggingface",
    "mistral": "mistral",
    "github": "github",
    "cloudflare": "cloudflare",
}

CAPABILITY_FIELDS = (
    "supports_function_calling",
    "supports_tool_choice",
    "supports_response_schema",
    "supports_vision",
    "supports_system_messages",
    "supports_reasoning",
    "supports_prompt_caching",
)


def fetch_litellm_db():
    try:
        d = _get(LITELLM_DB_URL)
        return d if isinstance(d, dict) else json.loads(d)
    except Exception as e:
        print(f"  [litellm-db] fetch failed: {e}")
        return {}


def enrich_with_litellm(results, db):
    """Add context window + capability flags from litellm's model database."""
    if not db:
        return
    # Index by litellm_provider for fast lookups
    by_prov = {}
    for key, val in db.items():
        if not isinstance(val, dict):
            continue
        p = val.get("litellm_provider")
        if not p:
            continue
        by_prov.setdefault(p, {})[key] = val

    enriched = 0
    for our_key, llm_prov in LITELLM_PROVIDER_MAP.items():
        models = results.get(our_key, [])
        prov_db = by_prov.get(llm_prov, {})
        if not models or not prov_db:
            continue
        for m in models:
            mid = m["id"]
            entry = (prov_db.get(f"{llm_prov}/{mid}")
                     or prov_db.get(mid)
                     or prov_db.get(f"{our_key}/{mid}"))
            if not entry:
                continue
            ctx = (entry.get("max_input_tokens")
                   or entry.get("max_tokens"))
            if ctx and not m.get("context"):
                m["context"] = ctx
            caps = [f.removeprefix("supports_") for f in CAPABILITY_FIELDS
                    if entry.get(f) is True]
            if caps:
                m["capabilities"] = caps
            mode = entry.get("mode")
            if mode and mode != "chat":
                m["mode"] = mode
            enriched += 1
    if enriched:
        print(f"  Enriched {enriched} models from litellm DB")


# ── Community cross-reference ─────────────────────────────────────────────────

def fetch_cheahjs():
    try:
        return _get(CHEAHJS_URL)
    except Exception as e:
        print(f"[community] {e}")
        return ""


# ── Model utilities ───────────────────────────────────────────────────────────

_QUALITY_RE = re.compile(
    r'[-_](instruct|chat|it|free|versatile|preview|latest|exp|experimental'
    r'|hf|awq|gptq|gguf|fp16|bf16|int4|int8)$',
    re.IGNORECASE,
)
_DATE_RE = re.compile(r'-\d{8}$')
_FREE_RE  = re.compile(r':free$')


def canonical_name(model_id):
    """Normalize a model ID to a provider-agnostic key for cross-provider grouping."""
    s = model_id.split('/')[-1].lower()
    s = _FREE_RE.sub('', s)
    for _ in range(5):
        ns = _QUALITY_RE.sub('', s)
        ns = _DATE_RE.sub('', ns)
        if ns == s:
            break
        s = ns
    return re.sub(r'[-_.]', '-', s).strip('-')


_TAG_RULES = [
    (["coder", "-code-", "starcoder", "codestral", "deepseek-coder"], "coding",    "#a78bfa"),
    (["reason", "think",  "qwq",   "deepseek-r", "-r1", "r1-"],       "reasoning", "#f59e0b"),
    (["vision", "-vl",    "llava",  "pixtral",    "qvq",  "visual"],   "vision",    "#06b6d4"),
    (["flash",  "turbo",  "lite",   "mini",       "small","haiku",
      "nano",   "fast"],                                                "fast",      "#10b981"),
]


_CAPABILITY_CHIPS = {
    "function_calling": ("tools", "#a78bfa"),
    "tool_choice":      ("tools", "#a78bfa"),
    "response_schema":  ("json", "#22d3ee"),
    "vision":           ("vision", "#f59e0b"),
    "reasoning":        ("reasoning", "#34d399"),
    "prompt_caching":   ("cache", "#94a3b8"),
}


def get_tags(model_id, context=None, capabilities=None):
    tags = []
    mid = model_id.lower()
    for keywords, label, color in _TAG_RULES:
        if any(kw in mid for kw in keywords):
            tags.append((label, color))
    if context and int(context) >= 128_000:
        tags.append(("128k+", "#38bdf8"))
    seen_chips = set()
    for cap in capabilities or []:
        chip = _CAPABILITY_CHIPS.get(cap)
        if not chip or chip[0] in seen_chips:
            continue
        seen_chips.add(chip[0])
        tags.append(chip)
    return tags


PROVIDERS = [
    {"key": "openrouter",  "label": "OpenRouter",   "env": "OPENROUTER_API_KEY", "fetch": fetch_openrouter, "color": "#6366f1", "url": "https://openrouter.ai",                       "key_url": "https://openrouter.ai/keys"},
    {"key": "groq",        "label": "Groq",         "env": "GROQ_API_KEY",       "fetch": fetch_groq,       "color": "#f59e0b", "url": "https://console.groq.com",                    "key_url": "https://console.groq.com/keys"},
    {"key": "cerebras",    "label": "Cerebras",     "env": "CEREBRAS_API_KEY",   "fetch": fetch_cerebras,   "color": "#10b981", "url": "https://cloud.cerebras.ai",                   "key_url": "https://cloud.cerebras.ai/platform"},
    {"key": "gemini",      "label": "Gemini",       "env": "GEMINI_API_KEY",     "fetch": fetch_gemini,     "color": "#3b82f6", "url": "https://aistudio.google.com",                 "key_url": "https://aistudio.google.com/apikey"},
    {"key": "sambanova",   "label": "SambaNova",    "env": "SAMBANOVA_API_KEY",  "fetch": fetch_sambanova,  "color": "#8b5cf6", "url": "https://cloud.sambanova.ai",                  "key_url": "https://cloud.sambanova.ai/"},
    {"key": "cohere",      "label": "Cohere",       "env": "COHERE_API_KEY",     "fetch": fetch_cohere,     "color": "#ec4899", "url": "https://cohere.com",                          "key_url": "https://dashboard.cohere.com/api-keys"},
    {"key": "together",    "label": "Together AI",  "env": "TOGETHER_API_KEY",   "fetch": fetch_together,   "color": "#14b8a6", "url": "https://api.together.ai",                     "key_url": "https://api.together.ai/settings/api-keys"},
    {"key": "nvidia",      "label": "NVIDIA NIM",   "env": "NVIDIA_NIM_API_KEY", "fetch": fetch_nvidia,     "color": "#22c55e", "url": "https://build.nvidia.com",                    "key_url": "https://build.nvidia.com/"},
    {"key": "huggingface", "label": "HuggingFace",  "env": "HF_TOKEN",           "fetch": fetch_huggingface,"color": "#f97316", "url": "https://huggingface.co",                      "key_url": "https://huggingface.co/settings/tokens"},
    {"key": "mistral",     "label": "Mistral",      "env": "MISTRAL_API_KEY",    "fetch": fetch_mistral,    "color": "#0ea5e9", "url": "https://console.mistral.ai",                  "key_url": "https://console.mistral.ai/api-keys/"},
    {"key": "github",      "label": "GitHub Models","env": "GH_MODELS_TOKEN",    "fetch": fetch_github,     "color": "#e2e8f0", "url": "https://github.com/marketplace/models",       "key_url": "https://github.com/settings/tokens"},
    {"key": "cloudflare",  "label": "Cloudflare AI","env": "CLOUDFLARE_API_KEY", "fetch": fetch_cloudflare, "color": "#f6821f", "url": "https://developers.cloudflare.com/workers-ai/","key_url": "https://dash.cloudflare.com/profile/api-tokens"},
    {"key": "pollinations","label": "Pollinations", "env": "POLLINATIONS_API_KEY", "fetch": fetch_pollinations, "color": "#ec4899", "url": "https://pollinations.ai",                       "key_url": "https://enter.pollinations.ai"},
    {"key": "kluster",     "label": "Kluster AI",   "env": "KLUSTER_API_KEY",     "fetch": fetch_kluster,     "color": "#a855f7", "url": "https://kluster.ai",                            "key_url": "https://platform.kluster.ai/apikeys"},
    {"key": "llm7",        "label": "LLM7",         "env": "LLM7_API_KEY",        "fetch": fetch_llm7,        "color": "#facc15", "url": "https://llm7.io",                               "key_url": "https://token.llm7.io", "anonymous_ok": True},
    {"key": "zai",         "label": "Z.ai (GLM)",   "env": "ZAI_API_KEY",         "fetch": fetch_zai,         "color": "#0ea5e9", "url": "https://open.bigmodel.cn",                     "key_url": "https://open.bigmodel.cn/usercenter/apikeys"},
]


# ── HTML generation ───────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Free LLM API Access — providers with free tokens</title>
<style>
  :root {{
    --bg: #0a0e1a;
    --surface: #131a2c;
    --surface-2: #1a2236;
    --border: #1f2940;
    --border-strong: #2c3a5a;
    --text: #e7eaf2;
    --muted: #8b95ad;
    --muted-2: #647084;
    --accent: #38bdf8;
    --accent-2: #7c3aed;
    --accent-glow: rgba(56,189,248,.16);
    --radius: 10px;
    --radius-sm: 6px;
    --sidebar-w: 264px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; }}
  body {{
    background:
      radial-gradient(1200px 460px at 0% -10%, rgba(56,189,248,.08), transparent 60%),
      radial-gradient(900px 360px at 100% -10%, rgba(124,58,237,.06), transparent 60%),
      var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Inter, sans-serif;
    -webkit-font-smoothing: antialiased;
    font-size: 14px;
    line-height: 1.45;
  }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  ::selection {{ background: var(--accent-glow); }}

  .app {{ max-width: 1280px; margin: 0 auto; padding: 1.5rem 1.25rem 3rem; }}

  /* ── Top bar ─────────────────────────── */
  .topbar {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 1.25rem;
    align-items: end;
    padding-bottom: 1.1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
  }}
  .brand h1 {{
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #fff;
    display: flex;
    align-items: center;
    gap: .6rem;
  }}
  .brand h1 .logo {{
    width: 28px; height: 28px;
    display: inline-grid; place-items: center;
    border-radius: 8px;
    background: linear-gradient(135deg, #38bdf8 0%, #7c3aed 100%);
    color: #0a0e1a;
    font-weight: 800;
    font-size: .95rem;
    box-shadow: 0 4px 14px -4px rgba(56,189,248,.55);
  }}
  .tagline {{
    margin-top: .4rem;
    color: var(--muted);
    font-size: .9rem;
    max-width: 64ch;
  }}
  .tagline strong {{ color: var(--text); font-weight: 600; }}
  .top-meta {{
    display: flex; align-items: center; gap: .4rem;
    flex-wrap: wrap; justify-content: flex-end;
  }}
  .top-pill {{
    display: inline-flex; align-items: baseline; gap: .35rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 999px; padding: .28rem .7rem;
    font-size: .78rem; color: var(--muted);
    font-feature-settings: "tnum";
    transition: border-color .15s, color .15s;
  }}
  .top-pill strong {{ color: #fff; font-weight: 700; font-size: .82rem; }}
  .top-pill a {{ color: var(--muted); }}
  .top-pill a:hover {{ color: var(--accent); text-decoration: none; }}
  .top-pill:hover {{ border-color: var(--border-strong); }}

  /* ── View tabs (segmented control) ──── */
  .view-tabs {{
    display: inline-flex; flex-wrap: nowrap;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 3px; gap: 2px;
    margin-bottom: 1.25rem;
  }}
  .vtab {{
    background: transparent; border: none;
    border-radius: 7px; color: var(--muted);
    padding: .42rem .95rem; font-size: .85rem;
    font-weight: 500; cursor: pointer;
    transition: color .15s, background .15s, box-shadow .15s;
    font-family: inherit; white-space: nowrap;
  }}
  .vtab:hover {{ color: var(--text); }}
  .vtab.active {{
    background: var(--surface-2); color: var(--text);
    box-shadow: 0 1px 0 rgba(255,255,255,.04) inset, 0 1px 2px rgba(0,0,0,.3);
  }}

  /* ── Layout (sidebar + main) ────────── */
  .layout {{
    display: grid;
    grid-template-columns: var(--sidebar-w) 1fr;
    gap: 1.25rem;
    align-items: flex-start;
  }}
  .layout.no-sidebar {{ grid-template-columns: 1fr; }}

  /* ── Sidebar ─────────────────────────── */
  .sidebar {{
    position: sticky; top: 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1rem;
    display: flex; flex-direction: column; gap: 1.1rem;
    max-height: calc(100vh - 2rem); overflow-y: auto;
  }}
  .sidebar::-webkit-scrollbar {{ width: 6px; }}
  .sidebar::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 3px; }}
  .sb-section {{ display: flex; flex-direction: column; gap: .55rem; }}
  .sb-title {{
    font-size: .68rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted-2);
  }}
  .sb-search input {{
    width: 100%; padding: .55rem .75rem .55rem 2.1rem;
    font-size: .85rem; background: var(--bg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--text); outline: none;
    transition: border-color .15s, box-shadow .15s;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' fill='none' stroke='%238b95ad' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: .65rem center;
    font-family: inherit;
  }}
  .sb-search input:focus {{
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }}
  .sb-search input::placeholder {{ color: var(--muted-2); }}
  .sb-pills {{ display: flex; flex-wrap: wrap; gap: .35rem; }}
  .sb-pill {{
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 999px; color: var(--muted);
    padding: .3rem .7rem; font-size: .78rem;
    cursor: pointer; font-family: inherit;
    transition: color .15s, border-color .15s, background .15s;
  }}
  .sb-pill:hover {{ color: var(--text); border-color: var(--border-strong); }}
  .sb-pill.active {{
    background: var(--accent); border-color: var(--accent);
    color: #0a0e1a; font-weight: 600;
  }}
  .sb-pill.tag-pill.active {{
    background: var(--tc, var(--accent));
    border-color: var(--tc, var(--accent));
    color: #0a0e1a;
  }}
  .sb-help {{
    font-size: .74rem; color: var(--muted-2); line-height: 1.55;
    border-top: 1px solid var(--border); padding-top: .8rem;
  }}

  /* ── Main column ─────────────────────── */
  main {{ display: flex; flex-direction: column; gap: 1rem; min-width: 0; }}
  .no-results {{
    display: none; text-align: center; padding: 2rem;
    color: var(--muted); font-size: .9rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius);
  }}

  /* ── Provider cards ──────────────────── */
  .provider-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
    transition: border-color .15s;
  }}
  .provider-card:hover {{ border-color: var(--border-strong); }}
  .provider-header {{
    display: flex; align-items: center; gap: .75rem;
    padding: .8rem 1.1rem; border-bottom: 1px solid var(--border);
  }}
  .provider-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(255,255,255,.04); }}
  .provider-name {{ font-weight: 600; font-size: .95rem; }}
  .provider-name a {{ color: inherit; }}
  .provider-name a:hover {{ color: var(--accent); text-decoration: none; }}
  .provider-count {{
    margin-left: auto; background: var(--bg);
    border: 1px solid var(--border); border-radius: 999px;
    padding: .14rem .6rem; font-size: .72rem; color: var(--muted);
    font-feature-settings: "tnum";
  }}
  .delta-add {{ color: #4ade80; font-size: .7rem; margin-left: .3rem; font-weight: 600; }}
  .delta-rem {{ color: #f87171; font-size: .7rem; margin-left: .15rem; font-weight: 600; }}
  .status-indicator {{ font-size: .5rem; flex-shrink: 0; }}
  .status-ok  {{ color: #22c55e; }}
  .status-err {{ color: #f87171; }}
  .tag-chip {{
    display: inline-block; border-radius: 999px;
    padding: .05rem .5rem; font-size: .68rem;
    font-weight: 500; margin-right: .2rem;
    margin-bottom: .1rem; white-space: nowrap;
  }}
  .api-key-link {{
    font-size: .72rem; color: var(--muted);
    border: 1px solid var(--border); border-radius: 5px;
    padding: .15rem .55rem; white-space: nowrap;
    transition: color .15s, border-color .15s;
  }}
  .api-key-link:hover {{ color: var(--accent); border-color: var(--accent); text-decoration: none; }}
  .collapse-btn {{
    background: none; border: none; cursor: pointer;
    color: var(--muted); padding: .1rem .2rem; line-height: 1;
    display: flex; align-items: center; transition: color .15s;
  }}
  .collapse-btn:hover {{ color: var(--text); }}
  .collapse-btn .chevron {{ transition: transform .2s; }}
  .provider-card.collapsed .chevron {{ transform: rotate(-90deg); }}
  .provider-card.collapsed .provider-header {{ border-bottom: none; }}
  .provider-card.collapsed .provider-body {{ display: none; }}
  .provider-error {{ padding: 1rem 1.1rem; color: #f87171; font-size: .85rem; }}

  /* ── Tables ──────────────────────────── */
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; table-layout: fixed; }}
  col.col-id      {{ width: 32%; }}
  col.col-name    {{ width: 20%; }}
  col.col-ctx     {{ width:  8%; }}
  col.col-tags    {{ width: 18%; }}
  col.col-limits  {{ width: 22%; }}
  th {{
    text-align: left; padding: .55rem 1.1rem;
    color: var(--muted-2); font-weight: 500;
    border-bottom: 1px solid var(--border);
    font-size: .68rem; text-transform: uppercase;
    letter-spacing: .07em;
    background: rgba(255,255,255,.012);
  }}
  td {{
    padding: .5rem 1.1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(56,189,248,.04); }}
  .model-id {{
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", monospace;
    font-size: .82rem; color: var(--accent);
    cursor: pointer; display: block;
    overflow: hidden; text-overflow: ellipsis;
  }}
  .model-id:hover {{ text-decoration: underline; }}
  .model-name {{ color: var(--text); overflow: hidden; text-overflow: ellipsis; }}
  .ctx {{ color: var(--muted); font-size: .78rem; font-feature-settings: "tnum"; }}
  .limits {{ color: var(--muted); font-size: .76rem; }}
  .copy-tip {{ font-size: .65rem; color: var(--muted-2); margin-left: .35rem; font-weight: 400; text-transform: none; letter-spacing: 0; }}

  /* ── Cross-provider groups ───────────── */
  .cross-group {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
  }}
  .cross-group-header {{
    padding: .75rem 1.1rem; border-bottom: 1px solid var(--border);
    font-weight: 600; font-size: .9rem;
    display: flex; align-items: center; gap: .55rem;
  }}
  .cross-group-count {{ font-size: .72rem; color: var(--muted); font-weight: 400; margin-left: auto; }}
  .provider-chip {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; }}

  /* ── Availability view ───────────────── */
  .uptime-badge {{
    display: inline-block; min-width: 3rem;
    padding: .14rem .55rem; border-radius: 5px;
    font-size: .72rem; font-weight: 600;
    text-align: center;
    font-family: ui-monospace, monospace;
    font-feature-settings: "tnum";
  }}
  .uptime-good {{ background: rgba(34,197,94,.14); color: #4ade80; border: 1px solid rgba(34,197,94,.28); }}
  .uptime-warn {{ background: rgba(245,158,11,.14); color: #fbbf24; border: 1px solid rgba(245,158,11,.28); }}
  .uptime-bad  {{ background: rgba(239,68,68,.14);  color: #f87171; border: 1px solid rgba(239,68,68,.28); }}
  .uptime-none {{ background: rgba(148,163,184,.08); color: var(--muted-2); border: 1px solid var(--border); }}
  .av-heatmap {{ display: inline-flex; align-items: flex-end; gap: 1px; height: 18px; vertical-align: middle; justify-self: start; }}
  .av-bar {{ width: 4px; height: 100%; background: var(--border); border-radius: 1px; cursor: help; }}
  .av-row {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 60px 130px 160px;
    gap: .85rem; align-items: center;
    padding: .5rem 1.1rem;
    border-bottom: 1px solid var(--border); font-size: .82rem;
  }}
  .av-row .uptime-badge {{ justify-self: start; }}
  .av-row:last-child {{ border-bottom: none; }}
  .av-row:hover {{ background: rgba(56,189,248,.03); }}
  .av-row .model-id {{ overflow: hidden; text-overflow: ellipsis; }}
  .av-meta {{ color: var(--muted); font-size: .72rem; font-family: ui-monospace, monospace; white-space: nowrap; }}
  .av-empty {{
    padding: 2rem; color: var(--muted); font-size: .9rem; text-align: center;
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  }}

  /* ── Changes view ────────────────────── */
  .change-entry {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: .85rem 1.1rem;
  }}
  .change-entry-header {{
    display: flex; align-items: center; gap: .55rem;
    flex-wrap: wrap; margin-bottom: .55rem; font-size: .9rem;
  }}
  .change-entry-header time {{ color: var(--muted); font-size: .78rem; font-family: ui-monospace, monospace; }}
  .change-entry-header .provider-name {{ font-weight: 600; }}
  .change-entry-header .change-summary {{ margin-left: auto; font-size: .76rem; color: var(--muted); }}
  .change-list {{ display: flex; flex-direction: column; gap: .25rem; }}
  .change-row {{ display: flex; align-items: center; gap: .5rem; font-family: ui-monospace, monospace; font-size: .8rem; }}
  .change-row.added .change-marker {{ color: #4ade80; }}
  .change-row.removed .change-marker {{ color: #f87171; }}
  .change-marker {{ font-weight: 700; flex-shrink: 0; width: 1rem; text-align: center; }}
  .change-id {{ color: var(--text); cursor: pointer; word-break: break-all; }}
  .change-id:hover {{ text-decoration: underline; color: var(--accent); }}
  .changes-empty {{
    text-align: center; padding: 2rem; color: var(--muted); font-size: .9rem;
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  }}

  /* ── Suggest card / footer ───────────── */
  .suggest-card {{
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    display: flex; align-items: center;
    gap: 1.25rem; flex-wrap: wrap;
    margin-top: .5rem;
  }}
  .suggest-card h2 {{ font-size: .95rem; font-weight: 600; color: #fff; margin-bottom: .25rem; }}
  .suggest-card p {{ font-size: .82rem; color: var(--muted); line-height: 1.5; max-width: 56ch; }}
  .suggest-btn {{
    display: inline-flex; align-items: center; gap: .35rem;
    background: #238636; color: #fff !important; border: none;
    border-radius: 6px; padding: .55rem 1rem;
    font-size: .85rem; font-weight: 600;
    text-decoration: none; white-space: nowrap; cursor: pointer;
    transition: background .15s, transform .1s;
    margin-left: auto;
  }}
  .suggest-btn:hover {{ background: #2ea043; text-decoration: none; }}
  .suggest-btn:active {{ transform: translateY(1px); }}
  footer {{
    margin-top: 2rem; padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    font-size: .76rem; color: var(--muted-2); text-align: center;
  }}
  footer a {{ color: var(--muted); }}
  footer a:hover {{ color: var(--accent); }}

  /* ── Mobile filters toggle ───────────── */
  .filters-toggle {{
    display: none;
    position: fixed; bottom: 1.25rem; right: 1.25rem;
    z-index: 30;
    background: var(--accent); color: #0a0e1a;
    border: none; border-radius: 999px;
    padding: .75rem 1.1rem;
    font-weight: 600; font-size: .85rem;
    font-family: inherit; cursor: pointer;
    align-items: center;
    box-shadow: 0 8px 24px -8px rgba(56,189,248,.55);
  }}
  body.sidebar-open .scrim {{
    position: fixed; inset: 0; z-index: 35;
    background: rgba(0,0,0,.55); backdrop-filter: blur(2px);
  }}
  .scrim {{ display: none; }}

  /* ── Responsive ──────────────────────── */
  @media (max-width: 900px) {{
    .layout, .layout.no-sidebar {{ grid-template-columns: 1fr; }}
    .sidebar {{
      position: fixed; inset: 0 auto 0 0;
      width: 280px; max-width: 85vw;
      max-height: 100vh; height: 100vh;
      border-radius: 0;
      transform: translateX(-100%);
      transition: transform .25s ease;
      z-index: 40;
    }}
    body.sidebar-open .sidebar {{ transform: translateX(0); }}
    body.sidebar-open .scrim {{ display: block; }}
    .filters-toggle {{ display: inline-flex; }}
    .topbar {{ grid-template-columns: 1fr; }}
    .top-meta {{ justify-content: flex-start; }}
  }}
  @media (max-width: 600px) {{
    body {{ font-size: 13.5px; }}
    .app {{ padding: 1rem .75rem 5rem; }}
    col.col-ctx, th:nth-child(3), td:nth-child(3),
    col.col-tags, th:nth-child(4), td:nth-child(4),
    col.col-limits, th:nth-child(5), td:nth-child(5) {{ display: none; }}
    col.col-id   {{ width: 50%; }}
    col.col-name {{ width: 50%; }}
    .suggest-card {{ flex-direction: column; align-items: flex-start; }}
    .suggest-btn {{ margin-left: 0; }}
    .view-tabs {{ width: 100%; overflow-x: auto; }}
    .av-row {{ grid-template-columns: 1fr auto; gap: .5rem 0; }}
    .av-row .av-heatmap, .av-row .av-meta {{ grid-column: 1 / -1; }}
  }}
</style>
</head>
<body>
<div class="app">
<header class="topbar">
  <div class="brand">
    <h1><span class="logo">⚡</span>Free LLM API Access</h1>
    <p class="tagline">
      Providers that give you <strong>free API tokens</strong> to call LLM models — no credit card required.
      Auto-updated every 8 hours from provider APIs.
    </p>
  </div>
  <div class="top-meta">
    <span class="top-pill"><strong>{total_models}</strong> models</span>
    <span class="top-pill"><strong>{total_providers}</strong> providers</span>
    <span class="top-pill" title="Last updated">{updated}</span>
    <span class="top-pill"><a href="https://tomaasz.github.io/litellm-free-models-proxy/models.json" target="_blank">JSON</a></span>
    <span class="top-pill"><a href="https://tomaasz.github.io/litellm-free-models-proxy/availability/stable_models.json" target="_blank" title="Models with 7d uptime ≥ 95%">JSON · stable</a></span>
    <span class="top-pill"><a href="https://tomaasz.github.io/litellm-free-models-proxy/availability/problems_models.json" target="_blank" title="Models with 7d uptime &lt; 95%">JSON · problems</a></span>
    <span class="top-pill"><a href="https://github.com/tomaasz/litellm-free-models-proxy" target="_blank">GitHub</a></span>
  </div>
</header>

<nav class="view-tabs" role="tablist">
  <button class="vtab active" data-target="view-provider" role="tab" aria-selected="true" aria-controls="view-provider">By Provider</button>
  <button class="vtab" data-target="view-model" role="tab" aria-selected="false" aria-controls="view-model">By Model</button>
  <button class="vtab" data-target="view-availability" role="tab" aria-selected="false" aria-controls="view-availability">Availability</button>
  <button class="vtab" data-target="view-changes" role="tab" aria-selected="false" aria-controls="view-changes">Changes</button>
</nav>

<div class="layout" id="layout">
  <aside class="sidebar" id="sidebar">
    <div class="sb-section sb-search" data-for="view-provider view-model view-availability">
      <input type="search" id="model-search" aria-label="Search models" placeholder="Search models…" autocomplete="off" spellcheck="false">
    </div>
    <div class="sb-section" data-for="view-provider view-model view-availability">
      <h3 class="sb-title">Context window</h3>
      <div class="sb-pills">
        <button class="sb-pill ctx-pill active" data-min="0">Any</button>
        <button class="sb-pill ctx-pill" data-min="32768">&#8805; 32k</button>
        <button class="sb-pill ctx-pill" data-min="131072">&#8805; 128k</button>
        <button class="sb-pill ctx-pill" data-min="1000000">&#8805; 1M</button>
      </div>
    </div>
    <div class="sb-section" data-for="view-provider view-model">
      <h3 class="sb-title">Tags</h3>
      <div class="sb-pills">
        <button class="sb-pill tag-pill active" data-tag="">All</button>
        <button class="sb-pill tag-pill" data-tag="coding"    style="--tc:#a78bfa">coding</button>
        <button class="sb-pill tag-pill" data-tag="reasoning" style="--tc:#f59e0b">reasoning</button>
        <button class="sb-pill tag-pill" data-tag="vision"    style="--tc:#06b6d4">vision</button>
        <button class="sb-pill tag-pill" data-tag="fast"      style="--tc:#10b981">fast</button>
        <button class="sb-pill tag-pill" data-tag="128k+"     style="--tc:#38bdf8">128k+</button>
      </div>
    </div>
    <div class="sb-section" data-for="view-availability">
      <h3 class="sb-title">Availability</h3>
      <div class="sb-pills">
        <button class="sb-pill av-pill active" data-av="all">All</button>
        <button class="sb-pill av-pill" data-av="problems">Only problems</button>
        <button class="sb-pill av-pill" data-av="stable">Only stable 7d</button>
      </div>
    </div>
    <p class="sb-help" data-for="view-provider view-model view-availability">
      Click any model ID to copy it to clipboard.
    </p>
  </aside>

  <main>
    <div id="view-provider" role="tabpanel">
      {provider_sections}
      <p class="no-results" id="no-results">No models match your search.</p>
    </div>
    <div id="view-model" style="display:none" role="tabpanel">
      {cross_provider_section}
    </div>
    <div id="view-availability" style="display:none" role="tabpanel">
      {availability_section}
    </div>
    <div id="view-changes" style="display:none" role="tabpanel">
      {changes_section}
    </div>
    <div class="suggest-card">
      <div>
        <h2>Know a provider we're missing?</h2>
        <p>If you know of a provider that gives free API tokens / free-tier access to LLM models and isn't listed here, open a GitHub issue — we'll add support for it.</p>
      </div>
      <a class="suggest-btn"
         href="https://github.com/tomaasz/litellm-free-models-proxy/issues/new?template=new-provider.yml"
         target="_blank">+ Suggest a provider</a>
    </div>
  </main>
</div>

<button class="filters-toggle" id="filters-toggle" aria-label="Toggle filters">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:.45rem"><line x1="4" y1="6" x2="20" y2="6"/><line x1="7" y1="12" x2="17" y2="12"/><line x1="10" y1="18" x2="14" y2="18"/></svg>
  Filters
</button>
<div class="scrim" id="scrim"></div>

<footer>
  <p>Auto-generated by <a href="https://github.com/tomaasz/litellm-free-models-proxy">litellm-free-models-proxy</a> · Cross-referenced with <a href="https://github.com/cheahjs/free-llm-api-resources">cheahjs/free-llm-api-resources</a>. Free-tier API tokens — not open-source or self-hostable models. Tiers may change without notice; not affiliated with any provider.</p>
</footer>
</div>
<script>
document.querySelectorAll('.model-id').forEach(el => {{
  el.title = 'Click to copy';
  el.addEventListener('click', () => {{
    navigator.clipboard.writeText(el.dataset.id || el.textContent.trim()).then(() => {{
      const orig = el.textContent;
      el.textContent = '✓ copied';
      setTimeout(() => el.textContent = orig, 1200);
    }});
  }});
}});

(function() {{
  const input = document.getElementById('model-search');
  const noResults = document.getElementById('no-results');
  const cards = Array.from(document.querySelectorAll('.provider-card[data-total], .cross-group[data-total]'));
  let ctxMin = 0;
  let tagFilter = '';

  function applyFilters() {{
    const q = input.value.toLowerCase().trim();
    const isFiltered = q || ctxMin > 0 || tagFilter;
    let totalVisible = 0;

    cards.forEach(card => {{
      const rows = card.querySelectorAll('tbody tr, .av-row');
      const countEl = card.querySelector('.provider-count');
      const total = parseInt(card.dataset.total, 10);
      let visible = 0;

      rows.forEach(row => {{
        const searchOk = !q || row.dataset.search.includes(q);
        const ctx = parseInt(row.dataset.ctx || '0', 10);
        const ctxOk = ctxMin === 0 || ctx >= ctxMin;
        const tagOk = !tagFilter || (row.dataset.tags || '').split(' ').includes(tagFilter);
        const match = searchOk && ctxOk && tagOk;
        row.style.display = match ? '' : 'none';
        if (match) visible++;
      }});

      const countNumEl = card.querySelector('.count-num');
      if (countNumEl) {{
        countNumEl.textContent = isFiltered ? visible + '/' + total : total;
      }}
      card.style.display = (rows.length === 0 || visible > 0) ? '' : 'none';
      if (visible > 0) totalVisible++;
    }});

    noResults.style.display = (isFiltered && totalVisible === 0) ? 'block' : 'none';
  }}

  input.addEventListener('input', applyFilters);

  document.querySelectorAll('.ctx-pill').forEach(pill => {{
    pill.addEventListener('click', () => {{
      document.querySelectorAll('.ctx-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      ctxMin = parseInt(pill.dataset.min, 10);
      applyFilters();
    }});
  }});

  document.querySelectorAll('.tag-pill').forEach(pill => {{
    pill.addEventListener('click', () => {{
      document.querySelectorAll('.tag-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      tagFilter = pill.dataset.tag;
      applyFilters();
    }});
  }});
}})();

document.querySelectorAll('.collapse-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    btn.closest('.provider-card').classList.toggle('collapsed');
  }});
}});

const TAB_TO_SLUG = {{
  'view-provider':     'by-provider',
  'view-model':        'by-model',
  'view-availability': 'availability',
  'view-changes':      'changes',
}};
const SLUG_TO_TAB = Object.fromEntries(Object.entries(TAB_TO_SLUG).map(([k,v]) => [v,k]));

function applyTab(t) {{
  document.querySelectorAll('.vtab').forEach(b => {{
    const isActive = b.dataset.target === t;
    b.classList.toggle('active', isActive);
    b.setAttribute('aria-selected', isActive ? 'true' : 'false');
  }});
  ['view-provider','view-model','view-availability','view-changes'].forEach(id => {{
    document.getElementById(id).style.display = id === t ? '' : 'none';
  }});
  let anyShown = false;
  document.querySelectorAll('#sidebar [data-for]').forEach(el => {{
    const ok = (el.dataset.for || '').split(' ').includes(t);
    el.style.display = ok ? '' : 'none';
    if (ok) anyShown = true;
  }});
  const layout = document.getElementById('layout');
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('filters-toggle');
  layout.classList.toggle('no-sidebar', !anyShown);
  sidebar.style.display = anyShown ? '' : 'none';
  if (toggle) toggle.style.display = anyShown ? '' : 'none';
  if (!anyShown) document.body.classList.remove('sidebar-open');
}}

function tabPath(t) {{
  const slug = TAB_TO_SLUG[t];
  // Strip trailing /<known-slug>(/availability sub-mode)? from current pathname.
  const rootPath = location.pathname.replace(
    /\\/(by-provider|by-model|availability(?:\\/(?:problems|stable))?|changes)\\/?$/, '/');
  return rootPath + (slug === 'by-provider' ? '' : slug + '/');
}}

document.querySelectorAll('.vtab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const t = btn.dataset.target;
    applyTab(t);
    if (history.pushState) history.pushState({{tab: t}}, '', tabPath(t));
  }});
}});

window.addEventListener('popstate', () => {{
  const p = location.pathname;
  const avSub = p.match(/\\/availability\\/(problems|stable)\\/?$/);
  if (avSub) {{
    applyTab('view-availability');
    applyAvFilter(avSub[1]);
    return;
  }}
  const m = p.match(/\\/(by-provider|by-model|availability|changes)\\/?$/);
  applyTab(m ? SLUG_TO_TAB[m[1]] : 'view-provider');
  if (m && m[1] === 'availability') applyAvFilter('all');
}});

(function() {{
  const toggle = document.getElementById('filters-toggle');
  const scrim = document.getElementById('scrim');
  if (toggle) toggle.addEventListener('click', (e) => {{
    e.stopPropagation();
    document.body.classList.toggle('sidebar-open');
  }});
  if (scrim) scrim.addEventListener('click', () => document.body.classList.remove('sidebar-open'));
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') document.body.classList.remove('sidebar-open');
  }});
}})();

applyTab(document.body.dataset.tab || 'view-provider');

function applyAvFilter(mode) {{
  document.querySelectorAll('.av-pill').forEach(p => p.classList.toggle('active', p.dataset.av === mode));
  document.querySelectorAll('#view-availability .av-row').forEach(row => {{
    const u = parseFloat(row.dataset.uptime || '-1');
    let show = true;
    if (mode === 'problems') show = (u >= 0 && u < 0.95);
    else if (mode === 'stable') show = (u >= 0.95);
    row.style.display = show ? '' : 'none';
  }});
}}

document.querySelectorAll('.av-pill').forEach(pill => {{
  pill.addEventListener('click', () => {{
    const mode = pill.dataset.av;
    applyAvFilter(mode);
    if (history.pushState) {{
      const root = location.pathname.replace(
        /\\/availability(?:\\/(?:problems|stable))?\\/?$/, '/');
      const next = mode === 'all'
        ? root + 'availability/'
        : root + 'availability/' + mode + '/';
      history.pushState({{tab: 'view-availability', av: mode}}, '', next);
    }}
  }});
}});

// Apply av filter from data-av attribute (set by per-view entry points).
if (document.body.dataset.av) applyAvFilter(document.body.dataset.av);

document.querySelectorAll('#view-changes .change-id').forEach(el => {{
  el.title = 'Click to copy';
  el.addEventListener('click', () => {{
    navigator.clipboard.writeText(el.dataset.id || el.textContent.trim()).then(() => {{
      const orig = el.textContent;
      el.textContent = '✓ copied';
      setTimeout(() => el.textContent = orig, 1200);
    }});
  }});
}});
</script>
</body>
</html>"""


def fmt_context(ctx):
    if not ctx:
        return ""
    ctx = int(ctx)
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    if ctx >= 1_000:
        return f"{ctx // 1_000}k"
    return str(ctx)


CHEVRON_SVG = (
    '<svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="6 9 12 15 18 9"/></svg>'
)


def render_provider(p, models, error=None, delta=None):
    color = p["color"]
    label = escape(p["label"])
    url = p["url"]
    key_url = p.get("key_url", "")
    count = len(models) if models else 0

    key_link = (
        f'<a class="api-key-link" href="{key_url}" target="_blank" title="Get API key">Get API key</a>'
        if key_url else ""
    )
    delta_html = ""
    if delta:
        if delta.get("added"):
            delta_html += f'<span class="delta-add">+{delta["added"]}</span>'
        if delta.get("removed"):
            delta_html += f'<span class="delta-rem">-{delta["removed"]}</span>'
    if error:
        status = f'<span class="status-indicator status-err" title="Error: {escape(str(error))}">&#9679;</span>'
    else:
        status = '<span class="status-indicator status-ok" title="API responding">&#9679;</span>'
    collapse_btn = f'<button class="collapse-btn" aria-label="Toggle models list">{CHEVRON_SVG}</button>'

    header = (
        f'<div class="provider-header">'
        f'<span class="provider-dot" style="background:{color}"></span>'
        f'<span class="provider-name"><a href="{url}" target="_blank">{label}</a></span>'
        f'{key_link}'
        f'<span class="provider-count"><span class="count-num">{count}</span> models{delta_html}</span>'
        f'{status}'
        f'{collapse_btn}'
        f'</div>'
    )

    if error:
        inner = f'<div class="provider-error">⚠ Could not fetch models: {escape(str(error))}</div>'
    elif not models:
        inner = '<div class="provider-error">No free models found.</div>'
    else:
        rows = ""
        for m in sorted(models, key=lambda x: x["id"]):
            ctx_raw = int(m.get("context") or 0)
            ctx = fmt_context(ctx_raw)
            mid = m["id"]
            name = m.get("name") or mid
            search_val = escape(f"{mid} {name}".lower())
            tag_list = get_tags(mid, m.get("context"), m.get("capabilities"))
            tags_html = "".join(
                f'<span class="tag-chip" style="background:{c}22;color:{c}">{escape(label)}</span>'
                for label, c in tag_list
            )
            tag_labels = escape(" ".join(label for label, _ in tag_list))
            rows += (
                f'<tr data-search="{search_val}" data-ctx="{ctx_raw}" data-tags="{tag_labels}">'
                f'<td><span class="model-id" data-id="{escape(mid)}">{escape(mid)}</span></td>'
                f'<td class="model-name">{escape(name)}</td>'
                f'<td class="ctx">{escape(ctx)}</td>'
                f'<td>{tags_html}</td>'
                f'<td class="limits">{escape(m.get("limits") or "")}</td>'
                f"</tr>"
            )
        colgroup = (
            '<colgroup>'
            '<col class="col-id"><col class="col-name"><col class="col-ctx">'
            '<col class="col-tags"><col class="col-limits">'
            '</colgroup>'
        )
        inner = (
            '<table>' + colgroup + '<thead><tr>'
            '<th>Model ID <span class="copy-tip">(click to copy)</span></th>'
            '<th>Name</th><th>Context</th><th>Tags</th><th>Limits</th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>'
        )

    body = f'<div class="provider-body">{inner}</div>'
    return f'<div class="provider-card" data-total="{count}">{header}{body}</div>'


def render_cross_provider(groups, provider_map):
    """Render the By Model view. groups = [(canonical, [model_entries])]"""
    if not groups:
        return '<p style="color:var(--muted);text-align:center;padding:2rem">No cross-provider models detected.</p>'
    html = ""
    for cname, entries in groups:
        providers_in_group = sorted({e["provider"] for e in entries})
        provider_dots = "".join(
            f'<span class="provider-chip" style="background:{provider_map[pr]["color"]}" title="{escape(pr)}"></span>'
            for pr in providers_in_group if pr in provider_map
        )
        header = (
            f'<div class="cross-group-header">'
            f'{provider_dots}'
            f'<span>{escape(cname)}</span>'
            f'<span class="cross-group-count">{len(providers_in_group)} providers · {len(entries)} variants</span>'
            f'</div>'
        )
        rows = ""
        rows_count = 0
        for e in sorted(entries, key=lambda x: x["provider"]):
            pcolor = provider_map.get(e["provider"], {}).get("color", "#94a3b8")
            ctx_raw = int(e.get("context") or 0)
            ctx = fmt_context(ctx_raw)
            tag_list = get_tags(e["model_id"], e.get("context"), e.get("capabilities"))
            tags_html = "".join(
                f'<span class="tag-chip" style="background:{c}22;color:{c}">{escape(label)}</span>'
                for label, c in tag_list
            )
            search_val = escape(f'{e["model_id"]} {e["provider"]} {cname}'.lower())
            tag_labels = escape(" ".join(label for label, _ in tag_list))
            rows += (
                f'<tr data-search="{search_val}" data-ctx="{ctx_raw}" data-tags="{tag_labels}">'
                f'<td><span class="provider-chip" style="background:{pcolor}"></span> {escape(e["provider"])}</td>'
                f'<td><span class="model-id" data-id="{escape(e["model_id"])}">{escape(e["model_id"])}</span></td>'
                f'<td class="ctx">{escape(ctx)}</td>'
                f'<td>{tags_html}</td>'
                f'<td class="limits">{escape(e.get("limits") or "")}</td>'
                f'</tr>'
            )
            rows_count += 1
        colgroup = (
            '<colgroup>'
            '<col style="width:18%"><col style="width:34%">'
            '<col style="width:8%"><col style="width:16%"><col style="width:24%">'
            '</colgroup>'
        )
        table = (
            '<table>' + colgroup + '<thead><tr>'
            '<th>Provider</th><th>Model ID</th><th>Context</th><th>Tags</th><th>Limits</th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>'
        )
        html += f'<div class="cross-group" data-total="{rows_count}">{header}{table}</div>'
    return html


def render_changes(history, provider_color_map):
    """Render the Changes view: most recent provider-level changes first."""
    if not history:
        return '<p class="changes-empty">No changes recorded yet. Subsequent sync runs will populate this list.</p>'
    html = ""
    for entry in reversed(history):
        ts = entry.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts_display = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            ts_display = ts
        pk = entry.get("provider", "")
        plabel = entry.get("provider_label", pk)
        color = provider_color_map.get(pk, "#94a3b8")
        added = entry.get("added", []) or []
        removed = entry.get("removed", []) or []
        rows = ""
        for mid in added:
            rows += (
                f'<div class="change-row added">'
                f'<span class="change-marker">+</span>'
                f'<span class="change-id" data-id="{escape(mid)}">{escape(mid)}</span>'
                f'</div>'
            )
        for mid in removed:
            rows += (
                f'<div class="change-row removed">'
                f'<span class="change-marker">−</span>'
                f'<span class="change-id" data-id="{escape(mid)}">{escape(mid)}</span>'
                f'</div>'
            )
        summary_parts = []
        if added:
            summary_parts.append(f'<span style="color:#22c55e">+{len(added)} added</span>')
        if removed:
            summary_parts.append(f'<span style="color:#f87171">−{len(removed)} removed</span>')
        summary = " · ".join(summary_parts)
        html += (
            f'<div class="change-entry">'
            f'<div class="change-entry-header">'
            f'<span class="provider-dot" style="background:{color}"></span>'
            f'<span class="provider-name">{escape(plabel)}</span>'
            f'<time>{escape(ts_display)}</time>'
            f'<span class="change-summary">{summary}</span>'
            f'</div>'
            f'<div class="change-list">{rows}</div>'
            f'</div>'
        )
    return html


def _uptime_class(u):
    if u is None:
        return "uptime-none"
    if u >= 0.95:
        return "uptime-good"
    if u >= 0.70:
        return "uptime-warn"
    return "uptime-bad"


def _uptime_text(u):
    if u is None:
        return "—"
    return f"{u * 100:.0f}%"


def render_availability(provider_list, results, availability):
    """Render the Availability view from docs/availability.json data."""
    if not availability:
        return ('<p class="av-empty" style="text-align:center;padding:2rem">'
                'No probe data yet. The probe workflow runs every 2h; data will '
                'appear after the first run completes.</p>')

    # Rotate so the rightmost bar = current UTC hour (freshest possible data).
    now_hour = datetime.now(timezone.utc).hour
    bar_order = [(now_hour + 1 + i) % 24 for i in range(24)]

    html = ""
    for p in provider_list:
        pk = p["key"]
        models = results.get(pk, [])
        if not models:
            continue
        avail_for_provider = availability.get(pk, {})
        # Header (reuses provider-header styling).
        header = (
            f'<div class="provider-header">'
            f'<span class="provider-dot" style="background:{p["color"]}"></span>'
            f'<span class="provider-name">{escape(p["label"])}</span>'
            f'<span class="provider-count">{len(models)} models</span>'
            f'</div>'
        )

        rows_html = ""
        # Order by uptime ascending (problems first), then by id.
        def sort_key(m):
            a = avail_for_provider.get(m["id"], {})
            u = a.get("uptime_7d")
            return (1 if u is None else 0, u if u is not None else 1.0, m["id"])

        for m in sorted(models, key=sort_key):
            mid = m["id"]
            a = avail_for_provider.get(mid, {})
            u7 = a.get("uptime_7d")
            samples = a.get("samples_7d", 0)
            rl = a.get("rate_limited_7d", 0)
            p50 = a.get("p50_latency_ms")
            hourly = a.get("hourly_uptime") or [{"ok": 0, "total": 0}] * 24

            badge = (
                f'<span class="uptime-badge {_uptime_class(u7)}" '
                f'title="7-day uptime · {samples} probes · '
                f'{rl} rate-limited">{_uptime_text(u7)}</span>'
            )
            bars = ""
            for h in bar_order:
                cell = hourly[h] if h < len(hourly) else {"ok": 0, "total": 0}
                tot = cell.get("total", 0)
                ok_n = cell.get("ok", 0)
                if tot == 0:
                    bg = "#334155"
                    title = f"{h:02d}:00 UTC · no data"
                else:
                    rate = ok_n / tot
                    if rate >= 0.95:
                        bg = "#22c55e"
                    elif rate >= 0.70:
                        bg = "#f59e0b"
                    else:
                        bg = "#ef4444"
                    title = f"{h:02d}:00 UTC · {rate*100:.0f}% · {ok_n}/{tot} probes"
                bars += f'<span class="av-bar" style="background:{bg}" title="{title}"></span>'
            heatmap = f'<span class="av-heatmap" aria-label="hourly uptime, UTC">{bars}</span>'
            meta_bits = []
            if p50 is not None:
                meta_bits.append(f"p50 {p50}ms")
            if rl:
                meta_bits.append(f"rl {rl}")
            if samples:
                meta_bits.append(f"n={samples}")
            meta = " · ".join(meta_bits) or "no probes"
            uptime_data = "" if u7 is None else f"{u7:.4f}"
            search_val = escape(f'{mid} {p["label"]}'.lower())
            tag_labels = escape(" ".join(label for label, _ in get_tags(mid, m.get("context"), m.get("capabilities"))))
            ctx_raw = int(m.get("context") or 0)
            rows_html += (
                f'<div class="av-row" data-uptime="{uptime_data}" '
                f'data-search="{search_val}" data-ctx="{ctx_raw}" data-tags="{tag_labels}">'
                f'<span class="model-id" data-id="{escape(mid)}">{escape(mid)}</span>'
                f'{badge}{heatmap}'
                f'<span class="av-meta">{escape(meta)}</span>'
                f'</div>'
            )

        body = f'<div class="provider-body">{rows_html}</div>'
        html += f'<div class="provider-card" data-total="{len(models)}">{header}{body}</div>'
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching community cross-reference...")
    # (used for Cohere fallback — already in sync_models.py logic)

    results = {}  # provider_key → list of model dicts
    errors = {}

    for p in PROVIDERS:
        key = os.environ.get(p["env"], "")
        if not key and not p.get("anonymous_ok"):
            print(f"  [{p['label']}] no API key, skipping")
            continue
        print(f"  [{p['label']}] fetching...", end=" ", flush=True)
        try:
            models = p["fetch"](key)
            results[p["key"]] = models
            print(f"{len(models)} models")
        except Exception as e:
            errors[p["key"]] = e
            results[p["key"]] = []
            print(f"ERROR: {e}")

    # Self-correcting layer: drop models that probe data confirms are broken.
    # A model is dropped when uptime_7d == 0 with samples_7d >= MIN_SAMPLES.
    # Once probes start succeeding again (e.g. provider restores credit), the
    # rolling 7d window will lift the model back onto the list.
    DROP_MIN_SAMPLES = 5
    avail_path = OUT_DIR / "availability.json"
    try:
        avail = json.loads(avail_path.read_text())
        dropped_total = 0
        for pk, models in list(results.items()):
            prov_avail = avail.get("providers", {}).get(pk, {})
            kept = []
            dropped = []
            for m in models:
                stats = prov_avail.get(m["id"], {})
                up = stats.get("uptime_7d")
                n = stats.get("samples_7d", 0) or 0
                if up == 0.0 and n >= DROP_MIN_SAMPLES:
                    dropped.append(m["id"])
                else:
                    kept.append(m)
            if dropped:
                print(f"  [{pk}] dropped {len(dropped)} models with sustained 0% uptime: "
                      f"{', '.join(dropped[:3])}{'...' if len(dropped) > 3 else ''}")
                dropped_total += len(dropped)
                results[pk] = kept
        if dropped_total:
            print(f"  Self-correcting layer dropped {dropped_total} broken models total")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  Self-correcting layer skipped: {e}")

    print("Enriching with litellm model DB...")
    enrich_with_litellm(results, fetch_litellm_db())

    # Read previous models.json for delta computation
    old_models_path = OUT_DIR / "models.json"
    old_model_ids = {}  # provider_key → set of model IDs
    try:
        old_json = json.loads(old_models_path.read_text())
        for pk, pdata in old_json.get("providers", {}).items():
            old_model_ids[pk] = {m["id"] for m in pdata.get("models", [])}
    except Exception:
        pass

    # Build JSON output
    json_out = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "providers": {}
    }
    for p in PROVIDERS:
        if p["key"] not in results:
            continue
        json_out["providers"][p["key"]] = {
            "name": p["label"],
            "url": p["url"],
            "models": results[p["key"]],
            "error": str(errors[p["key"]]) if p["key"] in errors else None,
        }

    old_models_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False))
    print("Written docs/models.json")

    # Filtered variants under /availability/ — same shape as models.json,
    # but each provider's "models" list is restricted by 7-day probe uptime.
    # Stable: uptime_7d >= 0.95; Problems: 0 <= uptime_7d < 0.95.
    # Both require samples_7d >= 5 so we don't include unproven models.
    try:
        avail_for_filter = json.loads(
            (OUT_DIR / "availability.json").read_text()
        ).get("providers", {})
    except Exception:
        avail_for_filter = {}

    def _filtered_json(predicate):
        out = {"updated": json_out["updated"], "providers": {}}
        for pk, pdata in json_out["providers"].items():
            prov_avail = avail_for_filter.get(pk, {})
            kept = []
            for m in pdata["models"]:
                a = prov_avail.get(m["id"], {})
                u = a.get("uptime_7d")
                n = a.get("samples_7d", 0) or 0
                if u is not None and n >= 5 and predicate(u):
                    kept.append(m)
            if kept:
                out["providers"][pk] = {**pdata, "models": kept}
        return out

    avail_subdir = OUT_DIR / "availability"
    avail_subdir.mkdir(parents=True, exist_ok=True)
    stable = _filtered_json(lambda u: u >= 0.95)
    problems = _filtered_json(lambda u: 0.0 <= u < 0.95)
    (avail_subdir / "stable_models.json").write_text(
        json.dumps(stable, indent=2, ensure_ascii=False))
    (avail_subdir / "problems_models.json").write_text(
        json.dumps(problems, indent=2, ensure_ascii=False))
    n_stable = sum(len(p["models"]) for p in stable["providers"].values())
    n_problems = sum(len(p["models"]) for p in problems["providers"].values())
    print(f"Written docs/availability/stable_models.json ({n_stable} models)")
    print(f"Written docs/availability/problems_models.json ({n_problems} models)")

    # Compute per-provider deltas (with full id sets for history)
    deltas = {}
    deltas_full = {}
    for p in PROVIDERS:
        pk = p["key"]
        if pk not in results or pk not in old_model_ids:
            continue
        current_ids = {m["id"] for m in results[pk]}
        old_ids = old_model_ids[pk]
        added_ids = sorted(current_ids - old_ids)
        removed_ids = sorted(old_ids - current_ids)
        if added_ids or removed_ids:
            deltas[pk] = {"added": len(added_ids), "removed": len(removed_ids)}
            deltas_full[pk] = {"added": added_ids, "removed": removed_ids}

    # Append to history.json (one entry per provider that changed this run)
    history_path = OUT_DIR / "history.json"
    try:
        history = json.loads(history_path.read_text()).get("entries", [])
    except Exception:
        history = []

    now_iso = datetime.now(timezone.utc).isoformat()
    provider_label_map = {p["key"]: p["label"] for p in PROVIDERS}
    for pk, d in deltas_full.items():
        history.append({
            "timestamp": now_iso,
            "provider": pk,
            "provider_label": provider_label_map.get(pk, pk),
            "added": d["added"],
            "removed": d["removed"],
        })
    # Cap to last 500 entries
    history = history[-500:]
    history_path.write_text(json.dumps({"entries": history}, indent=2, ensure_ascii=False))

    # Compute cross-provider model groups
    from collections import defaultdict
    provider_map = {p["label"]: p for p in PROVIDERS}
    groups_raw = defaultdict(list)
    for p in PROVIDERS:
        for m in results.get(p["key"], []):
            key = canonical_name(m["id"])
            if key:
                groups_raw[key].append({
                    "provider": p["label"],
                    "model_id": m["id"],
                    "context": m.get("context"),
                    "limits": m.get("limits", ""),
                    "capabilities": m.get("capabilities"),
                })
    cross_groups = [
        (cname, entries)
        for cname, entries in sorted(groups_raw.items())
        if len({e["provider"] for e in entries}) >= 2
    ]
    cross_groups.sort(key=lambda x: -len({e["provider"] for e in x[1]}))

    # Build HTML
    total_models = sum(len(v) for v in results.values())
    total_providers = sum(1 for v in results.values() if v)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = ""
    for p in PROVIDERS:
        if p["key"] not in results and p["key"] not in errors:
            continue
        sections += render_provider(
            p,
            results.get(p["key"], []),
            errors.get(p["key"]),
            delta=deltas.get(p["key"]),
        )

    cross_html = render_cross_provider(cross_groups, provider_map)
    provider_color_map = {p["key"]: p["color"] for p in PROVIDERS}
    changes_html = render_changes(history, provider_color_map)

    availability = {}
    avail_path = OUT_DIR / "availability.json"
    try:
        availability = json.loads(avail_path.read_text()).get("providers", {})
    except Exception:
        pass
    availability_html = render_availability(PROVIDERS, results, availability)

    html = HTML_TEMPLATE.format(
        updated=updated,
        total_models=total_models,
        total_providers=total_providers,
        provider_sections=sections,
        cross_provider_section=cross_html,
        availability_section=availability_html,
        changes_section=changes_html,
    )
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Written docs/index.html  ({total_models} models, {total_providers} providers, {len(cross_groups)} cross-provider groups)")

    # Per-view entry points (so /availability, /by-model, etc. work as URLs).
    # Each is the same HTML with <base href> pointing at the site root and a
    # data-tab marker so the JS activates the right tab on initial load.
    # Availability has shareable sub-filters (/availability/problems, /stable).
    entry_points = [
        ("by-provider",           "view-provider",     "../",    None),
        ("by-model",              "view-model",        "../",    None),
        ("availability",          "view-availability", "../",    None),
        ("availability/problems", "view-availability", "../../", "problems"),
        ("availability/stable",   "view-availability", "../../", "stable"),
        ("changes",               "view-changes",      "../",    None),
    ]
    for slug, tab, base, av in entry_points:
        sub_dir = OUT_DIR / slug
        sub_dir.mkdir(parents=True, exist_ok=True)
        body_attrs = f'data-tab="{tab}"'
        if av:
            body_attrs += f' data-av="{av}"'
        sub_html = (
            html
            .replace("<head>", f'<head>\n<base href="{base}">', 1)
            .replace("<body>", f'<body {body_attrs}>', 1)
        )
        (sub_dir / "index.html").write_text(sub_html, encoding="utf-8")
        # Mirror the JSON files so links resolve from any sub-path.
        for jf in ("models.json", "availability.json"):
            src = OUT_DIR / jf
            if src.exists():
                (sub_dir / jf).write_bytes(src.read_bytes())
    print(f"Written {len(entry_points)} per-view entry points")


if __name__ == "__main__":
    main()
