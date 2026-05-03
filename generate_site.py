#!/usr/bin/env python3
"""
Generates docs/index.html and docs/models.json from provider APIs.
Runs standalone (no LiteLLM needed) — used by GitHub Actions to build the site.
"""

import os, json, re, time, urllib.request, urllib.error
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
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


def get_tags(model_id, context=None):
    tags = []
    mid = model_id.lower()
    for keywords, label, color in _TAG_RULES:
        if any(kw in mid for kw in keywords):
            tags.append((label, color))
    if context and int(context) >= 128_000:
        tags.append(("128k+", "#38bdf8"))
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
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, sans-serif; padding: 2rem 1rem; }}
  header {{ max-width: 960px; margin: 0 auto 1.5rem; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; color: #fff; }}
  .subtitle {{ color: var(--muted); margin-top: .4rem; font-size: .95rem; }}
  .subtitle a {{ color: var(--accent); text-decoration: none; }}
  .subtitle a:hover {{ text-decoration: underline; }}
  .meta {{ margin-top: .6rem; font-size: .8rem; color: var(--muted); }}
  .stats {{ display: flex; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: .5rem; padding: .5rem 1rem; font-size: .85rem; }}
  .stat strong {{ color: #fff; font-size: 1.3rem; display: block; }}
  .search-wrap {{ max-width: 960px; margin: 0 auto 1.5rem; }}
  .search-wrap input {{
    width: 100%; padding: .65rem 1rem .65rem 2.5rem; font-size: .95rem;
    background: var(--surface); border: 1px solid var(--border); border-radius: .5rem;
    color: var(--text); outline: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: .75rem center;
    transition: border-color .15s;
  }}
  .search-wrap input:focus {{ border-color: var(--accent); }}
  .search-wrap input::placeholder {{ color: var(--muted); }}
  .no-results {{ display: none; text-align: center; padding: 2rem; color: var(--muted); font-size: .9rem; }}
  main {{ max-width: 960px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.5rem; }}
  .provider-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: .75rem; overflow: hidden; }}
  .provider-header {{ display: flex; align-items: center; gap: .75rem; padding: .85rem 1.25rem; border-bottom: 1px solid var(--border); }}
  .provider-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .provider-name {{ font-weight: 600; font-size: 1rem; }}
  .provider-name a {{ color: inherit; text-decoration: none; }}
  .provider-name a:hover {{ color: var(--accent); }}
  .provider-count {{ margin-left: auto; background: #0f172a; border-radius: 999px; padding: .15rem .6rem; font-size: .75rem; color: var(--muted); }}
  .delta-add {{ color: #22c55e; font-size: .7rem; margin-left: .3rem; }}
  .delta-rem {{ color: #f87171; font-size: .7rem; margin-left: .15rem; }}
  .status-indicator {{ font-size: .5rem; flex-shrink: 0; }}
  .status-ok  {{ color: #22c55e; }}
  .status-err {{ color: #f87171; }}
  .tag-chip {{ display: inline-block; border-radius: 999px; padding: .05rem .45rem; font-size: .68rem; font-weight: 500; margin-right: .2rem; margin-bottom: .1rem; white-space: nowrap; }}
  .view-tabs {{ max-width: 960px; margin: 0 auto 1.25rem; display: flex; gap: .5rem; }}
  .vtab {{ background: none; border: 1px solid var(--border); border-radius: .4rem; color: var(--muted); padding: .35rem .9rem; font-size: .85rem; cursor: pointer; transition: all .15s; }}
  .vtab:hover {{ color: var(--text); border-color: #64748b; }}
  .vtab.active {{ background: var(--surface); border-color: var(--accent); color: var(--accent); font-weight: 600; }}
  .cross-group {{ background: var(--surface); border: 1px solid var(--border); border-radius: .75rem; overflow: hidden; margin-bottom: 1.5rem; }}
  .cross-group-header {{ padding: .7rem 1.25rem; border-bottom: 1px solid var(--border); font-weight: 600; font-size: .95rem; display: flex; align-items: center; gap: .6rem; }}
  .cross-group-count {{ font-size: .75rem; color: var(--muted); font-weight: 400; }}
  .provider-chip {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; }}
  .api-key-link {{ font-size: .75rem; color: var(--muted); text-decoration: none; border: 1px solid var(--border); border-radius: .35rem; padding: .15rem .55rem; white-space: nowrap; transition: color .15s, border-color .15s; }}
  .api-key-link:hover {{ color: var(--accent); border-color: var(--accent); }}
  .collapse-btn {{ background: none; border: none; cursor: pointer; color: var(--muted); padding: .1rem .2rem; line-height: 1; display: flex; align-items: center; transition: color .15s; }}
  .collapse-btn:hover {{ color: var(--text); }}
  .collapse-btn .chevron {{ transition: transform .2s; }}
  .provider-card.collapsed .chevron {{ transform: rotate(-90deg); }}
  .provider-card.collapsed .provider-header {{ border-bottom: none; }}
  .provider-body {{ }}
  .provider-card.collapsed .provider-body {{ display: none; }}
  .provider-error {{ padding: 1rem 1.25rem; color: #f87171; font-size: .85rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; table-layout: fixed; }}
  col.col-id      {{ width: 32%; }}
  col.col-name    {{ width: 20%; }}
  col.col-ctx     {{ width:  8%; }}
  col.col-tags    {{ width: 18%; }}
  col.col-limits  {{ width: 22%; }}
  th {{ text-align: left; padding: .5rem 1.25rem; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); overflow: hidden; }}
  td {{ padding: .45rem 1.25rem; border-bottom: 1px solid #1e293b; vertical-align: middle; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,.03); }}
  .model-id {{ font-family: ui-monospace, monospace; font-size: .82rem; color: var(--accent); cursor: pointer; display: block; overflow: hidden; text-overflow: ellipsis; }}
  .model-id:hover {{ text-decoration: underline; }}
  .model-name {{ color: var(--text); overflow: hidden; text-overflow: ellipsis; }}
  .ctx {{ color: var(--muted); font-size: .78rem; }}
  .limits {{ color: var(--muted); font-size: .78rem; }}
  .copy-tip {{ font-size: .7rem; color: #475569; margin-left: .4rem; }}
  footer {{ max-width: 960px; margin: 2rem auto 0; font-size: .8rem; color: var(--muted); text-align: center; }}
  footer a {{ color: var(--accent); text-decoration: none; }}
  .suggest-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: .75rem; padding: 1.5rem 1.75rem; display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }}
  .suggest-card .suggest-text h2 {{ font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: .3rem; }}
  .suggest-card .suggest-text p {{ font-size: .85rem; color: var(--muted); line-height: 1.5; }}
  .suggest-btn {{ display: inline-flex; align-items: center; gap: .5rem; background: #238636; color: #fff; border: none; border-radius: .5rem; padding: .6rem 1.2rem; font-size: .875rem; font-weight: 600; text-decoration: none; white-space: nowrap; cursor: pointer; transition: background .15s; }}
  .suggest-btn:hover {{ background: #2ea043; }}
  .ctx-filters, .tag-filters {{ max-width: 960px; margin: 0 auto 1rem; display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }}
  .ctx-label {{ font-size: .8rem; color: var(--muted); white-space: nowrap; }}
  .ctx-pill, .tag-pill {{ background: none; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); padding: .25rem .75rem; font-size: .8rem; cursor: pointer; transition: color .15s, border-color .15s, background .15s; }}
  .ctx-pill:hover, .tag-pill:hover {{ color: var(--text); border-color: #64748b; }}
  .ctx-pill.active {{ background: var(--accent); border-color: var(--accent); color: #0f172a; font-weight: 600; }}
  .tag-pill.active {{ background: var(--tc,var(--accent)); border-color: var(--tc,var(--accent)); color: #0f172a; font-weight: 600; }}
  @media (max-width: 600px) {{
    col.col-ctx, th:nth-child(3), td:nth-child(3),
    col.col-tags, th:nth-child(4), td:nth-child(4),
    col.col-limits, th:nth-child(5), td:nth-child(5) {{ display: none; }}
    col.col-id   {{ width: 50%; }}
    col.col-name {{ width: 50%; }}
    .suggest-card {{ flex-direction: column; align-items: flex-start; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Free LLM API Access</h1>
  <p class="subtitle">
    Providers that give you <strong style="color:#e2e8f0">free API tokens</strong> to call LLM models — no credit card required.<br>
    List is auto-updated every 8 hours from provider APIs.<br>
    Source: provider APIs + <a href="https://github.com/cheahjs/free-llm-api-resources" target="_blank">cheahjs/free-llm-api-resources</a> ·
    <a href="models.json" target="_blank">models.json</a> ·
    <a href="https://github.com/tomaasz/litellm-free-models-proxy" target="_blank">GitHub</a>
  </p>
  <p class="meta">Updated: {updated}</p>
  <div class="stats">
    <div class="stat"><strong>{total_models}</strong>models with free API access</div>
    <div class="stat"><strong>{total_providers}</strong>providers</div>
  </div>
</header>
<div class="search-wrap">
  <input type="search" id="model-search" placeholder="Search models by ID or name…" autocomplete="off" spellcheck="false">
</div>
<div class="ctx-filters">
  <span class="ctx-label">Context:</span>
  <button class="ctx-pill active" data-min="0">Any</button>
  <button class="ctx-pill" data-min="32768">&#8805; 32k</button>
  <button class="ctx-pill" data-min="131072">&#8805; 128k</button>
  <button class="ctx-pill" data-min="1000000">&#8805; 1M</button>
</div>
<div class="tag-filters">
  <span class="ctx-label">Tags:</span>
  <button class="tag-pill active" data-tag="">All</button>
  <button class="tag-pill" data-tag="coding"    style="--tc:#a78bfa">coding</button>
  <button class="tag-pill" data-tag="reasoning" style="--tc:#f59e0b">reasoning</button>
  <button class="tag-pill" data-tag="vision"    style="--tc:#06b6d4">vision</button>
  <button class="tag-pill" data-tag="fast"      style="--tc:#10b981">fast</button>
  <button class="tag-pill" data-tag="128k+"     style="--tc:#38bdf8">128k+</button>
</div>
<div class="view-tabs">
  <button class="vtab active" data-target="view-provider">By Provider</button>
  <button class="vtab" data-target="view-model">By Model</button>
</div>
<main>
<div id="view-provider">
{provider_sections}
<p class="no-results" id="no-results">No models match your search.</p>
</div>
<div id="view-model" style="display:none">
{cross_provider_section}
</div>
<div class="suggest-card">
  <div class="suggest-text">
    <h2>Know a provider we're missing?</h2>
    <p>If you know of a provider that gives free API tokens / free-tier access to LLM models
    and isn't listed here, open a GitHub issue — we'll add support for it.</p>
  </div>
  <a class="suggest-btn"
     href="https://github.com/tomaasz/litellm-free-models-proxy/issues/new?template=new-provider.yml"
     target="_blank">
    &#43; Suggest a provider
  </a>
</div>
</main>
<footer>
  <p>Auto-generated by <a href="https://github.com/tomaasz/litellm-free-models-proxy">litellm-free-models-proxy</a>.
  Models listed here are accessible via free-tier API tokens — not open-source or self-hostable models.
  Not affiliated with any provider. Free tiers may change without notice.</p>
</footer>
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
  const cards = Array.from(document.querySelectorAll('.provider-card[data-total]'));
  let ctxMin = 0;
  let tagFilter = '';

  function applyFilters() {{
    const q = input.value.toLowerCase().trim();
    const isFiltered = q || ctxMin > 0 || tagFilter;
    let totalVisible = 0;

    cards.forEach(card => {{
      const rows = card.querySelectorAll('tbody tr');
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

document.querySelectorAll('.vtab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.vtab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const t = btn.dataset.target;
    document.getElementById('view-provider').style.display = t === 'view-provider' ? '' : 'none';
    document.getElementById('view-model').style.display   = t === 'view-model'    ? '' : 'none';
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
            tag_list = get_tags(mid, m.get("context"))
            tags_html = "".join(
                f'<span class="tag-chip" style="background:{c}22;color:{c}">{escape(l)}</span>'
                for l, c in tag_list
            )
            tag_labels = escape(" ".join(l for l, _ in tag_list))
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
        for e in sorted(entries, key=lambda x: x["provider"]):
            pcolor = provider_map.get(e["provider"], {}).get("color", "#94a3b8")
            ctx_raw = int(e.get("context") or 0)
            ctx = fmt_context(ctx_raw)
            tag_list = get_tags(e["model_id"], e.get("context"))
            tags_html = "".join(
                f'<span class="tag-chip" style="background:{c}22;color:{c}">{escape(l)}</span>'
                for l, c in tag_list
            )
            rows += (
                f'<tr>'
                f'<td><span class="provider-chip" style="background:{pcolor}"></span> {escape(e["provider"])}</td>'
                f'<td><span class="model-id" data-id="{escape(e["model_id"])}">{escape(e["model_id"])}</span></td>'
                f'<td class="ctx">{escape(ctx)}</td>'
                f'<td>{tags_html}</td>'
                f'<td class="limits">{escape(e.get("limits") or "")}</td>'
                f'</tr>'
            )
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
        html += f'<div class="cross-group">{header}{table}</div>'
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching community cross-reference...")
    # (used for Cohere fallback — already in sync_models.py logic)

    results = {}  # provider_key → list of model dicts
    errors = {}
    keys_configured = False

    for p in PROVIDERS:
        key = os.environ.get(p["env"], "")
        if not key:
            print(f"  [{p['label']}] no API key, skipping")
            continue
        keys_configured = True
        print(f"  [{p['label']}] fetching...", end=" ", flush=True)
        try:
            models = p["fetch"](key)
            results[p["key"]] = models
            print(f"{len(models)} models")
        except Exception as e:
            errors[p["key"]] = e
            results[p["key"]] = []
            print(f"ERROR: {e}")

    if not keys_configured:
        print("No API keys configured. Exiting early to avoid overwriting existing data.")
        return

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
    print(f"Written docs/models.json")

    # Compute per-provider deltas
    deltas = {}
    for p in PROVIDERS:
        pk = p["key"]
        if pk not in results or pk not in old_model_ids:
            continue
        current_ids = {m["id"] for m in results[pk]}
        old_ids = old_model_ids[pk]
        added = len(current_ids - old_ids)
        removed = len(old_ids - current_ids)
        if added or removed:
            deltas[pk] = {"added": added, "removed": removed}

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

    html = HTML_TEMPLATE.format(
        updated=updated,
        total_models=total_models,
        total_providers=total_providers,
        provider_sections=sections,
        cross_provider_section=cross_html,
    )
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Written docs/index.html  ({total_models} models, {total_providers} providers, {len(cross_groups)} cross-provider groups)")


if __name__ == "__main__":
    main()
