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


def fetch_chutes(key):
    data = _get("https://llm.chutes.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"})
    return [{"id": m["id"], "name": m["id"],
             "context": None, "limits": "free · rate-limited"}
            for m in data.get("data", [])
            if not any(x in m.get("id","").lower() for x in ("embed","tts","stt","image","vision"))]


# ── Community cross-reference ─────────────────────────────────────────────────

def fetch_cheahjs():
    try:
        return _get(CHEAHJS_URL)
    except Exception as e:
        print(f"[community] {e}")
        return ""


PROVIDERS = [
    {"key": "openrouter",   "label": "OpenRouter",   "env": "OPENROUTER_API_KEY",  "fetch": fetch_openrouter, "color": "#6366f1", "url": "https://openrouter.ai"},
    {"key": "groq",         "label": "Groq",         "env": "GROQ_API_KEY",        "fetch": fetch_groq,       "color": "#f59e0b", "url": "https://console.groq.com"},
    {"key": "cerebras",     "label": "Cerebras",     "env": "CEREBRAS_API_KEY",    "fetch": fetch_cerebras,   "color": "#10b981", "url": "https://cloud.cerebras.ai"},
    {"key": "gemini",       "label": "Gemini",       "env": "GEMINI_API_KEY",      "fetch": fetch_gemini,     "color": "#3b82f6", "url": "https://aistudio.google.com"},
    {"key": "sambanova",    "label": "SambaNova",    "env": "SAMBANOVA_API_KEY",   "fetch": fetch_sambanova,  "color": "#8b5cf6", "url": "https://cloud.sambanova.ai"},
    {"key": "cohere",       "label": "Cohere",       "env": "COHERE_API_KEY",      "fetch": fetch_cohere,     "color": "#ec4899", "url": "https://cohere.com"},
    {"key": "together",     "label": "Together AI",  "env": "TOGETHER_API_KEY",    "fetch": fetch_together,   "color": "#14b8a6", "url": "https://api.together.ai"},
    {"key": "nvidia",       "label": "NVIDIA NIM",   "env": "NVIDIA_NIM_API_KEY",  "fetch": fetch_nvidia,     "color": "#22c55e", "url": "https://build.nvidia.com"},
    {"key": "huggingface",  "label": "HuggingFace",  "env": "HF_TOKEN",            "fetch": fetch_huggingface,"color": "#f97316", "url": "https://huggingface.co"},
    {"key": "mistral",      "label": "Mistral",      "env": "MISTRAL_API_KEY",     "fetch": fetch_mistral,    "color": "#0ea5e9", "url": "https://console.mistral.ai"},
    {"key": "github",       "label": "GitHub Models","env": "GH_MODELS_TOKEN",      "fetch": fetch_github,     "color": "#e2e8f0", "url": "https://github.com/marketplace/models"},
    {"key": "cloudflare",   "label": "Cloudflare AI","env": "CLOUDFLARE_API_KEY",  "fetch": fetch_cloudflare, "color": "#f6821f", "url": "https://developers.cloudflare.com/workers-ai/"},
    {"key": "chutes",       "label": "Chutes",       "env": "CHUTES_API_KEY",      "fetch": fetch_chutes,     "color": "#06b6d4", "url": "https://chutes.ai"},
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
  header {{ max-width: 960px; margin: 0 auto 2rem; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; color: #fff; }}
  .subtitle {{ color: var(--muted); margin-top: .4rem; font-size: .95rem; }}
  .subtitle a {{ color: var(--accent); text-decoration: none; }}
  .subtitle a:hover {{ text-decoration: underline; }}
  .meta {{ margin-top: .6rem; font-size: .8rem; color: var(--muted); }}
  .stats {{ display: flex; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: .5rem; padding: .5rem 1rem; font-size: .85rem; }}
  .stat strong {{ color: #fff; font-size: 1.3rem; display: block; }}
  main {{ max-width: 960px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.5rem; }}
  .provider-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: .75rem; overflow: hidden; }}
  .provider-header {{ display: flex; align-items: center; gap: .75rem; padding: .85rem 1.25rem; border-bottom: 1px solid var(--border); }}
  .provider-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .provider-name {{ font-weight: 600; font-size: 1rem; }}
  .provider-name a {{ color: inherit; text-decoration: none; }}
  .provider-name a:hover {{ color: var(--accent); }}
  .provider-count {{ margin-left: auto; background: #0f172a; border-radius: 999px; padding: .15rem .6rem; font-size: .75rem; color: var(--muted); }}
  .provider-error {{ padding: 1rem 1.25rem; color: #f87171; font-size: .85rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ text-align: left; padding: .5rem 1.25rem; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); }}
  td {{ padding: .45rem 1.25rem; border-bottom: 1px solid #1e293b; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,.03); }}
  .model-id {{ font-family: ui-monospace, monospace; font-size: .82rem; color: var(--accent); cursor: pointer; }}
  .model-id:hover {{ text-decoration: underline; }}
  .model-name {{ color: var(--text); }}
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
  @media (max-width: 600px) {{
    td:nth-child(3), th:nth-child(3),
    td:nth-child(4), th:nth-child(4) {{ display: none; }}
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
<main>
{provider_sections}
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
    navigator.clipboard.writeText(el.textContent.trim()).then(() => {{
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


def render_provider(p, models, error=None):
    color = p["color"]
    label = escape(p["label"])
    url = p["url"]
    count = len(models) if models else 0

    header = (
        f'<div class="provider-header">'
        f'<span class="provider-dot" style="background:{color}"></span>'
        f'<span class="provider-name"><a href="{url}" target="_blank">{label}</a></span>'
        f'<span class="provider-count">{count} models</span>'
        f'</div>'
    )

    if error:
        body = f'<div class="provider-error">⚠ Could not fetch models: {escape(str(error))}</div>'
    elif not models:
        body = '<div class="provider-error">No free models found.</div>'
    else:
        rows = ""
        for m in sorted(models, key=lambda x: x["id"]):
            ctx = fmt_context(m.get("context"))
            rows += (
                f"<tr>"
                f'<td><span class="model-id">{escape(m["id"])}</span></td>'
                f'<td class="model-name">{escape(m.get("name") or m["id"])}</td>'
                f'<td class="ctx">{escape(ctx)}</td>'
                f'<td class="limits">{escape(m.get("limits") or "")}</td>'
                f"</tr>"
            )
        body = (
            '<table><thead><tr>'
            '<th>Model ID <span class="copy-tip">(click to copy)</span></th>'
            '<th>Name</th><th>Context</th><th>Limits</th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>'
        )

    return f'<div class="provider-card">{header}{body}</div>'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching community cross-reference...")
    # (used for Cohere fallback — already in sync_models.py logic)

    results = {}  # provider_key → list of model dicts
    errors = {}

    for p in PROVIDERS:
        key = os.environ.get(p["env"], "")
        if not key:
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

    (OUT_DIR / "models.json").write_text(json.dumps(json_out, indent=2, ensure_ascii=False))
    print(f"Written docs/models.json")

    # Build HTML
    total_models = sum(len(v) for v in results.values())
    total_providers = sum(1 for v in results.values() if v)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = ""
    for p in PROVIDERS:
        if p["key"] not in results and p["key"] not in errors:
            continue
        sections += render_provider(p, results.get(p["key"], []), errors.get(p["key"]))

    html = HTML_TEMPLATE.format(
        updated=updated,
        total_models=total_models,
        total_providers=total_providers,
        provider_sections=sections,
    )
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Written docs/index.html  ({total_models} models, {total_providers} providers)")


if __name__ == "__main__":
    main()
