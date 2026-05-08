#!/usr/bin/env python3
"""
Probes each (provider, model) listed in docs/models.json with a minimal
chat-completions request and records the result to docs/probes.jsonl.
Regenerates docs/availability.json (rolling 7d/30d aggregates).

Round-robin: each run probes ~1/3 of models (hash-bucketed by model_id);
full cycle = 6h with cron every 2h. Models that failed their last 3
probes are always included (watch list).
"""

import os, sys, json, time, gzip, hashlib, secrets, threading, urllib.request, urllib.error
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
MODELS_JSON   = DOCS / "models.json"
PROBES_JSONL  = DOCS / "probes.jsonl"
AVAIL_JSON    = DOCS / "availability.json"
ARCHIVE_DIR   = DOCS / "probes-archive"

PROBE_TIMEOUT = 15
PER_PROVIDER_CONCURRENCY = 2
PER_PROBE_PAUSE_S = 0.2
ROUND_ROBIN_BUCKETS = 3
WATCH_LIST_FAILS = 3

# ── Provider probe configs ────────────────────────────────────────────────────

# Each config: how to build a (url, headers, body_dict) for a chat probe.
# `style` distinguishes payload formats.

PROVIDER_PROBES = {
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "groq": {
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "cerebras": {
        "env": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "sambanova": {
        "env": "SAMBANOVA_API_KEY",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "together": {
        "env": "TOGETHER_API_KEY",
        "url": "https://api.together.ai/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "nvidia": {
        "env": "NVIDIA_NIM_API_KEY",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "huggingface": {
        "env": "HF_TOKEN",
        "url": "https://router.huggingface.co/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "mistral": {
        "env": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "github": {
        "env": "GH_MODELS_TOKEN",
        "url": "https://models.inference.ai.azure.com/chat/completions",
        "auth": "bearer",
        "style": "openai",
    },
    "cohere": {
        "env": "COHERE_API_KEY",
        "url": "https://api.cohere.com/v2/chat",
        "auth": "bearer",
        "style": "cohere_v2",
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "auth": "x-goog",
        "style": "gemini",
    },
    "cloudflare": {
        "env": "CLOUDFLARE_API_KEY",
        "url": "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
        "auth": "bearer",
        "style": "cloudflare",
    },
}


def build_request(provider, cfg, model_id, nonce):
    prompt = f"ping {nonce}"
    headers = {"Content-Type": "application/json", "User-Agent": "litellm-free-models-proxy/probe"}
    key = os.environ.get(cfg["env"], "")
    if cfg["auth"] == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif cfg["auth"] == "x-goog":
        headers["x-goog-api-key"] = key

    style = cfg["style"]
    if style == "openai":
        url = cfg["url"]
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
    elif style == "cohere_v2":
        url = cfg["url"]
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
    elif style == "gemini":
        # Strip leading "models/" if present.
        m = model_id.split("models/")[-1]
        url = cfg["url"].format(model=m)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1, "temperature": 0},
        }
    elif style == "cloudflare":
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        if not account:
            return None
        url = cfg["url"].format(account=account, model=model_id)
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
    else:
        return None
    return url, headers, json.dumps(body).encode()


def classify(http_status, body_text, exc):
    """Return one of: ok, rate_limited, auth_error, not_found, server_error,
    timeout, bad_response, network_error."""
    if exc is not None:
        msg = str(exc).lower()
        if "timed out" in msg or "timeout" in msg:
            return "timeout"
        return "network_error"
    if http_status is None:
        return "network_error"
    if http_status == 429:
        return "rate_limited"
    if http_status in (401, 403):
        return "auth_error"
    if http_status == 404:
        return "not_found"
    if 500 <= http_status < 600:
        return "server_error"
    if http_status == 200:
        # Heuristic: "rate limit" / "quota" wording inside 200 (rare) → rate_limited
        low = (body_text or "")[:600].lower()
        if "rate limit" in low or "quota" in low:
            return "rate_limited"
        try:
            data = json.loads(body_text or "")
        except Exception:
            return "bad_response"
        if not data:
            return "bad_response"
        # Provider-shaped sanity checks
        if isinstance(data, dict):
            if data.get("error"):
                err = str(data.get("error")).lower()
                if "model_not_found" in err or "not found" in err:
                    return "not_found"
                if "rate" in err or "quota" in err:
                    return "rate_limited"
                return "bad_response"
            if "choices" in data or "candidates" in data or "message" in data or "result" in data or "text" in data:
                return "ok"
        return "ok"
    # Other 4xx → treat as bad_response with http code preserved.
    return "bad_response"


def do_probe(url, headers, body_bytes):
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as r:
            body = r.read().decode(errors="replace")
            return r.status, body, None, int((time.monotonic() - t0) * 1000)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            pass
        return e.code, body, None, int((time.monotonic() - t0) * 1000)
    except Exception as e:
        return None, "", e, int((time.monotonic() - t0) * 1000)


# ── Round-robin selection ─────────────────────────────────────────────────────

def bucket_for(model_id):
    h = hashlib.md5(model_id.encode()).hexdigest()
    return int(h, 16) % ROUND_ROBIN_BUCKETS


def run_index_for(now):
    # 12 runs/day, repeating every 6h → bucket = (hour // 2) % 3.
    return (now.hour // 2) % ROUND_ROBIN_BUCKETS


def load_recent_statuses(path, lookback_runs=WATCH_LIST_FAILS, max_lines=200_000):
    """Return {(provider, model): deque-of-recent-statuses (newest first)}."""
    recent = defaultdict(lambda: deque(maxlen=lookback_runs))
    if not path.exists():
        return recent
    # We want last N for each (provider,model). Read in chronological order
    # then keep only the tail per key.
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = (row.get("provider"), row.get("model"))
            recent[key].append(row.get("status"))
    # Trim each deque to lookback_runs (newest at right)
    trimmed = {}
    for k, q in recent.items():
        items = list(q)[-lookback_runs:]
        trimmed[k] = items
    return trimmed


def is_on_watch_list(recent_statuses):
    if len(recent_statuses) < WATCH_LIST_FAILS:
        return False
    return all(s != "ok" for s in recent_statuses[-WATCH_LIST_FAILS:])


# ── Aggregation ───────────────────────────────────────────────────────────────

def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def aggregate(probes_path):
    now = datetime.now(timezone.utc)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    # Per (provider, model) collect:
    #   samples_7d, ok_7d, rate_limited_7d, samples_30d, ok_30d
    #   latencies_7d (ok only), hourly_counts[24] = [ok, total]
    #   last_ts, last_status
    bucket = defaultdict(lambda: {
        "samples_7d": 0, "ok_7d": 0, "rl_7d": 0,
        "samples_30d": 0, "ok_30d": 0,
        "latencies": [],
        "hourly": [[0, 0] for _ in range(24)],
        "last_ts": None, "last_status": None,
    })

    if not probes_path.exists():
        return {}

    with probes_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            try:
                ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
            except Exception:
                continue
            if ts < cutoff_30d:
                continue
            key = (row.get("provider"), row.get("model"))
            b = bucket[key]
            status = row.get("status")
            b["samples_30d"] += 1
            if status == "ok":
                b["ok_30d"] += 1
            if ts >= cutoff_7d:
                b["samples_7d"] += 1
                if status == "ok":
                    b["ok_7d"] += 1
                    if row.get("latency_ms") is not None:
                        b["latencies"].append(row["latency_ms"])
                if status == "rate_limited":
                    b["rl_7d"] += 1
                hour = ts.hour
                b["hourly"][hour][1] += 1
                if status == "ok":
                    b["hourly"][hour][0] += 1
            if b["last_ts"] is None or ts > b["last_ts"]:
                b["last_ts"] = ts
                b["last_status"] = status

    out = defaultdict(dict)
    for (provider, model), b in bucket.items():
        if not provider or not model:
            continue
        uptime_7d = (b["ok_7d"] / b["samples_7d"]) if b["samples_7d"] else None
        uptime_30d = (b["ok_30d"] / b["samples_30d"]) if b["samples_30d"] else None
        hourly = []
        for ok_n, tot in b["hourly"]:
            hourly.append({"ok": ok_n, "total": tot})
        out[provider][model] = {
            "uptime_7d": round(uptime_7d, 4) if uptime_7d is not None else None,
            "uptime_30d": round(uptime_30d, 4) if uptime_30d is not None else None,
            "rate_limited_7d": b["rl_7d"],
            "samples_7d": b["samples_7d"],
            "samples_30d": b["samples_30d"],
            "p50_latency_ms": percentile(b["latencies"], 50),
            "p95_latency_ms": percentile(b["latencies"], 95),
            "hourly_uptime": hourly,
            "last_probe_ts": b["last_ts"].isoformat() if b["last_ts"] else None,
            "last_status": b["last_status"],
        }
    return dict(out)


# ── Rotation ─────────────────────────────────────────────────────────────────

def rotate_old(probes_path, keep_days=30):
    if not probes_path.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    keep_lines = []
    archive_lines = defaultdict(list)  # "YYYY-MM" → [lines]
    with probes_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
            except Exception:
                continue
            if ts >= cutoff:
                keep_lines.append(line)
            else:
                archive_lines[ts.strftime("%Y-%m")].append(line)
    if not archive_lines:
        return
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for ym, lines in archive_lines.items():
        path = ARCHIVE_DIR / f"{ym}.jsonl.gz"
        existing = b""
        if path.exists():
            with gzip.open(path, "rb") as g:
                existing = g.read()
        with gzip.open(path, "wb") as g:
            g.write(existing)
            g.write("".join(lines).encode())
    probes_path.write_text("".join(keep_lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not MODELS_JSON.exists():
        print("docs/models.json missing — run generate_site.py first", file=sys.stderr)
        sys.exit(1)

    models_data = json.loads(MODELS_JSON.read_text())
    providers_data = models_data.get("providers", {})

    now = datetime.now(timezone.utc)
    bucket_idx = run_index_for(now)
    print(f"Run bucket {bucket_idx}/{ROUND_ROBIN_BUCKETS} at {now.isoformat()}")

    recent = load_recent_statuses(PROBES_JSONL)

    # Build the probe targets
    targets = []  # list of (provider, model_id)
    for provider, pdata in providers_data.items():
        if provider not in PROVIDER_PROBES:
            continue
        if not os.environ.get(PROVIDER_PROBES[provider]["env"]):
            continue
        for m in pdata.get("models", []):
            mid = m.get("id")
            if not mid:
                continue
            in_bucket = bucket_for(mid) == bucket_idx
            on_watch = is_on_watch_list(recent.get((provider, mid), []))
            if in_bucket or on_watch:
                targets.append((provider, mid))

    print(f"{len(targets)} probes scheduled")

    # Group by provider to enforce per-provider concurrency.
    by_provider = defaultdict(list)
    for p, m in targets:
        by_provider[p].append(m)

    write_lock = threading.Lock()
    PROBES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    out_f = PROBES_JSONL.open("a")

    def probe_one(provider, model_id):
        cfg = PROVIDER_PROBES[provider]
        nonce = secrets.token_hex(4)
        built = build_request(provider, cfg, model_id, nonce)
        if built is None:
            return None
        url, headers, body = built
        http, body_text, exc, latency = do_probe(url, headers, body)
        status = classify(http, body_text, exc)
        err = None
        if status not in ("ok", "rate_limited") and body_text:
            err = body_text[:200]
        elif exc is not None:
            err = str(exc)[:200]
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model_id,
            "status": status,
            "http": http,
            "latency_ms": latency,
            "err": err,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with write_lock:
            out_f.write(line)
            out_f.flush()
        return row

    def run_provider(provider, model_ids):
        with ThreadPoolExecutor(max_workers=PER_PROVIDER_CONCURRENCY) as ex:
            futures = []
            for mid in model_ids:
                futures.append(ex.submit(probe_one, provider, mid))
                time.sleep(PER_PROBE_PAUSE_S)
            counts = defaultdict(int)
            for fut in futures:
                row = fut.result()
                if row:
                    counts[row["status"]] += 1
            print(f"  [{provider}] " + " ".join(f"{k}={v}" for k, v in counts.items()))

    # Run providers in parallel; each provider has its own pool.
    threads = []
    for provider, model_ids in by_provider.items():
        t = threading.Thread(target=run_provider, args=(provider, model_ids), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    out_f.close()

    # Rotate older entries off the live file.
    rotate_old(PROBES_JSONL, keep_days=30)

    # Regenerate aggregate.
    avail = aggregate(PROBES_JSONL)
    AVAIL_JSON.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(),
        "providers": avail,
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {AVAIL_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
