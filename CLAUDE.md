# Project conventions

## Workflow

- **Always merge and deploy.** When a PR is created for this repo, do not stop at the draft stage — flip it to ready-for-review, merge it (squash), and let the deploy workflow run. Do not ask "should I merge?" for routine changes; just ship.
- The site is built by `update-models.yml` (cron hourly + on-push when `generate_site.py` changes) and deployed via GitHub Pages from `docs/`.
- Probe data comes from `probe_models.py` via `probe.yml` (cron every 30 min).
