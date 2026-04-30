# Claude-based supervisor

Periodic + event-driven autonomous supervisor for the gated-lora project.
Uses Claude Code CLI Opus 4.7 (Max plan, no per-call cost) on the VPS.

## When it runs

| Trigger | How |
|---|---|
| **Hourly checkup** | `cron @hourly bash supervise.sh` |
| **Job failure** | `cron_check.sh` calls `supervise.sh --on-failure <job_id>` if a job exits without `TRAINING_DONE` |
| **Loss divergence** | `gather_state.sh` flags suspicious `final_results.json` content; supervisor decides |

## What it can do autonomously

- ✅ `sbatch` resubmit a transient-failure job
- ✅ `scancel` a diverging run (its own jobs only)
- ✅ Trigger `cron_check.sh` to free up Ensimag disk space
- ✅ Read SLURM logs (.err / .out tails) to diagnose
- ✅ Launch the next experiment from `experiments/queue.txt` if defined
- ✅ Send email alerts via `notify.py`

## What it never does

- ❌ Destructive commands (`rm -rf`, `git push --force`, dropping HF dataset)
- ❌ Spamming SSH (single connection per check)
- ❌ Running training on the VPS (no GPU)

## Files

| File | Role |
|---|---|
| `supervise.sh` | Entrypoint. Gathers state + invokes Claude CLI |
| `gather_state.sh` | One SSH call; collects `df`, `squeue`, run dirs, log tails |
| `SUPERVISOR_PROMPT.md` | The system prompt loaded by Claude |
| `notify.py` | stdout + log + optional SMTP email |

## VPS setup

```bash
# 1. Authenticate Claude Code CLI (one-time, opens browser flow)
claude /login

# 2. SMTP env (Gmail with app password example) — add to ~/.env
cat >> ~/.env <<'EOF'
GLR_SMTP_HOST=smtp.gmail.com
GLR_SMTP_PORT=587
GLR_SMTP_USER=floflog777@gmail.com
GLR_SMTP_PASSWORD=<gmail app password>
GLR_NOTIFY_TO=floflog777@gmail.com
EOF

# 3. Smoke test
bash scripts/supervisor/supervise.sh

# 4. Add cron entries
( crontab -l 2>/dev/null; cat <<'EOF'
# Hourly supervisor checkup
0 * * * * bash /home/debian/gated-lora-research-paper-001/scripts/supervisor/supervise.sh routine >> /var/log/glr-supervisor.log 2>&1
# Email-only test once a day (optional, validates SMTP is alive)
0 9 * * * /home/debian/gated-lora-research-paper-001/scripts/supervisor/notify.py --level warn --subject "Daily SMTP heartbeat" --message "If you see this, the supervisor email path works."
EOF
) | crontab -
```

## Cost

- Claude Code CLI on Opus 4.7 with **Claude Max x20** subscription = no per-call cost.
- Each supervisor invocation reads ~2-5k tokens of state + emits ~1-3k of analysis.
- 24 hourly invocations + ~5 event-triggered ones = ~30 invocations/day.

## Tuning the prompt

Edit `SUPERVISOR_PROMPT.md`. Changes take effect on next invocation (no restart needed).
