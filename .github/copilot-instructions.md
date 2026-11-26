# Catanthropic: shared rules for every Copilot agent

Passion project: reinforcement learning for Catan, ending in an LLM post-trained (LoRA, GRPO/PPO, single GPU) to play better. Non-commercial, GPL-3.0. Owner: Yash Soni.

## Hard rules (never break)
- `v1/` is archived and read-only. Never modify it. Before finishing any task, `git status` must show nothing under `v1/`.
- The engine fork at `~/Desktop/catanatron-fork` (branch `rng-streams`) is untouched except for planned, explicitly requested patches.
- Never reuse reserved seed ranges: eval 0-124 (training evals 0-62); BC dataset 200000-202999; PPO training from 1,000,000; fresh final eval 5000-5249 (already used to pick checkpoints); B2 validation 6000-6249 and 7000-7099. New experiments need a NEW range, recorded in `docs/baselines.md` before use.
- Confidence intervals: seed-clustered bootstrap only. Never Wilson (ignores within-seed correlation).
- Eval protocol: seed x 4 cyclic rotations, each agent plays each seat once per seed, paired dice/board/deck.
- No information leaks: observations are per-viewer. The determinizer and belief features must never expose dev-buy or steal identities. Any change to `env/`, `belief/` or feature extraction needs a leak test.
- Python hash-seed nondeterminism is fixed and test-covered. Do not introduce hash-order-dependent iteration.
- Do not suggest or attempt to drive VS Code through computer use. It cannot type into it.

## Environment
- Repo `~/Desktop/catanthropic`, Python 3.12 venv `.venv-rl`. Packages: `catan_rl/{belief,env,eval,winprob,policy,rl,agents}`, `docs/`, `checkpoints/` (git-ignored), `runs/` (git-ignored).
- Use sklearn HistGradientBoosting (LightGBM needs libomp, not installed).
- Tests: run the full suite before any commit (95 passing at last report). Never commit with failing tests.
- Long jobs (more than a couple of minutes): launch them yourself via `scripts/notify_run.sh <name> -- <cmd>`. It runs the command detached with `caffeinate -dims nohup`, logs to `runs/<name>/stdout.log`, and pushes a phone notification (via a Telegram bot, env vars `TG_BOT_TOKEN`/`TG_CHAT_ID`) on start and on completion/failure with a log tail. Never run a long job as a plain foreground terminal command that blocks the chat. Mac plugged in, nothing else heavy running, 8 workers on the 10-core Mac. Quick commands (tests, short evals, git, builds) can run directly and don't need the notifier.

## Working style
- Gate-based: every experiment has explicit pass/fail thresholds and a stated action for each outcome, written BEFORE the run.
- Be honest about noise. Report clustered CIs, flag selection bias (best-of-N on the same seeds), never oversell.
- Reports back to Yash: short, numbers first, then verdict against the gate, then the next action.
- Any drafted prose for Yash (emails, messages): no em dashes, natural tone.
- Record every gate number in `docs/baselines.md`. Commit checkpoints' sha256 in docs, not the weights.
- Commit messages: imperative, one line summary, body lists what and why.

## Agent roster (see `.github/agents/`)
- `pm`: plans, writes specs and gates, owns roadmap and decisions. Never edits code.
- `dev`: implements one spec at a time, with tests.
- `reviewer`: audits diffs against the hard rules before merge.
- `experiment-runner`: turns a spec into exact run commands, then collects outputs.
- `analyst`: turns raw eval output into gate verdicts with honest statistics.
- `scribe`: keeps docs and the handoff current.
- `llm-stage`: designs and builds the LLM post-training stage (roadmap item 5, later).
