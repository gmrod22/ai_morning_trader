ai_morning_trader

A lightweight, opinionated morning trading bot that scans at the open, places/manage orders with Alpaca, and sends Slack notifications. It’s designed to be simple to operate, safe-by-default with a DRY_RUN toggle, and easy to run on a schedule via GitHub Actions.
	•	Brokerage: Alpaca (live or paper)
	•	Signals: Pluggable strategy module (e.g., LightGBM / rules)
	•	Notifications: Slack Incoming Webhook
	•	Scheduler: GitHub Actions (AM open, PM close, weekly summary)
	•	Capital profile: Works with small accounts ($1k–$2k); supports intraday exits

⸻

Features
	•	Morning open flow: fetch pre-open context and first prints, generate orders, place bracket OCO (TP/SL) when enabled.
	•	Intraday risk controls: position sizing cap, max concurrent orders, and optional stop/take-profit.
	•	Slack updates: executions, errors, P/L snapshots, Friday performance recap.
	•	One-switch safety: DRY_RUN=true simulates everything without placing orders.
	•	Pluggable strategy: swap out strategy/ (e.g., train_lgbm) without touching the runner.
	•	Actionable logs: CSV trade log + structured console output for quick audits.

⸻

Repository structure

ai_morning_trader/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ trade_open.py                # Morning run: fetch data → decide → (simulate|submit) orders
├─ trade_close.py               # Optional PM run: manage/flatten positions, end-of-day tasks
├─ weekly_summary.py            # Friday recap: performance summary + Slack
├─ strategy/
│  ├─ __init__.py
│  ├─ signals.py                # Signal generation helper(s)
│  └─ train_lgbm.py             # Example model training / scoring function(s)
├─ broker/
│  ├─ __init__.py
│  ├─ alpaca_client.py          # Thin wrapper around Alpaca Trading API
│  └─ data.py                   # Market data fetch (e.g., Alpaca/YFinance fallback)
├─ notifier/
│  ├─ __init__.py
│  └─ notify_slack.py           # Slack webhook helper
├─ config/
│  ├─ symbols.yaml              # Universe / watchlist
│  └─ settings.yaml             # Risk, sizing, TP/SL, thresholds
├─ utils/
│  ├─ timeutils.py              # Timezones, session windows
│  ├─ mathutils.py              # Sizing, rounding, safe arithmetic
│  └─ io.py                     # Logging (CSV), safe file ops
├─ trade_log.csv                # Appended trade/activity log (auto-created)
└─ .github/workflows/
   ├─ morning_open.yml          # 9:30 ET (14:30 UTC) run
   ├─ afternoon_close.yml       # 15:55 ET (20:55 UTC) run
   └─ friday_summary.yml        # Fri recap

Your exact filenames may differ; update this tree to match your repo.

⸻

Quick start

1) Install

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

2) Configure environment

Copy the template and fill in secrets:

cp .env.example .env

.env.example (edit as needed):

# --- Mode ---
DRY_RUN=true                 # true = simulate; false = place real orders

# --- Alpaca ---
ALPACA_KEY_ID=your_key_id
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true           # true = paper API; false = live API

# --- Slack ---
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ

# --- Behavior / Risk ---
MAX_POSITION_DOLLARS=500
MAX_CONCURRENT_POSITIONS=3
DEFAULT_TAKE_PROFIT_PCT=0.05
DEFAULT_STOP_LOSS_PCT=0.03

3) Configure strategy & symbols
	•	Edit config/symbols.yaml for your universe (e.g., ["HOOG"]).
	•	Adjust thresholds and risk in config/settings.yaml.
	•	Make sure your strategy functions return actionable signals (e.g., BUY/SELL, target qty, TP/SL).

4) Run locally

Morning open (simulated by default):

python trade_open.py

PM close (optional, rebalance/flatten):

python trade_close.py

Weekly recap:

python weekly_summary.py

Switch to live trading by editing .env:

DRY_RUN=false
ALPACA_PAPER=true   # keep true for paper, set to false for a funded live account


⸻

GitHub Actions (recommended)

This repo includes workflows to run hands-free on a schedule.
	1.	Add repo secrets in GitHub → Settings → Secrets and variables → Actions:

	•	ALPACA_KEY_ID
	•	ALPACA_SECRET_KEY
	•	SLACK_WEBHOOK_URL
	•	Optional: DRY_RUN (true/false), ALPACA_PAPER (true/false)

	2.	Check cron schedules in .github/workflows/:

	•	morning_open.yml → triggers near 9:30 AM ET to run trade_open.py
	•	afternoon_close.yml → triggers near 3:55 PM ET to run trade_close.py
	•	friday_summary.yml → triggers Fridays for weekly_summary.py

If you are not in US/Eastern, adjust crons carefully— Actions use UTC. 9:30 ET is 14:30 UTC (or 13:30 UTC during DST changes—verify each season).

⸻

How it works (high level)
	1.	Data
	•	Pulls pre-open or latest quotes (Alpaca market data preferred).
	•	Falls back to secondary sources if configured (e.g., limited use of yfinance).
	2.	Signals
	•	Calls strategy/* (e.g., train_lgbm.py) which returns per-symbol intents and sizes.
	•	Applies guardrails: min price, max allocation, max open positions, etc.
	3.	Orders
	•	In live mode, submits market or limit orders via Alpaca’s Trading API.
	•	Optional bracket (take-profit / stop-loss) using OrderClass.BRACKET.
	4.	Logging & Notify
	•	Appends to trade_log.csv and posts Slack messages for visibility.
	•	Captures errors with stack traces (posted to Slack in redacted form).
	5.	Close & Recap
	•	PM script can trim/exit positions if desired.
	•	Friday script compiles weekly P/L and sends a summary to Slack.

⸻

Configuration notes
	•	Universe: keep it small at first (e.g., 1–5 tickers) while validating behavior.
	•	Sizing: MAX_POSITION_DOLLARS and MAX_CONCURRENT_POSITIONS enforce account-appropriate risk.
	•	Stops/Targets: Defaults are in .env and/or config/settings.yaml and can be overridden per-symbol.
	•	Intraday exits: If your rules allow, the PM script can close early winners/losers to keep capital nimble.

⸻

Logs & artifacts
	•	trade_log.csv: append-only activity log with timestamp, symbol(s), decision, and order detail.
	•	GitHub Actions artifacts: optional upload of logs after each run (configure in workflows).
	•	Slack: every decision, order result, and summary is posted for quick review.

⸻

Troubleshooting
	•	“subscription does not permit querying recent SIP data”
Your market data plan may be insufficient for that endpoint. Use Alpaca’s free plan endpoints or adjust calls to use last trade/quote supported by your plan.
	•	404 ... /v2/v2/account
Double-check you’re using the correct base URL and client (paper vs live). Ensure you don’t duplicate /v2/ in your paths and that your keys match the mode.
	•	Orders not placed
Confirm DRY_RUN=false. Check Slack logs for validation failures (sizing cap, market closed, universe empty).
	•	Wrong session time
The bot uses US/Eastern by default. Ensure the container/runner timezones are handled and you’re mapping cron to UTC correctly.

⸻

Extending the bot
	•	Add a new strategy: create strategy/my_strategy.py exposing generate_signals(); wire it in trade_open.py.
	•	More risk controls: add kill-switches (max daily loss, max slippage), and position cool-downs.
	•	Data enrichment: plug in fundamentals/alt-data before market open to guide selection.

⸻

Safety & disclaimers

This code is for educational purposes. Trading involves risk; past performance does not guarantee future results. Use DRY_RUN=true until you are fully confident, and prefer paper trading while validating.

⸻

FAQ

Can I run it 24/7?
Yes via GitHub Actions schedules (recommended) or a small VPS/VM with cron.

How do I switch to paper/live?
Set ALPACA_PAPER=true for paper, false for live, and use the matching keys.

Where do I change TP/SL?
Update .env percent defaults or per-symbol overrides in config/settings.yaml.

Can it re-enter after selling?
Yes—strategy logic controls re-entry. Add cool-down rules to avoid churn.

⸻

License

MIT (or your preferred license)

⸻

Credits

Built by Grant Rodny & helpers. Uses Alpaca APIs and Slack webhooks.
