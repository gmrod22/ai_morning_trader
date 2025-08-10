
AI Morning Trader

An automated Alpaca-based trading bot that uses LightGBM machine learning to rank stocks at market open, place bracket orders with take-profit and stop-loss levels, and manage positions through the day using a hybrid close logic that can selectively hold winners overnight.

Built for hands-off daily execution with Slack notifications, P&L tracking, and optional GitHub Actions automation.

⸻

Features
	•	📈 Machine Learning Ranking — LightGBM model ranks tickers daily based on recent performance, volatility, and trend indicators
	•	🛑 Bracket Orders — Automatic stop-loss and take-profit protection
	•	💤 Hybrid Close Logic — Option to hold strong performers overnight
	•	📊 Performance Logging — Tracks trades, P&L, and daily returns in trade_log.csv
	•	🔔 Slack Notifications — Real-time updates for trade signals, executions, and end-of-day summaries
	•	⚙️ Fully Configurable via config.yaml
	•	🤖 Optional Automation with GitHub Actions — runs at open and close without manual intervention

⸻

Workflow Diagram

flowchart TD
    A[Market Open] --> B[Fetch Historical Prices from Yahoo Finance]
    B --> C[Compute Features: returns, volatility, SMA ratios]
    C --> D[Train LightGBM Model]
    D --> E[Predict Next-Day Returns for Each Ticker]
    E --> F[Rank by Predicted Return]
    F --> G[Build Bracket Orders (stop-loss & take-profit)]
    G --> H[Submit Orders via Alpaca API]
    H --> I[Log Trades in trade_log.csv]
    I --> J[Send Slack Morning Notification]
    J --> K[Market Trading Day]
    K --> L[Market Close Trigger]
    L --> M[Fetch Positions & P&L]
    M --> N[Hybrid Close Logic: Hold or Close]
    N --> O[Update trade_log.csv with P&L]
    O --> P[Send Slack Close Summary]


⸻

Project Structure

ai_morning_trader/
│
├── trade_open.py        # Market open logic — builds orders & submits trades
├── trade_close.py       # Market close logic — hybrid close or flatten positions
├── strategy.py          # LightGBM model training & predictions
├── notifier.py          # Slack notification helper
├── config.yaml          # Trading configuration
├── trade_log.csv        # Auto-generated log of trades & P&L
└── .github/workflows/   # GitHub Actions automation


⸻

Setup

1. Clone the repo

git clone https://github.com/YOUR_USERNAME/ai_morning_trader.git
cd ai_morning_trader

2. Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Set environment variables

For local runs:

export APCA_API_KEY_ID="your_api_key"
export APCA_API_SECRET_KEY="your_secret_key"
export APCA_PAPER="true"        # true = paper trading, false = live trading
export DRY_RUN="true"           # true = simulate, false = submit real orders
export SLACK_WEBHOOK_URL="your_slack_webhook"

For GitHub Actions, set the same variables as Secrets in your repository settings.

⸻

Configuration (config.yaml)

Example:

tickers: ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"]
top_n: 3
per_trade_budget: 500
stop_atr_mult: 1.5
take_atr_mult: 3.0
atr_lookback: 14
dry_run: true

hold:
  enabled: true
  max_overnight_positions: 3
  max_overnight_notional: 2000
  min_today_ret: 0.002
  require_sma_trend: true
  skip_friday: true
  stop_atr_mult: 1.2
  atr_lookback: 14


⸻

Trading Logic

Morning (trade_open.py)
	1.	Pull historical prices from Yahoo Finance
	2.	Compute features:
	•	1-day & 5-day returns
	•	Volatility (5-day std dev)
	•	SMA(5) & SMA(20) relative position
	3.	Train LightGBM on all but the latest day
	4.	Predict next-day returns for each ticker
	5.	Rank by predicted return
	6.	Place bracket orders for top N:
	•	Stop-loss at stop_atr_mult × ATR
	•	Take-profit at take_atr_mult × ATR
	7.	Log trades and send Slack notification

⸻

Close (trade_close.py)
	1.	Pull open positions
	2.	Log unrealized P&L into trade_log.csv
	3.	Hybrid close logic:
	•	If hold.enabled = true, keep positions meeting criteria:
	•	Positive momentum (min_today_ret)
	•	SMA trend up (if enabled)
	•	Unrealized P&L ≥ 0
	•	Under position & notional limits
	•	Skip Friday holds (if enabled)
	4.	Close all other positions
	5.	Send Slack summary

⸻

Logging

trade_log.csv fields:
	•	date
	•	dry_run
	•	n_orders
	•	symbols
	•	details (stop/take)
	•	pnl
	•	daily_return

Example:

date,dry_run,n_orders,symbols,details,pnl,daily_return
2025-08-11,False,3,AAPL,qty=5 stop=170.20 take=180.50,125.50,0.021


⸻

Slack Notifications

Example morning message:

DRY_RUN=False | Top 3 orders for today:
• AAPL: qty 5 | ref 175.00 | stop 170.20 | take 180.50
• MSFT: qty 3 | ref 320.10 | stop 315.00 | take 330.00
• NVDA: qty 2 | ref 450.50 | stop 440.00 | take 470.00

Example close message:

Close script finished: Held 2 positions overnight (AAPL, MSFT), closed 1 (NVDA).
Today's P&L snapshot: $125.50


⸻

Automation with GitHub Actions

Two workflows run daily:
	•	.github/workflows/trade_open.yml — triggers at market open
	•	.github/workflows/trade_close.yml — triggers before market close

These workflows:
	•	Pull the latest code
	•	Install dependencies
	•	Run the scripts with your API keys & config

⸻

Switching Between Paper & Live Trading
	•	Paper trading (safe, no real money):

export APCA_PAPER="true"


	•	Live trading (real money):

export APCA_PAPER="false"

Make sure your API key/secret is for a live Alpaca account.

⸻

Disclaimer

⚠️ This bot is for educational purposes. Live trading involves risk. Use at your own discretion.


