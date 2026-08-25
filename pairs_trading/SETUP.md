# Where do I actually run this?

**There is no website that runs this.** It is a program, not a web app — there
is nothing to log into and no dashboard hosted on the internet. You run it on a
computer, and it prints results in a terminal.

Two websites *are* involved, but neither of them runs the code:

| Site | What it's for |
|---|---|
| [app.alpaca.markets](https://app.alpaca.markets) | Your paper account. Get API keys here, and see the trades the program places. |
| [github.com/lucagiustip-star/TradingAgents](https://github.com/lucagiustip-star/TradingAgents) | Where the code is stored. |

You have two ways to run it. Pick one.

---

## Option A — In your browser, nothing to install (easiest)

**GitHub Codespaces** gives you a full development machine in a browser tab. It
is free for personal use within a monthly quota. This repository is already
configured for it, so everything installs itself.

1. Go to the repository:
   **https://github.com/lucagiustip-star/TradingAgents**
2. Click the green **Code** button → **Codespaces** tab → **Create codespace on
   `claude/pairs-trading-system-ymt44t`**.
3. Wait a minute or two. A code editor opens in your browser and installs
   everything automatically. When it says `Ready.` you can start.
4. Find the **terminal** panel at the bottom. That is where you type commands.

Then add your keys. In a Codespace, use **Codespaces secrets** rather than a
file — they survive rebuilds, and a Codespace is disposable:

1. On GitHub: repo **Settings** → **Secrets and variables** → **Codespaces**
2. **New repository secret**, twice:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
3. Rebuild the Codespace (Command Palette → *Codespaces: Rebuild Container*),
   or just stop and restart it.

Check it worked:

```bash
python -m pairs_trading.main --check-alpaca
```

---

## Option B — On your own computer

You need **Python 3.10 or newer** and a terminal. The terminal is called:

- **Windows** — "PowerShell" or "Terminal" (press Start, type `powershell`)
- **macOS** — "Terminal" (press ⌘+Space, type `terminal`)
- **Linux** — you already know

Check whether you have Python:

```bash
python3 --version
```

If that errors, install it from [python.org/downloads](https://www.python.org/downloads/).
On Windows, tick **"Add Python to PATH"** during installation.

Then:

```bash
git clone https://github.com/lucagiustip-star/TradingAgents.git
cd TradingAgents
git checkout claude/pairs-trading-system-ymt44t
pip install -r pairs_trading/requirements.txt
python -m pairs_trading.main --init-env
```

That last command creates the file for your keys and tells you how to open it.
Paste your Alpaca paper keys in, save, then:

```bash
python -m pairs_trading.main --check-alpaca
```

---

## Getting your Alpaca keys

1. Sign up at [alpaca.markets](https://alpaca.markets) — free, no deposit needed
   for paper trading.
2. On the dashboard, find the **Paper / Live** toggle and make sure it says
   **Paper**. This is the step people get wrong: paper and live keys are
   different credentials, and live keys will not work here.
3. **API Keys** → **Generate New Key**.
4. You get two values. The **secret is shown once** — copy it before closing the
   dialog. If you lose it, generate a new key; you cannot retrieve the old one.

Never paste these into a chat, an email, or a screenshot. If one is ever
exposed, revoke it at Alpaca and generate a new one — it takes seconds.

---

## Then what?

```bash
# Is the pair even tradeable?
python -m pairs_trading.main --check-only --pair KO/PEP

# What would it have made historically?
python -m pairs_trading.main --backtest --pair KO/PEP

# Any news that breaks the pair? (read-only, no orders)
python -m pairs_trading.main --news-check --pair KO/PEP

# Full pipeline including risk checks, but places NO orders
python -m pairs_trading.main --paper-trade --pair KO/PEP --dry-run

# For real (on the paper account)
python -m pairs_trading.main --paper-trade --pair KO/PEP

# See everything on one page
python -m pairs_trading.main --dashboard --pair KO/PEP --live
```

The dashboard writes `logs/dashboard.html`. In a Codespace, right-click that
file in the sidebar and choose **Download** to open it locally; on your own
machine, add `--open` and it opens by itself.

Panic button, at any time:

```bash
python -m pairs_trading.kill_switch --reason "stopping"
```

---

## If something goes wrong

Run the diagnostic first — it names the one thing to fix and masks your keys, so
its output is safe to share:

```bash
python -m pairs_trading.main --check-alpaca
```

| It says | What to do |
|---|---|
| `No .env file was found` | Run `--init-env`. On Windows check the file isn't `.env.txt`. |
| `Found a .env, but it does not set…` | The file exists but a line is wrong. No quotes, no spaces around `=`. |
| Credentials rejected | You used live keys. Regenerate from the **Paper** side. |
| `Shorting enabled: FAIL` | Your paper account is a cash account. Every pairs trade shorts one leg, so reset it as a **margin** account. |
| `Market clock: market is CLOSED` | Not a setup problem. Orders are refused outside market hours by design. |
