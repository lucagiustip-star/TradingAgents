# Pairs Trading (Statistical Arbitrage)

A self-contained statistical-arbitrage system: backtesting on free Yahoo Finance
data, and Alpaca **paper** trading. It is independent of the LLM agent framework
in the rest of this repository — no shared imports, no shared config.

> **Paper trading only.** There is no code path in this package that can place a
> live order. See [Safety](#safety) for the five independent checks that enforce
> this.
>
> **Every order passes a risk gate.** Position-size and exposure caps, a
> latching daily-loss circuit breaker, and a kill switch — enforced by a
> clearance token that `_submit` requires, so the layer cannot be routed around.

---

## Install

```bash
pip install -r pairs_trading/requirements.txt   # pinned, reproducible
# or, as an extra of the parent project:
pip install ".[pairs]"
```

For paper trading, create the credentials file and fill in your Alpaca **paper**
keys:

```bash
python -m pairs_trading.main --init-env
```

That writes a blank `.env` at the repository root, mode `0600`, and prints the
command to open it. It refuses to overwrite an existing `.env` — that file may
hold working credentials — and `--overwrite` takes a timestamped backup first.
Both `.env` and its backups are gitignored.

Verify without pasting a key anywhere:

```bash
python -m pairs_trading.main --check-alpaca
```

---

## Quick start

```bash
# 1. Is the pair even tradeable? Run the statistical tests and stop.
python -m pairs_trading.main --check-only --pair KO/PEP

# 2. Backtest it.
python -m pairs_trading.main --backtest --pair KO/PEP

# 3. Compare spread constructions and thresholds.
python -m pairs_trading.main --backtest --pair GOOGL/MSFT --method log_ratio --entry-z 2.5

# 4. Verify the Alpaca setup (read-only; places no orders).
python -m pairs_trading.main --check-alpaca

# 5. Check the risk limits and halt state before going anywhere near the broker.
python -m pairs_trading.main --risk-status

# 6. Paper trade — dry run first. Runs every risk check, submits nothing.
python -m pairs_trading.main --paper-trade --pair KO/PEP --dry-run
python -m pairs_trading.main --paper-trade --pair KO/PEP

# 7. Build the dashboard and open it.
python -m pairs_trading.main --dashboard --pair KO/PEP --open

# 8. Emergency stop, any time.
python -m pairs_trading.kill_switch --reason "stopping for the day"
```

An installed copy also exposes the console script `pairs-trading`.

---

## How it works

### 1. Data (`data.py`)

Daily adjusted closes from Yahoo Finance, split/dividend-adjusted
(`auto_adjust=True`). Raw closes drop on ex-dividend dates and the spread would
read those drops as divergence, firing entries on a corporate action.

The two legs are **inner-joined** on the date index. Unmatched sessions are
dropped, never forward-filled: a repeated price contributes a zero return, which
deflates the volatility the z-score divides by and so exaggerates every `|z|`.

### 2. Cointegration (`cointegration.py`) — the gate

Correlation is not the property this strategy needs. Correlation measures
co-movement of *returns*; cointegration measures whether two series stay together
in *level*. Two random walks can correlate at 0.9 and still diverge forever.

A non-cointegrated pair does not produce an error — it produces trades, and
losses. So the tests run **before** the backtest, and a failure aborts it:

| Test | Question | Failure means |
|---|---|---|
| ADF on each leg | Is each leg I(1)? | A leg is already stationary; cointegration is vacuous |
| Engle-Granger | Is the OLS residual mean-reverting? | No equilibrium to revert to |
| Johansen | What is the cointegration rank? | Rank 0 = no relationship |
| Half-life | *How fast* does it revert? | Too slow to trade on this horizon |
| Hurst | Mean-reverting or trending? | `H > 0.55` contradicts the verdict |
| Sub-period scan | Does it still hold *now*? | The relationship has already broken |

Override with `--force` only to investigate; the results carry no statistical
meaning.

### 3. Strategy (`strategy.py`)

Three spread constructions, selectable so you can compare them:

| `spread.method` | Formula | Use when |
|---|---|---|
| `price_diff` | `S = Py - Px` | Prices are similar in magnitude and stay that way |
| `log_ratio` | `S = ln(Py/Px)` | The relationship is proportional; immune to price level |
| `ols` | `S = Py - (α + β·Px)` | General case: β is measured on a rolling window, not assumed |

Then `z = (S - rolling_mean) / rolling_std`, and a state machine over `z`:

```
flat,  z <= -entry_z  ->  LONG spread   (buy y, sell x)
flat,  z >= +entry_z  ->  SHORT spread  (sell y, buy x)
long,  z >= -exit_z   ->  flat
short, z <= +exit_z   ->  flat
       |z| >= stop_z  ->  flat, and re-entry locked on that side
       held too long  ->  flat, and re-entry locked on that side
```

Two details that are easy to get wrong:

- **Exits are asymmetric** (`z >= -exit_z`, not `|z| <= exit_z`). If the spread
  gaps from −2.5 straight through the mean to +2.5, the symmetric form misses
  the exit band entirely and holds a wrong-sided position.
- **A protective exit locks re-entry** until `|z|` returns inside the exit band.
  Without it, stopping out at `z = 3.6` immediately re-enters at `z = 3.7`,
  turning one bounded loss into an unbounded run of them.

### 4. Backtest (`backtest.py`)

Cash accounting, marked to market daily. Slippage and commission are charged per
leg, per side — four legs of cost per round trip, which matters twice as much
here as in a directional strategy.

Reports Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, average
holding period, exposure, and a breakdown of exits by reason. `--trades` prints
the full round-trip ledger. The chart stacks rebased prices, the z-score with
every fill marked, and cumulative P&L with drawdown.

**No look-ahead:** signals are shifted forward by `signal.execution_lag` bars
(default 1), so a signal computed at today's close fills at tomorrow's. The
rolling hedge ratio and z-score use only trailing windows — both properties are
asserted in the test suite.

### 5. Risk controls (`risk_manager.py`, `kill_switch.py`)

Every order passes through the risk layer before reaching Alpaca. The interlock
is structural, not conventional: `validate_order` returns a **clearance token**
that fingerprints the exact orders it approved, and `_submit` refuses to act
without one that covers the order in its hand. A future code path that forgets
to call the risk manager gets a `RiskError`, not a fill.

Validation is **atomic over the whole pair**. Clearing leg one, filling it, then
rejecting leg two on an exposure cap would leave a naked directional position —
so clearance covers both legs or neither.

**Pre-trade checklist** (all must pass):

| Check | Applies to | On failure |
|---|---|---|
| `TRADING_HALTED` flag absent | entries | `TradingHaltedError` |
| Market open | all | reject |
| Daily loss limit not breached | entries | reject + trip breaker |
| Notional ≤ `max_position_size_usd` | entries | reject (never resize) |
| Exposure + order ≤ `max_total_exposure_usd` | entries | reject |
| Sufficient buying power | all | reject |

Exits deliberately skip the halt, loss and exposure checks. Those exist to stop
the book *growing*; applying them to closing orders would block the very trades
that shrink it, trapping a position exactly when you most need out.

**Circuit breaker.** Daily P&L is `equity − start_of_day_equity`; since Alpaca's
equity already marks open positions to market, that one number covers realised
*and* unrealised. The baseline is persisted, so a mid-session restart does not
reset the measurement. On breach it logs, alerts, optionally flattens
(`close_positions_on_breach`), and **latches** — it writes the same
`TRADING_HALTED` file the kill switch uses, so trading cannot resume when the
date rolls over and today's loss resets to zero. Clearing it is a manual act.

**Kill switch.**

```bash
python -m pairs_trading.kill_switch                    # cancel, flatten, halt
python -m pairs_trading.kill_switch --status
python -m pairs_trading.kill_switch --clear            # manual resume
python -m pairs_trading.kill_switch --halt-only        # stop without flattening
python -m pairs_trading.kill_switch --dry-run          # drill; still halts
```

It writes the halt flag **before** any broker call, so a network failure still
leaves the system stopped. Broker errors are collected rather than raised — one
symbol that will not close must not prevent the others from closing.

**Alerting** is transport-agnostic: `AlertChannel` subclasses plug into an
`AlertDispatcher`. Email over SMTP ships; SMS or Slack is one subclass with no
change to calling code. Channel failures are swallowed and logged — a dead mail
server must never stop a halt.

Everything lands in `logs/risk_events.csv` with full account context at the
moment of the decision.

### 6. Dashboard (`dashboard.py`)

```bash
python -m pairs_trading.main --dashboard --pair KO/PEP --open
python -m pairs_trading.main --dashboard --pair KO/PEP --live   # + paper account state
python -m pairs_trading.dashboard --no-backtest                 # logs and risk state only
```

Writes one self-contained HTML file (`logs/dashboard.html`) — all CSS, JS and
chart geometry inlined, so it opens from disk, survives being emailed, and makes
no external request. Nothing about what you trade reaches a CDN.

It shows, in scanning order: halt state, headline metrics, risk-limit usage
meters, cumulative P&L, drawdown, the recent z-score window with every fill
marked, then the trade and risk-event logs.

Charts are hand-built SVG with a crosshair-and-tooltip hover layer. Two rules
they follow that are easy to get wrong:

- **One y-axis per chart.** Drawdown gets its own panel rather than a second
  scale on the P&L chart. Two scales in one frame invite the reader to read
  meaning into where the curves cross, and those crossings are an artefact of
  the independent scalings.
- **The z-score panel is windowed** to the last 250 bars. Compressing a
  multi-year history into one frame renders the line as a solid block and stacks
  every trade marker on top of the last, misrepresenting how often the signal
  actually fires.

Every section has an empty state, so the page is useful on a fresh install, and
it still renders when the data vendor or broker is unreachable — which is
exactly when you want to look at it.

### 7. News guard (`news_guard.py`)

```bash
python -m pairs_trading.main --news-check --pair KO/PEP
```

A **veto, never a signal**: it can stop the strategy trading, and can never tell
it what to trade. The strategy decides on daily bars with a one-bar lag, so a
headline cannot usefully inform *direction* — there is no path from news to a
better z-score. What news can do is reveal that the spread's equilibrium has
structurally changed, which the statistics only discover later, after the loss.

The distinction that makes it useful rather than an off-switch:

| Event | Verdict | Why |
|---|---|---|
| Merger, acquisition, take-private, spin-off | **halt** | The pair stops being two independent companies |
| Bankruptcy, delisting, trading suspension | **halt** | One leg stops being continuously tradeable |
| Restatement, auditor resignation, SEC probe | **halt** | Reported fundamentals stop being trustworthy |
| Index add/remove, CEO exit, antitrust | warn | Real flow effects, relationship survives |
| Earnings, guidance, upgrades, dividends | ignore | **This is the divergence the strategy trades** |

Halting on earnings would defeat the strategy, so it doesn't. Only structural
events halt, and they write the same `TRADING_HALTED` flag the circuit breaker
and kill switch use — one way to resume, not three.

Runs automatically before each paper-trading session (`news.enabled`), **before**
the cointegration test: a merger announced yesterday ends the relationship today,
but the statistics are computed from historical prices and will happily still
pass.

Uses Alpaca's news API, so it needs **no credentials beyond the paper keys you
already have**. Classification is rule-based — deterministic, inspectable, and
testable without a model in the loop; `NewsGuard.classify` is the single
extension point if you later want an LLM to read ambiguous cases.

Two deliberate biases, both documented in the module: it **fails open** when the
feed is unreachable (a veto that halts on its own outage hands the news vendor an
off-switch, and missing news is not evidence), and it **over-triggers rather than
under-triggers** on ambiguous deal language, because a false positive costs a
pause you clear in one command while a false negative costs money for as long as
the spread keeps not reverting.

### 8. Execution (`execution_alpaca.py`)

`sync_to_signal` is a **reconciler**, not an order generator: it reads the target
and the broker's actual position and issues only the difference. Running it twice
is a no-op the second time, so a double-fired cron job cannot double a position.

Quantities are floored to whole shares (Alpaca cannot short fractional shares).
If either leg floors to zero, the whole pair aborts rather than sending one leg —
an unhedged single leg is a naked directional bet.

---

## Safety

### Risk limits

All limits live in `config.yaml` under `risk:` and are overridable per run
(`--max-position-size`, `--max-total-exposure`, `--max-daily-loss`).
`--risk-status` prints the current limits and halt state without touching the
broker. `--dry-run` runs the full pipeline including every risk check and logs
`WOULD HAVE PLACED ORDER` instead of calling Alpaca — the way to exercise the
risk layer in isolation.

Exit codes: `3` halted, `4` circuit breaker tripped, `5` order rejected by risk,
`6` setup check found only non-blocking issues.

### Connecting to Alpaca

```bash
python -m pairs_trading.main --check-alpaca
```

A read-only diagnostic that walks the setup in dependency order and stops at the
first blocking problem, so the output names the one thing to fix rather than
cascading a missing credential into six confusing errors. It checks the SDK,
credentials (masked in output, never printed), that the endpoint is the paper
one, connectivity, account health, **that the account can sell short** — a cash
account cannot, and every pairs trade shorts one leg — buying power against your
configured trade size, the market clock, and the halt flag. It places no orders.

### Paper-trading guarantee

Five independent checks, each sufficient on its own:

1. `ExecutionConfig` rejects a `base_url` without `paper`, at config-load time.
2. `assert_paper_endpoint` re-checks immediately before the client is built,
   comparing the **parsed hostname** — `api.alpaca.markets` is a substring of
   `paper-api.alpaca.markets`, so naive substring matching fails in both
   directions.
3. `TradingClient(..., paper=True)` — a literal, not a variable, not an argument.
4. The URL the SDK actually resolved is read back off the client and re-checked.
5. The account is fetched and verified active and unblocked before any order.

`tests/test_pairs_trading.py` additionally asserts by source inspection that no
file in this package contains `paper=False` or references the SDK's live-endpoint
enum.

---

## Configuration

Everything tunable lives in `config.yaml` — no thresholds or windows are
hardcoded in the strategy. Precedence:

```
CLI flag  >  environment variable  >  config.yaml  >  built-in default
```

Unknown keys are rejected rather than ignored: a silently ignored typo is the
worst kind of config bug, because the run succeeds using a parameter you thought
you had changed.

`main.py` also prints **advisories** — combinations that run but whose results
are likely to mislead (zero execution lag, a static full-sample hedge ratio,
zero costs, a half-life longer than the z-score window).

### Logs

Two CSVs, deliberately separate. `trades.csv` records what the account **did**;
`risk_events.csv` records what it was **stopped from doing**. The second is the
one you read after a bad day.

### Trade log

Every fill, backtest and paper alike, appends to `logs/trades.csv` with the same
schema, so a paper session can be diffed against the backtest that justified it:

```
timestamp, action, ticker, quantity, price, zscore,
mode, pair, side, reason, spread, hedge_ratio, notional, commission, group_id, order_id
```

The two legs of one trade share a `group_id`.

---

## Tests

```bash
python -m pytest tests/test_pairs_trading.py tests/test_risk_manager.py -q
```

140 tests, fully offline — no network, no API keys. Synthetic series are built
with a known half-life (`-ln(2)/ln(φ)` for an AR(1) with coefficient φ), so the
diagnostics are checked against analytically known values rather than against
themselves.

---

## Known limitations

- **Short costs are not modelled.** Short proceeds are credited in full with no
  borrow fee or margin interest. Real short books pay both, and for
  hard-to-borrow names the borrow rate can exceed the spread's entire edge.
- **Yahoo data is not survivorship-bias free** and is unsuitable for screening
  across a delisted universe.
- **In-sample parameter selection.** Tuning thresholds on the same window you
  evaluate on will overfit. Hold out a period.
- **Cointegration is not stationary in time.** The sub-period scan flags a broken
  relationship, but passing it is not a guarantee it holds tomorrow.
- **Whole-share rounding** makes small-notional live positions imperfectly
  dollar-neutral in a way the backtest (fractional shares) does not model.
