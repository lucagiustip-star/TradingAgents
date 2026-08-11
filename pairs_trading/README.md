# Pairs Trading (Statistical Arbitrage)

A self-contained statistical-arbitrage system: backtesting on free Yahoo Finance
data, and Alpaca **paper** trading. It is independent of the LLM agent framework
in the rest of this repository — no shared imports, no shared config.

> **Paper trading only.** There is no code path in this package that can place a
> live order. See [Safety](#safety) for the five independent checks that enforce
> this.

---

## Install

```bash
pip install -r pairs_trading/requirements.txt   # pinned, reproducible
# or, as an extra of the parent project:
pip install ".[pairs]"
```

For paper trading, copy the env template to the **repository root** and fill in
your Alpaca paper keys:

```bash
cp pairs_trading/.env.example .env
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

# 4. Paper trade — dry run first, which computes orders but submits nothing.
python -m pairs_trading.main --paper-trade --pair KO/PEP --dry-run
python -m pairs_trading.main --paper-trade --pair KO/PEP
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

### 5. Execution (`execution_alpaca.py`)

`sync_to_signal` is a **reconciler**, not an order generator: it reads the target
and the broker's actual position and issues only the difference. Running it twice
is a no-op the second time, so a double-fired cron job cannot double a position.

Quantities are floored to whole shares (Alpaca cannot short fractional shares).
If either leg floors to zero, the whole pair aborts rather than sending one leg —
an unhedged single leg is a naked directional bet.

---

## Safety

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
python -m pytest tests/test_pairs_trading.py -q
```

90 tests, fully offline — no network, no API keys. Synthetic series are built
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
