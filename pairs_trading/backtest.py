"""Event-driven backtest of the pairs strategy, with performance metrics and charts.

ACCOUNTING MODEL
----------------
Cash accounting, marked to market every bar:

* Buying ``q`` shares at fill ``f``: ``cash -= q*f``, and ``shares += q``.
* Selling (including selling short): ``cash += q*f``, ``shares -= q``.
* ``equity_t = cash_t + shares_y * P_y,t + shares_x * P_x,t``

Short proceeds are credited to cash and short positions carry no borrow fee or
margin interest. That is the conventional simplification for a daily-bar equity
backtest, and it is optimistic: a real short book pays a borrow rate, which for
hard-to-borrow names can exceed the spread's entire edge.

Frictions that *are* modelled, per leg and per side:

* **Slippage** as an adverse fill -- buys fill ``slippage_bps`` above the close,
  sells the same distance below it.
* **Commission** as ``commission_bps`` of traded notional.

Both are charged on entry and on exit, so a round trip pays four legs of cost.
This matters more in pairs trading than in directional strategies: every signal
trades two instruments, so the cost per unit of signal is double.

POSITION SIZING
---------------
``dollar_neutral`` commits ``gross_exposure_per_leg`` dollars to each leg, so
the position is market-neutral in dollar terms at entry:
``q_y = N / P_y`` and ``q_x = N / P_x``.

``beta_neutral`` instead sets ``q_x = beta * q_y``, which neutralises exposure to
the *spread's* regression relationship rather than to dollars. When ``beta`` is
far from ``P_y/P_x`` the two differ materially, and beta-neutral is the more
faithful hedge of the thing actually being traded.

Share counts are fixed at entry and held constant until exit -- the position is
not rebalanced as prices move, matching what the execution module does live.

LOOK-AHEAD
----------
The simulation consumes ``signals.position``, which ``strategy.py`` has already
shifted by ``config.signal.execution_lag`` bars. Fills happen at the close of the
bar on which the position changes, using only prices from that bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, resolve_path
from .data import PairData
from .strategy import EXIT_EOD, SignalFrame
from .trade_log import TradeLogger, TradeRecord, make_group_id

logger = logging.getLogger(__name__)

BPS = 1e-4


class BacktestError(RuntimeError):
    """Raised when a backtest cannot be run or produced no usable result."""


@dataclass
class Trade:
    """One completed round trip in the spread (both legs, entry to exit)."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int              # +1 long spread, -1 short spread
    entry_zscore: float
    exit_zscore: float
    entry_price_y: float
    entry_price_x: float
    exit_price_y: float
    exit_price_x: float
    quantity_y: float
    quantity_x: float
    gross_pnl: float
    costs: float
    net_pnl: float
    bars_held: int
    exit_reason: str
    hedge_ratio: float

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def notional(self) -> float:
        """Gross capital committed at entry, both legs."""
        return abs(self.quantity_y * self.entry_price_y) + abs(self.quantity_x * self.entry_price_x)

    @property
    def return_on_notional(self) -> float:
        return self.net_pnl / self.notional if self.notional else 0.0

    @property
    def direction_label(self) -> str:
        return "long spread" if self.direction > 0 else "short spread"


@dataclass
class Metrics:
    """Performance statistics for a completed backtest."""

    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_date: pd.Timestamp | None
    calmar: float
    volatility: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_bars_held: float
    exposure: float
    total_costs: float
    gross_pnl: float
    net_pnl: float
    exit_reasons: dict[str, int] = field(default_factory=dict)

    def render(self, pair: str, start: str, end: str, n_bars: int) -> str:
        """Format the metrics as a console summary block."""
        width = 74
        pct = lambda v: f"{v:>10.2%}"  # noqa: E731
        usd = lambda v: f"{v:>10,.2f}"  # noqa: E731

        lines = [
            "=" * width,
            f" BACKTEST RESULTS  --  {pair}",
            "=" * width,
            f" Period        : {start} -> {end}  ({n_bars} bars)",
            "",
            " Returns",
            " " + "-" * (width - 2),
            f"  Initial capital        {usd(self.initial_capital)}",
            f"  Final equity           {usd(self.final_equity)}",
            f"  Net P&L                {usd(self.net_pnl)}",
            f"  Total return           {pct(self.total_return)}",
            f"  CAGR                   {pct(self.cagr)}",
            "",
            " Risk",
            " " + "-" * (width - 2),
            f"  Sharpe ratio           {self.sharpe:>10.3f}",
            f"  Sortino ratio          {self.sortino:>10.3f}",
            f"  Annualised volatility  {pct(self.volatility)}",
            f"  Max drawdown           {pct(-self.max_drawdown)}"
            + (f"   (trough {self.max_drawdown_date.date()})" if self.max_drawdown_date is not None else ""),
            f"  Calmar ratio           {self.calmar:>10.3f}",
            "",
            " Trading",
            " " + "-" * (width - 2),
            f"  Round trips            {self.n_trades:>10d}",
            f"  Wins / losses          {self.n_wins:>10d} / {self.n_losses}",
            f"  Win rate               {pct(self.win_rate)}",
            f"  Average win            {usd(self.avg_win)}",
            f"  Average loss           {usd(self.avg_loss)}",
            f"  Profit factor          {self.profit_factor:>10.3f}",
            f"  Average holding period {self.avg_bars_held:>10.1f} bars",
            f"  Time in market         {pct(self.exposure)}",
            f"  Total costs            {usd(self.total_costs)}",
            f"  Gross P&L              {usd(self.gross_pnl)}",
        ]
        if self.exit_reasons:
            lines += ["", " Exits by reason", " " + "-" * (width - 2)]
            for reason, count in sorted(self.exit_reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {reason:<24} {count:>10d}")
        lines.append("=" * width)
        return "\n".join(lines)


@dataclass
class BacktestResult:
    """Everything a completed run produced."""

    equity: pd.Series
    returns: pd.Series
    trades: list[Trade]
    metrics: Metrics
    signals: SignalFrame
    config: Config
    pair: str
    trade_log_path: Path | None = None

    @property
    def trades_frame(self) -> pd.DataFrame:
        """Round trips as a DataFrame, one row per trade."""
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "direction": t.direction_label,
                    "entry_z": t.entry_zscore,
                    "exit_z": t.exit_zscore,
                    "qty_y": t.quantity_y,
                    "qty_x": t.quantity_x,
                    "gross_pnl": t.gross_pnl,
                    "costs": t.costs,
                    "net_pnl": t.net_pnl,
                    "return": t.return_on_notional,
                    "bars_held": t.bars_held,
                    "exit_reason": t.exit_reason,
                }
            )
        return pd.DataFrame(rows).set_index("entry_date")

    def summary(self) -> str:
        return self.metrics.render(
            self.pair,
            str(self.equity.index[0].date()),
            str(self.equity.index[-1].date()),
            len(self.equity),
        )


def _fill_price(price: float, is_buy: bool, slippage_bps: float) -> float:
    """Apply adverse slippage to a close price.

    Buys fill above the close and sells below it, both by ``slippage_bps``. This
    is a crude but unbiased model of paying the spread and some market impact;
    it never works in the strategy's favour, which is the property that matters
    for not fooling yourself.
    """
    adjustment = 1.0 + (slippage_bps * BPS if is_buy else -slippage_bps * BPS)
    return price * adjustment


def _position_sizes(
    price_y: float, price_x: float, hedge_ratio: float, config: Config
) -> tuple[float, float]:
    """Compute share quantities for one leg each, per the configured sizing rule.

    Returns:
        ``(qty_y, qty_x)``, both positive. The caller applies signs from the
        position direction.
    """
    notional = config.backtest.gross_exposure_per_leg
    qty_y = notional / price_y

    if config.backtest.sizing == "beta_neutral":
        beta = abs(hedge_ratio)
        if not np.isfinite(beta) or beta <= 0:
            # A degenerate or unavailable beta would size the hedge leg at zero
            # (or infinity); fall back to dollar-neutral for this entry.
            logger.warning(
                "Hedge ratio %.4f is unusable for beta-neutral sizing; "
                "falling back to dollar-neutral for this entry.", hedge_ratio,
            )
            return qty_y, notional / price_x
        return qty_y, beta * qty_y

    return qty_y, notional / price_x


def _compute_metrics(
    equity: pd.Series, trades: list[Trade], exposure: float, config: Config
) -> Metrics:
    """Derive the performance statistics from the equity curve and trade list.

    Sharpe ratio::

        sharpe = (mean(r_daily) - rf_daily) / std(r_daily) * sqrt(periods_per_year)

    where ``r_daily`` are simple returns of the equity curve. The annualisation
    by ``sqrt(252)`` assumes returns are serially uncorrelated; a mean-reverting
    strategy's returns often are not, so treat the figure as comparative rather
    than absolute.

    Sortino replaces the denominator with the standard deviation of *negative*
    returns only, on the argument that upside dispersion is not risk.

    Max drawdown is the largest peak-to-trough decline of the equity curve::

        dd_t = equity_t / cummax(equity)_t - 1
    """
    bt = config.backtest
    periods = bt.trading_days_per_year
    returns = equity.pct_change().dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max(len(equity) / periods, 1e-9)
    # Guard against a wiped-out account, where the fractional power is complex.
    ratio = float(equity.iloc[-1] / equity.iloc[0])
    cagr = float(ratio ** (1.0 / years) - 1.0) if ratio > 0 else -1.0

    if len(returns) > 1 and returns.std() > 0:
        rf_daily = bt.risk_free_rate / periods
        excess = returns - rf_daily
        sharpe = float(excess.mean() / returns.std() * np.sqrt(periods))
        volatility = float(returns.std() * np.sqrt(periods))
        downside = returns[returns < 0]
        sortino = (
            float(excess.mean() / downside.std() * np.sqrt(periods))
            if len(downside) > 1 and downside.std() > 0
            else float("inf") if excess.mean() > 0 else 0.0
        )
    else:
        sharpe = sortino = volatility = 0.0

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(-drawdown.min()) if len(drawdown) else 0.0
    max_dd_date = drawdown.idxmin() if len(drawdown) and max_dd > 0 else None
    calmar = float(cagr / max_dd) if max_dd > 1e-12 else 0.0

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))

    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return Metrics(
        initial_capital=float(equity.iloc[0]),
        final_equity=float(equity.iloc[-1]),
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        max_drawdown_date=max_dd_date,
        calmar=calmar,
        volatility=volatility,
        n_trades=len(trades),
        n_wins=len(wins),
        n_losses=len(losses),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        avg_win=gross_profit / len(wins) if wins else 0.0,
        avg_loss=-gross_loss / len(losses) if losses else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 1e-12
        else (float("inf") if gross_profit > 0 else 0.0),
        avg_bars_held=float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        exposure=exposure,
        total_costs=sum(t.costs for t in trades),
        gross_pnl=sum(t.gross_pnl for t in trades),
        net_pnl=sum(t.net_pnl for t in trades),
        exit_reasons=exit_reasons,
    )


def run_backtest(
    data: PairData,
    signals: SignalFrame,
    config: Config,
    trade_logger: TradeLogger | None = None,
) -> BacktestResult:
    """Simulate the strategy bar by bar over the signal frame.

    Walks the tradeable position series. Whenever the target exposure changes,
    the existing position (if any) is closed and the new one opened at that
    bar's close, with slippage and commission charged on every leg. Equity is
    marked to market on every bar, including bars with no activity.

    Any position still open on the final bar is closed there, so the reported
    P&L contains no unrealised component and ``sum(trade.net_pnl)`` reconciles
    with the change in equity.

    Args:
        data: Aligned price history.
        signals: Output of :func:`pairs_trading.strategy.generate_signals`.
        config: Full configuration.
        trade_logger: Optional CSV logger; every leg of every fill is written.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        BacktestError: if the signal frame has no bars with a defined z-score.
    """
    bt = config.backtest
    frame = signals.frame
    y_col, x_col = signals.pair_y, signals.pair_x
    pair_label = f"{y_col}/{x_col}"

    usable = frame[frame["zscore"].notna()]
    if usable.empty:
        raise BacktestError(
            "No bar has a defined z-score, so no position could ever be taken. "
            "The sample is shorter than the combined hedge and z-score warm-up."
        )

    cash = bt.initial_capital
    shares_y = 0.0
    shares_x = 0.0

    # Open-position state.
    direction = 0
    entry_idx: pd.Timestamp | None = None
    entry_bar = 0
    entry_z = 0.0
    entry_py = entry_px = 0.0
    entry_costs = 0.0
    entry_hedge = float("nan")
    # Signed share counts fixed at entry and held until exit.
    held_y = held_x = 0.0

    trades: list[Trade] = []
    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []
    bars_in_market = 0

    positions = frame["position"].to_numpy(dtype=int)
    zscores = frame["zscore"].to_numpy(dtype=float)
    hedges = frame["hedge_ratio"].to_numpy(dtype=float)
    reasons = frame["reason"].to_numpy(dtype=object)
    prices_y = frame[y_col].to_numpy(dtype=float)
    prices_x = frame[x_col].to_numpy(dtype=float)
    index = frame.index
    n = len(frame)

    def trade_legs(
        ts: pd.Timestamp, qty_y: float, qty_x: float, py: float, px: float,
        z: float, spread_v: float, hedge: float, side: str, reason: str,
    ) -> float:
        """Execute both legs, update cash/shares, log them, return total cost.

        ``qty_y``/``qty_x`` are signed *changes* in share count.
        """
        nonlocal cash, shares_y, shares_x
        total_cost = 0.0
        group = make_group_id(pair_label, ts.isoformat())
        records: list[TradeRecord] = []

        for ticker, qty, close in ((y_col, qty_y, py), (x_col, qty_x, px)):
            if abs(qty) < 1e-12:
                continue
            is_buy = qty > 0
            fill = _fill_price(close, is_buy, bt.slippage_bps)
            notional = abs(qty) * fill
            commission = notional * bt.commission_bps * BPS
            cash += -qty * fill - commission
            total_cost += commission
            # Slippage is a cost too: the gap between the close we would have
            # liked and the fill we modelled.
            total_cost += abs(qty) * abs(fill - close)

            if ticker == y_col:
                shares_y += qty
            else:
                shares_x += qty

            records.append(
                TradeRecord(
                    timestamp=ts.isoformat(),
                    action="BUY" if is_buy else "SELL",
                    ticker=ticker,
                    quantity=abs(qty),
                    price=fill,
                    zscore=z,
                    mode="backtest",
                    pair=pair_label,
                    side=side,
                    reason=reason,
                    spread=spread_v,
                    hedge_ratio=hedge,
                    notional=notional,
                    commission=commission,
                    group_id=group,
                )
            )

        if trade_logger and records:
            trade_logger.log_many(records)
        return total_cost

    for i in range(n):
        ts = index[i]
        py, px = prices_y[i], prices_x[i]
        z = zscores[i] if np.isfinite(zscores[i]) else 0.0
        spread_v = float(frame["spread"].iloc[i])
        hedge = hedges[i] if np.isfinite(hedges[i]) else 1.0
        target = int(positions[i])

        # The reason code sits on the bar that *generated* the signal; the fill
        # happens `execution_lag` bars later, so look back for the label.
        lag = config.signal.execution_lag
        reason = str(reasons[i - lag]) if i - lag >= 0 else ""
        is_last = i == n - 1

        # Force a flat position on the final bar so nothing is left unrealised.
        if is_last and direction != 0:
            target = 0
            reason = reason or EXIT_EOD

        if target != direction:
            # 1. Close the existing position, if any.
            if direction != 0:
                cost = trade_legs(
                    ts, -shares_y, -shares_x, py, px, z, spread_v, hedge,
                    side="exit", reason=reason or EXIT_EOD,
                )
                # Computed from the actual signed share counts held, so it is
                # exact regardless of which sizing rule was used at entry.
                gross = _gross_pnl(entry_py, entry_px, py, px, held_y, held_x)
                total_costs = entry_costs + cost
                trades.append(
                    Trade(
                        entry_date=entry_idx,
                        exit_date=ts,
                        direction=direction,
                        entry_zscore=entry_z,
                        exit_zscore=z,
                        entry_price_y=entry_py,
                        entry_price_x=entry_px,
                        exit_price_y=py,
                        exit_price_x=px,
                        quantity_y=held_y,
                        quantity_x=held_x,
                        gross_pnl=gross,
                        costs=total_costs,
                        net_pnl=gross - total_costs,
                        bars_held=i - entry_bar,
                        exit_reason=reason or EXIT_EOD,
                        hedge_ratio=entry_hedge,
                    )
                )
                direction = 0

            # 2. Open the new position, if the target is not flat.
            if target != 0 and not is_last:
                if not bt.allow_short:
                    logger.debug("Shorting disabled; skipping entry at %s.", ts.date())
                else:
                    qty_y, qty_x = _position_sizes(py, px, hedge, config)
                    signed_y = target * qty_y
                    signed_x = -target * qty_x
                    entry_costs = trade_legs(
                        ts, signed_y, signed_x, py, px, z, spread_v, hedge,
                        side="entry", reason=reason,
                    )
                    direction = target
                    entry_idx, entry_bar = ts, i
                    entry_z, entry_py, entry_px = z, py, px
                    entry_hedge = hedge
                    held_y, held_x = signed_y, signed_x

        if direction != 0:
            bars_in_market += 1

        equity_values.append(cash + shares_y * py + shares_x * px)
        equity_index.append(ts)

    equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index), name="equity")
    returns = equity.pct_change().fillna(0.0).rename("returns")
    exposure = bars_in_market / n if n else 0.0
    metrics = _compute_metrics(equity, trades, exposure, config)

    return BacktestResult(
        equity=equity,
        returns=returns,
        trades=trades,
        metrics=metrics,
        signals=signals,
        config=config,
        pair=pair_label,
        trade_log_path=trade_logger.path if trade_logger else None,
    )


def _gross_pnl(
    entry_py: float, entry_px: float, exit_py: float, exit_px: float,
    held_y: float, held_x: float,
) -> float:
    """P&L before costs from holding signed share counts between two price points."""
    return held_y * (exit_py - entry_py) + held_x * (exit_px - entry_px)


def plot_results(result: BacktestResult, output: str | Path | None = None, show: bool = False):
    """Render the diagnostic chart: prices, z-score with fills, and cumulative P&L.

    Three stacked panels sharing a date axis:

    1. **Normalised prices** -- both legs rebased to 100 at the start, which is
       the only way to eyeball whether the two actually track each other. A pair
       whose lines visibly separate and never rejoin is one whose cointegration
       verdict deserves suspicion regardless of the p-value.
    2. **Z-score** with the entry, exit and stop thresholds drawn, and every fill
       marked: green up-triangles for long-spread entries, red down-triangles for
       short-spread entries, and crosses for exits (black for mean-reversion
       exits, red for stops).
    3. **Cumulative P&L** with the drawdown shaded underneath.

    Args:
        result: A completed backtest.
        output: PNG path. Parent directories are created.
        show: Open an interactive window as well (needs a display).

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib

    if not show:
        # Headless-safe: pick a non-interactive backend before pyplot loads.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    frame = result.signals.frame
    y_col, x_col = result.signals.pair_y, result.signals.pair_x
    sig = result.config.signal

    fig, axes = plt.subplots(4, 1, figsize=(14, 13.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.3, 1.2, 0.7]})
    fig.suptitle(
        f"Pairs trading: {result.pair}   "
        f"[{result.config.spread.method}, z-window {sig.zscore_window}, "
        f"entry +/-{sig.entry_z:g}, exit +/-{sig.exit_z:g}, stop +/-{sig.stop_z:g}]",
        fontsize=13, fontweight="bold",
    )

    # --- Panel 1: normalised prices ----------------------------------------
    ax = axes[0]
    for col, colour in ((y_col, "#1f77b4"), (x_col, "#ff7f0e")):
        ax.plot(frame.index, frame[col] / frame[col].iloc[0] * 100.0,
                label=f"{col} (rebased)", linewidth=1.2, color=colour)
    ax.set_ylabel("Price (start = 100)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Both legs, rebased to 100", fontsize=10, loc="left")

    # --- Panel 2: z-score with thresholds and fills -------------------------
    ax = axes[1]
    ax.plot(frame.index, frame["zscore"], color="#333333", linewidth=1.0, label="z-score")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    for level, style, colour, label in (
        (sig.entry_z, "--", "#2ca02c", f"entry +/-{sig.entry_z:g}"),
        (-sig.entry_z, "--", "#2ca02c", None),
        (sig.exit_z, ":", "#7f7f7f", f"exit +/-{sig.exit_z:g}"),
        (-sig.exit_z, ":", "#7f7f7f", None),
        (sig.stop_z, "-.", "#d62728", f"stop +/-{sig.stop_z:g}"),
        (-sig.stop_z, "-.", "#d62728", None),
    ):
        ax.axhline(level, linestyle=style, color=colour, linewidth=0.9, label=label, alpha=0.8)

    entries_long = [t for t in result.trades if t.direction > 0]
    entries_short = [t for t in result.trades if t.direction < 0]
    stops = [t for t in result.trades if t.exit_reason == "exit_stop_loss"]
    normal_exits = [t for t in result.trades if t.exit_reason != "exit_stop_loss"]

    if entries_long:
        ax.scatter([t.entry_date for t in entries_long], [t.entry_zscore for t in entries_long],
                   marker="^", s=70, color="#2ca02c", zorder=5, label="long-spread entry",
                   edgecolors="black", linewidths=0.4)
    if entries_short:
        ax.scatter([t.entry_date for t in entries_short], [t.entry_zscore for t in entries_short],
                   marker="v", s=70, color="#d62728", zorder=5, label="short-spread entry",
                   edgecolors="black", linewidths=0.4)
    if normal_exits:
        ax.scatter([t.exit_date for t in normal_exits], [t.exit_zscore for t in normal_exits],
                   marker="x", s=55, color="black", zorder=5, label="exit")
    if stops:
        ax.scatter([t.exit_date for t in stops], [t.exit_zscore for t in stops],
                   marker="X", s=85, color="#d62728", zorder=6, label="stop-loss exit",
                   edgecolors="black", linewidths=0.5)

    ax.set_ylabel("z-score of spread")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title("Spread z-score with entries and exits", fontsize=10, loc="left")

    # --- Panel 3: cumulative P&L and drawdown -------------------------------
    ax = axes[2]
    pnl = result.equity - result.equity.iloc[0]
    ax.plot(result.equity.index, pnl, color="#1f77b4", linewidth=1.4, label="cumulative P&L")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.fill_between(pnl.index, pnl, 0, where=(pnl >= 0), color="#2ca02c", alpha=0.15)
    ax.fill_between(pnl.index, pnl, 0, where=(pnl < 0), color="#d62728", alpha=0.15)

    m = result.metrics
    ax.set_ylabel("Cumulative P&L ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"Cumulative P&L  |  Sharpe {m.sharpe:.2f}  |  max DD {-m.max_drawdown:.1%}  |  "
        f"{m.n_trades} trades  |  win rate {m.win_rate:.0%}",
        fontsize=10, loc="left",
    )

    # --- Panel 4: drawdown, on its own axis ---------------------------------
    # Deliberately a separate panel rather than a second y-axis on the P&L
    # chart. Two scales sharing one frame invite the reader to read meaning
    # into where the curves cross, and those crossings are an artefact of the
    # independent scalings, not a fact about the strategy.
    ax = axes[3]
    drawdown = (result.equity / result.equity.cummax() - 1.0) * 100.0
    ax.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.25)
    ax.plot(drawdown.index, drawdown, color="#d62728", linewidth=1.0)
    ax.set_ylabel("Drawdown (%)")
    ax.set_ylim(min(drawdown.min() * 1.15, -0.5), 0.5)
    ax.grid(alpha=0.3)
    ax.set_title("Drawdown from running peak", fontsize=10, loc="left")
    ax.set_xlabel("Date")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output:
        path = resolve_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Chart written to %s", path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
