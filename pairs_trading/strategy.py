"""Spread construction and z-score signal generation.

This module is deliberately pure: it takes a price frame and a config, and
returns a signal frame. It knows nothing about brokers, order fills, P&L or
plotting, which is what makes it testable against synthetic series with known
answers (see ``tests/test_pairs_strategy.py``).

THE SIGNAL, END TO END
----------------------
1. **Spread.** Collapse two price series into one that should be stationary.
   Three constructions, selected by ``config.spread.method``:

   ``price_diff``  ``S_t = P_y,t - P_x,t``
       Assumes a 1:1 share hedge. Only sensible when the two prices are of
       similar magnitude and stay that way; otherwise the more expensive leg
       dominates the spread's variance entirely.

   ``log_ratio``   ``S_t = ln(P_y,t) - ln(P_x,t) = ln(P_y,t / P_x,t)``
       Scale-free: equivalent to a 1:1 hedge in *percentage* terms, so it is
       immune to the two legs having different price levels. The natural choice
       when you believe the relationship is proportional rather than additive.

   ``ols``         ``S_t = P_y,t - (alpha_t + beta_t * P_x,t)``
       The regression residual, with ``beta_t`` re-estimated on a rolling
       window. This is the general case: it *measures* the hedge ratio instead
       of assuming one, and it adapts as the relationship drifts.

2. **Z-score.** Standardise the spread against its own recent history::

       z_t = (S_t - mean(S, w)_t) / std(S, w)_t

   The z-score answers "how unusual is today's spread, in units of its own
   recent volatility". Using a *rolling* mean and standard deviation rather than
   full-sample constants is what keeps this honest: a full-sample mean would
   embed the future, telling the strategy on day 10 what the average spread will
   be over the whole backtest.

3. **Position.** A state machine over ``z``:

   * flat and ``z <= -entry_z``  -> **long the spread** (+1): buy y, sell x.
     The spread is unusually low, so we bet it rises back to the mean.
   * flat and ``z >= +entry_z``  -> **short the spread** (-1): sell y, buy x.
   * long and ``z >= -exit_z``   -> flat (mean reversion realised).
   * short and ``z <= +exit_z``  -> flat.
   * ``|z| >= stop_z``           -> flat (stop-loss; the relationship is not
     reverting and may have broken).
   * held ``max_holding_days``   -> flat (time stop).

4. **Execution lag.** The target position is shifted forward by
   ``config.signal.execution_lag`` bars before it becomes tradeable. A signal
   computed from today's close cannot be filled at today's close. With the
   default lag of 1, the backtest fills at the *next* bar's close.

The exit rules are written as ``z >= -exit_z`` rather than ``|z| <= exit_z``
deliberately. If the spread gaps straight from -2.5 through the mean to +2.5 in
a single bar, the symmetric form would miss the exit band entirely and leave a
long position on while the spread sat far above its mean -- holding a winner
into a reversal. The asymmetric form exits on any move back through the mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from .config import Config
from .data import PairData

logger = logging.getLogger(__name__)


class Position(IntEnum):
    """Target exposure to the spread."""

    SHORT_SPREAD = -1   # short y, long x  -- entered when z is high
    FLAT = 0
    LONG_SPREAD = 1     # long y, short x  -- entered when z is low


# Reason codes recorded on the bar a position changes; they flow through to the
# trade log so a post-mortem can separate mean-reversion exits from stops.
ENTRY_LONG = "entry_long_spread"
ENTRY_SHORT = "entry_short_spread"
EXIT_MEAN = "exit_mean_reversion"
EXIT_STOP = "exit_stop_loss"
EXIT_TIME = "exit_time_stop"
EXIT_EOD = "exit_end_of_data"


class StrategyError(ValueError):
    """Raised when a signal cannot be computed from the given inputs."""


@dataclass
class SignalFrame:
    """The strategy's output: one row per bar, everything needed to simulate.

    Columns of ``frame``:
        ``<y>``, ``<x>``   raw prices for each leg
        ``hedge_ratio``    beta_t used to build the spread (1.0 for non-OLS methods)
        ``intercept``      alpha_t from the rolling regression (0.0 otherwise)
        ``spread``         S_t
        ``spread_mean``    rolling mean of S over the z-score window
        ``spread_std``     rolling std of S over the same window
        ``zscore``         z_t
        ``target``         state-machine output, *before* the execution lag
        ``position``       tradeable exposure, ``target`` shifted by the lag
        ``reason``         why the position changed on this bar (else "")
    """

    frame: pd.DataFrame
    pair_y: str
    pair_x: str
    method: str
    zscore_window: int

    @property
    def zscore(self) -> pd.Series:
        return self.frame["zscore"]

    @property
    def position(self) -> pd.Series:
        return self.frame["position"]

    @property
    def spread(self) -> pd.Series:
        return self.frame["spread"]

    @property
    def tradeable(self) -> pd.DataFrame:
        """Rows where the z-score is defined (i.e. past the rolling warm-up)."""
        return self.frame[self.frame["zscore"].notna()]

    def position_changes(self) -> pd.DataFrame:
        """Bars where the tradeable position changed -- one row per fill event."""
        pos = self.frame["position"]
        changed = pos != pos.shift(1).fillna(0)
        return self.frame[changed & (pos.notna())]

    def summary(self) -> str:
        pos = self.frame["position"]
        n_entries = int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum())
        exposure = float((pos != 0).mean())
        return (
            f"{self.pair_y}/{self.pair_x} [{self.method}, z-window {self.zscore_window}]: "
            f"{n_entries} entries, {exposure:.1%} of bars in a position"
        )


def rolling_hedge_ratio(
    y: pd.Series, x: pd.Series, window: int | None, use_log_prices: bool = False
) -> tuple[pd.Series, pd.Series]:
    """Estimate the hedge ratio ``beta_t`` by rolling OLS of y on x.

    Fits ``y_t = alpha_t + beta_t * x_t + e_t`` over a trailing window ending at
    each bar ``t``. Because the window ends at ``t`` and never extends past it,
    ``beta_t`` uses only information available at ``t`` -- no look-ahead.

    ``beta`` is the number of units of x that offset one unit of y. It is the
    quantity that makes the residual stationary, and it is *not* the same thing
    as the ratio of prices or the CAPM beta of one stock on the other.

    Args:
        y: Dependent leg.
        x: Independent leg.
        window: Trailing window in bars. ``None`` fits a single static beta over
            the whole sample -- convenient for research, but it embeds future
            prices into every historical bar and will flatter a backtest.
        use_log_prices: Regress log prices, making ``beta`` an elasticity.

    Returns:
        ``(beta, alpha)`` as Series aligned to ``y``'s index. The first
        ``window - 1`` entries are NaN while the regression warms up.
    """
    yy = np.log(y) if use_log_prices else y
    xx = np.log(x) if use_log_prices else x

    if window is None:
        fit = sm.OLS(yy.values, sm.add_constant(xx.values)).fit()
        alpha, beta = float(fit.params[0]), float(fit.params[1])
        logger.debug("Static hedge ratio: beta=%.4f alpha=%.4f", beta, alpha)
        return (
            pd.Series(beta, index=y.index, name="hedge_ratio"),
            pd.Series(alpha, index=y.index, name="intercept"),
        )

    if window > len(y):
        raise StrategyError(
            f"hedge_window ({window}) exceeds the {len(y)} available bars; "
            "the rolling regression would never produce a single estimate."
        )

    exog = sm.add_constant(xx.values)
    fit = RollingOLS(yy.values, exog, window=window, min_nobs=window).fit()
    params = np.asarray(fit.params)
    alpha = pd.Series(params[:, 0], index=y.index, name="intercept")
    beta = pd.Series(params[:, 1], index=y.index, name="hedge_ratio")
    return beta, alpha


def compute_spread(data: PairData, config: Config) -> pd.DataFrame:
    """Build the spread series according to ``config.spread.method``.

    See the module docstring for the three constructions and when each is
    appropriate. All three return the same frame shape so the downstream
    z-score and position logic is method-agnostic.

    Args:
        data: Aligned prices for the pair.
        config: ``config.spread`` selects the method and the hedge window.

    Returns:
        A DataFrame with ``hedge_ratio``, ``intercept`` and ``spread`` columns,
        indexed like the input prices.

    Raises:
        StrategyError: on an unknown method or a window longer than the sample.
    """
    method = config.spread.method
    y, x = data.y, data.x

    if method == "price_diff":
        # S = Py - Px. Implicit 1:1 share hedge.
        beta = pd.Series(1.0, index=y.index, name="hedge_ratio")
        alpha = pd.Series(0.0, index=y.index, name="intercept")
        spread = (y - x).rename("spread")

    elif method == "log_ratio":
        # S = ln(Py/Px). A one-unit move in S is a 100% change in the price
        # ratio, so the spread is comparable across pairs of any price level.
        beta = pd.Series(1.0, index=y.index, name="hedge_ratio")
        alpha = pd.Series(0.0, index=y.index, name="intercept")
        spread = (np.log(y) - np.log(x)).rename("spread")

    elif method == "ols":
        # S = Py - (alpha + beta*Px): the regression residual.
        beta, alpha = rolling_hedge_ratio(
            y, x, config.spread.hedge_window, config.spread.use_log_prices
        )
        yy = np.log(y) if config.spread.use_log_prices else y
        xx = np.log(x) if config.spread.use_log_prices else x
        spread = (yy - (alpha + beta * xx)).rename("spread")

    else:  # pragma: no cover - config validation rejects this earlier
        raise StrategyError(f"Unknown spread method {method!r}.")

    return pd.DataFrame({"hedge_ratio": beta, "intercept": alpha, "spread": spread})


def rolling_zscore(spread: pd.Series, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Standardise a spread against its own trailing distribution.

    ``z_t = (S_t - mu_t) / sigma_t`` where ``mu_t`` and ``sigma_t`` are the mean
    and sample standard deviation of ``S`` over the ``window`` bars ending at
    ``t`` (inclusive).

    Including bar ``t`` in its own window is standard and is not look-ahead --
    ``S_t`` is known at ``t``. What *would* be look-ahead is using a centred
    window or full-sample statistics, and neither is used here.

    Bars where ``sigma_t`` is zero or vanishing yield NaN rather than a huge or
    infinite z-score. A flat window means the spread has not moved at all, which
    on real data indicates stale prices or a trading halt, not a signal.

    Args:
        spread: The spread series.
        window: Trailing window length in bars.
        min_periods: Minimum observations before a value is produced. Defaults to
            ``window``, so the first ``window - 1`` bars are NaN and the strategy
            stays flat through the warm-up.

    Returns:
        A DataFrame with ``spread_mean``, ``spread_std`` and ``zscore``.
    """
    if window < 2:
        raise StrategyError("z-score window must be at least 2 bars.")
    min_periods = min_periods or window

    rolling = spread.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    # ddof=1 (sample std) is pandas' default and the right choice: the mean was
    # estimated from the same window, so one degree of freedom is spent.
    std = rolling.std(ddof=1)

    # Guard against a degenerate window. Scale the floor to the spread's own
    # magnitude so it is meaningful for both a log-ratio spread (~0.01) and a
    # price-difference spread (~10).
    scale = float(spread.abs().median()) or 1.0
    floor = max(1e-12, scale * 1e-9)
    safe_std = std.where(std > floor)

    z = ((spread - mean) / safe_std).rename("zscore")
    n_degenerate = int((std.notna() & (std <= floor)).sum())
    if n_degenerate:
        logger.warning(
            "%d bar(s) had a near-zero rolling spread standard deviation and were left "
            "without a z-score (likely stale prices or a halt).", n_degenerate,
        )
    return pd.DataFrame({"spread_mean": mean, "spread_std": std, "zscore": z})


def generate_positions(zscore: pd.Series, config: Config) -> pd.DataFrame:
    """Run the entry/exit state machine over a z-score series.

    Walks the series bar by bar holding one piece of state -- the current target
    position -- and applies, in order: time stop, stop-loss, mean-reversion exit,
    then entry. Exits are evaluated before entries on the same bar, so a spread
    that reverts through the mean and immediately diverges the other way can
    reverse in one bar rather than idling for one.

    **Re-entry lock after a protective exit.** After a stop fires at, say,
    ``z = +3.6``, the entry condition ``z >= +2`` is still satisfied, so a naive
    state machine would re-enter the position it just stopped out of on the very
    same bar, and keep doing so all the way up -- turning one bounded loss into
    an unbounded sequence of them and making the stop a no-op. The same applies
    to the time stop. After either, re-entry *in that direction* is blocked
    until ``|z|`` comes back inside the exit band, the earliest point at which
    the original mean-reversion thesis has been re-established. A mean-reversion
    exit imposes no lock, since it leaves ``|z|`` inside the band already.

    Args:
        zscore: Rolling z-score of the spread; NaN during warm-up.
        config: ``config.signal`` supplies the thresholds and the execution lag.

    Returns:
        A DataFrame with ``target`` (pre-lag), ``position`` (post-lag, the
        tradeable exposure), ``reason``, and ``bars_held``.
    """
    sig = config.signal
    entry_z, exit_z, stop_z = sig.entry_z, sig.exit_z, sig.stop_z
    max_hold = sig.max_holding_days

    n = len(zscore)
    targets = np.zeros(n, dtype=int)
    reasons: list[str] = [""] * n
    bars_held = np.zeros(n, dtype=int)

    state = int(Position.FLAT)
    held = 0
    # Direction we are forbidden from re-entering until |z| re-enters the exit
    # band: +1 blocks long-spread, -1 blocks short-spread, 0 blocks nothing.
    locked_out = 0

    for i, z in enumerate(zscore.to_numpy(dtype=float)):
        reason = ""

        if not np.isfinite(z):
            # No signal (warm-up or degenerate window). Hold whatever we have;
            # we cannot evaluate any rule without a z-score.
            targets[i] = state
            bars_held[i] = held = held + 1 if state != 0 else 0
            continue

        # Clear the lock once the spread has come back to its mean.
        if locked_out and abs(z) <= exit_z:
            locked_out = 0

        if state != 0:
            held += 1
            if max_hold is not None and held >= max_hold:
                # Lock out re-entry for the same reason as a stop-loss: the
                # entry condition is usually still true when a time stop fires,
                # so without the lock the position would be reopened on the same
                # bar and the time stop would never actually close anything.
                locked_out = state
                state, reason = int(Position.FLAT), EXIT_TIME
            elif abs(z) >= stop_z:
                # The divergence widened past the point where we still believe
                # in reversion. Exit, and lock out re-entry on this side.
                locked_out = state
                state, reason = int(Position.FLAT), EXIT_STOP
            elif (state == Position.LONG_SPREAD and z >= -exit_z) or (
                state == Position.SHORT_SPREAD and z <= exit_z
            ):
                # Reverted through (or past) the mean -- the trade worked.
                state, reason = int(Position.FLAT), EXIT_MEAN

            if state == 0:
                held = 0

        if state == 0:
            # Spread far below its mean -> expect it to rise -> long the spread.
            if z <= -entry_z and locked_out != Position.LONG_SPREAD:
                state, held = int(Position.LONG_SPREAD), 1
                reason = ENTRY_LONG
            # Spread far above its mean -> expect it to fall -> short the spread.
            elif z >= entry_z and locked_out != Position.SHORT_SPREAD:
                state, held = int(Position.SHORT_SPREAD), 1
                reason = ENTRY_SHORT

        targets[i] = state
        reasons[i] = reason
        bars_held[i] = held

    target = pd.Series(targets, index=zscore.index, name="target")

    # A signal computed from bar t's close is filled at bar t+lag. Shifting the
    # *target* (rather than lagging the z-score) keeps the diagnostics aligned to
    # the bar that generated them while making the exposure honest.
    position = target.shift(sig.execution_lag).fillna(0).astype(int).rename("position")

    return pd.DataFrame(
        {
            "target": target,
            "position": position,
            "reason": pd.Series(reasons, index=zscore.index, name="reason"),
            "bars_held": pd.Series(bars_held, index=zscore.index, name="bars_held"),
        }
    )


def generate_signals(data: PairData, config: Config) -> SignalFrame:
    """Full pipeline: prices -> spread -> z-score -> tradeable positions.

    This is the function the backtester and the paper-trading loop both call, so
    that live signals are produced by exactly the same code path that was
    backtested.

    Args:
        data: Aligned price history for the pair.
        config: Full configuration.

    Returns:
        A :class:`SignalFrame`.

    Raises:
        StrategyError: if the sample is too short for the configured windows.
    """
    spread_frame = compute_spread(data, config)
    window = config.signal.zscore_window

    valid_spread = spread_frame["spread"].notna().sum()
    if valid_spread < window + 1:
        raise StrategyError(
            f"Only {valid_spread} usable spread observations for a {window}-bar z-score window. "
            f"With spread.method={config.spread.method!r} and hedge_window="
            f"{config.spread.hedge_window}, the warm-up consumes "
            f"{len(spread_frame) - valid_spread} bars. Fetch more history or shorten the windows."
        )

    z_frame = rolling_zscore(spread_frame["spread"], window)
    pos_frame = generate_positions(z_frame["zscore"], config)

    frame = pd.concat([data.prices, spread_frame, z_frame, pos_frame], axis=1)
    frame.index.name = "date"

    signals = SignalFrame(
        frame=frame,
        pair_y=data.pair.y,
        pair_x=data.pair.x,
        method=config.spread.method,
        zscore_window=window,
    )
    logger.info(signals.summary())
    return signals


def latest_signal(data: PairData, config: Config) -> dict[str, float | str | int]:
    """Compute the current actionable signal for live (paper) trading.

    Runs the same pipeline as the backtest and returns the *most recent* bar's
    state. Note that ``target`` is used here rather than ``position``: the lagged
    ``position`` column exists so the backtest fills one bar after the signal,
    but in live trading "now" is that next bar -- the signal from yesterday's
    close is the one being acted on today.

    Returns:
        A dict with the date, both prices, hedge ratio, spread, z-score, the
        target position and the reason code.
    """
    signals = generate_signals(data, config)
    valid = signals.tradeable
    if valid.empty:
        raise StrategyError(
            "No bar has a defined z-score; the sample is shorter than the warm-up windows."
        )
    row = valid.iloc[-1]
    return {
        "date": str(valid.index[-1].date()),
        "y_ticker": signals.pair_y,
        "x_ticker": signals.pair_x,
        "y_price": float(row[signals.pair_y]),
        "x_price": float(row[signals.pair_x]),
        "hedge_ratio": float(row["hedge_ratio"]),
        "spread": float(row["spread"]),
        "zscore": float(row["zscore"]),
        "target": int(row["target"]),
        "reason": str(row["reason"]),
    }
