"""Pre-trade statistical tests: is this pair actually cointegrated?

WHY THIS MODULE EXISTS
----------------------
Pairs trading is a bet that a spread is *stationary* -- that it has a fixed mean
it keeps returning to. If that is false, the z-score is still computable and
still crosses +/-2 regularly, so the strategy will still generate trades. It
will simply be shorting a spread that is drifting away forever, and every
"divergence" it fades is the beginning of a permanent repricing. A backtest on a
non-cointegrated pair does not report an error; it reports a loss, or worse, a
profit that came from one lucky sample. This module is the gate that stops that
run before it starts.

Correlation is not the property we need. Correlation measures whether two series
move together *day to day* (co-movement of returns). Cointegration measures
whether they stay together *in level* over time. Two random walks can have
0.9 return correlation and still diverge without bound. Only the second property
makes a spread revert.

THE MATH, IN ORDER
------------------
1. **Order of integration.** Both legs should be I(1): non-stationary in level,
   stationary in first differences. This is the precondition everything else
   rests on. If a leg is already I(0) -- stationary on its own -- then a linear
   combination of it with anything is trivially "stationary" and the
   cointegration test is measuring nothing. :func:`test_stationarity` checks it.

2. **Engle-Granger, step one.** Regress one leg on the other by OLS::

       y_t = alpha + beta * x_t + e_t

   ``beta`` is the *hedge ratio*: how many units of x offset one unit of y. The
   residual ``e_t = y_t - alpha - beta*x_t`` is the spread.

3. **Engle-Granger, step two.** Run an Augmented Dickey-Fuller test on ``e_t``::

       delta_e_t = phi * e_{t-1} + sum(gamma_i * delta_e_{t-i}) + u_t

   Null hypothesis: ``phi = 0``, i.e. the residual has a unit root and is *not*
   mean-reverting. Rejecting the null (small p-value) is evidence of
   cointegration. The critical values are *not* the standard ADF ones, because
   ``e_t`` is itself estimated -- OLS chose ``beta`` to make the residual look as
   small as possible, which biases the test toward rejection. ``statsmodels``'
   :func:`~statsmodels.tsa.stattools.coint` applies the MacKinnon critical
   values that correct for this, which is why we use it rather than running
   ``adfuller`` on our own residuals.

4. **Johansen.** Engle-Granger picks a direction (y on x) and inherits whatever
   bias that choice carries; a pair can pass one way and fail the other. The
   Johansen procedure treats the system symmetrically, estimating a VECM and
   testing how many independent cointegrating vectors exist via the trace and
   maximum-eigenvalue statistics. For two assets there can be at most one, so
   the verdict we want is "reject r=0, fail to reject r<=1".

5. **Half-life.** Cointegration is a yes/no answer; it says nothing about
   *speed*. Fit the discrete Ornstein-Uhlenbeck / AR(1) form::

       delta_s_t = a + b * s_{t-1} + eps   =>   phi = 1 + b
       half_life = -ln(2) / ln(phi)

   This is how many bars the spread takes to close half its gap to the mean. A
   half-life of 200 days on a strategy holding for 20 is a losing proposition
   even though the pair is genuinely cointegrated.

6. **Stability.** A single full-sample verdict hides regime change. We re-run
   the test on sub-periods; a pair that only passes on the whole sample, or only
   in its first half, is one whose relationship has already broken.

The public entry point is :func:`test_pair`, which returns a
:class:`CointegrationResult` carrying every statistic plus a single
``is_cointegrated`` verdict, and :func:`assert_tradeable`, which raises.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .config import CointegrationConfig, Config
from .data import PairData

logger = logging.getLogger(__name__)

# Johansen critical values are tabulated at these three levels only.
_JOHANSEN_LEVELS = {0.10: 0, 0.05: 1, 0.01: 2}


class NotCointegratedError(RuntimeError):
    """Raised when a pair fails the cointegration gate and enforcement is on.

    Carries the full :class:`CointegrationResult` so the caller can print the
    diagnostics rather than just the message.
    """

    def __init__(self, message: str, result: CointegrationResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class StationarityResult:
    """ADF test on a single series (used to confirm each leg is I(1))."""

    name: str
    level_pvalue: float
    diff_pvalue: float
    significance: float

    @property
    def level_is_stationary(self) -> bool:
        return self.level_pvalue < self.significance

    @property
    def diff_is_stationary(self) -> bool:
        return self.diff_pvalue < self.significance

    @property
    def is_i1(self) -> bool:
        """True when the series is non-stationary in level but stationary in differences."""
        return (not self.level_is_stationary) and self.diff_is_stationary

    def describe(self) -> str:
        order = "I(1)" if self.is_i1 else ("I(0)" if self.level_is_stationary else "I(2)+ or unclear")
        return (
            f"{self.name:<8} level p={self.level_pvalue:6.4f}  "
            f"diff p={self.diff_pvalue:6.4f}  -> {order}"
        )


@dataclass
class EngleGrangerResult:
    """Outcome of the two-step Engle-Granger procedure, in one direction."""

    direction: str          # e.g. "KO ~ PEP"
    pvalue: float
    statistic: float
    critical_values: dict[str, float]
    hedge_ratio: float      # beta from the OLS first step
    intercept: float
    residuals: pd.Series = field(repr=False)
    significance: float = 0.05

    @property
    def is_cointegrated(self) -> bool:
        return self.pvalue < self.significance

    def describe(self) -> str:
        verdict = "PASS" if self.is_cointegrated else "FAIL"
        return (
            f"{self.direction:<16} p={self.pvalue:6.4f}  stat={self.statistic:8.4f}  "
            f"beta={self.hedge_ratio:8.4f}  [{verdict}]"
        )


@dataclass
class JohansenResult:
    """Outcome of the Johansen trace / maximum-eigenvalue tests."""

    trace_stats: list[float]
    trace_crit: list[float]
    eigen_stats: list[float]
    eigen_crit: list[float]
    hedge_ratio: float
    significance: float

    @property
    def rejects_r0_trace(self) -> bool:
        """True when the trace test rejects "no cointegrating vector"."""
        return self.trace_stats[0] > self.trace_crit[0]

    @property
    def rejects_r0_eigen(self) -> bool:
        """True when the max-eigenvalue test rejects "no cointegrating vector"."""
        return self.eigen_stats[0] > self.eigen_crit[0]

    @property
    def rejects_r1_trace(self) -> bool:
        """True when the trace test also rejects "at most one vector".

        In a bivariate system this should *not* happen under genuine
        cointegration: rejecting r<=1 implies rank 2, which means both series
        are already stationary and there is no unit root to share.
        """
        return len(self.trace_stats) > 1 and self.trace_stats[1] > self.trace_crit[1]

    @property
    def is_cointegrated(self) -> bool:
        return self.rejects_r0_trace and not self.rejects_r1_trace

    def describe(self) -> str:
        lines = [
            f"  trace   r=0:  stat={self.trace_stats[0]:8.4f}  crit={self.trace_crit[0]:8.4f}  "
            f"[{'reject' if self.rejects_r0_trace else 'fail to reject'}]",
            f"  maxeig  r=0:  stat={self.eigen_stats[0]:8.4f}  crit={self.eigen_crit[0]:8.4f}  "
            f"[{'reject' if self.rejects_r0_eigen else 'fail to reject'}]",
        ]
        if len(self.trace_stats) > 1:
            lines.append(
                f"  trace   r<=1: stat={self.trace_stats[1]:8.4f}  crit={self.trace_crit[1]:8.4f}  "
                f"[{'reject' if self.rejects_r1_trace else 'fail to reject'}]"
            )
        lines.append(f"  Johansen hedge ratio (normalised eigenvector): {self.hedge_ratio:.4f}")
        return "\n".join(lines)


@dataclass
class CointegrationResult:
    """Everything the pre-trade tests found, plus the single go/no-go verdict."""

    pair: str
    n_observations: int
    start: str
    end: str
    significance: float
    stationarity: dict[str, StationarityResult]
    engle_granger: EngleGrangerResult | None
    engle_granger_reverse: EngleGrangerResult | None
    johansen: JohansenResult | None
    half_life: float | None
    hurst: float | None
    correlation: float
    spread_std: float
    stability: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def is_cointegrated(self) -> bool:
        """The gate. True only when no test that was run returned a failure."""
        return not self.failures

    @property
    def hedge_ratio(self) -> float | None:
        """Best available static hedge ratio, preferring Engle-Granger's OLS beta."""
        if self.engle_granger is not None:
            return self.engle_granger.hedge_ratio
        if self.johansen is not None:
            return self.johansen.hedge_ratio
        return None

    def report(self) -> str:
        """A printable diagnostic block covering every test performed."""
        width = 74
        lines = [
            "=" * width,
            f" COINTEGRATION REPORT  --  {self.pair}",
            "=" * width,
            f" Sample      : {self.n_observations} bars  [{self.start} -> {self.end}]",
            f" Significance: {self.significance:.0%}",
            f" Return corr : {self.correlation:+.4f}   (co-movement, NOT cointegration)",
            "",
            " Order of integration (ADF on each leg)",
            " " + "-" * (width - 2),
        ]
        for res in self.stationarity.values():
            lines.append(f"  {res.describe()}")

        if self.engle_granger or self.engle_granger_reverse:
            lines += ["", " Engle-Granger (OLS residual + ADF, MacKinnon critical values)",
                      " " + "-" * (width - 2)]
            for eg in (self.engle_granger, self.engle_granger_reverse):
                if eg:
                    lines.append(f"  {eg.describe()}")

        if self.johansen:
            lines += ["", " Johansen (VECM rank test)", " " + "-" * (width - 2),
                      self.johansen.describe()]

        lines += ["", " Spread dynamics", " " + "-" * (width - 2)]
        hl = f"{self.half_life:.1f} bars" if self.half_life is not None else "not estimable"
        lines.append(f"  Half-life of mean reversion : {hl}")
        if self.hurst is not None:
            tendency = (
                "mean-reverting" if self.hurst < 0.45
                else "trending" if self.hurst > 0.55
                else "random walk"
            )
            lines.append(f"  Hurst exponent              : {self.hurst:.3f}  ({tendency})")
        lines.append(f"  Spread std dev              : {self.spread_std:.4f}")

        if self.stability:
            lines += ["", " Sub-period stability (Engle-Granger p-value per window)",
                      " " + "-" * (width - 2)]
            for window, pval in self.stability.items():
                mark = "pass" if pval < self.significance else "FAIL"
                lines.append(f"  {window:<24} p={pval:6.4f}  [{mark}]")

        lines.append("")
        if self.is_cointegrated:
            lines.append(" VERDICT: PASS -- the spread is statistically mean-reverting.")
        else:
            lines.append(" VERDICT: FAIL -- do NOT trade this pair.")
            for f in self.failures:
                lines.append(f"   x {f}")
        for w in self.warnings:
            lines.append(f"   ! {w}")
        lines.append("=" * width)
        return "\n".join(lines)


def test_stationarity(series: pd.Series, name: str, significance: float = 0.05) -> StationarityResult:
    """Augmented Dickey-Fuller test on a series in level and in first differences.

    The ADF null hypothesis is "this series has a unit root" (is non-stationary).
    A *small* p-value rejects that null and says the series is stationary.

    For pairs trading we want each individual leg to be I(1): failing to reject
    in level (it wanders) but rejecting in differences (its returns do not). If a
    leg comes back I(0) -- stationary in level all by itself -- then the
    cointegration machinery below is inapplicable, because any linear
    combination involving a stationary series is stationary for trivial reasons
    and tells you nothing about a shared equilibrium.

    Args:
        series: Price series (levels, not returns).
        name: Label for reporting.
        significance: Alpha for the pass/fail interpretation.

    Returns:
        A :class:`StationarityResult` with both p-values.
    """
    clean = series.dropna()
    # autolag="AIC" lets the test choose its own lag order, which matters: too
    # few lags leaves serial correlation in the residual and over-rejects.
    level_p = float(adfuller(clean, autolag="AIC")[1])
    diff_p = float(adfuller(clean.diff().dropna(), autolag="AIC")[1])
    return StationarityResult(name, level_p, diff_p, significance)


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    significance: float = 0.05,
    use_log_prices: bool = False,
) -> EngleGrangerResult:
    """Run the two-step Engle-Granger cointegration test for ``y`` on ``x``.

    Step 1 -- estimate the long-run relationship by OLS::

        y_t = alpha + beta * x_t + e_t

    ``beta`` is the hedge ratio and ``e_t`` the spread. Note that OLS here is
    *not* a normal regression: under cointegration the estimator is
    "superconsistent", converging faster than the usual sqrt(n) rate, which is
    why a simple OLS beta is good enough for a long-run relationship even though
    the regressors are non-stationary.

    Step 2 -- test ``e_t`` for a unit root. We call ``statsmodels.tsa.stattools.coint``
    rather than running ``adfuller`` on our own residuals, because the residual
    was constructed by minimising its own variance; the standard ADF critical
    values would over-reject. ``coint`` applies MacKinnon's critical values,
    which account for the estimated cointegrating vector.

    Args:
        y: Dependent leg.
        x: Independent leg.
        significance: Alpha for the pass/fail verdict.
        use_log_prices: Regress log prices, making beta an elasticity. Stabilises
            the fit when the two price levels are far apart in magnitude.

    Returns:
        An :class:`EngleGrangerResult` with the p-value, hedge ratio and residuals.
    """
    yy = np.log(y) if use_log_prices else y
    xx = np.log(x) if use_log_prices else x

    design = sm.add_constant(xx.values)
    fit = sm.OLS(yy.values, design).fit()
    intercept, beta = float(fit.params[0]), float(fit.params[1])
    residuals = pd.Series(yy.values - intercept - beta * xx.values, index=y.index, name="spread")

    stat, pvalue, crit = coint(yy.values, xx.values, trend="c", autolag="AIC")
    return EngleGrangerResult(
        direction=f"{y.name} ~ {x.name}",
        pvalue=float(pvalue),
        statistic=float(stat),
        critical_values={"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
        hedge_ratio=beta,
        intercept=intercept,
        residuals=residuals,
        significance=significance,
    )


def johansen(
    prices: pd.DataFrame,
    significance: float = 0.05,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Run the Johansen cointegration rank test on a two-column price frame.

    Johansen estimates a vector error-correction model::

        delta_P_t = Pi * P_{t-1} + sum(Gamma_i * delta_P_{t-i}) + mu + eps_t

    and tests the rank of ``Pi``. The rank *is* the number of independent
    cointegrating relationships:

    * rank 0 -- no cointegration; the series share no equilibrium.
    * rank 1 -- exactly one cointegrating vector. This is what a tradeable pair
      looks like.
    * rank 2 (full, for two assets) -- both series are already stationary
      individually, so there is no common stochastic trend to arbitrage.

    Two statistics test this, both compared against tabulated critical values:
    the **trace** statistic tests "rank <= r" against "rank > r", and the
    **maximum-eigenvalue** statistic tests "rank = r" against "rank = r+1".

    Unlike Engle-Granger this is symmetric in the two assets -- no choice of
    dependent variable, so no direction-dependent verdict.

    Args:
        prices: Two-column frame of price levels, ordered ``[y, x]``.
        significance: One of 0.10, 0.05, 0.01 (the only tabulated levels).
        det_order: Deterministic trend order. 0 = constant term in the
            cointegrating relation, which is the right default for price spreads
            that have a non-zero mean.
        k_ar_diff: Number of lagged differences in the VECM.

    Returns:
        A :class:`JohansenResult` with both statistics and the implied hedge ratio.
    """
    if significance not in _JOHANSEN_LEVELS:
        # Fall back to the nearest tabulated level rather than failing the run.
        nearest = min(_JOHANSEN_LEVELS, key=lambda lv: abs(lv - significance))
        logger.warning(
            "Johansen critical values exist only at 10%%/5%%/1%%; using %.0f%% for a "
            "requested %.1f%%.", nearest * 100, significance * 100,
        )
        significance = nearest
    col = _JOHANSEN_LEVELS[significance]

    res = coint_johansen(prices.values, det_order, k_ar_diff)

    # The first eigenvector is the cointegrating relation, defined up to scale.
    # Normalising by its first element gives  1*y - beta*x  ~  I(0), so the
    # hedge ratio is the negated second element.
    vec = res.evec[:, 0]
    hedge_ratio = float(-vec[1] / vec[0]) if vec[0] != 0 else float("nan")

    return JohansenResult(
        trace_stats=[float(v) for v in res.lr1],
        trace_crit=[float(v) for v in res.cvt[:, col]],
        eigen_stats=[float(v) for v in res.lr2],
        eigen_crit=[float(v) for v in res.cvm[:, col]],
        hedge_ratio=hedge_ratio,
        significance=significance,
    )


def half_life(spread: pd.Series) -> float | None:
    """Estimate the mean-reversion half-life of a spread, in bars.

    Fits the discrete Ornstein-Uhlenbeck process in its AR(1) form::

        s_t - s_{t-1} = a + b * s_{t-1} + eps_t

    so that ``s_t = a + (1 + b) * s_{t-1} + eps_t`` with autoregressive
    coefficient ``phi = 1 + b``. Starting from a displacement ``d`` and ignoring
    noise, the expected displacement after ``k`` bars is ``d * phi^k``. Setting
    that to ``d/2`` and solving::

        half_life = -ln(2) / ln(phi)

    Interpretation: the number of bars for the spread to close half the distance
    back to its mean. ``b`` must be negative (``0 < phi < 1``) for this to be
    meaningful -- a non-negative ``b`` means the spread is diverging, not
    reverting, and the function returns ``None``.

    This number is the practical counterpart to the cointegration verdict: it
    answers "how long will my capital be tied up", and it should be comfortably
    shorter than the z-score window and any holding-period limit.
    """
    s = spread.dropna()
    if len(s) < 20:
        return None

    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    lagged, delta = lagged.align(delta, join="inner")
    if len(lagged) < 10:
        return None

    fit = sm.OLS(delta.values, sm.add_constant(lagged.values)).fit()
    b = float(fit.params[1])
    phi = 1.0 + b
    if b >= 0 or phi <= 0:
        # Diverging (or oscillating past the mean each bar); no half-life exists.
        return None
    return float(-math.log(2.0) / math.log(phi))


def hurst_exponent(series: pd.Series, max_lag: int = 60) -> float | None:
    """Estimate the Hurst exponent H of a series.

    For a series whose increments scale as ``std(s_{t+k} - s_t) ~ k^H``, a
    log-log regression of that dispersion against the lag ``k`` recovers ``H`` as
    the slope:

    * ``H < 0.5`` -- mean-reverting (anti-persistent): a move up is more likely
      to be followed by a move down. This is the regime pairs trading needs.
    * ``H = 0.5`` -- random walk; no exploitable structure.
    * ``H > 0.5`` -- trending (persistent); fading divergences here loses money.

    A useful cross-check on the cointegration tests, because it is estimated in a
    completely different way and makes no distributional assumptions.
    """
    s = series.dropna().values
    n = len(s)
    if n < 100:
        return None
    max_lag = min(max_lag, n // 4)
    if max_lag < 10:
        return None

    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        sd = float(np.std(diff))
        tau.append(sd if sd > 0 else np.nan)

    tau_arr = np.asarray(tau, dtype=float)
    ok = np.isfinite(tau_arr) & (tau_arr > 0)
    if ok.sum() < 5:
        return None
    slope = np.polyfit(np.log(lags[ok]), np.log(tau_arr[ok]), 1)[0]
    return float(slope)


def _stability_scan(
    data: PairData, significance: float, use_log_prices: bool, n_windows: int = 3
) -> dict[str, float]:
    """Re-run Engle-Granger on contiguous sub-periods.

    A full-sample pass can be produced by a relationship that held strongly for
    the first few years and has since broken; the pooled test averages over
    both regimes. Splitting the sample exposes that. A pair that passes overall
    but fails its most recent window is the classic trap: the statistics say yes,
    the present says no.
    """
    out: dict[str, float] = {}
    n = len(data.prices)
    if n < 3 * 120:  # need enough points per window for the test to have power
        return out

    bounds = np.linspace(0, n, n_windows + 1).astype(int)
    for i in range(n_windows):
        chunk = data.prices.iloc[bounds[i]:bounds[i + 1]]
        if len(chunk) < 60:
            continue
        label = f"{chunk.index[0].date()}..{chunk.index[-1].date()}"
        try:
            res = engle_granger(
                chunk[data.pair.y], chunk[data.pair.x], significance, use_log_prices
            )
            out[label] = res.pvalue
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("Stability window %s failed: %s", label, exc)
    return out


def test_pair(data: PairData, config: Config) -> CointegrationResult:
    """Run the full pre-trade battery on a pair and return a single verdict.

    Executes, in order: ADF on each leg, Engle-Granger in both directions,
    Johansen, half-life, Hurst, and a sub-period stability scan. Which
    cointegration tests run is controlled by ``config.cointegration.method``.

    The verdict in ``result.is_cointegrated`` is conjunctive -- every test that
    was requested must pass. Issues that should not block trading but that you
    ought to see (both legs already stationary, a direction-dependent
    Engle-Granger verdict, an unstable sub-period) land in ``result.warnings``
    instead of ``result.failures``.

    Args:
        data: Aligned price history from :func:`pairs_trading.data.fetch_pair`.
        config: Full configuration; ``config.cointegration`` supplies the
            thresholds and ``config.spread.use_log_prices`` the regression scale.

    Returns:
        A :class:`CointegrationResult`. Call ``.report()`` for the printable
        diagnostics, or pass it to :func:`assert_tradeable` to enforce the gate.
    """
    cc: CointegrationConfig = config.cointegration
    alpha = cc.significance
    use_log = config.spread.use_log_prices
    y, x = data.y, data.x

    failures: list[str] = []
    warnings: list[str] = []

    # --- 1. Order of integration -------------------------------------------
    stationarity = {
        name: test_stationarity(series, name, alpha)
        for name, series in ((data.pair.y, y), (data.pair.x, x))
    }
    already_stationary = [r.name for r in stationarity.values() if r.level_is_stationary]
    if len(already_stationary) == 2:
        warnings.append(
            f"Both legs ({', '.join(already_stationary)}) test as stationary in level. "
            "Cointegration is then vacuous -- any linear combination of two I(0) series is "
            "I(0) -- so the tests below are not evidence of a shared equilibrium."
        )
    elif already_stationary:
        warnings.append(
            f"{already_stationary[0]} tests as stationary in level (I(0)) while its partner "
            "does not. Engle-Granger assumes both legs are I(1); treat its p-value with caution."
        )

    # --- 2/3. Engle-Granger, both directions --------------------------------
    eg = eg_rev = None
    if cc.method in ("engle_granger", "both"):
        eg = engle_granger(y, x, alpha, use_log)
        eg_rev = engle_granger(x, y, alpha, use_log)
        if not eg.is_cointegrated:
            failures.append(
                f"Engle-Granger ({eg.direction}) p={eg.pvalue:.4f} >= {alpha:.2f}: cannot reject "
                "a unit root in the spread, so there is no evidence it mean-reverts."
            )
        if eg.is_cointegrated != eg_rev.is_cointegrated:
            warnings.append(
                f"Engle-Granger is direction-dependent here ({eg.direction} p={eg.pvalue:.4f} vs "
                f"{eg_rev.direction} p={eg_rev.pvalue:.4f}). This is a known weakness of the "
                "two-step test; the Johansen result is the more reliable arbiter."
            )

    # --- 4. Johansen --------------------------------------------------------
    joh = None
    if cc.method in ("johansen", "both"):
        try:
            joh = johansen(data.prices, alpha)
            if not joh.rejects_r0_trace:
                failures.append(
                    f"Johansen trace statistic {joh.trace_stats[0]:.4f} does not exceed the "
                    f"{alpha:.0%} critical value {joh.trace_crit[0]:.4f}: cointegration rank 0, "
                    "i.e. no cointegrating relationship."
                )
            elif joh.rejects_r1_trace:
                warnings.append(
                    "Johansen rejects r<=1 as well as r=0, implying full rank: both series are "
                    "individually stationary, so there is no common trend to arbitrage."
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.append(f"Johansen test could not be computed: {exc}")

    # --- 5. Spread dynamics -------------------------------------------------
    if eg is not None:
        spread = eg.residuals
    else:
        # Johansen-only mode: build the spread from the Johansen hedge ratio.
        beta = joh.hedge_ratio if joh else 1.0
        spread = (y - beta * x).rename("spread")

    hl = half_life(spread)
    hurst = hurst_exponent(spread)

    if hl is None:
        failures.append(
            "The spread has no estimable half-life: the fitted AR(1) coefficient is >= 1, "
            "meaning it diverges rather than reverts."
        )
    else:
        if cc.max_half_life is not None and hl > cc.max_half_life:
            failures.append(
                f"Half-life {hl:.1f} bars exceeds the configured maximum of {cc.max_half_life:g}. "
                "The spread does mean-revert, but too slowly to trade on this horizon: capital "
                "would sit in the position far longer than the signal window assumes."
            )
        if cc.min_half_life is not None and hl < cc.min_half_life:
            warnings.append(
                f"Half-life {hl:.2f} bars is below {cc.min_half_life:g}. Reversion this fast is "
                "usually bid-ask bounce rather than a tradeable dislocation, and transaction "
                "costs will consume it."
            )
    if hurst is not None and hurst > 0.55:
        warnings.append(
            f"Hurst exponent {hurst:.3f} > 0.55 indicates the spread trends rather than reverts, "
            "which contradicts the cointegration verdict. Prefer the more pessimistic reading."
        )

    # --- 6. Stability -------------------------------------------------------
    stability = _stability_scan(data, alpha, use_log)
    if stability:
        failed_windows = [w for w, p in stability.items() if p >= alpha]
        if failed_windows and len(failed_windows) < len(stability):
            last_window = list(stability)[-1]
            if last_window in failed_windows:
                warnings.append(
                    f"The pair fails Engle-Granger in its most recent sub-period ({last_window}, "
                    f"p={stability[last_window]:.4f}) despite passing overall. The relationship "
                    "may have already broken; the full-sample verdict is averaging over regimes."
                )
            else:
                warnings.append(
                    f"Cointegration is unstable across sub-periods ({len(failed_windows)} of "
                    f"{len(stability)} windows fail)."
                )
        elif len(failed_windows) == len(stability):
            warnings.append(
                "Every sub-period fails Engle-Granger even though the full sample may pass; "
                "the pooled result is likely an artefact of the longer window."
            )

    returns = data.prices.pct_change().dropna()
    result = CointegrationResult(
        pair=str(data.pair),
        n_observations=len(data.prices),
        start=str(data.start.date()),
        end=str(data.end.date()),
        significance=alpha,
        stationarity=stationarity,
        engle_granger=eg,
        engle_granger_reverse=eg_rev,
        johansen=joh,
        half_life=hl,
        hurst=hurst,
        correlation=float(returns.corr().iloc[0, 1]),
        spread_std=float(spread.std()),
        stability=stability,
        warnings=warnings,
        failures=failures,
    )
    return result


def assert_tradeable(result: CointegrationResult, config: Config) -> None:
    """Enforce the cointegration gate.

    Args:
        result: Output of :func:`test_pair`.
        config: ``config.cointegration.enforce`` decides whether a failure is
            fatal or merely loud.

    Raises:
        NotCointegratedError: when the pair failed and enforcement is enabled.
    """
    if result.is_cointegrated:
        for w in result.warnings:
            logger.warning("%s: %s", result.pair, w)
        return

    reasons = "\n  - ".join(result.failures)
    message = (
        f"{result.pair} failed the cointegration gate:\n  - {reasons}\n"
        "Backtesting a non-cointegrated pair produces a spread with no equilibrium to revert "
        "to; the z-score will still cross its thresholds, so the run would silently report "
        "results for a strategy with no statistical basis."
    )
    if config.cointegration.enforce:
        raise NotCointegratedError(message, result)

    logger.error("%s\n(cointegration.enforce is false -- continuing anyway.)", message)
