"""Historical price data for a ticker pair.

Backtesting data comes from Yahoo Finance via ``yfinance``: free, no API key,
and adequate for daily-bar research. The module's job is not just "download two
series" but "hand the rest of the system two series that are safe to regress
against each other", which means:

* **Adjusted prices.** ``auto_adjust=True`` back-adjusts for splits and
  dividends. Raw closes contain discrete drops on ex-dividend dates; a spread
  built from raw closes reads those drops as divergence and would fire entry
  signals on a corporate action rather than on a mispricing.
* **Calendar alignment.** The two legs are inner-joined on the date index. If
  one ticker halted, was delisted, or trades on a different holiday calendar,
  the unmatched dates are dropped rather than forward-filled -- a forward-filled
  price produces a zero return that damps measured volatility and shrinks the
  denominator of every subsequent z-score.
* **Loud failure.** A pair with too little overlapping history, or a series with
  suspicious gaps, raises instead of quietly returning a short frame that the
  cointegration test would then evaluate with almost no power.

The public entry point is :func:`fetch_pair`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import Config, DataConfig, PairConfig, resolve_path

logger = logging.getLogger(__name__)

# yfinance occasionally returns an empty frame on transient upstream errors.
_MAX_DOWNLOAD_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (2, 4, 8)


class DataError(RuntimeError):
    """Raised when usable price history cannot be assembled for a pair."""


@dataclass
class PairData:
    """Aligned closing prices for a pair, plus provenance metadata.

    Attributes:
        prices: DataFrame indexed by date with exactly two columns, named after
            the tickers, ordered ``[y, x]``.
        pair: The pair definition these prices belong to.
        source: ``"yfinance"`` or ``"cache"``.
        requested_start / requested_end: What was asked for, which may be wider
            than what the exchange actually provided.
    """

    prices: pd.DataFrame
    pair: PairConfig
    source: str = "yfinance"
    requested_start: str | None = None
    requested_end: str | None = None

    @property
    def y(self) -> pd.Series:
        """Price series of the dependent leg."""
        return self.prices[self.pair.y]

    @property
    def x(self) -> pd.Series:
        """Price series of the independent leg."""
        return self.prices[self.pair.x]

    @property
    def start(self) -> pd.Timestamp:
        return self.prices.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.prices.index[-1]

    def __len__(self) -> int:
        return len(self.prices)

    def summary(self) -> str:
        return (
            f"{self.pair}: {len(self)} aligned daily bars "
            f"[{self.start.date()} -> {self.end.date()}] from {self.source}"
        )


def _cache_path(cache_dir: Path, ticker: str, interval: str) -> Path:
    return cache_dir / f"{ticker.upper()}_{interval}.csv"


def _read_cache(path: Path, max_age_hours: float) -> pd.Series | None:
    """Return a cached series if it exists and is fresh enough, else ``None``.

    Staleness matters more than it looks: a cache written mid-session holds a
    provisional close that Yahoo revises after the official settlement print.
    The default ceiling of 12 hours means an intraday cache is never reused on a
    later day.
    """
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > max_age_hours:
        logger.debug("Cache %s is %.1fh old (limit %.1fh); refetching.", path, age_hours, max_age_hours)
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Ignoring unreadable cache %s: %s", path, exc)
        return None
    if frame.empty or frame.shape[1] < 1:
        return None
    series = frame.iloc[:, 0]
    series.name = path.stem.split("_")[0]
    return series


def _write_cache(path: Path, series: pd.Series) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame(name=series.name or "price").to_csv(path)
    except OSError as exc:  # caching is a convenience, never a hard requirement
        logger.warning("Could not write cache %s: %s", path, exc)


def _extract_price_column(raw: pd.DataFrame, ticker: str, price_field: str) -> pd.Series:
    """Pull one price column out of whatever shape yfinance returned.

    yfinance returns a flat column index for a single ticker but a MultiIndex
    ``(field, ticker)`` when ``group_by`` defaults change or multiple tickers are
    requested, and the capitalisation of field names has varied across releases.
    This normalises all of those into a single named Series.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        # Try (field, ticker) then (ticker, field).
        for level in (0, 1):
            fields = {str(c).lower() for c in raw.columns.get_level_values(level)}
            if price_field.lower() in fields:
                matches = [c for c in raw.columns if str(c[level]).lower() == price_field.lower()]
                if matches:
                    series = raw[matches[0]]
                    break
        else:
            raise DataError(
                f"{ticker}: no {price_field!r} column in downloaded data "
                f"(columns: {list(raw.columns)[:8]})."
            )
    else:
        lookup = {str(c).lower(): c for c in raw.columns}
        if price_field.lower() not in lookup:
            raise DataError(
                f"{ticker}: no {price_field!r} column in downloaded data "
                f"(columns: {list(raw.columns)})."
            )
        series = raw[lookup[price_field.lower()]]

    if isinstance(series, pd.DataFrame):  # duplicate column names
        series = series.iloc[:, 0]
    series = series.astype(float)
    series.name = ticker.upper()
    return series


def _download_one(ticker: str, cfg: DataConfig) -> pd.Series:
    """Download a single adjusted price series, with retries.

    Raises:
        DataError: if every attempt returns empty or the symbol is unknown.
    """
    import yfinance  # imported lazily so unit tests need no network stack

    end = cfg.end
    # yfinance treats `end` as exclusive; nudge it forward so the requested
    # final session is actually included.
    if end:
        end = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")

    last_error: Exception | None = None
    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        try:
            raw = yfinance.download(
                ticker,
                start=cfg.start,
                end=end,
                interval=cfg.interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return _extract_price_column(raw, ticker, cfg.price_field)
            last_error = DataError(f"{ticker}: empty response")
        except DataError:
            raise
        except Exception as exc:  # network/parse errors from the vendor
            last_error = exc
            logger.debug("Download attempt %d for %s failed: %s", attempt + 1, ticker, exc)
        if attempt < _MAX_DOWNLOAD_ATTEMPTS - 1:
            time.sleep(_RETRY_BACKOFF_SECONDS[attempt])

    raise DataError(
        f"Could not download price history for {ticker!r} after "
        f"{_MAX_DOWNLOAD_ATTEMPTS} attempts. Check the symbol is valid and that "
        f"Yahoo Finance is reachable. Last error: {last_error}"
    )


def fetch_series(ticker: str, cfg: DataConfig, cache_dir: Path | None = None) -> tuple[pd.Series, str]:
    """Fetch one ticker's adjusted close series, using the on-disk cache if fresh.

    Returns:
        ``(series, source)`` where source is ``"cache"`` or ``"yfinance"``.
    """
    ticker = ticker.upper().strip()
    cache_dir = cache_dir or resolve_path(cfg.cache_dir)
    path = _cache_path(cache_dir, ticker, cfg.interval)

    if cfg.use_cache:
        cached = _read_cache(path, cfg.max_cache_age_hours)
        if cached is not None:
            # The cache may span a wider window than this request; slice it, and
            # only accept it if it actually covers the requested start.
            window = cached.loc[cached.index >= pd.Timestamp(cfg.start)]
            if cfg.end:
                window = window.loc[window.index <= pd.Timestamp(cfg.end)]
            if not window.empty and window.index[0] <= pd.Timestamp(cfg.start) + timedelta(days=10):
                logger.debug("Using cached history for %s (%d bars).", ticker, len(window))
                return window, "cache"

    series = _download_one(ticker, cfg)
    if cfg.use_cache:
        _write_cache(path, series)
    return series, "yfinance"


def _validate_series(series: pd.Series, ticker: str) -> pd.Series:
    """Clean and sanity-check a single price series.

    Drops duplicate timestamps and non-positive prices (a zero or negative
    close is a vendor artefact, and ``log_ratio`` spreads would produce ``-inf``
    from it), then verifies the index is sorted and unique.
    """
    if series.empty:
        raise DataError(f"{ticker}: empty price series.")

    series = series[~series.index.duplicated(keep="last")].sort_index()

    n_null = int(series.isna().sum())
    if n_null:
        logger.debug("%s: dropping %d NaN observations.", ticker, n_null)
        series = series.dropna()

    bad = series <= 0
    if bad.any():
        logger.warning("%s: dropping %d non-positive prices.", ticker, int(bad.sum()))
        series = series[~bad]

    if series.empty:
        raise DataError(f"{ticker}: no valid observations remain after cleaning.")
    return series


def _check_gaps(prices: pd.DataFrame, pair: PairConfig) -> None:
    """Warn about calendar gaps large enough to distort the rolling statistics.

    A multi-week hole (halt, delisting, symbol change) leaves the rolling mean
    and standard deviation straddling two regimes that were never contiguous, so
    a z-score computed across the gap compares prices from different worlds.
    """
    if len(prices) < 3:
        return
    deltas = prices.index.to_series().diff().dropna()
    if deltas.empty:
        return
    big = deltas[deltas > timedelta(days=10)]
    if not big.empty:
        worst = big.max()
        logger.warning(
            "%s: %d calendar gap(s) longer than 10 days in the aligned history "
            "(largest %d days, ending %s). Rolling statistics spanning a gap of this size "
            "mix pre- and post-gap regimes.",
            pair,
            len(big),
            worst.days,
            big.idxmax().date(),
        )


def fetch_pair(config: Config, pair: PairConfig | None = None) -> PairData:
    """Download and align daily price history for both legs of a pair.

    The two series are inner-joined on their date index so every row holds a
    price for both legs observed on the same session. Rows where either leg is
    missing are dropped, never forward-filled: a synthetic repeated price
    contributes a zero return, which deflates the volatility estimate that the
    z-score divides by and so systematically exaggerates ``|z|``.

    Args:
        config: The full configuration; ``config.data`` controls the window,
            interval, and caching.
        pair: Override the pair in ``config.pair`` (used by screening loops).

    Returns:
        A :class:`PairData` with aligned prices in ``[y, x]`` column order.

    Raises:
        DataError: if either leg cannot be downloaded, or the overlapping
            history is shorter than ``config.data.min_observations``.
    """
    pair = pair or config.pair
    cfg = config.data
    cache_dir = resolve_path(cfg.cache_dir)

    logger.info("Fetching %s from %s to %s...", pair, cfg.start, cfg.end or "today")

    series: dict[str, pd.Series] = {}
    sources: set[str] = set()
    for ticker in pair.tickers:
        raw, source = fetch_series(ticker, cfg, cache_dir)
        series[ticker] = _validate_series(raw, ticker)
        sources.add(source)

    y, x = series[pair.y], series[pair.x]
    prices = pd.concat([y, x], axis=1, join="inner").dropna()
    prices.columns = [pair.y, pair.x]
    prices.index.name = "date"

    if prices.empty:
        raise DataError(
            f"{pair}: the two series have no overlapping trading days between "
            f"{cfg.start} and {cfg.end or 'today'}. Check that both symbols were listed "
            "over this window."
        )

    dropped = max(len(y), len(x)) - len(prices)
    if dropped > 0:
        logger.info(
            "%s: dropped %d unmatched session(s) when aligning calendars "
            "(%d and %d bars in, %d aligned).",
            pair, dropped, len(y), len(x), len(prices),
        )

    if len(prices) < cfg.min_observations:
        raise DataError(
            f"{pair}: only {len(prices)} aligned observations, below the configured minimum "
            f"of {cfg.min_observations}. A cointegration test on this little data has very "
            "low power -- it will usually fail to reject the null even for a genuinely "
            "cointegrated pair. Widen data.start or lower data.min_observations."
        )

    _check_gaps(prices, pair)

    data = PairData(
        prices=prices,
        pair=pair,
        source="+".join(sorted(sources)),
        requested_start=cfg.start,
        requested_end=cfg.end,
    )
    logger.info(data.summary())
    return data


def load_prices_csv(path: str | Path, pair: PairConfig) -> PairData:
    """Load a pair's prices from a local CSV instead of Yahoo Finance.

    Expects a date index in the first column and one column per ticker. Useful
    for reproducible tests and for backtesting against a vendor extract when you
    do not want yfinance in the loop.
    """
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    missing = [t for t in pair.tickers if t not in frame.columns]
    if missing:
        raise DataError(f"{path}: missing column(s) for {missing}; found {list(frame.columns)}.")
    prices = frame[[pair.y, pair.x]].dropna()
    prices.index.name = "date"
    return PairData(prices=prices, pair=pair, source=f"csv:{path}")


def describe(data: PairData) -> pd.DataFrame:
    """Summary statistics for both legs -- a quick eyeball before modelling.

    Reports the annualised volatility of each leg alongside the correlation of
    their daily returns. High return correlation is what makes a pair a
    *candidate*; it is emphatically not the same thing as cointegration, which
    is what makes the spread tradeable. Two series can co-move day to day and
    still drift apart without bound.
    """
    returns = data.prices.pct_change().dropna()
    out = pd.DataFrame(
        {
            "first": data.prices.iloc[0],
            "last": data.prices.iloc[-1],
            "min": data.prices.min(),
            "max": data.prices.max(),
            "mean": data.prices.mean(),
            "ann_vol": returns.std() * (252 ** 0.5),
        }
    )
    out["total_return"] = data.prices.iloc[-1] / data.prices.iloc[0] - 1.0
    out.attrs["return_correlation"] = float(returns.corr().iloc[0, 1])
    out.attrs["observations"] = len(data.prices)
    return out


def default_end_date() -> str:
    """Today's date as ``YYYY-MM-DD`` (the implicit end of an open-ended window)."""
    return datetime.now().strftime("%Y-%m-%d")
