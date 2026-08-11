"""Typed configuration objects for the pairs-trading system.

Everything tunable lives in ``config.yaml``; this module turns that YAML into
frozen dataclasses so the rest of the codebase gets attribute access and an
early, readable error when a key is missing or malformed. The precedence chain
is::

    CLI flag  >  environment variable  >  config.yaml  >  dataclass default

Only credentials and the Alpaca base URL are read from the environment (via a
``.env`` file); strategy parameters deliberately are not, so a backtest is fully
described by its config file plus its command line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"

# Spread construction methods understood by strategy.compute_spread().
SPREAD_METHODS = ("price_diff", "log_ratio", "ols")
SIZING_METHODS = ("dollar_neutral", "beta_neutral")
COINT_METHODS = ("engle_granger", "johansen", "both")


class ConfigError(ValueError):
    """Raised when the configuration is internally inconsistent or unusable."""


class _Null:
    """Sentinel for :meth:`Config.with_overrides` meaning "set this key to None".

    ``with_overrides`` treats ``None`` as "caller did not supply this", so that
    an argparse namespace full of unset flags can be passed straight through.
    That makes ``None`` unusable for the settings where it is a *meaningful*
    value -- ``hedge_window: null`` (static full-sample beta) and
    ``max_holding_days: null`` (no time stop). Pass :data:`NULL` for those.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NULL"


NULL = _Null()


@dataclass(frozen=True)
class PairConfig:
    """The ordered ticker pair.

    Order matters. ``y`` is the dependent variable and ``x`` the independent one
    in the hedge-ratio regression ``y = alpha + beta * x + residual``. Swapping
    them yields a different residual series (approximately, but not exactly, a
    rescaling by ``1/beta``) and can flip a marginal cointegration verdict, so
    ``cointegration.py`` reports the test in both directions.
    """

    y: str = "KO"
    x: str = "PEP"

    def __post_init__(self) -> None:
        if not self.y or not self.x:
            raise ConfigError("Both legs of the pair must be non-empty tickers.")
        if self.y.upper() == self.x.upper():
            raise ConfigError(
                f"A pair needs two distinct tickers; got {self.y!r} twice. "
                "The spread of a series against itself is identically zero."
            )
        object.__setattr__(self, "y", self.y.upper().strip())
        object.__setattr__(self, "x", self.x.upper().strip())

    @property
    def tickers(self) -> tuple[str, str]:
        return (self.y, self.x)

    def __str__(self) -> str:
        return f"{self.y}/{self.x}"


@dataclass(frozen=True)
class DataConfig:
    """Where price history comes from and how much of it we insist on."""

    start: str = "2015-01-01"
    end: str | None = None
    interval: str = "1d"
    price_field: str = "close"
    cache_dir: str = ".cache/pairs_data"
    use_cache: bool = True
    max_cache_age_hours: float = 12.0
    min_observations: int = 252

    def __post_init__(self) -> None:
        if self.min_observations < 30:
            raise ConfigError(
                "min_observations below 30 leaves too few points for the ADF "
                "test to have any power; raise it."
            )


@dataclass(frozen=True)
class SpreadConfig:
    """How two price series are collapsed into one tradeable spread."""

    method: str = "ols"
    hedge_window: int | None = 60
    use_log_prices: bool = False

    def __post_init__(self) -> None:
        if self.method not in SPREAD_METHODS:
            raise ConfigError(
                f"Unknown spread method {self.method!r}; expected one of {SPREAD_METHODS}."
            )
        if self.hedge_window is not None and self.hedge_window < 10:
            raise ConfigError(
                "hedge_window below 10 bars produces a beta dominated by noise; "
                "use null for a static full-sample beta or raise the window."
            )


@dataclass(frozen=True)
class SignalConfig:
    """Z-score thresholds and windows -- the strategy's entire decision surface."""

    zscore_window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.25
    stop_z: float = 3.5
    max_holding_days: int | None = None
    execution_lag: int = 1

    def __post_init__(self) -> None:
        if self.zscore_window < 5:
            raise ConfigError("zscore_window must be at least 5 bars.")
        if self.entry_z <= 0:
            raise ConfigError("entry_z must be positive (it is compared against |z|).")
        if self.exit_z < 0:
            raise ConfigError("exit_z must be non-negative.")
        if self.exit_z >= self.entry_z:
            raise ConfigError(
                f"exit_z ({self.exit_z}) must be below entry_z ({self.entry_z}); "
                "otherwise every position closes on the bar it opens."
            )
        if self.stop_z <= self.entry_z:
            raise ConfigError(
                f"stop_z ({self.stop_z}) must exceed entry_z ({self.entry_z}); "
                "otherwise the stop fires immediately on entry."
            )
        if self.execution_lag < 0:
            raise ConfigError("execution_lag cannot be negative.")
        if self.max_holding_days is not None and self.max_holding_days < 1:
            raise ConfigError("max_holding_days must be at least 1 bar, or null.")


@dataclass(frozen=True)
class CointegrationConfig:
    """Gatekeeping thresholds for the pre-trade statistical tests."""

    method: str = "both"
    significance: float = 0.05
    enforce: bool = True
    max_half_life: float | None = 60.0
    min_half_life: float | None = 1.0

    def __post_init__(self) -> None:
        if self.method not in COINT_METHODS:
            raise ConfigError(
                f"Unknown cointegration method {self.method!r}; expected one of {COINT_METHODS}."
            )
        if not 0.0 < self.significance < 0.5:
            raise ConfigError("significance must lie in (0, 0.5); 0.05 is conventional.")


@dataclass(frozen=True)
class BacktestConfig:
    """Capital, sizing and friction assumptions for the simulation."""

    initial_capital: float = 100_000.0
    gross_exposure_per_leg: float = 10_000.0
    sizing: str = "dollar_neutral"
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.0
    trading_days_per_year: int = 252
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.sizing not in SIZING_METHODS:
            raise ConfigError(
                f"Unknown sizing method {self.sizing!r}; expected one of {SIZING_METHODS}."
            )
        if self.initial_capital <= 0:
            raise ConfigError("initial_capital must be positive.")
        if self.gross_exposure_per_leg <= 0:
            raise ConfigError("gross_exposure_per_leg must be positive.")
        if self.gross_exposure_per_leg > self.initial_capital:
            raise ConfigError(
                f"gross_exposure_per_leg ({self.gross_exposure_per_leg:,.0f}) exceeds "
                f"initial_capital ({self.initial_capital:,.0f}); the position could not be funded."
            )


@dataclass(frozen=True)
class ExecutionConfig:
    """Alpaca paper-trading settings.

    ``base_url`` is validated twice: once here, and again inside
    ``execution_alpaca.py`` immediately before the client is constructed. Both
    checks require the substring ``paper``.
    """

    base_url: str = "https://paper-api.alpaca.markets"
    notional_per_leg: float = 1000.0
    time_in_force: str = "day"
    require_market_open: bool = True
    verify_paper_account: bool = True

    def __post_init__(self) -> None:
        if "paper" not in self.base_url.lower():
            raise ConfigError(
                f"Refusing to accept execution.base_url={self.base_url!r}: this project is "
                "paper-trading only and the endpoint must contain 'paper'."
            )
        if self.notional_per_leg <= 0:
            raise ConfigError("notional_per_leg must be positive.")


@dataclass(frozen=True)
class RiskConfig:
    """Hard limits enforced by ``risk_manager.py`` before any order is sent.

    Every value here is a *cap*, not a target. Breaching one rejects the order
    outright rather than resizing it, on the principle that a silently shrunk
    position is a different trade from the one the strategy asked for.

    Attributes:
        max_position_size_usd: Cap on the gross notional of a single pairs
            trade (both legs summed). ``None`` disables the check.
        max_total_exposure_usd: Cap on gross notional across all open positions
            plus the proposed order.
        max_daily_loss_usd: Absolute daily loss that trips the circuit breaker.
        max_daily_loss_pct: Same limit as a fraction of start-of-day equity.
            When both are set the tighter one binds.
        close_positions_on_breach: Flatten open positions when the breaker
            trips.
        halt_file: Flag file written by the breaker and the kill switch. Its
            presence blocks all new entries until manually cleared.
        state_file: Persists the start-of-day equity baseline, so a mid-session
            restart does not reset the daily-loss measurement.
        require_market_open: Reject orders while the market is closed.
        buying_power_buffer_usd: Headroom left unspent when checking buying
            power.
        alert_on_rejection: Send an alert for routine limit rejections, not just
            breaches and kill-switch activations.
    """

    max_position_size_usd: float | None = 5_000.0
    max_total_exposure_usd: float | None = 20_000.0
    max_daily_loss_usd: float | None = 1_000.0
    max_daily_loss_pct: float | None = None
    close_positions_on_breach: bool = True
    halt_file: str = "logs/TRADING_HALTED"
    state_file: str = "logs/risk_state.json"
    risk_log: str = "logs/risk_events.csv"
    require_market_open: bool = True
    buying_power_buffer_usd: float = 0.0
    alert_on_rejection: bool = True

    def __post_init__(self) -> None:
        for name in ("max_position_size_usd", "max_total_exposure_usd", "max_daily_loss_usd"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(f"risk.{name} must be positive, or null to disable; got {value}.")
        if self.max_daily_loss_pct is not None and not 0 < self.max_daily_loss_pct < 1:
            raise ConfigError(
                f"risk.max_daily_loss_pct must be a fraction in (0, 1) -- 0.02 for 2% -- "
                f"got {self.max_daily_loss_pct}."
            )
        if (
            self.max_position_size_usd is not None
            and self.max_total_exposure_usd is not None
            and self.max_position_size_usd > self.max_total_exposure_usd
        ):
            raise ConfigError(
                f"risk.max_position_size_usd ({self.max_position_size_usd:,.0f}) exceeds "
                f"risk.max_total_exposure_usd ({self.max_total_exposure_usd:,.0f}); a single "
                "trade could never be opened without breaching the aggregate cap."
            )
        if self.buying_power_buffer_usd < 0:
            raise ConfigError("risk.buying_power_buffer_usd cannot be negative.")


@dataclass(frozen=True)
class NewsConfig:
    """Settings for the news-driven trading veto.

    The guard can only ever *stop* trading, never start or direct it. See
    ``news_guard.py`` for why that asymmetry is deliberate.

    Attributes:
        enabled: Run the guard automatically before each paper-trading run.
        lookback_hours: How far back to scan. Should comfortably exceed the gap
            between runs, so a weekend cannot hide a Friday-evening merger
            announcement.
        max_items: Cap on headlines fetched per scan.
        max_symbols: Items tagged with more symbols than this are treated as
            market roundups rather than news about the pair, and skipped.
        halt_on_blocking: Write the halt flag when a structural event is found.
        fail_closed: Halt when the news feed itself is unreachable. Off by
            default -- a veto that stops on its own outage hands the news vendor
            an off-switch for your strategy, and missing news is not evidence.
    """

    enabled: bool = True
    lookback_hours: float = 96.0
    max_items: int = 50
    max_symbols: int = 8
    halt_on_blocking: bool = True
    fail_closed: bool = False

    def __post_init__(self) -> None:
        if self.lookback_hours <= 0:
            raise ConfigError("news.lookback_hours must be positive.")
        if self.max_items < 1:
            raise ConfigError("news.max_items must be at least 1.")
        if self.max_symbols < 1:
            raise ConfigError("news.max_symbols must be at least 1.")


@dataclass(frozen=True)
class LoggingConfig:
    trade_log: str = "logs/trades.csv"
    level: str = "INFO"


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = True
    output: str = "logs/backtest_plot.png"
    show: bool = False


@dataclass(frozen=True)
class Config:
    """Root configuration object handed to every module in the package."""

    pair: PairConfig = field(default_factory=PairConfig)
    data: DataConfig = field(default_factory=DataConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    cointegration: CointegrationConfig = field(default_factory=CointegrationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    source_path: str | None = None

    def cross_validate(self) -> None:
        """Check constraints that span more than one config section.

        Only genuinely unrunnable combinations raise here. Configurations that
        are merely questionable are reported by :meth:`advisories` instead, so a
        deliberate experiment is never blocked by a heuristic.
        """
        if self.backtest.sizing == "beta_neutral" and self.spread.method != "ols":
            raise ConfigError(
                f"backtest.sizing='beta_neutral' needs a hedge ratio, but spread.method is "
                f"{self.spread.method!r}, which does not estimate one. Use spread.method='ols' "
                "or switch to sizing='dollar_neutral'."
            )
        if self.data.min_observations <= self.signal.zscore_window:
            raise ConfigError(
                f"data.min_observations ({self.data.min_observations}) must exceed "
                f"signal.zscore_window ({self.signal.zscore_window}); otherwise the rolling "
                "z-score is undefined over the entire sample and no signal can ever fire."
            )

    def advisories(self) -> list[str]:
        """Return human-readable warnings about questionable parameter combinations.

        These are modelling smells, not errors: each one describes a setup that
        will run and produce numbers, but where the numbers are likely to mean
        something other than what they appear to. ``main.py`` prints them before
        every run.
        """
        notes: list[str] = []
        sig, coint, spread = self.signal, self.cointegration, self.spread

        if coint.max_half_life is not None and coint.max_half_life > sig.zscore_window:
            notes.append(
                f"cointegration.max_half_life ({coint.max_half_life:g}) exceeds "
                f"signal.zscore_window ({sig.zscore_window}). A spread that reverts more slowly "
                "than the window measuring it has its own drift absorbed into the rolling mean, "
                "so real dislocations can read as z~0. Consider a longer z-score window."
            )
        if spread.hedge_window is not None and spread.hedge_window < sig.zscore_window:
            notes.append(
                f"spread.hedge_window ({spread.hedge_window}) is shorter than "
                f"signal.zscore_window ({sig.zscore_window}). The hedge ratio then moves faster "
                "than the z-score, letting the spread re-anchor out of a dislocation before the "
                "signal fires."
            )
        if spread.hedge_window is None and spread.method == "ols":
            notes.append(
                "spread.hedge_window is null, so beta is fitted once on the whole sample. "
                "That beta embeds future prices, which flatters any backtest run over the same "
                "period. Use a finite window for results you intend to trade."
            )
        if sig.execution_lag == 0:
            notes.append(
                "signal.execution_lag is 0: signals are filled at the same bar's close that "
                "generated them, which is not achievable in live trading and inflates returns."
            )
        if self.backtest.commission_bps == 0 and self.backtest.slippage_bps == 0:
            notes.append(
                "Both commission_bps and slippage_bps are 0. Pairs trading turns over two legs "
                "per round trip, so frictionless results overstate a real strategy materially."
            )

        # A pair entry is two legs of roughly notional_per_leg each. If that
        # always exceeds the per-trade cap, every order is rejected and the
        # system looks broken rather than merely conservative.
        risk, execu = self.risk, self.execution
        pair_notional = 2 * execu.notional_per_leg
        if risk.max_position_size_usd is not None and pair_notional > risk.max_position_size_usd:
            notes.append(
                f"A pair entry costs about ${pair_notional:,.0f} (2 x execution.notional_per_leg) "
                f"but risk.max_position_size_usd is ${risk.max_position_size_usd:,.0f}, so nearly "
                "every order will be rejected on size. Raise the cap or lower notional_per_leg."
            )
        if (
            risk.max_total_exposure_usd is not None
            and pair_notional > risk.max_total_exposure_usd
        ):
            notes.append(
                f"A single pair entry (~${pair_notional:,.0f}) exceeds "
                f"risk.max_total_exposure_usd (${risk.max_total_exposure_usd:,.0f}); no position "
                "can ever be opened."
            )
        if risk.max_daily_loss_usd is None and risk.max_daily_loss_pct is None:
            notes.append(
                "No daily loss limit is configured (both risk.max_daily_loss_usd and "
                "max_daily_loss_pct are null). The circuit breaker will never trip."
            )
        return notes

    def with_overrides(self, **sections: dict[str, Any]) -> Config:
        """Return a copy with per-section keyword overrides applied.

        Used by ``main.py`` to layer CLI flags on top of the YAML file::

            cfg.with_overrides(signal={"entry_z": 2.5}, pair={"y": "GOOGL", "x": "MSFT"})

        ``None`` values are dropped, so an argparse namespace of mostly-unset
        flags can be passed through unfiltered. To set a key *to* ``None``, pass
        :data:`NULL`::

            cfg.with_overrides(spread={"hedge_window": NULL})   # static beta
        """
        updates: dict[str, Any] = {}
        for section, values in sections.items():
            clean = {
                k: (None if isinstance(v, _Null) else v)
                for k, v in (values or {}).items()
                if v is not None
            }
            if not clean:
                continue
            current = getattr(self, section)
            unknown = set(clean) - {f.name for f in fields(current)}
            if unknown:
                raise ConfigError(f"Unknown key(s) for section {section!r}: {sorted(unknown)}")
            updates[section] = replace(current, **clean)
        new = replace(self, **updates)
        new.cross_validate()
        return new


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Config section {name!r} must be a mapping, got {type(value).__name__}.")
    return value


def _build(cls: type, raw: dict[str, Any], name: str) -> Any:
    """Instantiate a config dataclass, rejecting unknown keys loudly.

    A silently ignored typo in a config file is the most expensive kind of bug
    in a backtest: the run succeeds and reports numbers produced by a parameter
    you thought you had changed.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in config section {name!r}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}."
        )
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config.yaml``.

    Args:
        path: Config file to read. Defaults to the ``config.yaml`` shipped
            beside this module.

    Returns:
        A fully validated :class:`Config`.

    Raises:
        ConfigError: if the file is missing, malformed, contains unknown keys,
            or specifies a combination of parameters that cannot be traded.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a top-level mapping.")

    known_sections = {f.name for f in fields(Config)} - {"source_path"}
    unknown_sections = set(raw) - known_sections
    if unknown_sections:
        raise ConfigError(
            f"Unknown config section(s): {sorted(unknown_sections)}. "
            f"Valid sections: {sorted(known_sections)}."
        )

    cfg = Config(
        pair=_build(PairConfig, _section(raw, "pair"), "pair"),
        data=_build(DataConfig, _section(raw, "data"), "data"),
        spread=_build(SpreadConfig, _section(raw, "spread"), "spread"),
        signal=_build(SignalConfig, _section(raw, "signal"), "signal"),
        cointegration=_build(CointegrationConfig, _section(raw, "cointegration"), "cointegration"),
        backtest=_build(BacktestConfig, _section(raw, "backtest"), "backtest"),
        execution=_build(ExecutionConfig, _section(raw, "execution"), "execution"),
        risk=_build(RiskConfig, _section(raw, "risk"), "risk"),
        news=_build(NewsConfig, _section(raw, "news"), "news"),
        logging=_build(LoggingConfig, _section(raw, "logging"), "logging"),
        plot=_build(PlotConfig, _section(raw, "plot"), "plot"),
        source_path=str(cfg_path),
    )
    cfg.cross_validate()
    return cfg


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a config-relative path against the repository root.

    Relative paths in ``config.yaml`` (log files, caches, plots) are interpreted
    relative to the project root rather than the process working directory, so
    ``python -m pairs_trading.main`` writes to the same place no matter which
    directory it was launched from.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or PACKAGE_DIR.parent) / p


def alpaca_credentials() -> tuple[str, str]:
    """Read Alpaca paper-trading credentials from the environment.

    Loads ``.env`` (via python-dotenv) if present, then reads the key/secret.
    Alpaca's own SDK naming has drifted over versions, so both the current
    ``ALPACA_API_KEY``/``ALPACA_SECRET_KEY`` and the older
    ``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY`` spellings are accepted.

    Returns:
        ``(api_key, secret_key)``.

    Raises:
        ConfigError: if either credential is missing.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(PACKAGE_DIR.parent / ".env")
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        pass

    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    missing = [n for n, v in (("ALPACA_API_KEY", key), ("ALPACA_SECRET_KEY", secret)) if not v]
    if missing:
        raise ConfigError(
            f"Missing Alpaca credential(s): {', '.join(missing)}. "
            "Copy pairs_trading/.env.example to .env at the repo root and fill in your "
            "PAPER trading keys from https://app.alpaca.markets/paper/dashboard/overview"
        )
    return key, secret
