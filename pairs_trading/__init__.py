"""Statistical arbitrage (pairs trading): backtesting and Alpaca PAPER trading.

Modules:
    config              Typed configuration loaded from ``config.yaml``.
    data                Historical daily prices for a ticker pair (yfinance).
    cointegration       Engle-Granger / Johansen tests, half-life, Hurst.
    strategy            Spread construction and the z-score signal state machine.
    backtest            Event-driven simulation, metrics and charts.
    risk_manager        Pre-trade limits, daily-loss breaker, halt flag, alerts.
    execution_alpaca    Alpaca PAPER trading. No live-order code path exists.
    kill_switch         Emergency stop: cancel, flatten, halt.
    trade_log           Shared CSV trade logging.
    main                CLI entry point.

Typical use::

    from pairs_trading import load_config, fetch_pair, test_pair, generate_signals, run_backtest

    config = load_config()
    data = fetch_pair(config)
    result = test_pair(data, config)
    if result.is_cointegrated:
        backtest = run_backtest(data, generate_signals(data, config), config)
        print(backtest.summary())
"""

from .backtest import BacktestResult, run_backtest
from .cointegration import CointegrationResult, NotCointegratedError, assert_tradeable, test_pair
from .config import Config, ConfigError, load_config
from .data import PairData, fetch_pair
from .risk_manager import (
    AccountSnapshot,
    Alert,
    AlertChannel,
    AlertDispatcher,
    ProposedOrder,
    RiskClearance,
    RiskError,
    RiskManager,
    RiskRejection,
    TradingHaltedError,
)
from .strategy import SignalFrame, generate_signals, latest_signal
from .trade_log import TradeLogger, TradeRecord

__all__ = [
    "AccountSnapshot",
    "Alert",
    "AlertChannel",
    "AlertDispatcher",
    "BacktestResult",
    "CointegrationResult",
    "Config",
    "ConfigError",
    "NotCointegratedError",
    "PairData",
    "ProposedOrder",
    "RiskClearance",
    "RiskError",
    "RiskManager",
    "RiskRejection",
    "SignalFrame",
    "TradeLogger",
    "TradeRecord",
    "TradingHaltedError",
    "assert_tradeable",
    "fetch_pair",
    "generate_signals",
    "latest_signal",
    "load_config",
    "run_backtest",
    "test_pair",
]

__version__ = "0.1.0"
