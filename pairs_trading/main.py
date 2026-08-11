"""CLI entry point for the pairs-trading system.

Usage::

    python -m pairs_trading.main --backtest --pair KO/PEP
    python -m pairs_trading.main --backtest --pair GOOGL/MSFT --method log_ratio --entry-z 2.5
    python -m pairs_trading.main --check-only --pair KO/PEP
    python -m pairs_trading.main --paper-trade --pair KO/PEP --dry-run

Exactly one mode must be chosen. Every strategy parameter has a config-file
default and an optional flag that overrides it, so a parameter sweep is a shell
loop rather than a series of edits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import NULL, Config, ConfigError, load_config, resolve_path

logger = logging.getLogger("pairs_trading")


def _parse_pair(value: str) -> tuple[str, str]:
    """Parse a ``Y/X`` (or ``Y,X`` / ``Y:X``) pair specification."""
    for sep in ("/", ",", ":"):
        if sep in value:
            parts = [p.strip().upper() for p in value.split(sep) if p.strip()]
            if len(parts) == 2:
                return parts[0], parts[1]
            break
    raise argparse.ArgumentTypeError(
        f"--pair expects two tickers separated by '/', e.g. KO/PEP; got {value!r}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairs_trading",
        description="Statistical arbitrage (pairs trading): backtest and Alpaca PAPER trading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Safety: this tool trades ONLY against Alpaca's paper endpoint. There is no\n"
            "live-order code path, and the configured base URL must contain 'paper'.\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backtest", action="store_true",
                      help="Run the historical backtest and print metrics.")
    mode.add_argument("--paper-trade", action="store_true",
                      help="Reconcile the Alpaca PAPER account to the current signal.")
    mode.add_argument("--check-only", action="store_true",
                      help="Run the cointegration tests and exit without trading.")
    mode.add_argument("--close-all", action="store_true",
                      help="Flatten both legs of the pair on the PAPER account and exit.")
    mode.add_argument("--risk-status", action="store_true",
                      help="Print the current risk limits and halt state, then exit.")
    mode.add_argument("--dashboard", action="store_true",
                      help="Build the HTML dashboard from the logs, risk state and a backtest.")
    mode.add_argument("--check-alpaca", action="store_true",
                      help="Verify the Alpaca paper setup end to end (read-only) and exit.")

    parser.add_argument("--pair", type=_parse_pair, metavar="Y/X",
                        help="Ticker pair, dependent leg first (e.g. KO/PEP).")
    parser.add_argument("--config", type=Path, help="Path to config.yaml.")

    data = parser.add_argument_group("data")
    data.add_argument("--start", help="History start date, YYYY-MM-DD.")
    data.add_argument("--end", help="History end date, YYYY-MM-DD.")
    data.add_argument("--no-cache", action="store_true", help="Bypass the on-disk price cache.")
    data.add_argument("--csv", type=Path,
                      help="Load prices from a local CSV instead of Yahoo Finance.")

    spread = parser.add_argument_group("spread")
    spread.add_argument("--method", choices=["price_diff", "log_ratio", "ols"],
                        help="Spread construction method.")
    spread.add_argument("--hedge-window", type=int,
                        help="Rolling window for the OLS hedge ratio (0 = static full-sample).")
    spread.add_argument("--log-prices", action="store_true",
                        help="Fit the hedge ratio on log prices.")

    signal = parser.add_argument_group("signal")
    signal.add_argument("--zscore-window", type=int, help="Rolling z-score window in bars.")
    signal.add_argument("--entry-z", type=float, help="|z| at which to open a position.")
    signal.add_argument("--exit-z", type=float, help="|z| at which to close it.")
    signal.add_argument("--stop-z", type=float, help="|z| at which to stop out.")
    signal.add_argument("--max-hold", type=int, help="Time stop, in bars.")
    signal.add_argument("--execution-lag", type=int,
                        help="Bars between signal and fill (default 1; 0 is look-ahead).")

    bt = parser.add_argument_group("backtest")
    bt.add_argument("--capital", type=float, help="Initial capital.")
    bt.add_argument("--exposure", type=float, help="Dollar notional committed per leg.")
    bt.add_argument("--sizing", choices=["dollar_neutral", "beta_neutral"], help="Sizing rule.")
    bt.add_argument("--commission-bps", type=float, help="Commission per leg, in bps.")
    bt.add_argument("--slippage-bps", type=float, help="Slippage per leg, in bps.")

    coint = parser.add_argument_group("cointegration")
    coint.add_argument("--coint-method", choices=["engle_granger", "johansen", "both"],
                       help="Which cointegration test(s) to run.")
    coint.add_argument("--significance", type=float, help="Significance level (default 0.05).")
    coint.add_argument("--force", action="store_true",
                       help="Run even if the pair FAILS the cointegration gate. The results "
                            "will have no statistical basis; use only for investigation.")

    out = parser.add_argument_group("output")
    out.add_argument("--no-plot", action="store_true", help="Skip the matplotlib chart.")
    out.add_argument("--show-plot", action="store_true", help="Open the chart interactively.")
    out.add_argument("--plot-output", help="Path for the chart PNG.")
    out.add_argument("--trade-log", help="Path for the CSV trade log.")
    out.add_argument("--trades", action="store_true", help="Print the full round-trip ledger.")
    out.add_argument("--dashboard-output", help="Path for the dashboard HTML.")
    out.add_argument("--live", action="store_true",
                     help="With --dashboard, include live paper-account state.")
    out.add_argument("--open", action="store_true", dest="open_browser",
                     help="With --dashboard, open the page in a browser when done.")
    out.add_argument("--verbose", "-v", action="store_true", help="Debug-level logging.")
    out.add_argument("--quiet", "-q", action="store_true", help="Warnings and errors only.")

    live = parser.add_argument_group("paper trading")
    live.add_argument("--dry-run", action="store_true",
                      help="Run the full pipeline including every risk check, but log "
                           "'WOULD HAVE PLACED ORDER' instead of calling Alpaca. The way to "
                           "exercise the risk layer in isolation.")

    risk = parser.add_argument_group("risk controls")
    risk.add_argument("--max-position-size", type=float, metavar="USD",
                      help="Cap on gross notional per pairs trade.")
    risk.add_argument("--max-total-exposure", type=float, metavar="USD",
                      help="Cap on gross notional across all open positions.")
    risk.add_argument("--max-daily-loss", type=float, metavar="USD",
                      help="Daily loss that trips the circuit breaker.")

    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Layer CLI flags on top of the loaded config file."""
    pair = {}
    if args.pair:
        pair = {"y": args.pair[0], "x": args.pair[1]}

    data = {"start": args.start, "end": args.end}
    if args.no_cache:
        data["use_cache"] = False

    spread = {"method": args.method}
    if args.hedge_window is not None:
        # 0 is the CLI spelling of "static full-sample beta" (YAML null); NULL
        # is the sentinel that survives with_overrides' drop-None filtering.
        spread["hedge_window"] = NULL if args.hedge_window == 0 else args.hedge_window
    if args.log_prices:
        spread["use_log_prices"] = True

    signal = {
        "zscore_window": args.zscore_window,
        "entry_z": args.entry_z,
        "exit_z": args.exit_z,
        "stop_z": args.stop_z,
        "max_holding_days": args.max_hold,
        "execution_lag": args.execution_lag,
    }

    backtest = {
        "initial_capital": args.capital,
        "gross_exposure_per_leg": args.exposure,
        "sizing": args.sizing,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
    }

    coint = {"method": args.coint_method, "significance": args.significance}
    if args.force:
        coint["enforce"] = False

    plot = {"output": args.plot_output}
    if args.no_plot:
        plot["enabled"] = False
    if args.show_plot:
        plot["show"] = True

    logging_cfg = {"trade_log": args.trade_log}

    risk = {
        "max_position_size_usd": args.max_position_size,
        "max_total_exposure_usd": args.max_total_exposure,
        "max_daily_loss_usd": args.max_daily_loss,
    }

    return config.with_overrides(
        pair=pair, data=data, spread=spread, signal=signal, backtest=backtest,
        cointegration=coint, plot=plot, logging=logging_cfg, risk=risk,
    )


def setup_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)-7s %(name)s: %(message)s")
    # yfinance is chatty at INFO and its progress noise obscures our own output.
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _load_data(config: Config, args: argparse.Namespace):
    from .data import fetch_pair, load_prices_csv

    if args.csv:
        return load_prices_csv(args.csv, config.pair)
    return fetch_pair(config)


def _run_cointegration(data, config: Config, args: argparse.Namespace):
    """Run the gate and print the report. Returns the result."""
    from .cointegration import assert_tradeable, test_pair

    result = test_pair(data, config)
    print(result.report())

    if not result.is_cointegrated and args.force:
        print(
            "\n!! --force is set: continuing on a pair that FAILED the cointegration gate.\n"
            "!! The spread has no demonstrated equilibrium, so any P&L below is a property\n"
            "!! of this particular sample and should not be read as an edge.\n"
        )
    assert_tradeable(result, config)
    return result


def cmd_check(config: Config, args: argparse.Namespace) -> int:
    data = _load_data(config, args)
    result = _run_cointegration(data, config, args)
    return 0 if result.is_cointegrated else 1


def cmd_backtest(config: Config, args: argparse.Namespace) -> int:
    from .backtest import plot_results, run_backtest
    from .strategy import generate_signals
    from .trade_log import TradeLogger

    data = _load_data(config, args)
    _run_cointegration(data, config, args)

    signals = generate_signals(data, config)
    log_path = resolve_path(config.logging.trade_log)
    trade_logger = TradeLogger(log_path, mode="backtest")

    result = run_backtest(data, signals, config, trade_logger)
    print()
    print(result.summary())
    print(f"\n Trade log: {log_path}  ({trade_logger.count} legs)")

    if args.trades and result.trades:
        import pandas as pd

        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print("\n Round trips")
            print(result.trades_frame.to_string(float_format=lambda v: f"{v:,.4f}"))

    if config.plot.enabled:
        plot_results(result, config.plot.output, config.plot.show)
        print(f" Chart    : {resolve_path(config.plot.output)}")

    return 0


def _check_not_halted(config: Config) -> int | None:
    """Refuse to trade while the ``TRADING_HALTED`` flag is present.

    The flag is written by the circuit breaker and by ``kill_switch.py``, and
    nothing removes it automatically -- resuming is a deliberate human act, so
    that a breached loss limit cannot quietly un-breach itself overnight.
    """
    from .risk_manager import RiskManager

    risk = RiskManager(config)
    if not risk.is_halted():
        return None

    print(
        f"\n  TRADING IS HALTED -- refusing to trade.\n\n"
        f"  flag  : {risk.halt_file}\n"
        f"  reason: {risk.halt_reason()}\n\n"
        f"  Review what happened (risk log: {resolve_path(config.risk.risk_log)}), then clear it:\n"
        f"      python -m pairs_trading.kill_switch --clear\n",
        file=sys.stderr,
    )
    return 3


def cmd_paper_trade(config: Config, args: argparse.Namespace) -> int:
    from .execution_alpaca import build_trader

    halted = _check_not_halted(config)
    if halted is not None:
        return halted

    data = _load_data(config, args)
    _run_cointegration(data, config, args)

    trader = build_trader(config, config.logging.trade_log, dry_run=args.dry_run)
    print()
    print(trader.account_summary())
    print()
    print(_risk_summary(config, trader))
    print()

    outcome = trader.sync_to_signal(data)
    signal = outcome.get("signal")
    if signal:
        print(
            f" Signal   : {signal['date']}  z={signal['zscore']:+.3f}  "
            f"target={outcome['target_direction']:+d}  ({signal['reason'] or 'no change'})"
        )
    print(f" Action   : {outcome['action']}")
    if outcome.get("orders"):
        print(f" Orders   : {', '.join(outcome['orders'])}")
    if outcome.get("rejected_reasons"):
        print(" BLOCKED by risk controls:")
        for reason in outcome["rejected_reasons"]:
            print(f"   - {reason}")
    if args.dry_run:
        print(" (dry run -- risk checks ran in full; no order was submitted)")

    if outcome["action"] == "circuit_breaker":
        return 4
    if outcome.get("rejected_reasons"):
        return 5
    return 0


def _risk_summary(config: Config, trader) -> str:
    """Render the current risk state before acting."""
    risk_cfg = config.risk
    try:
        snapshot = trader.account_snapshot()
        limit = trader.risk.daily_loss_limit(snapshot)
        return (
            f" Risk     : daily P&L ${snapshot.daily_pnl:,.2f} of ${limit:,.2f} limit  |  "
            f"exposure ${snapshot.open_exposure:,.2f} of "
            f"${risk_cfg.max_total_exposure_usd or float('inf'):,.2f}  |  "
            f"per-trade cap ${risk_cfg.max_position_size_usd or float('inf'):,.2f}"
        )
    except Exception as exc:  # reporting must not block trading
        return f" Risk     : (could not read account state: {exc})"


def cmd_check_alpaca(config: Config, args: argparse.Namespace) -> int:
    """Run the read-only Alpaca setup diagnostic.

    Exit 0 when everything passes, 1 when a blocking problem was found, 6 when
    only non-blocking issues remain.
    """
    from .execution_alpaca import diagnose, render_diagnosis

    checks = diagnose(config)
    print(render_diagnosis(checks, config))

    failures = [c for c in checks if not c.ok]
    if not failures:
        return 0
    return 1 if any(c.fatal for c in failures) else 6


def cmd_dashboard(config: Config, args: argparse.Namespace) -> int:
    """Build the HTML dashboard.

    Runs a backtest for the charts when price data is reachable, and falls back
    to a logs-only page when it is not -- a dashboard that refuses to render
    because the data vendor is down is useless exactly when you want it.
    """
    import webbrowser

    from .dashboard import generate

    result = None
    try:
        from .backtest import run_backtest
        from .strategy import generate_signals

        data = _load_data(config, args)
        result = run_backtest(data, generate_signals(data, config), config)
    except Exception as exc:
        logger.warning("Backtest unavailable (%s); rendering logs and risk state only.", exc)

    path = generate(config, args.dashboard_output, result, live=args.live)
    print(f"\n Dashboard: {path}")
    if args.open_browser:
        webbrowser.open(path.resolve().as_uri())
    return 0


def cmd_risk_status(config: Config, args: argparse.Namespace) -> int:
    """Print the configured limits and the current halt state, without trading.

    Deliberately makes no broker call, so it answers "am I halted and what are
    my limits" even when Alpaca is unreachable.
    """
    from .risk_manager import RiskManager

    risk = RiskManager(config)
    cfg = config.risk

    def money(value):
        return f"${value:,.2f}" if value is not None else "disabled"

    print("=" * 68)
    print(" RISK CONTROLS")
    print("=" * 68)
    print(f"  Max position size (per trade) : {money(cfg.max_position_size_usd)}")
    print(f"  Max total exposure            : {money(cfg.max_total_exposure_usd)}")
    print(f"  Max daily loss                : {money(cfg.max_daily_loss_usd)}")
    print(
        "  Max daily loss (%)            : "
        + (f"{cfg.max_daily_loss_pct:.2%} of start-of-day equity"
           if cfg.max_daily_loss_pct is not None else "disabled")
    )
    print(f"  Close positions on breach     : {cfg.close_positions_on_breach}")
    print(f"  Require market open           : {cfg.require_market_open}")
    print()
    print(f"  Risk event log                : {resolve_path(cfg.risk_log)}")
    print(f"  Halt flag                     : {risk.halt_file}")
    print(f"  State file                    : {risk.state_file}")
    print()
    if risk.is_halted():
        print("  STATUS: HALTED -- no new positions will be opened.")
        print(f"  Reason: {risk.halt_reason()}")
        print("  Clear with: python -m pairs_trading.kill_switch --clear")
        if args.verbose:
            print("\n  Halt file contents:")
            for line in risk.halt_details().splitlines():
                print(f"    {line}")
    else:
        print("  STATUS: ACTIVE")
    print("=" * 68)
    return 3 if risk.is_halted() else 0


def cmd_close_all(config: Config, args: argparse.Namespace) -> int:
    from .execution_alpaca import build_trader

    # Deliberately NOT halt-gated: flattening must work while halted.
    trader = build_trader(config, config.logging.trade_log, dry_run=args.dry_run)
    print(trader.account_summary())
    ids = trader.close_all()
    print(f"\n Closed {len(ids)} leg(s): {', '.join(ids) if ids else 'nothing to close'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args)

    try:
        config = apply_overrides(load_config(args.config), args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    for note in config.advisories():
        logger.warning("Config advisory: %s", note)

    try:
        if args.risk_status:
            return cmd_risk_status(config, args)
        if args.check_alpaca:
            return cmd_check_alpaca(config, args)
        if args.dashboard:
            return cmd_dashboard(config, args)
        if args.check_only:
            return cmd_check(config, args)
        if args.backtest:
            return cmd_backtest(config, args)
        if args.paper_trade:
            return cmd_paper_trade(config, args)
        if args.close_all:
            return cmd_close_all(config, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Surface the failure class, since "not cointegrated" and "network down"
        # call for very different responses.
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        print("\n(Re-run with --verbose for the full traceback.)", file=sys.stderr)
        return 1

    parser.error("No mode selected.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
