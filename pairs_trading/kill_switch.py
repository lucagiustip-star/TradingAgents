"""Emergency stop: cancel everything, flatten everything, halt trading.

Run it at any time::

    python -m pairs_trading.kill_switch                       # cancel, flatten, halt
    python -m pairs_trading.kill_switch --reason "spread broke"
    python -m pairs_trading.kill_switch --halt-only           # halt without flattening
    python -m pairs_trading.kill_switch --status              # is trading halted?
    python -m pairs_trading.kill_switch --clear               # manual resume

DESIGN CONSTRAINTS
------------------
This script runs when something has already gone wrong, so it is built to work
when other things are broken:

* **The halt flag is written first**, before any broker call. If cancelling or
  closing then fails -- network down, API rejecting -- the system is still
  stopped. Halting is the part that must never fail; flattening is best-effort.
* **Failures are collected, not raised.** One symbol that will not close must
  not prevent the others from closing. Everything is attempted, and the exit
  code reports whether it all worked.
* **It does not import the strategy.** No prices, no signals, no cointegration
  -- only the broker and the risk log. A kill switch that needed a working data
  feed would be useless exactly when you need it.

Exit codes: ``0`` success, ``1`` partial failure (halt set, some broker action
failed), ``2`` configuration error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from .config import Config, ConfigError, load_config, resolve_path
from .risk_manager import (
    EVENT_KILL_SWITCH,
    Alert,
    AlertDispatcher,
    RiskEventLogger,
    RiskManager,
)

logger = logging.getLogger("pairs_trading.kill_switch")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def activate(
    config: Config,
    reason: str = "manual kill switch",
    close_positions: bool = True,
    cancel_orders: bool = True,
    dry_run: bool = False,
) -> dict:
    """Halt trading, then cancel open orders and flatten open positions.

    Order of operations matters. The halt flag goes down first so that even a
    total broker failure leaves the system stopped rather than running
    unsupervised.

    Args:
        config: Loaded configuration.
        reason: Recorded in the halt file, the risk log and the alert.
        close_positions: Liquidate all open positions at market.
        cancel_orders: Cancel all open orders.
        dry_run: Report what would happen; still writes the halt flag, because
            a dry run that leaves trading enabled would be a trap.

    Returns:
        A summary dict with ``halted``, ``cancelled``, ``closed`` and ``errors``.
    """
    summary: dict = {
        "timestamp": _utc_now(),
        "reason": reason,
        "halted": False,
        "cancelled": 0,
        "closed": 0,
        "errors": [],
        "dry_run": dry_run,
    }

    alerts = AlertDispatcher.from_env()
    events = RiskEventLogger(resolve_path(config.risk.risk_log))
    risk = RiskManager(config, alerts=alerts, event_logger=events)

    # 1. Halt first. This must succeed even if everything else fails.
    try:
        risk.halt(f"KILL SWITCH: {reason}", None, event=EVENT_KILL_SWITCH)
        summary["halted"] = True
    except Exception as exc:
        summary["errors"].append(f"could not write halt flag: {exc}")
        logger.critical("FAILED to write the halt flag: %s", exc)

    # 2. Broker actions, best effort.
    client = None
    if cancel_orders or close_positions:
        try:
            from .execution_alpaca import AlpacaPaperTrader

            trader = AlpacaPaperTrader(config, trade_logger=None, dry_run=dry_run)
            client = trader.client
        except Exception as exc:
            summary["errors"].append(f"could not connect to Alpaca: {exc}")
            logger.error("Could not connect to Alpaca: %s", exc)

    if client is not None and cancel_orders:
        try:
            open_orders = client.get_orders()
            count = len(list(open_orders or []))
            if dry_run:
                logger.warning("[DRY RUN] WOULD HAVE CANCELLED %d open order(s)", count)
            else:
                client.cancel_orders()
                logger.warning("Cancelled %d open order(s)", count)
            summary["cancelled"] = count
        except Exception as exc:
            summary["errors"].append(f"cancel orders failed: {exc}")
            logger.error("Could not cancel open orders: %s", exc)

    if client is not None and close_positions:
        try:
            positions = list(client.get_all_positions() or [])
            if dry_run:
                for pos in positions:
                    logger.warning(
                        "[DRY RUN] WOULD HAVE CLOSED %s %s",
                        getattr(pos, "qty", "?"), getattr(pos, "symbol", "?"),
                    )
            else:
                # close_all_positions is atomic server-side; fall back to
                # per-symbol closes if the bulk call is unavailable.
                try:
                    client.close_all_positions(cancel_orders=True)
                except AttributeError:  # pragma: no cover - SDK variation
                    for pos in positions:
                        client.close_position(pos.symbol)
                logger.warning("Closed %d position(s) at market", len(positions))
            summary["closed"] = len(positions)
        except Exception as exc:
            summary["errors"].append(f"close positions failed: {exc}")
            logger.error("Could not close positions: %s", exc)

    # 3. One consolidated alert.
    severity = "critical" if not summary["errors"] else "critical"
    alerts.send(
        Alert(
            severity=severity,
            title="KILL SWITCH ACTIVATED",
            message=(
                f"The kill switch was activated.\n\nReason: {reason}\n\n"
                f"Orders cancelled: {summary['cancelled']}\n"
                f"Positions closed: {summary['closed']}\n"
                f"Trading halted: {summary['halted']}\n"
                + ("\nERRORS:\n- " + "\n- ".join(summary["errors"]) if summary["errors"] else "")
            ),
            context={
                "timestamp": summary["timestamp"],
                "dry_run": dry_run,
                "halt_file": str(resolve_path(config.risk.halt_file)),
                "resume_with": "python -m pairs_trading.kill_switch --clear",
            },
        )
    )
    return summary


def status(config: Config) -> dict:
    """Report whether trading is currently halted, and why."""
    risk = RiskManager(config)
    return {
        "halted": risk.is_halted(),
        "halt_file": str(risk.halt_file),
        "reason": risk.halt_reason(),
    }


def clear(config: Config, note: str = "manual clear via kill_switch --clear") -> bool:
    """Remove the halt flag so trading can resume.

    Intentionally the only way to resume. The circuit breaker and the kill
    switch both latch, and neither expires on a timer or a date change.
    """
    return RiskManager(config).clear_halt(note)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairs_trading.kill_switch",
        description="Emergency stop: cancel all orders, close all positions, halt trading.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="Report halt state and exit.")
    mode.add_argument("--clear", action="store_true",
                      help="Remove the halt flag and allow trading to resume.")
    parser.add_argument("--reason", default="manual kill switch",
                        help="Recorded in the halt file, risk log and alert.")
    parser.add_argument("--halt-only", action="store_true",
                        help="Set the halt flag without touching orders or positions.")
    parser.add_argument("--keep-positions", action="store_true",
                        help="Cancel orders and halt, but leave positions open.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report broker actions without performing them. Still halts.")
    parser.add_argument("--config", help="Path to config.yaml.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.status:
        state = status(config)
        if state["halted"]:
            print(f"TRADING IS HALTED\n  flag  : {state['halt_file']}\n  reason: {state['reason']}")
        else:
            print(f"Trading is ACTIVE (no halt flag at {state['halt_file']})")
        return 0

    if args.clear:
        if clear(config):
            print("Halt flag cleared. Trading may resume on the next run.")
        else:
            print("No halt flag was present; nothing to clear.")
        return 0

    # Confirm, unless told not to. This closes real (paper) positions.
    if not args.yes and sys.stdin.isatty():
        action = "halt trading" if args.halt_only else "CANCEL ALL ORDERS and CLOSE ALL POSITIONS"
        answer = input(f"This will {action} on the paper account. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    summary = activate(
        config,
        reason=args.reason,
        close_positions=not (args.halt_only or args.keep_positions),
        cancel_orders=not args.halt_only,
        dry_run=args.dry_run,
    )

    print("\n=== KILL SWITCH ===")
    print(f"  reason           : {summary['reason']}")
    print(f"  trading halted   : {summary['halted']}")
    print(f"  orders cancelled : {summary['cancelled']}")
    print(f"  positions closed : {summary['closed']}")
    if summary["dry_run"]:
        print("  (dry run -- no broker action was taken)")
    if summary["errors"]:
        print("  errors:")
        for err in summary["errors"]:
            print(f"    - {err}")
        print("\n  Trading is halted, but review the errors above.")
        return 1
    print("\n  Resume with: python -m pairs_trading.kill_switch --clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
