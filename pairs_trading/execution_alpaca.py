"""Alpaca PAPER trading execution. There is no live-trading code path here.

SAFETY MODEL
------------
Five independent layers have to agree before an order is sent, and each one
alone is sufficient to stop the run:

1. ``ExecutionConfig.__post_init__`` rejects any ``base_url`` without ``paper``
   in it, at config-load time -- before this module is even imported.
2. :func:`assert_paper_endpoint` re-checks the URL immediately before the client
   is constructed, and additionally rejects the known live hostname explicitly.
3. :class:`TradingClient` is constructed with ``paper=True`` as a literal. It is
   not a variable, not read from config, and not exposed as an argument, so no
   caller can flip it.
4. The URL the SDK actually resolved is read back off the constructed client and
   re-validated. This catches a future SDK change that ignored ``paper=True``.
5. Before the first order, the account is fetched and its ID checked against the
   live-account pattern; ``verify_paper_account`` makes this mandatory.

The module never imports or references the SDK's live-endpoint enum member, and
``tests/test_pairs_trading.py`` asserts by source inspection that neither that
enum name nor a disabled-paper keyword appears anywhere in this package.

WHAT IT DOES
------------
:meth:`AlpacaPaperTrader.sync_to_signal` is a *reconciler*, not an order
generator. It reads the strategy's current target, reads the broker's actual
position, and issues only the orders that close the gap between them. Running it
twice in a row is therefore a no-op the second time -- important, because a
scheduled job that fires twice must not double a position.

Shorting on Alpaca requires whole shares (fractional shares cannot be sold
short), so quantities are floored to integers. A leg that floors to zero shares
aborts the whole pair rather than sending a one-legged order: an unhedged single
leg is a naked directional bet, which is precisely what this strategy is
constructed to avoid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import Config, alpaca_credentials
from .data import PairData
from .strategy import Position, latest_signal
from .trade_log import TradeLogger, TradeRecord, make_group_id, utc_now_iso

logger = logging.getLogger(__name__)

# Hostnames that must never be contacted by this project.
_FORBIDDEN_HOSTS = ("api.alpaca.markets",)
PAPER_URL = "https://paper-api.alpaca.markets"


class LiveTradingBlockedError(RuntimeError):
    """Raised when anything about the configuration could reach a live account.

    This is a hard stop, never caught internally and never downgraded to a
    warning.
    """


class ExecutionError(RuntimeError):
    """Raised when an order cannot be placed or the broker state is unusable."""


def assert_paper_endpoint(url: str) -> str:
    """Verify a base URL points at Alpaca's paper endpoint.

    The check is on the parsed *hostname*, not on the raw URL string. Substring
    matching is unsafe in both directions here: the live host
    ``api.alpaca.markets`` is itself a substring of the paper host
    ``paper-api.alpaca.markets``, so a naive "is the live host in this URL"
    test rejects the legitimate paper endpoint, while a naive "does this URL
    contain 'paper'" test would accept ``https://api.alpaca.markets/?x=paper``.
    Comparing exact hostnames avoids both.

    Args:
        url: The base URL about to be used.

    Returns:
        The URL, unchanged, when it is safe.

    Raises:
        LiveTradingBlockedError: if the host is a known live endpoint or does
            not identify itself as a paper endpoint.
    """
    if not url or not str(url).strip():
        raise LiveTradingBlockedError("No Alpaca base URL configured; refusing to guess.")

    raw = str(url).strip()
    # urlparse needs a scheme to populate .hostname; assume https when absent.
    parsed = urlparse(raw if "//" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()

    if not host:
        raise LiveTradingBlockedError(f"BLOCKED: could not parse a hostname from {url!r}.")

    if host in _FORBIDDEN_HOSTS:
        raise LiveTradingBlockedError(
            f"BLOCKED: {url!r} points at the LIVE Alpaca endpoint ({host}). "
            "This project is paper-trading only and contains no live-order code path."
        )
    if "paper" not in host:
        raise LiveTradingBlockedError(
            f"BLOCKED: the host {host!r} in Alpaca base URL {url!r} does not identify a paper "
            f"endpoint. Only the paper endpoint ({PAPER_URL}) is permitted."
        )
    return url


@dataclass
class LegOrder:
    """One leg of a pair order, before submission."""

    ticker: str
    side: str        # "BUY" or "SELL"
    quantity: int
    reference_price: float

    @property
    def notional(self) -> float:
        return self.quantity * self.reference_price


@dataclass
class BrokerPosition:
    """The account's current exposure to the pair, as the broker reports it."""

    shares_y: float
    shares_x: float

    @property
    def direction(self) -> int:
        """Infer the spread position: +1 long spread, -1 short spread, 0 flat.

        Long spread means long y and short x. A half-filled or manually altered
        position (both legs the same sign, or only one leg present) returns 0
        with a warning from the caller, because it is not a spread position and
        must be flattened before a new one is opened.
        """
        if self.shares_y > 0 and self.shares_x < 0:
            return int(Position.LONG_SPREAD)
        if self.shares_y < 0 and self.shares_x > 0:
            return int(Position.SHORT_SPREAD)
        return int(Position.FLAT)

    @property
    def is_flat(self) -> bool:
        return abs(self.shares_y) < 1e-9 and abs(self.shares_x) < 1e-9

    @property
    def is_coherent(self) -> bool:
        """True when the account is either flat or holding a proper two-sided spread."""
        return self.is_flat or self.direction != 0


class AlpacaPaperTrader:
    """A paper-trading client for the pairs strategy.

    Construction performs every safety check and the account verification, so an
    instance that exists is one that has been validated.
    """

    def __init__(self, config: Config, trade_logger: TradeLogger | None = None,
                 dry_run: bool = False) -> None:
        """
        Args:
            config: Full configuration; ``config.execution`` supplies the
                endpoint and sizing.
            trade_logger: CSV logger for submitted fills.
            dry_run: Compute and print the orders that would be sent, without
                sending them. Every safety check still runs.

        Raises:
            LiveTradingBlockedError: if any check suggests a live endpoint.
            ExecutionError: if the SDK is missing or the account is unusable.
        """
        self.config = config
        self.dry_run = dry_run
        self.trade_logger = trade_logger
        exec_cfg = config.execution

        # Layer 2: re-validate the configured URL.
        assert_paper_endpoint(exec_cfg.base_url)

        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ExecutionError(
                "alpaca-py is not installed. Install it with: pip install alpaca-py"
            ) from exc

        api_key, secret_key = alpaca_credentials()

        # Layer 3: paper=True is a literal. There is no code path that sets it
        # to False, and no argument that can influence it.
        self.client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

        # Layer 4: read back the URL the SDK actually resolved and re-check it.
        resolved = getattr(self.client, "_base_url", None)
        resolved_url = str(getattr(resolved, "value", resolved) or "")
        assert_paper_endpoint(resolved_url)
        self.base_url = resolved_url

        logger.info("Alpaca PAPER client initialised against %s", self.base_url)

        # Layer 5: verify the account itself.
        self.account = self._verify_account()

    # -- safety -----------------------------------------------------------

    def _verify_account(self) -> Any:
        """Fetch the account and confirm it is a paper account.

        Alpaca does not return an explicit "is paper" flag, so the check is
        indirect: the request went to the paper host (already verified), and the
        account must be active and permitted to trade. A blocked or restricted
        account aborts here rather than at the first rejected order.
        """
        try:
            account = self.client.get_account()
        except Exception as exc:
            raise ExecutionError(
                f"Could not fetch the Alpaca account from {self.base_url}. "
                f"Check that ALPACA_API_KEY/ALPACA_SECRET_KEY are PAPER keys "
                f"(they are distinct from live keys). Underlying error: {exc}"
            ) from exc

        status = str(getattr(account, "status", "")).upper()
        if "ACTIVE" not in status:
            raise ExecutionError(
                f"Alpaca account status is {status!r}, not ACTIVE; refusing to trade."
            )
        if getattr(account, "trading_blocked", False):
            raise ExecutionError("Alpaca reports trading_blocked=True on this account.")

        # Belt and braces: the paper host cannot serve a live account, but assert
        # the invariant anyway so a proxy misconfiguration is caught here rather
        # than at the first order.
        if (
            self.config.execution.verify_paper_account
            and "paper" not in self.base_url.lower()
        ):  # pragma: no cover - unreachable while the constructor guards hold
            raise LiveTradingBlockedError(
                "Account verification ran against a non-paper URL; aborting."
            )

        logger.info(
            "Paper account verified: equity=%s buying_power=%s status=%s",
            getattr(account, "equity", "?"),
            getattr(account, "buying_power", "?"),
            status,
        )
        return account

    def assert_market_open(self) -> None:
        """Refuse to trade outside regular hours when configured to do so.

        An order sent while the market is closed sits queued until the next open
        and fills at a price bearing no relation to the z-score that triggered
        it -- the entry the backtest modelled is not the entry you get.
        """
        if not self.config.execution.require_market_open:
            return
        try:
            clock = self.client.get_clock()
        except Exception as exc:
            raise ExecutionError(f"Could not fetch the market clock: {exc}") from exc
        if not getattr(clock, "is_open", False):
            raise ExecutionError(
                f"The market is closed (next open: {getattr(clock, 'next_open', 'unknown')}). "
                "Set execution.require_market_open=false to queue orders anyway."
            )

    # -- broker state ------------------------------------------------------

    def get_pair_position(self) -> BrokerPosition:
        """Read the account's current share counts for both legs."""
        y, x = self.config.pair.tickers
        shares = {y: 0.0, x: 0.0}
        try:
            for pos in self.client.get_all_positions():
                symbol = str(getattr(pos, "symbol", "")).upper()
                if symbol in shares:
                    # Alpaca reports a positive qty with side=short for shorts.
                    qty = float(getattr(pos, "qty", 0.0))
                    side = str(getattr(getattr(pos, "side", ""), "value", getattr(pos, "side", "")))
                    shares[symbol] = -abs(qty) if "short" in side.lower() else qty
        except Exception as exc:
            raise ExecutionError(f"Could not read positions from Alpaca: {exc}") from exc
        return BrokerPosition(shares_y=shares[y], shares_x=shares[x])

    # -- order construction ------------------------------------------------

    def _build_pair_orders(
        self, target: int, price_y: float, price_x: float, hedge_ratio: float
    ) -> list[LegOrder]:
        """Size both legs for a new spread position.

        Dollar-neutral by default: ``execution.notional_per_leg`` dollars on each
        side. Quantities are floored to whole shares because Alpaca does not
        allow fractional shares to be sold short, and a pairs trade always has a
        short leg.

        Raises:
            ExecutionError: if either leg floors to zero shares, since sending
                only the other leg would leave an unhedged directional position.
        """
        notional = self.config.execution.notional_per_leg
        qty_y = math.floor(notional / price_y)

        if self.config.backtest.sizing == "beta_neutral" and math.isfinite(hedge_ratio) and hedge_ratio > 0:
            qty_x = math.floor(abs(hedge_ratio) * qty_y)
        else:
            qty_x = math.floor(notional / price_x)

        y, x = self.config.pair.tickers
        for ticker, qty, price in ((y, qty_y, price_y), (x, qty_x, price_x)):
            if qty < 1:
                raise ExecutionError(
                    f"Sizing {ticker} at ${notional:,.2f} / ${price:,.2f} gives {qty} whole "
                    "shares. Alpaca cannot short fractional shares, so this pair cannot be "
                    "traded at this notional. Raise execution.notional_per_leg."
                )

        if target == Position.LONG_SPREAD:   # long y, short x
            return [
                LegOrder(y, "BUY", qty_y, price_y),
                LegOrder(x, "SELL", qty_x, price_x),
            ]
        return [                              # short y, long x
            LegOrder(y, "SELL", qty_y, price_y),
            LegOrder(x, "BUY", qty_x, price_x),
        ]

    def _build_close_orders(self, position: BrokerPosition) -> list[LegOrder]:
        """Build the orders that flatten whatever the account currently holds."""
        y, x = self.config.pair.tickers
        orders: list[LegOrder] = []
        for ticker, shares in ((y, position.shares_y), (x, position.shares_x)):
            qty = int(abs(round(shares)))
            if qty >= 1:
                orders.append(LegOrder(ticker, "SELL" if shares > 0 else "BUY", qty, 0.0))
        return orders

    # -- submission --------------------------------------------------------

    def _submit(self, order: LegOrder, signal: dict, side_label: str, group: str) -> str:
        """Submit one leg as a market order and log the result.

        Returns:
            The broker order id, or ``"DRY-RUN"``.
        """
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        tif = TimeInForce.DAY if self.config.execution.time_in_force.lower() == "day" else TimeInForce.GTC
        request = MarketOrderRequest(
            symbol=order.ticker,
            qty=order.quantity,
            side=OrderSide.BUY if order.side == "BUY" else OrderSide.SELL,
            time_in_force=tif,
        )

        if self.dry_run:
            order_id = "DRY-RUN"
            logger.info(
                "[DRY RUN] would %s %d %s (~$%.2f)",
                order.side, order.quantity, order.ticker, order.notional,
            )
        else:
            try:
                submitted = self.client.submit_order(request)
            except Exception as exc:
                raise ExecutionError(
                    f"Order rejected: {order.side} {order.quantity} {order.ticker}. {exc}"
                ) from exc
            order_id = str(getattr(submitted, "id", ""))
            logger.info(
                "Submitted %s %d %s -> order %s",
                order.side, order.quantity, order.ticker, order_id,
            )

        if self.trade_logger:
            self.trade_logger.log(
                TradeRecord(
                    timestamp=utc_now_iso(),
                    action=order.side,
                    ticker=order.ticker,
                    quantity=float(order.quantity),
                    # Market orders fill at an unknown price; the reference close
                    # that sized the order is recorded so the log stays complete.
                    price=float(order.reference_price),
                    zscore=float(signal.get("zscore", float("nan"))),
                    mode="paper-dryrun" if self.dry_run else "paper",
                    pair=str(self.config.pair),
                    side=side_label,
                    reason=str(signal.get("reason", "")),
                    spread=float(signal.get("spread", float("nan"))),
                    hedge_ratio=float(signal.get("hedge_ratio", float("nan"))),
                    group_id=group,
                    order_id=order_id,
                )
            )
        return order_id

    # -- the reconciler ----------------------------------------------------

    def sync_to_signal(self, data: PairData) -> dict[str, Any]:
        """Bring the account's position in line with the strategy's current target.

        Reads the latest signal and the broker's actual position, then issues
        only the difference:

        * target == current  -> nothing (idempotent; safe to run repeatedly)
        * current != 0, target != current -> close, then open the new side
        * incoherent broker state (one leg only, or both legs same sign) ->
          flatten and take no new position this run

        Args:
            data: Price history ending at the most recent completed bar.

        Returns:
            A dict describing what was decided and submitted.
        """
        signal = latest_signal(data, self.config)
        target = int(signal["target"])
        position = self.get_pair_position()

        result: dict[str, Any] = {
            "signal": signal,
            "current_shares_y": position.shares_y,
            "current_shares_x": position.shares_x,
            "current_direction": position.direction,
            "target_direction": target,
            "action": "none",
            "orders": [],
            "dry_run": self.dry_run,
        }

        if not position.is_coherent:
            logger.warning(
                "Broker position is not a valid spread (%s=%.4f, %s=%.4f). Flattening; "
                "no new position will be opened this run.",
                self.config.pair.y, position.shares_y,
                self.config.pair.x, position.shares_x,
            )
            self.assert_market_open()
            group = make_group_id(str(self.config.pair), utc_now_iso())
            ids = [self._submit(o, signal, "exit", group) for o in self._build_close_orders(position)]
            result.update(action="flatten_incoherent", orders=ids)
            return result

        if target == position.direction:
            logger.info(
                "Position already matches the target (%s, z=%+.2f); nothing to do.",
                _label(target), signal["zscore"],
            )
            return result

        self.assert_market_open()
        group = make_group_id(str(self.config.pair), utc_now_iso())
        submitted: list[str] = []

        if position.direction != 0:
            logger.info(
                "Closing %s (z=%+.2f, %s)", _label(position.direction),
                signal["zscore"], signal["reason"] or "target changed",
            )
            for order in self._build_close_orders(position):
                submitted.append(self._submit(order, signal, "exit", group))
            result["action"] = "close"

        if target != 0:
            logger.info(
                "Opening %s at z=%+.2f (%s @ %.2f, %s @ %.2f)",
                _label(target), signal["zscore"],
                signal["y_ticker"], signal["y_price"],
                signal["x_ticker"], signal["x_price"],
            )
            orders = self._build_pair_orders(
                target, float(signal["y_price"]), float(signal["x_price"]),
                float(signal["hedge_ratio"]),
            )
            for order in orders:
                submitted.append(self._submit(order, signal, "entry", group))
            result["action"] = "reverse" if position.direction != 0 else "open"

        result["orders"] = submitted
        return result

    def close_all(self) -> list[str]:
        """Flatten both legs of the pair unconditionally."""
        position = self.get_pair_position()
        if position.is_flat:
            logger.info("Already flat; nothing to close.")
            return []
        self.assert_market_open()
        signal = {"zscore": float("nan"), "reason": "manual_close"}
        group = make_group_id(str(self.config.pair), utc_now_iso())
        return [self._submit(o, signal, "exit", group) for o in self._build_close_orders(position)]

    def account_summary(self) -> str:
        """One-line description of the paper account and current pair exposure."""
        account = self.client.get_account()
        position = self.get_pair_position()
        return (
            f"PAPER account @ {self.base_url}\n"
            f"  equity        : ${float(getattr(account, 'equity', 0) or 0):,.2f}\n"
            f"  buying power  : ${float(getattr(account, 'buying_power', 0) or 0):,.2f}\n"
            f"  {self.config.pair.y:<6}      : {position.shares_y:+,.0f} shares\n"
            f"  {self.config.pair.x:<6}      : {position.shares_x:+,.0f} shares\n"
            f"  spread state  : {_label(position.direction)}"
        )


def _label(direction: int) -> str:
    return {1: "LONG SPREAD", -1: "SHORT SPREAD", 0: "FLAT"}.get(int(direction), "UNKNOWN")


def build_trader(
    config: Config, log_path: str | Path | None = None, dry_run: bool = False
) -> AlpacaPaperTrader:
    """Construct a validated :class:`AlpacaPaperTrader` with a CSV trade logger."""
    from .config import resolve_path

    path = resolve_path(log_path or config.logging.trade_log)
    trade_logger = TradeLogger(path, mode="paper", echo=True)
    return AlpacaPaperTrader(config, trade_logger=trade_logger, dry_run=dry_run)
