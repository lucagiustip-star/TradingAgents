"""Pre-trade risk control. No order reaches Alpaca without clearance from here.

WHY A TOKEN INSTEAD OF A FUNCTION CALL
--------------------------------------
"Every order call routes through validate_order() first" is easy to state and
easy to violate: someone adds a new order path six months from now and forgets.
So the interlock here is structural rather than conventional.

:meth:`RiskManager.validate_order` does not return a boolean. It returns a
:class:`RiskClearance` -- a token that fingerprints the exact orders it
approved. ``AlpacaPaperTrader._submit`` requires one, and verifies that the
order in its hand is among the ones the token approved. A caller who skips the
risk manager has no token, cannot forge one (the fingerprint is checked, not
just the token's presence), and gets :class:`RiskError` instead of a fill.

VALIDATION IS ATOMIC OVER A PAIR
--------------------------------
A pairs trade is two legs that are only safe together. If leg one filled and
leg two were then rejected for breaching an exposure cap, the account would be
left holding a naked directional position -- precisely the exposure the strategy
exists to avoid. So clearance is granted over the *whole* leg set at once,
before any of it is submitted, or not at all.

EXITS ARE NOT ENTRIES
---------------------
Every check here exists to stop the book getting *bigger*. Applying them to
closing orders would be actively dangerous: a breached daily-loss limit would
block the very orders that flatten the book, and a halt flag would trap an open
position indefinitely. So orders declared ``intent="exit"`` skip the exposure,
loss-limit and halt checks. They are still logged, and still require the market
to be open, because an exit filled at an unknown price hours later is its own
kind of risk.

THE CIRCUIT BREAKER LATCHES
---------------------------
Breaching the daily loss limit writes the same ``TRADING_HALTED`` file the kill
switch uses. That is deliberate: a flag on disk survives the process, so the
system cannot quietly resume when the clock rolls past midnight and "today's
loss" resets to zero. Clearing it is a manual act (``kill_switch.py --clear``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import smtplib
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Literal

from .config import Config, resolve_path

logger = logging.getLogger(__name__)

Intent = Literal["entry", "exit"]

# Event types written to risk_events.csv.
EVENT_REJECTED = "order_rejected"
EVENT_APPROVED = "order_approved"
EVENT_BREACH = "daily_loss_breach"
EVENT_HALT_SET = "trading_halted"
EVENT_HALT_CLEARED = "halt_cleared"
EVENT_KILL_SWITCH = "kill_switch"


class RiskError(RuntimeError):
    """Raised when an order is submitted without valid risk clearance."""


class RiskRejection(RuntimeError):
    """Raised when a proposed order fails one or more pre-trade checks.

    Attributes:
        reasons: One human-readable string per failed check.
        snapshot: Account state at the time of the decision.
    """

    def __init__(self, reasons: list[str], snapshot: AccountSnapshot | None = None) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.snapshot = snapshot


class TradingHaltedError(RiskRejection):
    """Raised when the ``TRADING_HALTED`` flag is present.

    Distinct from a generic rejection because it needs a manual clear rather
    than waiting for conditions to change.
    """


# ---------------------------------------------------------------------------
# alerting
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A notification about a risk event.

    Deliberately transport-agnostic: ``severity``/``title``/``message``/
    ``context`` map cleanly onto email, SMS and Slack alike, so adding a channel
    means writing one :class:`AlertChannel` subclass and touching nothing else.
    """

    severity: str            # "info" | "warning" | "critical"
    title: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        lines = [self.message, ""]
        if self.context:
            lines.append("Context:")
            lines += [f"  {k}: {v}" for k, v in self.context.items()]
        return "\n".join(lines)


class AlertChannel(ABC):
    """One delivery mechanism for alerts.

    Implement :meth:`send` and add the instance to an :class:`AlertDispatcher`.
    An SMS or Slack channel is a subclass with the same three-line surface --
    no change to any calling code.
    """

    name: str = "channel"

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver one alert. Return True on success."""

    def is_configured(self) -> bool:
        """Whether this channel has what it needs to deliver."""
        return True


class LogAlertChannel(AlertChannel):
    """Fallback channel that writes alerts to the application log.

    Always configured, so an alert is never silently lost when SMTP is not set
    up -- which is the normal state during paper trading.
    """

    name = "log"

    def send(self, alert: Alert) -> bool:
        level = {"critical": logging.CRITICAL, "warning": logging.WARNING}.get(
            alert.severity, logging.INFO
        )
        logger.log(level, "ALERT [%s] %s\n%s", alert.severity.upper(), alert.title, alert.as_text())
        return True


class EmailAlertChannel(AlertChannel):
    """Email delivery over SMTP, configured entirely from the environment.

    Reads ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``, ``SMTP_PASSWORD``,
    ``SMTP_FROM``, ``SMTP_TO`` (comma-separated) and ``SMTP_USE_TLS``. If the
    host or recipients are absent the channel reports itself unconfigured and is
    skipped rather than raising -- risk alerting must never be the thing that
    breaks a trading run.
    """

    name = "email"

    def __init__(self, timeout: float = 10.0) -> None:
        self.host = os.getenv("SMTP_HOST", "").strip()
        self.port = int(os.getenv("SMTP_PORT", "587") or 587)
        self.user = os.getenv("SMTP_USER", "").strip()
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.sender = os.getenv("SMTP_FROM", "").strip() or self.user
        self.recipients = [
            r.strip() for r in os.getenv("SMTP_TO", "").split(",") if r.strip()
        ]
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() not in ("false", "0", "no")
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.host and self.recipients and self.sender)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            logger.debug("Email alerts not configured (SMTP_HOST/SMTP_TO unset); skipping.")
            return False

        message = EmailMessage()
        message["Subject"] = f"[pairs-trading][{alert.severity.upper()}] {alert.title}"
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(alert.as_text())

        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as smtp:
                if self.use_tls:
                    smtp.starttls()
                if self.user and self.password:
                    smtp.login(self.user, self.password)
                smtp.send_message(message)
            logger.info("Risk alert emailed to %s", ", ".join(self.recipients))
            return True
        except Exception as exc:
            # Never propagate: a dead mail server must not stop the kill switch.
            logger.error("Could not send risk alert email: %s", exc)
            return False


class AlertDispatcher:
    """Fans an alert out to every configured channel.

    Channel failures are swallowed and logged. Alerting is a reporting
    obligation, not a trading precondition -- if the mail server is down, the
    circuit breaker must still trip and the kill switch must still fire.
    """

    def __init__(self, channels: list[AlertChannel] | None = None) -> None:
        self.channels = channels if channels is not None else [LogAlertChannel()]

    @classmethod
    def from_env(cls, include_log: bool = True) -> AlertDispatcher:
        """Build the default dispatcher: the log, plus email when configured."""
        channels: list[AlertChannel] = [LogAlertChannel()] if include_log else []
        email = EmailAlertChannel()
        if email.is_configured():
            channels.append(email)
        return cls(channels)

    def send(self, alert: Alert) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for channel in self.channels:
            try:
                results[channel.name] = channel.send(alert)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Alert channel %r raised: %s", channel.name, exc)
                results[channel.name] = False
        return results


# ---------------------------------------------------------------------------
# risk event log
# ---------------------------------------------------------------------------

@dataclass
class RiskEvent:
    """One row of ``risk_events.csv``: what happened, why, and the state then."""

    timestamp: str
    event: str
    reason: str
    severity: str = "info"
    pair: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    notional: float = 0.0
    intent: str = ""
    equity: float = float("nan")
    daily_pnl: float = float("nan")
    daily_pnl_pct: float = float("nan")
    open_exposure: float = float("nan")
    buying_power: float = float("nan")
    market_open: str = ""
    zscore: float = float("nan")
    detail: str = ""


RISK_CSV_COLUMNS = [f.name for f in fields(RiskEvent)]


class RiskEventLogger:
    """Appends :class:`RiskEvent` rows to a CSV, flushing each one immediately.

    Kept separate from the trade log on purpose: ``trades.csv`` records what the
    account did, ``risk_events.csv`` records what it was stopped from doing. The
    second is the one you read after a bad day.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=RISK_CSV_COLUMNS).writeheader()

    def log(self, event: RiskEvent) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=RISK_CSV_COLUMNS).writerow(asdict(event))
            fh.flush()
            os.fsync(fh.fileno())

    def read(self):
        import pandas as pd

        if not self.path.exists():
            return pd.DataFrame(columns=RISK_CSV_COLUMNS)
        return pd.read_csv(self.path)


# ---------------------------------------------------------------------------
# account state
# ---------------------------------------------------------------------------

@dataclass
class AccountSnapshot:
    """Account state at one moment, as every check sees it.

    Attributes:
        equity: Total account value including unrealised P&L.
        buying_power: What the broker will let you deploy right now.
        open_exposure: Absolute market value across all open positions, both
            legs and both directions. Gross, not net -- a dollar-neutral pair
            has ~zero net exposure but consumes real capital and real risk.
        start_of_day_equity: Baseline for the daily P&L calculation.
        daily_pnl: ``equity - start_of_day_equity``. Because Alpaca's equity
            already marks open positions to market, this single number captures
            realised *and* unrealised P&L for the session.
        market_open: None when unknown (the clock was not consulted).
    """

    timestamp: str
    equity: float
    buying_power: float
    cash: float = 0.0
    open_exposure: float = 0.0
    position_count: int = 0
    start_of_day_equity: float = 0.0
    market_open: bool | None = None

    @property
    def daily_pnl(self) -> float:
        if not self.start_of_day_equity:
            return 0.0
        return self.equity - self.start_of_day_equity

    @property
    def daily_pnl_pct(self) -> float:
        if not self.start_of_day_equity:
            return 0.0
        return self.daily_pnl / self.start_of_day_equity

    def describe(self) -> str:
        return (
            f"equity=${self.equity:,.2f} daily_pnl=${self.daily_pnl:,.2f} "
            f"({self.daily_pnl_pct:+.2%}) exposure=${self.open_exposure:,.2f} "
            f"buying_power=${self.buying_power:,.2f}"
        )


@dataclass
class ProposedOrder:
    """A leg awaiting clearance. Mirrors ``execution_alpaca.LegOrder``."""

    symbol: str
    side: str          # BUY | SELL
    quantity: float
    price: float       # reference price used for sizing

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.price)

    def fingerprint(self) -> str:
        """Stable identity used to bind a clearance to specific orders."""
        return f"{self.symbol}|{self.side}|{self.quantity:.6f}"


@dataclass(frozen=True)
class RiskClearance:
    """Proof that a specific set of orders passed every pre-trade check.

    Required by the submission path. The fingerprints are what make it real
    proof rather than a rubber stamp: a clearance issued for 100 shares does not
    authorise 10,000.
    """

    token: str
    fingerprints: frozenset[str]
    intent: Intent
    issued_at: str
    snapshot: AccountSnapshot
    checks_passed: tuple[str, ...] = ()

    def approves(self, order: ProposedOrder) -> bool:
        return order.fingerprint() in self.fingerprints


# ---------------------------------------------------------------------------
# the risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    """Enforces position limits, exposure caps, a daily-loss breaker and a halt flag.

    Args:
        config: Full configuration; ``config.risk`` holds every limit.
        alerts: Alert dispatcher. Defaults to log + email-if-configured.
        event_logger: Risk-event CSV logger. Defaults to ``risk.risk_log``.
        now_fn: Injectable clock, so tests can cross a day boundary.
    """

    def __init__(
        self,
        config: Config,
        alerts: AlertDispatcher | None = None,
        event_logger: RiskEventLogger | None = None,
        now_fn=None,
    ) -> None:
        self.config = config
        self.risk = config.risk
        self.alerts = alerts if alerts is not None else AlertDispatcher.from_env()
        self.events = event_logger or RiskEventLogger(resolve_path(self.risk.risk_log))
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.halt_file = resolve_path(self.risk.halt_file)
        self.state_file = resolve_path(self.risk.state_file)

    # -- halt flag ---------------------------------------------------------

    def is_halted(self) -> bool:
        """Whether the ``TRADING_HALTED`` flag is present."""
        return self.halt_file.exists()

    def halt_reason(self) -> str:
        """The recorded reason for the current halt, or ``""``.

        Returns just the reason line, not the whole flag file -- the reason gets
        embedded in rejection messages and alerts, and pasting the full file
        (including its own "clear this with..." instructions) into every one of
        them makes the actual cause hard to find.
        """
        if not self.is_halted():
            return ""
        try:
            text = self.halt_file.read_text()
        except OSError:
            return "(halt file present but unreadable)"

        for line in text.splitlines():
            if line.lower().startswith("reason:"):
                return line.split(":", 1)[1].strip()
        return text.strip().splitlines()[0] if text.strip() else "(no reason recorded)"

    def halt_details(self) -> str:
        """The full contents of the halt file, for status displays."""
        if not self.is_halted():
            return ""
        try:
            return self.halt_file.read_text().strip()
        except OSError:
            return "(halt file present but unreadable)"

    def halt(self, reason: str, snapshot: AccountSnapshot | None = None,
             event: str = EVENT_HALT_SET) -> Path:
        """Write the halt flag, log the event and alert.

        The flag is a file rather than in-memory state precisely so it survives
        the process. Nothing in this codebase removes it automatically.
        """
        stamp = self._now().isoformat(timespec="seconds")
        self.halt_file.parent.mkdir(parents=True, exist_ok=True)
        body = (
            f"TRADING HALTED\ntimestamp: {stamp}\nreason: {reason}\n"
            + (f"account: {snapshot.describe()}\n" if snapshot else "")
            + "\nClear this file to resume trading:\n"
            "    python -m pairs_trading.kill_switch --clear\n"
        )
        self.halt_file.write_text(body)

        self._record(event, reason, snapshot, severity="critical")
        self.alerts.send(
            Alert(
                severity="critical",
                title="Trading halted",
                message=f"Trading has been halted and will not resume until manually cleared.\n\nReason: {reason}",
                context={
                    "timestamp": stamp,
                    "halt_file": str(self.halt_file),
                    "account": snapshot.describe() if snapshot else "unknown",
                    "clear_with": "python -m pairs_trading.kill_switch --clear",
                },
            )
        )
        logger.critical("TRADING HALTED: %s (flag: %s)", reason, self.halt_file)
        return self.halt_file

    def clear_halt(self, note: str = "manual clear") -> bool:
        """Remove the halt flag. Only ever called from an explicit human action."""
        if not self.is_halted():
            return False
        self.halt_file.unlink()
        self._record(EVENT_HALT_CLEARED, note, None, severity="warning")
        logger.warning("Trading halt cleared: %s", note)
        return True

    # -- daily P&L baseline ------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable risk state file %s: %s", self.state_file, exc)
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))

    def start_of_day_equity(self, current_equity: float, last_equity: float | None = None) -> float:
        """Return today's opening equity, establishing it on first call of the day.

        Persisted to disk so that a process restarted mid-session measures the
        day's loss from the true session open rather than from the restart --
        otherwise a crash-and-restart would silently reset the circuit breaker.

        ``last_equity`` (Alpaca's previous-close equity) seeds the baseline when
        available, which is more accurate than the equity at whatever time the
        process happened to start.
        """
        today = self._now().date().isoformat()
        state = self._load_state()

        if state.get("date") == today and state.get("start_of_day_equity"):
            return float(state["start_of_day_equity"])

        baseline = self._plausible_baseline(float(current_equity), last_equity)
        state.update({"date": today, "start_of_day_equity": baseline})
        self._save_state(state)
        logger.info("Start-of-day equity baseline for %s: $%,.2f", today, baseline)
        return baseline

    @staticmethod
    def _plausible_baseline(current_equity: float, last_equity: float | None) -> float:
        """Choose a defensible start-of-day baseline.

        ``last_equity`` (the broker's previous close) is preferred, but only
        when it is plausible. A garbage value here is quietly catastrophic: the
        baseline is the denominator of every daily-loss decision, so a
        nonsensical one either disables the circuit breaker entirely or trips it
        on the first check. Anything outside half to double the current equity
        is rejected in favour of the equity we can actually see, since a real
        session does not move an account by more than that.
        """
        if current_equity <= 0:
            return max(float(last_equity or 0.0), 0.0)
        if last_equity is None:
            return current_equity

        candidate = float(last_equity)
        if candidate > 0 and 0.5 <= candidate / current_equity <= 2.0:
            return candidate

        logger.warning(
            "Ignoring implausible last_equity %.2f against current equity %.2f; using current "
            "equity as today's baseline. A bad baseline would corrupt every daily-loss check.",
            candidate, current_equity,
        )
        return current_equity

    def snapshot(
        self,
        equity: float,
        buying_power: float,
        cash: float = 0.0,
        open_exposure: float = 0.0,
        position_count: int = 0,
        last_equity: float | None = None,
        market_open: bool | None = None,
    ) -> AccountSnapshot:
        """Build an :class:`AccountSnapshot`, resolving the daily baseline."""
        baseline = self.start_of_day_equity(equity, last_equity)
        return AccountSnapshot(
            timestamp=self._now().isoformat(timespec="seconds"),
            equity=float(equity),
            buying_power=float(buying_power),
            cash=float(cash),
            open_exposure=float(open_exposure),
            position_count=int(position_count),
            start_of_day_equity=baseline,
            market_open=market_open,
        )

    # -- limits ------------------------------------------------------------

    def daily_loss_limit(self, snapshot: AccountSnapshot) -> float:
        """Resolve the loss limit to a positive dollar figure.

        Both an absolute cap and a percentage-of-equity cap may be configured;
        when both are set the *tighter* one wins, which is the conservative
        reading of two limits that were presumably both meant to bind.
        """
        limits = []
        if self.risk.max_daily_loss_usd is not None:
            limits.append(abs(float(self.risk.max_daily_loss_usd)))
        if self.risk.max_daily_loss_pct is not None:
            base = snapshot.start_of_day_equity or snapshot.equity
            limits.append(abs(float(self.risk.max_daily_loss_pct)) * base)
        return min(limits) if limits else float("inf")

    def check_daily_loss(self, snapshot: AccountSnapshot, auto_halt: bool = True) -> bool:
        """Test the circuit breaker. Returns True when the limit is breached.

        On breach: logs, alerts, and (when ``auto_halt``) latches the halt flag
        so trading cannot resume without a manual clear. The caller is
        responsible for closing positions if
        ``risk.close_positions_on_breach`` is set -- doing it here would make
        this method's contract far larger than "check a number".
        """
        limit = self.daily_loss_limit(snapshot)
        loss = -snapshot.daily_pnl  # positive when losing
        if loss < limit:
            return False

        reason = (
            f"Daily loss limit breached: lost ${loss:,.2f} today "
            f"({snapshot.daily_pnl_pct:+.2%}), limit ${limit:,.2f}. "
            f"Start-of-day equity ${snapshot.start_of_day_equity:,.2f}, "
            f"now ${snapshot.equity:,.2f}."
        )
        self._record(EVENT_BREACH, reason, snapshot, severity="critical")
        self.alerts.send(
            Alert(
                severity="critical",
                title="Daily loss circuit breaker tripped",
                message=reason,
                context={
                    "timestamp": snapshot.timestamp,
                    "daily_pnl": f"${snapshot.daily_pnl:,.2f}",
                    "limit": f"${limit:,.2f}",
                    "equity": f"${snapshot.equity:,.2f}",
                    "close_positions": self.risk.close_positions_on_breach,
                },
            )
        )
        if auto_halt and not self.is_halted():
            self.halt(reason, snapshot, event=EVENT_BREACH)
        return True

    # -- the pre-trade checklist ------------------------------------------

    def validate_order(
        self,
        orders: list[ProposedOrder],
        snapshot: AccountSnapshot,
        intent: Intent = "entry",
        context: dict[str, Any] | None = None,
    ) -> RiskClearance:
        """Run every pre-trade check over a complete order set.

        Checks, in order (cheapest and most decisive first):

        1. ``TRADING_HALTED`` flag absent            (entries only)
        2. Market is open
        3. Daily loss limit not breached             (entries only)
        4. Per-trade notional within ``max_position_size_usd``   (entries only)
        5. Resulting total exposure within ``max_total_exposure_usd`` (entries only)
        6. Sufficient buying power for the order

        Exits skip 1, 3, 4 and 5 -- see the module docstring. A failing check
        does not resize the order; it rejects it, because a silently shrunk
        position is a different trade from the one the strategy asked for.

        Returns:
            A :class:`RiskClearance` binding this exact order set.

        Raises:
            TradingHaltedError: when the halt flag is present.
            RiskRejection: when any other check fails. ``.reasons`` lists all
                failures, not just the first, so one round trip surfaces
                everything wrong.
        """
        context = context or {}
        reasons: list[str] = []
        passed: list[str] = []
        is_entry = intent == "entry"
        total_notional = sum(o.notional for o in orders)

        if not orders:
            raise RiskRejection(["No orders proposed."], snapshot)

        # 1. halt flag
        if is_entry and self.is_halted():
            reason = (
                f"TRADING_HALTED flag is present ({self.halt_file}): {self.halt_reason()}. "
                "Clear it with `python -m pairs_trading.kill_switch --clear` to resume."
            )
            self._record_orders(EVENT_REJECTED, reason, orders, snapshot, intent, context, "critical")
            raise TradingHaltedError([reason], snapshot)
        passed.append("halt_flag")

        # 2. market hours
        if self.risk.require_market_open and snapshot.market_open is False:
            reasons.append(
                "The market is closed. An order queued now fills at the next open, at a price "
                "unrelated to the signal that triggered it."
            )
        else:
            passed.append("market_open")

        # 3. daily loss breaker
        if is_entry:
            if self.check_daily_loss(snapshot):
                reasons.append(
                    f"Daily loss limit breached (P&L ${snapshot.daily_pnl:,.2f}, "
                    f"limit ${self.daily_loss_limit(snapshot):,.2f}); no new positions."
                )
            else:
                passed.append("daily_loss")

            # 4. per-trade size cap
            cap = self.risk.max_position_size_usd
            if cap is not None and total_notional > cap:
                reasons.append(
                    f"Order notional ${total_notional:,.2f} exceeds max_position_size_usd "
                    f"${cap:,.2f}. Rejected rather than resized -- a shrunk position is a "
                    "different trade from the one the strategy signalled."
                )
            else:
                passed.append("position_size")

            # 5. aggregate exposure cap
            total_cap = self.risk.max_total_exposure_usd
            projected = snapshot.open_exposure + total_notional
            if total_cap is not None and projected > total_cap:
                reasons.append(
                    f"Projected total exposure ${projected:,.2f} (open ${snapshot.open_exposure:,.2f} "
                    f"+ new ${total_notional:,.2f}) exceeds max_total_exposure_usd ${total_cap:,.2f}."
                )
            else:
                passed.append("total_exposure")

        # 6. buying power
        required = total_notional + float(self.risk.buying_power_buffer_usd or 0.0)
        if snapshot.buying_power < required:
            reasons.append(
                f"Insufficient buying power: need ${required:,.2f} "
                f"(order ${total_notional:,.2f} + buffer "
                f"${float(self.risk.buying_power_buffer_usd or 0.0):,.2f}), "
                f"have ${snapshot.buying_power:,.2f}."
            )
        else:
            passed.append("buying_power")

        if reasons:
            self._record_orders(
                EVENT_REJECTED, " | ".join(reasons), orders, snapshot, intent, context, "warning"
            )
            self.alerts.send(
                Alert(
                    severity="warning",
                    title=f"Order rejected by risk checks ({context.get('pair', 'pair')})",
                    message="A proposed order was blocked before submission:\n\n- "
                    + "\n- ".join(reasons),
                    context={
                        "timestamp": snapshot.timestamp,
                        "intent": intent,
                        "orders": ", ".join(
                            f"{o.side} {o.quantity:g} {o.symbol} (${o.notional:,.2f})"
                            for o in orders
                        ),
                        "account": snapshot.describe(),
                    },
                )
            )
            raise RiskRejection(reasons, snapshot)

        clearance = RiskClearance(
            token=uuid.uuid4().hex,
            fingerprints=frozenset(o.fingerprint() for o in orders),
            intent=intent,
            issued_at=snapshot.timestamp,
            snapshot=snapshot,
            checks_passed=tuple(passed),
        )
        self._record_orders(
            EVENT_APPROVED,
            f"passed {len(passed)} checks: {', '.join(passed)}",
            orders, snapshot, intent, context, "info",
        )
        logger.info(
            "Risk clearance %s issued for %d %s leg(s), $%,.2f notional",
            clearance.token[:8], len(orders), intent, total_notional,
        )
        return clearance

    def require_clearance(self, order: ProposedOrder, clearance: RiskClearance | None) -> None:
        """Assert that ``order`` is covered by ``clearance``.

        Called at the submission boundary. This is the check that makes the risk
        layer impossible to route around: no token, or a token issued for
        different orders, means no fill.
        """
        if clearance is None:
            raise RiskError(
                f"Refusing to submit {order.side} {order.quantity:g} {order.symbol}: no risk "
                "clearance. Every order must pass RiskManager.validate_order() first."
            )
        if not clearance.approves(order):
            raise RiskError(
                f"Risk clearance {clearance.token[:8]} does not cover "
                f"{order.side} {order.quantity:g} {order.symbol}. A clearance authorises only "
                "the exact orders it was issued for."
            )

    # -- logging helpers ---------------------------------------------------

    def _record(
        self, event: str, reason: str, snapshot: AccountSnapshot | None,
        severity: str = "info", **extra: Any,
    ) -> None:
        self.events.log(
            RiskEvent(
                timestamp=(snapshot.timestamp if snapshot else self._now().isoformat(timespec="seconds")),
                event=event,
                reason=reason,
                severity=severity,
                equity=snapshot.equity if snapshot else float("nan"),
                daily_pnl=snapshot.daily_pnl if snapshot else float("nan"),
                daily_pnl_pct=snapshot.daily_pnl_pct if snapshot else float("nan"),
                open_exposure=snapshot.open_exposure if snapshot else float("nan"),
                buying_power=snapshot.buying_power if snapshot else float("nan"),
                market_open="" if not snapshot or snapshot.market_open is None
                else str(snapshot.market_open),
                **extra,
            )
        )

    def _record_orders(
        self, event: str, reason: str, orders: list[ProposedOrder],
        snapshot: AccountSnapshot, intent: str, context: dict[str, Any], severity: str,
    ) -> None:
        """Write one risk-event row per leg, so the CSV is joinable with trades.csv."""
        for order in orders:
            self._record(
                event, reason, snapshot, severity,
                pair=str(context.get("pair", "")),
                symbol=order.symbol,
                side=order.side,
                quantity=float(order.quantity),
                notional=order.notional,
                intent=intent,
                zscore=float(context.get("zscore", float("nan"))),
                detail=str(context.get("detail", "")),
            )


def orders_from_legs(legs) -> list[ProposedOrder]:
    """Adapt ``execution_alpaca.LegOrder`` objects into :class:`ProposedOrder`.

    Keeps the risk layer free of any dependency on the execution module, so it
    can be tested -- and reused against another broker -- on its own.
    """
    return [
        ProposedOrder(
            symbol=leg.ticker,
            side=leg.side,
            quantity=float(leg.quantity),
            price=float(leg.reference_price),
        )
        for leg in legs
    ]


def today_iso() -> str:
    return date.today().isoformat()
