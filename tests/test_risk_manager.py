"""Tests for the pre-trade risk layer.

Covers the three scenarios that matter most and a few that are easy to get
wrong:

* a trade exceeding the position-size cap is rejected, not resized
* a sequence of losses breaches the daily limit, halts trading, and does not
  un-halt when the calendar rolls over
* a kill switch mid-session blocks all subsequent entries
* no order can reach the broker without a clearance token issued for that exact
  order

Everything runs offline against a fake Alpaca client and tmp-path config, so no
network, no credentials, and no files outside the test's own directory.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from pairs_trading.config import ConfigError, RiskConfig, load_config
from pairs_trading.risk_manager import (
    EVENT_BREACH,
    EVENT_REJECTED,
    Alert,
    AlertChannel,
    AlertDispatcher,
    EmailAlertChannel,
    ProposedOrder,
    RiskError,
    RiskEventLogger,
    RiskManager,
    RiskRejection,
    TradingHaltedError,
)

pytestmark = pytest.mark.unit


class CapturingChannel(AlertChannel):
    """Records alerts instead of sending them."""

    name = "capture"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True

    def titles(self) -> list[str]:
        return [a.title for a in self.alerts]


@pytest.fixture
def alerts():
    return CapturingChannel()


@pytest.fixture
def config(tmp_path):
    """Config with every risk artefact redirected into tmp_path."""
    cfg = load_config()
    return replace(
        cfg,
        risk=replace(
            cfg.risk,
            max_position_size_usd=5_000.0,
            max_total_exposure_usd=20_000.0,
            max_daily_loss_usd=1_000.0,
            halt_file=str(tmp_path / "TRADING_HALTED"),
            state_file=str(tmp_path / "risk_state.json"),
            risk_log=str(tmp_path / "risk_events.csv"),
        ),
    )


@pytest.fixture
def clock():
    """A mutable clock so tests can cross a day boundary deliberately."""

    class Clock:
        def __init__(self):
            self.now = datetime(2026, 8, 11, 15, 0, tzinfo=timezone.utc)

        def __call__(self):
            return self.now

        def advance_days(self, n=1):
            self.now = self.now + timedelta(days=n)

    return Clock()


@pytest.fixture
def risk(config, alerts, clock):
    return RiskManager(config, alerts=AlertDispatcher([alerts]), now_fn=clock)


def snap(
    risk, equity=100_000.0, buying_power=200_000.0, exposure=0.0,
    market_open=True, last_equity=100_000.0,
):
    """Build a snapshot. `last_equity` seeds the baseline on a fresh day."""
    return risk.snapshot(
        equity=equity, buying_power=buying_power, open_exposure=exposure,
        market_open=market_open, last_equity=last_equity,
    )


def orders(notional=2_000.0, price=100.0):
    """A two-leg pair order totalling `notional` dollars gross."""
    qty = notional / 2 / price
    return [
        ProposedOrder("AAA", "BUY", qty, price),
        ProposedOrder("BBB", "SELL", qty, price),
    ]


# --------------------------------------------------------------------------
# 1. position size and exposure limits
# --------------------------------------------------------------------------

class TestPositionSizeLimits:
    def test_order_within_limits_is_cleared(self, risk):
        clearance = risk.validate_order(orders(2_000.0), snap(risk))
        assert clearance.token
        assert "position_size" in clearance.checks_passed
        assert "total_exposure" in clearance.checks_passed

    def test_oversized_trade_is_rejected(self, risk):
        with pytest.raises(RiskRejection) as exc:
            risk.validate_order(orders(9_000.0), snap(risk))
        assert any("max_position_size_usd" in r for r in exc.value.reasons)

    def test_oversized_trade_is_not_silently_resized(self, risk):
        """The rejection must be total -- no clearance for a smaller size."""
        with pytest.raises(RiskRejection):
            risk.validate_order(orders(9_000.0), snap(risk))
        # Nothing was approved, so nothing could be submitted at any size.
        events = RiskEventLogger(risk.events.path).read()
        assert (events["event"] == EVENT_REJECTED).any()
        assert not (events["event"] == "order_approved").any()

    def test_total_exposure_cap_accounts_for_open_positions(self, risk):
        """A trade that fits on its own can still breach the aggregate cap."""
        # 4,000 alone is fine; on top of 18,000 already open it is not.
        risk.validate_order(orders(4_000.0), snap(risk, exposure=0.0))
        with pytest.raises(RiskRejection) as exc:
            risk.validate_order(orders(4_000.0), snap(risk, exposure=18_000.0))
        assert any("max_total_exposure_usd" in r for r in exc.value.reasons)

    def test_insufficient_buying_power_is_rejected(self, risk):
        with pytest.raises(RiskRejection) as exc:
            risk.validate_order(orders(4_000.0), snap(risk, buying_power=100.0))
        assert any("buying power" in r.lower() for r in exc.value.reasons)

    def test_closed_market_is_rejected(self, risk):
        with pytest.raises(RiskRejection) as exc:
            risk.validate_order(orders(2_000.0), snap(risk, market_open=False))
        assert any("market is closed" in r.lower() for r in exc.value.reasons)

    def test_all_failures_are_reported_together(self, risk):
        """One round trip should surface everything wrong, not just the first."""
        with pytest.raises(RiskRejection) as exc:
            risk.validate_order(
                orders(9_000.0), snap(risk, buying_power=10.0, market_open=False)
            )
        assert len(exc.value.reasons) >= 3

    def test_rejection_is_logged_with_account_context(self, risk):
        with pytest.raises(RiskRejection):
            risk.validate_order(orders(9_000.0), snap(risk, equity=98_500.0))
        events = risk.events.read()
        row = events[events["event"] == EVENT_REJECTED].iloc[0]
        assert row["equity"] == pytest.approx(98_500.0)
        assert row["symbol"] in ("AAA", "BBB")
        assert row["notional"] > 0

    def test_rejection_raises_an_alert(self, risk, alerts):
        with pytest.raises(RiskRejection):
            risk.validate_order(orders(9_000.0), snap(risk))
        assert any("rejected" in t.lower() for t in alerts.titles())

    def test_limits_can_be_disabled(self, config, alerts, clock):
        cfg = replace(config, risk=replace(
            config.risk, max_position_size_usd=None, max_total_exposure_usd=None
        ))
        manager = RiskManager(cfg, alerts=AlertDispatcher([alerts]), now_fn=clock)
        assert manager.validate_order(orders(500_000.0), snap(manager, buying_power=1e9))


# --------------------------------------------------------------------------
# 2. daily loss circuit breaker
# --------------------------------------------------------------------------

class TestDailyLossCircuitBreaker:
    def test_losses_below_the_limit_do_not_trip(self, risk):
        assert not risk.check_daily_loss(snap(risk, equity=99_500.0))
        assert not risk.is_halted()

    def test_sequence_of_losses_breaches_the_limit(self, risk, alerts):
        """Walk equity down across several checks until the limit is crossed."""
        for equity in (99_800.0, 99_400.0, 99_100.0):
            assert not risk.check_daily_loss(snap(risk, equity=equity))
            assert not risk.is_halted()

        # This one crosses the $1,000 limit.
        assert risk.check_daily_loss(snap(risk, equity=98_900.0))
        assert risk.is_halted()
        assert any("circuit breaker" in t.lower() for t in alerts.titles())

    def test_breach_is_logged_with_pnl_and_timestamp(self, risk):
        risk.check_daily_loss(snap(risk, equity=98_000.0))
        events = risk.events.read()
        breach = events[events["event"] == EVENT_BREACH].iloc[0]
        assert breach["daily_pnl"] == pytest.approx(-2_000.0)
        assert breach["timestamp"]
        assert "Daily loss limit breached" in breach["reason"]

    def test_breach_blocks_subsequent_entries(self, risk):
        risk.check_daily_loss(snap(risk, equity=98_000.0))
        with pytest.raises(TradingHaltedError):
            risk.validate_order(orders(1_000.0), snap(risk, equity=98_000.0))

    def test_breach_does_not_block_exits(self, risk):
        """Flattening must still work -- the breaker exists to shrink the book."""
        risk.check_daily_loss(snap(risk, equity=98_000.0))
        assert risk.is_halted()
        clearance = risk.validate_order(
            orders(1_000.0), snap(risk, equity=98_000.0), intent="exit"
        )
        assert clearance.intent == "exit"

    def test_halt_does_not_auto_clear_the_next_day(self, risk, clock):
        """The whole point of latching: a new day must not resume trading.

        Daily P&L resets when the date rolls over, so a breaker that only
        checked today's loss would silently re-enable trading overnight.
        """
        risk.check_daily_loss(snap(risk, equity=98_000.0))
        assert risk.is_halted()

        # Next session: yesterday's close (98k) becomes today's baseline, so
        # the day's P&L is flat and a naive breaker would happily resume.
        clock.advance_days(1)
        fresh = snap(risk, equity=98_000.0, last_equity=98_000.0)
        assert fresh.daily_pnl == pytest.approx(0.0)   # new day, flat P&L
        assert risk.is_halted()                        # but still halted
        with pytest.raises(TradingHaltedError):
            risk.validate_order(orders(1_000.0), fresh)

    def test_manual_clear_resumes_trading(self, risk, clock):
        risk.check_daily_loss(snap(risk, equity=98_000.0))
        assert risk.is_halted()
        assert risk.clear_halt("reviewed and resuming")
        assert not risk.is_halted()
        clock.advance_days(1)
        assert risk.validate_order(orders(1_000.0), snap(risk))

    def test_percentage_limit_binds(self, config, alerts, clock):
        cfg = replace(config, risk=replace(
            config.risk, max_daily_loss_usd=None, max_daily_loss_pct=0.005
        ))
        manager = RiskManager(cfg, alerts=AlertDispatcher([alerts]), now_fn=clock)
        assert not manager.check_daily_loss(snap(manager, equity=99_600.0))
        assert manager.check_daily_loss(snap(manager, equity=99_400.0))

    def test_tighter_of_two_limits_wins(self, config, alerts, clock):
        cfg = replace(config, risk=replace(
            config.risk, max_daily_loss_usd=1_000.0, max_daily_loss_pct=0.002
        ))
        manager = RiskManager(cfg, alerts=AlertDispatcher([alerts]), now_fn=clock)
        # 0.2% of 100k = $200, tighter than the $1,000 absolute cap.
        assert manager.daily_loss_limit(snap(manager)) == pytest.approx(200.0)

    def test_baseline_survives_a_restart(self, config, alerts, clock):
        """A process restarted mid-session must not reset the loss measurement."""
        first = RiskManager(config, alerts=AlertDispatcher([alerts]), now_fn=clock)
        snap(first, equity=100_000.0)          # establishes the baseline

        # New manager instance = new process; equity has since fallen.
        second = RiskManager(config, alerts=AlertDispatcher([alerts]), now_fn=clock)
        state = snap(second, equity=99_000.0)
        assert state.start_of_day_equity == pytest.approx(100_000.0)
        assert state.daily_pnl == pytest.approx(-1_000.0)

    def test_baseline_resets_on_a_new_day(self, risk, clock):
        snap(risk, equity=100_000.0)
        clock.advance_days(1)
        # A new session seeds its baseline from the previous close.
        state = snap(risk, equity=95_000.0, last_equity=96_000.0)
        assert state.start_of_day_equity == pytest.approx(96_000.0)
        assert state.daily_pnl == pytest.approx(-1_000.0)


# --------------------------------------------------------------------------
# 3. kill switch / halt flag
# --------------------------------------------------------------------------

class TestKillSwitch:
    def test_halt_flag_blocks_entries(self, risk):
        risk.halt("manual test halt")
        with pytest.raises(TradingHaltedError) as exc:
            risk.validate_order(orders(1_000.0), snap(risk))
        assert "TRADING_HALTED" in str(exc.value)

    def test_halt_file_records_the_reason(self, risk):
        risk.halt("spread relationship broke down")
        assert risk.is_halted()
        assert "spread relationship broke down" in risk.halt_reason()
        assert "kill_switch --clear" in risk.halt_file.read_text()

    def test_kill_switch_mid_session_stops_trading(self, config, risk, monkeypatch):
        """Trade freely, fire the real kill switch, then find entries blocked."""
        from pairs_trading import kill_switch

        assert risk.validate_order(orders(1_000.0), snap(risk))     # before

        broker = MagicMock()
        broker.get_orders.return_value = [MagicMock(), MagicMock()]
        broker.get_all_positions.return_value = [MagicMock(symbol="AAA")]
        trader = MagicMock()
        trader.client = broker
        monkeypatch.setattr(
            "pairs_trading.execution_alpaca.AlpacaPaperTrader", MagicMock(return_value=trader)
        )

        summary = kill_switch.activate(config, reason="mid-session stop")
        assert summary["halted"] is True
        assert summary["cancelled"] == 2
        assert summary["closed"] == 1
        assert not summary["errors"]
        broker.cancel_orders.assert_called_once()
        broker.close_all_positions.assert_called_once()

        with pytest.raises(TradingHaltedError):
            risk.validate_order(orders(1_000.0), snap(risk))         # after

    def test_kill_switch_dry_run_halts_but_touches_nothing(self, config, monkeypatch):
        """A dry run that left trading enabled would be a trap, so it still halts."""
        from pairs_trading import kill_switch

        broker = MagicMock()
        broker.get_orders.return_value = [MagicMock()]
        broker.get_all_positions.return_value = [MagicMock(symbol="AAA")]
        trader = MagicMock()
        trader.client = broker
        monkeypatch.setattr(
            "pairs_trading.execution_alpaca.AlpacaPaperTrader", MagicMock(return_value=trader)
        )

        summary = kill_switch.activate(config, reason="drill", dry_run=True)
        assert summary["halted"] is True
        broker.cancel_orders.assert_not_called()
        broker.close_all_positions.assert_not_called()

    def test_kill_switch_activate_halts_even_when_broker_fails(self, config, monkeypatch):
        """Halting must not depend on the broker being reachable."""
        from pairs_trading import kill_switch

        monkeypatch.setattr(
            "pairs_trading.execution_alpaca.AlpacaPaperTrader",
            MagicMock(side_effect=RuntimeError("alpaca unreachable")),
        )
        summary = kill_switch.activate(config, reason="test", dry_run=False)
        assert summary["halted"] is True
        assert any("alpaca" in e.lower() for e in summary["errors"])

    def test_status_and_clear_round_trip(self, config):
        from pairs_trading import kill_switch

        assert kill_switch.status(config)["halted"] is False
        RiskManager(config).halt("test")
        assert kill_switch.status(config)["halted"] is True
        assert kill_switch.clear(config) is True
        assert kill_switch.status(config)["halted"] is False

    def test_clearing_when_not_halted_is_a_noop(self, config):
        from pairs_trading import kill_switch

        assert kill_switch.clear(config) is False

    def test_exits_still_allowed_while_halted(self, risk):
        risk.halt("halted for test")
        assert risk.validate_order(orders(1_000.0), snap(risk), intent="exit")


# --------------------------------------------------------------------------
# 4. the clearance interlock
# --------------------------------------------------------------------------

class TestClearanceInterlock:
    def test_submitting_without_clearance_raises(self, risk):
        with pytest.raises(RiskError, match="no risk clearance"):
            risk.require_clearance(ProposedOrder("AAA", "BUY", 10, 100.0), None)

    def test_clearance_does_not_cover_a_different_order(self, risk):
        clearance = risk.validate_order(orders(2_000.0), snap(risk))
        forged = ProposedOrder("AAA", "BUY", 10_000, 100.0)
        with pytest.raises(RiskError, match="does not cover"):
            risk.require_clearance(forged, clearance)

    def test_clearance_covers_exactly_what_it_approved(self, risk):
        proposed = orders(2_000.0)
        clearance = risk.validate_order(proposed, snap(risk))
        for order in proposed:
            risk.require_clearance(order, clearance)   # must not raise

    def test_clearance_does_not_cover_an_inflated_quantity(self, risk):
        proposed = orders(2_000.0)
        clearance = risk.validate_order(proposed, snap(risk))
        inflated = ProposedOrder(
            proposed[0].symbol, proposed[0].side, proposed[0].quantity * 10, proposed[0].price
        )
        with pytest.raises(RiskError):
            risk.require_clearance(inflated, clearance)

    def test_execution_submit_requires_clearance(self):
        """`_submit` must call the interlock -- asserted at source level."""
        import inspect

        from pairs_trading.execution_alpaca import AlpacaPaperTrader

        source = inspect.getsource(AlpacaPaperTrader._submit)
        assert "require_clearance" in source
        # And it must happen before the order is built/sent.
        assert source.index("require_clearance") < source.index("submit_order")

    def test_no_submit_path_bypasses_the_batch_validator(self):
        """Every `_submit` call site must go through `_submit_batch`."""
        import re
        from pathlib import Path

        source = Path("pairs_trading/execution_alpaca.py").read_text()
        # Strip the definition and the one authorised call inside _submit_batch.
        calls = re.findall(r"self\._submit\(", source)
        definition = re.findall(r"def _submit\(", source)
        assert len(calls) == 1, (
            f"expected exactly one self._submit(...) call site (inside _submit_batch), "
            f"found {len(calls)}"
        )
        assert len(definition) == 1


# --------------------------------------------------------------------------
# 5. alerting
# --------------------------------------------------------------------------

class TestAlerting:
    def test_dispatcher_fans_out_to_every_channel(self):
        a, b = CapturingChannel(), CapturingChannel()
        b.name = "capture2"
        AlertDispatcher([a, b]).send(Alert("info", "t", "m"))
        assert len(a.alerts) == 1 and len(b.alerts) == 1

    def test_a_failing_channel_does_not_break_the_others(self):
        class Broken(AlertChannel):
            name = "broken"

            def send(self, alert):
                raise RuntimeError("smtp down")

        good = CapturingChannel()
        results = AlertDispatcher([Broken(), good]).send(Alert("critical", "t", "m"))
        assert results["broken"] is False
        assert len(good.alerts) == 1

    def test_alert_failure_never_stops_a_halt(self, config, clock):
        """A dead mail server must not prevent trading from being halted."""

        class Broken(AlertChannel):
            name = "broken"

            def send(self, alert):
                raise RuntimeError("smtp down")

        manager = RiskManager(config, alerts=AlertDispatcher([Broken()]), now_fn=clock)
        manager.halt("test halt")
        assert manager.is_halted()

    def test_email_channel_unconfigured_by_default(self, monkeypatch):
        for var in ("SMTP_HOST", "SMTP_TO", "SMTP_FROM", "SMTP_USER"):
            monkeypatch.delenv(var, raising=False)
        assert EmailAlertChannel().is_configured() is False

    def test_email_channel_configured_from_env(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_FROM", "bot@example.com")
        monkeypatch.setenv("SMTP_TO", "me@example.com,ops@example.com")
        channel = EmailAlertChannel()
        assert channel.is_configured()
        assert channel.recipients == ["me@example.com", "ops@example.com"]

    def test_unconfigured_email_send_returns_false_not_raise(self, monkeypatch):
        monkeypatch.delenv("SMTP_HOST", raising=False)
        assert EmailAlertChannel().send(Alert("info", "t", "m")) is False

    def test_alert_renders_context(self):
        text = Alert("info", "t", "body", {"equity": "$100"}).as_text()
        assert "body" in text and "equity: $100" in text


# --------------------------------------------------------------------------
# 6. config validation
# --------------------------------------------------------------------------

class TestRiskConfig:
    def test_negative_limits_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            RiskConfig(max_position_size_usd=-1)

    def test_position_cap_above_total_cap_rejected(self):
        with pytest.raises(ConfigError, match="exceeds"):
            RiskConfig(max_position_size_usd=50_000, max_total_exposure_usd=10_000)

    def test_percentage_must_be_a_fraction(self):
        with pytest.raises(ConfigError, match="fraction"):
            RiskConfig(max_daily_loss_pct=25)

    def test_limits_may_be_disabled_with_null(self):
        cfg = RiskConfig(max_position_size_usd=None, max_total_exposure_usd=None)
        assert cfg.max_position_size_usd is None

    def test_advisory_when_every_trade_would_breach_the_cap(self, config):
        cfg = replace(
            config,
            risk=replace(config.risk, max_position_size_usd=100.0),
            execution=replace(config.execution, notional_per_leg=1_000.0),
        )
        assert any("rejected on size" in n for n in cfg.advisories())

    def test_advisory_when_no_loss_limit_is_set(self, config):
        cfg = replace(config, risk=replace(
            config.risk, max_daily_loss_usd=None, max_daily_loss_pct=None
        ))
        assert any("circuit breaker will never trip" in n for n in cfg.advisories())


# --------------------------------------------------------------------------
# 7. risk event log
# --------------------------------------------------------------------------

class TestRiskEventLog:
    def test_schema_has_context_columns(self, tmp_path):
        from pairs_trading.risk_manager import RISK_CSV_COLUMNS

        for column in ("timestamp", "event", "reason", "equity", "daily_pnl", "open_exposure"):
            assert column in RISK_CSV_COLUMNS

    def test_events_accumulate_across_instances(self, config, alerts, clock):
        first = RiskManager(config, alerts=AlertDispatcher([alerts]), now_fn=clock)
        first.halt("one")
        first.clear_halt("two")
        second = RiskManager(config, alerts=AlertDispatcher([alerts]), now_fn=clock)
        second.halt("three")
        assert len(second.events.read()) >= 3
