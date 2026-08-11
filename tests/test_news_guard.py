"""Tests for the news-driven trading veto.

The property that matters most is the *boundary*: structural events must halt,
and ordinary news must not. A guard that halts on earnings is an off-switch, not
a risk control, and one that misses a merger is worse than no guard at all.

Everything runs offline against a fake news client.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from pairs_trading.config import ConfigError, NewsConfig, load_config
from pairs_trading.news_guard import (
    BLOCKING,
    INFO,
    WARNING,
    NewsGuard,
    NewsItem,
    NewsUnavailableError,
)
from pairs_trading.risk_manager import AlertChannel, AlertDispatcher, RiskManager

pytestmark = pytest.mark.unit


class CapturingChannel(AlertChannel):
    name = "capture"

    def __init__(self):
        self.alerts = []

    def send(self, alert):
        self.alerts.append(alert)
        return True


@pytest.fixture
def config(tmp_path):
    cfg = load_config().with_overrides(pair={"y": "AAA", "x": "BBB"})
    return replace(
        cfg,
        risk=replace(
            cfg.risk,
            halt_file=str(tmp_path / "TRADING_HALTED"),
            state_file=str(tmp_path / "risk_state.json"),
            risk_log=str(tmp_path / "risk_events.csv"),
        ),
    )


def item(headline: str, symbols=("AAA",), summary: str = "", **kw) -> NewsItem:
    return NewsItem(
        id=kw.get("id", "1"),
        headline=headline,
        summary=summary,
        symbols=list(symbols),
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        source="test",
        url="https://example.test/1",
    )


def guard(config, items=None, alerts=None, error=False):
    """A NewsGuard backed by a fake feed."""
    client = MagicMock()
    if error:
        client.get_news.side_effect = RuntimeError("feed down")
    else:
        client.get_news.return_value = MagicMock(data=list(items or []))
    dispatcher = AlertDispatcher([alerts]) if alerts else AlertDispatcher([])
    return NewsGuard(config, alerts=dispatcher, client=client)


# --------------------------------------------------------------------------
# classification -- the boundary that matters
# --------------------------------------------------------------------------

class TestBlockingEvents:
    """Events that end the pair's price relationship."""

    @pytest.mark.parametrize("headline", [
        "AAA to acquire rival in $12bn all-cash deal",
        "BBB agrees to merger with CCC",
        "AAA receives takeover approach from private equity",
        "AAA announces take-private transaction",
        "Board approves tender offer for AAA shares",
    ])
    def test_merger_and_acquisition(self, config, headline):
        finding = guard(config).classify(item(headline))
        assert finding is not None and finding.severity == BLOCKING
        assert finding.category == "merger_acquisition"

    @pytest.mark.parametrize("headline", [
        "AAA to spin off its beverage unit",
        "AAA announces spinoff of European business",
        "AAA plans to separate into two public companies",
    ])
    def test_spin_off(self, config, headline):
        finding = guard(config).classify(item(headline))
        assert finding.severity == BLOCKING and finding.category == "spin_off"

    @pytest.mark.parametrize("headline", [
        "AAA files for Chapter 11 bankruptcy protection",
        "Auditors raise going concern doubt over AAA",
        "AAA defaults on senior notes",
    ])
    def test_distress(self, config, headline):
        assert guard(config).classify(item(headline)).severity == BLOCKING

    @pytest.mark.parametrize("headline", [
        "AAA to be delisted from the NYSE",
        "Exchange announces suspension of trading in AAA",
    ])
    def test_delisting(self, config, headline):
        assert guard(config).classify(item(headline)).severity == BLOCKING

    @pytest.mark.parametrize("headline", [
        "AAA to restate three years of financial statements",
        "AAA discloses material weakness in internal controls",
        "SEC investigation into AAA accounting practices",
        "AAA auditor resigns citing disagreements",
    ])
    def test_accounting(self, config, headline):
        assert guard(config).classify(item(headline)).severity == BLOCKING


class TestOrdinaryNewsDoesNotHalt:
    """The divergence this strategy exists to trade must not stop it."""

    @pytest.mark.parametrize("headline", [
        "AAA beats Q3 earnings estimates",
        "AAA misses revenue expectations, cuts guidance",
        "AAA raises full-year outlook",
        "Analyst upgrades AAA, lifts price target to $80",
        "AAA downgraded to hold",
        "AAA declares quarterly dividend",
        "AAA announces $2bn share repurchase",
    ])
    def test_is_informational_only(self, config, headline):
        finding = guard(config).classify(item(headline))
        assert finding is not None
        assert finding.severity == INFO, f"{headline!r} must not block trading"

    def test_earnings_do_not_halt_the_strategy(self, config):
        news = [item("AAA beats Q3 earnings, shares jump 8%")]
        result = guard(config, news).run()
        assert result.blocking == []
        assert not result.halted
        assert not RiskManager(config).is_halted()


class TestWordBoundaries:
    """Substring matching would fire constantly on innocent words."""

    @pytest.mark.parametrize("headline,expect_blocking", [
        ("AAA emerges as sector leader", False),          # 'emerges' vs 'merger'
        ("Halton Industries partners with AAA", False),   # 'Halton' vs 'halt'
        ("AAA merger completed", True),
        ("AAA restates guidance for clarity", True),      # 'restate' is blocking
    ])
    def test_no_false_positives_from_substrings(self, config, headline, expect_blocking):
        finding = guard(config).classify(item(headline))
        is_blocking = finding is not None and finding.severity == BLOCKING
        assert is_blocking == expect_blocking

    def test_unmatched_headline_returns_none(self, config):
        assert guard(config).classify(item("AAA opens new distribution centre")) is None

    def test_empty_headline_returns_none(self, config):
        assert guard(config).classify(item("", summary="")) is None


class TestPrecedence:
    def test_structural_wins_over_ordinary(self, config):
        """A merger mentioned alongside earnings must classify as the merger.

        Under-reacting to a structural event is the expensive error.
        """
        finding = guard(config).classify(
            item("AAA posts Q3 earnings ahead of shareholder merger vote")
        )
        assert finding.severity == BLOCKING
        assert finding.category == "merger_acquisition"

    def test_warning_beats_info(self, config):
        finding = guard(config).classify(item("AAA added to the S&P 500, earnings due"))
        assert finding.severity == WARNING


# --------------------------------------------------------------------------
# relevance filtering
# --------------------------------------------------------------------------

class TestRelevance:
    def test_ignores_news_about_other_tickers(self, config):
        result = guard(config, [item("ZZZ to be acquired", symbols=("ZZZ",))]).run()
        assert result.scanned == 0
        assert result.blocking == []

    def test_ignores_market_roundups(self, config):
        """An item tagged with forty tickers is not news about your pair."""
        roundup = item(
            "Market wrap: merger activity picks up across the sector",
            symbols=("AAA", *[f"T{i}" for i in range(30)]),
        )
        result = guard(config, [roundup]).run()
        assert result.scanned == 0
        assert not result.halted

    def test_untagged_items_are_skipped(self, config):
        assert guard(config, [item("AAA to be acquired", symbols=())]).run().scanned == 0

    def test_either_leg_counts(self, config):
        result = guard(config, [item("BBB to be acquired", symbols=("BBB",))]).run()
        assert result.scanned == 1
        assert result.blocking


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------

class TestEnforcement:
    def test_blocking_event_halts_trading(self, config):
        result = guard(config, [item("AAA agrees to merger with CCC")]).run()
        assert result.blocking
        assert result.halted
        risk = RiskManager(config)
        assert risk.is_halted()
        assert "NEWS GUARD" in risk.halt_reason()

    def test_halt_uses_the_same_flag_as_the_kill_switch(self, config):
        """One way to resume trading, not three."""
        guard(config, [item("AAA files for bankruptcy")]).run()
        from pairs_trading import kill_switch

        assert kill_switch.status(config)["halted"] is True
        assert kill_switch.clear(config) is True
        assert not RiskManager(config).is_halted()

    def test_blocking_event_alerts(self, config):
        alerts = CapturingChannel()
        guard(config, [item("AAA to be delisted")], alerts=alerts).run()
        assert alerts.alerts
        assert alerts.alerts[0].severity == "critical"

    def test_halt_can_be_disabled(self, config):
        cfg = replace(config, news=replace(config.news, halt_on_blocking=False))
        result = guard(cfg, [item("AAA agrees to merger")]).run()
        assert result.blocking
        assert not result.halted
        assert not RiskManager(cfg).is_halted()

    def test_findings_are_logged_to_the_risk_log(self, config):
        guard(config, [item("AAA agrees to merger with CCC")]).run()
        events = RiskManager(config).events.read()
        assert (events["event"] == "news_scan").any()

    def test_warning_does_not_halt(self, config):
        result = guard(config, [item("AAA added to the S&P 500")]).run()
        assert result.warnings
        assert not result.halted


class TestFeedFailure:
    def test_fails_open_by_default(self, config):
        """An unreachable feed must not hand the vendor an off-switch."""
        result = guard(config, error=True).run()
        assert result.error
        assert not result.halted
        assert not RiskManager(config).is_halted()

    def test_fail_closed_halts_when_configured(self, config):
        cfg = replace(config, news=replace(config.news, fail_closed=True))
        result = guard(cfg, error=True).run()
        assert result.halted
        assert RiskManager(cfg).is_halted()

    def test_error_is_not_reported_as_ok(self, config):
        assert guard(config, error=True).run().ok is False

    def test_fetch_wraps_the_underlying_error(self, config):
        with pytest.raises(NewsUnavailableError):
            guard(config, error=True).fetch()


class TestReport:
    def test_clear_verdict(self, config):
        report = guard(config, [item("AAA opens new plant")]).run().report()
        assert "clear" in report.lower()

    def test_blocking_verdict_names_the_headline(self, config):
        result = guard(config, [item("AAA agrees to merger with CCC")]).run()
        report = result.report()
        assert "TRADING HALTED" in report
        assert "merger" in report.lower()

    def test_explains_why_earnings_are_ignored(self, config):
        report = guard(config, [item("AAA beats Q3 earnings")]).run().report()
        assert "divergence this strategy trades" in report


class TestConfig:
    def test_rejects_bad_lookback(self):
        with pytest.raises(ConfigError):
            NewsConfig(lookback_hours=0)

    def test_rejects_bad_limits(self):
        with pytest.raises(ConfigError):
            NewsConfig(max_items=0)
        with pytest.raises(ConfigError):
            NewsConfig(max_symbols=0)

    def test_defaults_cover_a_long_weekend(self):
        """A Friday-evening merger must not be missed by a Monday run."""
        assert NewsConfig().lookback_hours >= 72

    def test_cli_exposes_the_mode(self):
        from pairs_trading.main import build_parser

        assert build_parser().parse_args(["--news-check"]).news_check is True

    def test_no_news_flag_disables_the_guard(self):
        from pairs_trading.main import apply_overrides, build_parser

        args = build_parser().parse_args(["--paper-trade", "--no-news"])
        assert apply_overrides(load_config(), args).news.enabled is False
