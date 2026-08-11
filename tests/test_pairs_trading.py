"""Tests for the pairs-trading (statistical arbitrage) package.

Everything here runs offline against synthetic series with known properties, so
the suite never touches Yahoo Finance or Alpaca. The synthetic construction is
the point: a spread built as ``y = beta*x + OU(phi)`` has an analytically known
half-life, ``-ln(2)/ln(phi)``, which lets the cointegration diagnostics be
checked against a real number rather than against themselves.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs_trading.backtest import run_backtest
from pairs_trading.cointegration import (
    NotCointegratedError,
    assert_tradeable,
    engle_granger,
    half_life,
    johansen,
    # Aliased: pytest would otherwise collect `test_pair` as a test function and
    # error on its unfilled `data`/`config` arguments.
    test_pair as run_cointegration_tests,
)
from pairs_trading.config import (
    NULL,
    Config,
    ConfigError,
    ExecutionConfig,
    PairConfig,
    SignalConfig,
    load_config,
)
from pairs_trading.data import DataError, PairData, _extract_price_column
from pairs_trading.strategy import (
    ENTRY_LONG,
    ENTRY_SHORT,
    EXIT_STOP,
    compute_spread,
    generate_positions,
    generate_signals,
    rolling_zscore,
)
from pairs_trading.trade_log import TradeLogger, TradeRecord

pytestmark = pytest.mark.unit

PACKAGE = Path(__file__).resolve().parent.parent / "pairs_trading"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_cointegrated(n: int = 1200, phi: float = 0.94, beta: float = 1.0, seed: int = 42):
    """y = 50 + beta*x + OU(phi), with x a random walk.

    The OU residual has half-life ``-ln(2)/ln(phi)`` by construction, so the
    pair is cointegrated with a known reversion speed.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    x = 100 + np.cumsum(rng.normal(0, 1.0, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = phi * ou[i - 1] + rng.normal(0, 1.0)
    y = 50 + beta * x + ou
    frame = pd.DataFrame({"AAA": y, "BBB": x}, index=idx)
    frame.index.name = "date"
    return frame


def make_independent(n: int = 1200, seed: int = 7):
    """Two independent random walks: correlated in places, never cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    frame = pd.DataFrame(
        {
            "AAA": 100 + np.cumsum(rng.normal(0.03, 1.0, n)),
            "BBB": 100 + np.cumsum(rng.normal(0.02, 1.0, n)),
        },
        index=idx,
    )
    frame.index.name = "date"
    return frame


@pytest.fixture
def config(tmp_path) -> Config:
    """Test config with every risk artefact redirected into tmp_path.

    The halt flag, risk state and risk log must never be written to the real
    ``logs/`` directory: a leftover halt file from a test run would block a
    subsequent real run, and a stale equity baseline would corrupt the daily
    loss calculation.
    """
    from dataclasses import replace

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


@pytest.fixture
def coint_data(config) -> PairData:
    return PairData(prices=make_cointegrated(), pair=config.pair, source="synthetic")


@pytest.fixture
def random_data(config) -> PairData:
    return PairData(prices=make_independent(), pair=config.pair, source="synthetic")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

class TestConfig:
    def test_default_config_loads_and_validates(self):
        cfg = load_config()
        assert cfg.execution.base_url == "https://paper-api.alpaca.markets"
        assert cfg.signal.exit_z < cfg.signal.entry_z < cfg.signal.stop_z

    def test_exit_threshold_must_be_inside_entry(self):
        with pytest.raises(ConfigError, match="must be below entry_z"):
            SignalConfig(entry_z=2.0, exit_z=2.0)

    def test_stop_must_be_outside_entry(self):
        with pytest.raises(ConfigError, match="must exceed entry_z"):
            SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=1.5)

    def test_pair_rejects_duplicate_tickers(self):
        with pytest.raises(ConfigError, match="two distinct tickers"):
            PairConfig(y="KO", x="ko")

    def test_pair_normalises_case(self):
        assert PairConfig(y="ko", x=" pep ").tickers == ("KO", "PEP")

    def test_beta_neutral_sizing_requires_ols_spread(self, config):
        with pytest.raises(ConfigError, match="needs a hedge ratio"):
            config.with_overrides(
                spread={"method": "log_ratio"}, backtest={"sizing": "beta_neutral"}
            )

    def test_unknown_key_is_rejected_not_ignored(self, config):
        # A silently ignored typo is the worst kind of config bug: the run
        # succeeds using a parameter you thought you had changed.
        with pytest.raises(ConfigError, match="Unknown key"):
            config.with_overrides(signal={"entry_zscore": 3.0})

    def test_advisories_flag_zero_execution_lag(self, config):
        notes = config.with_overrides(signal={"execution_lag": 0}).advisories()
        assert any("execution_lag is 0" in n for n in notes)

    def test_advisories_flag_static_hedge_ratio(self, config):
        notes = config.with_overrides(spread={"hedge_window": NULL}).advisories()
        assert any("future prices" in n for n in notes)


# --------------------------------------------------------------------------
# paper-trading safety
# --------------------------------------------------------------------------

class TestPaperTradingSafety:
    """The guarantee that no live order can be placed."""

    def test_config_rejects_live_endpoint(self):
        with pytest.raises(ConfigError, match="paper-trading only"):
            ExecutionConfig(base_url="https://api.alpaca.markets")

    def test_config_rejects_url_without_paper(self):
        with pytest.raises(ConfigError, match="must contain 'paper'"):
            ExecutionConfig(base_url="https://example.com/trading")

    def test_assert_paper_endpoint_blocks_live_host(self):
        from pairs_trading.execution_alpaca import LiveTradingBlockedError, assert_paper_endpoint

        with pytest.raises(LiveTradingBlockedError, match="LIVE Alpaca endpoint"):
            assert_paper_endpoint("https://api.alpaca.markets")

    def test_assert_paper_endpoint_blocks_empty_and_foreign_urls(self):
        from pairs_trading.execution_alpaca import LiveTradingBlockedError, assert_paper_endpoint

        for bad in ("", "https://broker.example.com", "http://localhost:9000"):
            with pytest.raises(LiveTradingBlockedError):
                assert_paper_endpoint(bad)

    def test_assert_paper_endpoint_accepts_paper_url(self):
        from pairs_trading.execution_alpaca import PAPER_URL, assert_paper_endpoint

        assert assert_paper_endpoint(PAPER_URL) == PAPER_URL

    def test_no_source_file_can_construct_a_live_client(self):
        """Source-level guarantee: no ``paper=False`` and no live-endpoint enum."""
        offenders = []
        for path in PACKAGE.glob("*.py"):
            source = path.read_text()
            if re.search(r"paper\s*=\s*False", source):
                offenders.append(f"{path.name}: paper=False")
            if "TRADING_LIVE" in source:
                offenders.append(f"{path.name}: references TRADING_LIVE")
        assert not offenders, f"Live-trading code path found: {offenders}"

    def test_trading_client_is_always_constructed_with_paper_true(self):
        source = (PACKAGE / "execution_alpaca.py").read_text()
        calls = re.findall(r"TradingClient\((.*?)\)", source, re.DOTALL)
        constructions = [c for c in calls if "api_key" in c]
        assert constructions, "expected a TradingClient construction to inspect"
        for call in constructions:
            assert "paper=True" in call

    def test_config_yaml_ships_a_paper_endpoint(self):
        text = (PACKAGE / "config.yaml").read_text()
        assert "paper-api.alpaca.markets" in text

    def test_live_host_with_paper_in_the_query_string_is_blocked(self):
        """A substring check on the whole URL would wrongly allow this."""
        from pairs_trading.execution_alpaca import LiveTradingBlockedError, assert_paper_endpoint

        with pytest.raises(LiveTradingBlockedError):
            assert_paper_endpoint("https://api.alpaca.markets/?note=paper")

    def test_paper_host_is_not_mistaken_for_the_live_host(self):
        """'api.alpaca.markets' is a substring of 'paper-api.alpaca.markets'.

        A naive "is the live host in this URL" check rejects the legitimate
        paper endpoint, which would make the whole module unusable.
        """
        from pairs_trading.execution_alpaca import assert_paper_endpoint

        assert assert_paper_endpoint("https://paper-api.alpaca.markets")
        assert assert_paper_endpoint("paper-api.alpaca.markets")


# --------------------------------------------------------------------------
# execution against a mocked broker
# --------------------------------------------------------------------------

@pytest.fixture
def fake_alpaca(monkeypatch):
    """A stand-in Alpaca client that reports an active, empty paper account."""
    from unittest.mock import MagicMock

    monkeypatch.setenv("ALPACA_API_KEY", "pk_test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "sk_test")

    client = MagicMock()
    client._base_url = "https://paper-api.alpaca.markets"
    client.get_account.return_value = MagicMock(
        status="ACTIVE", trading_blocked=False, equity="100000", buying_power="200000"
    )
    client.get_clock.return_value = MagicMock(is_open=True)
    client.get_all_positions.return_value = []
    client.submit_order.return_value = MagicMock(id="order-123")
    return client


def _hold(symbol_y: str, qty_y: int, side_y: str, symbol_x: str, qty_x: int, side_x: str):
    from unittest.mock import MagicMock

    return [
        MagicMock(symbol=symbol_y, qty=str(qty_y), side=MagicMock(value=side_y)),
        MagicMock(symbol=symbol_x, qty=str(qty_x), side=MagicMock(value=side_x)),
    ]


class TestExecution:
    def _trader(self, config, fake_alpaca, **kwargs):
        from unittest.mock import patch

        from pairs_trading.execution_alpaca import AlpacaPaperTrader

        with patch("alpaca.trading.client.TradingClient", return_value=fake_alpaca):
            return AlpacaPaperTrader(config, dry_run=True, **kwargs)

    def test_rejects_a_blocked_account(self, config, fake_alpaca):
        from pairs_trading.execution_alpaca import ExecutionError

        fake_alpaca.get_account.return_value.trading_blocked = True
        with pytest.raises(ExecutionError, match="trading_blocked"):
            self._trader(config, fake_alpaca)

    def test_rejects_an_inactive_account(self, config, fake_alpaca):
        from pairs_trading.execution_alpaca import ExecutionError

        fake_alpaca.get_account.return_value.status = "SUBMITTED"
        with pytest.raises(ExecutionError, match="not ACTIVE"):
            self._trader(config, fake_alpaca)

    def test_refuses_to_trade_when_the_market_is_closed(self, config, fake_alpaca, coint_data):
        """An order sent while closed fills at an unrelated price on the next open.

        Since the risk layer landed, a closed market is a *reported* rejection
        rather than an exception: a scheduled run should record the refusal and
        exit cleanly instead of crashing.
        """
        from unittest.mock import MagicMock

        fake_alpaca.get_clock.return_value = MagicMock(
            is_open=False, next_open="2026-01-02T14:30:00Z"
        )
        trader = self._trader(config, fake_alpaca)
        outcome = trader.sync_to_signal(coint_data)
        assert outcome["action"] == "rejected"
        assert any("market is closed" in r.lower() for r in outcome["rejected_reasons"])
        fake_alpaca.submit_order.assert_not_called()

    def test_opens_a_position_from_flat(self, config, fake_alpaca, coint_data):
        trader = self._trader(config, fake_alpaca)
        out = trader.sync_to_signal(coint_data)
        if out["target_direction"] != 0:
            assert out["action"] == "open"
            assert len(out["orders"]) == 2

    def test_is_idempotent_when_already_in_the_target_position(
        self, config, fake_alpaca, coint_data
    ):
        """Running twice must not double the position.

        A scheduled job that fires twice, or a retried run, must be harmless.
        """
        trader = self._trader(config, fake_alpaca)
        first = trader.sync_to_signal(coint_data)
        target = first["target_direction"]
        if target == 0:
            pytest.skip("no position targeted on the final bar of this fixture")

        y, x = config.pair.tickers
        fake_alpaca.get_all_positions.return_value = _hold(
            y, 10, "long" if target > 0 else "short",
            x, 10, "short" if target > 0 else "long",
        )
        second = trader.sync_to_signal(coint_data)
        assert second["action"] == "none"
        assert second["orders"] == []

    def test_flattens_an_incoherent_book_without_reopening(
        self, config, fake_alpaca, coint_data
    ):
        """Both legs long is not a spread; flatten and take no new position."""
        y, x = config.pair.tickers
        fake_alpaca.get_all_positions.return_value = _hold(y, 10, "long", x, 10, "long")
        trader = self._trader(config, fake_alpaca)
        out = trader.sync_to_signal(coint_data)
        assert out["action"] == "flatten_incoherent"

    def test_dry_run_submits_nothing(self, config, fake_alpaca, coint_data):
        trader = self._trader(config, fake_alpaca)
        trader.sync_to_signal(coint_data)
        fake_alpaca.submit_order.assert_not_called()

    def test_position_direction_is_inferred_from_leg_signs(self):
        from pairs_trading.execution_alpaca import BrokerPosition

        assert BrokerPosition(10, -10).direction == 1
        assert BrokerPosition(-10, 10).direction == -1
        assert BrokerPosition(0, 0).direction == 0
        assert BrokerPosition(10, 10).direction == 0        # not a spread
        assert not BrokerPosition(10, 10).is_coherent
        assert BrokerPosition(0, 0).is_coherent

    def test_sizing_refuses_a_one_legged_order(self, config, fake_alpaca):
        """If a leg floors to zero shares the whole pair must abort.

        Sending only the other leg would leave a naked directional position --
        precisely the exposure this strategy exists to avoid.
        """
        from pairs_trading.execution_alpaca import ExecutionError

        cfg = config.with_overrides(execution={"notional_per_leg": 10.0})
        trader = self._trader(cfg, fake_alpaca)
        with pytest.raises(ExecutionError, match="whole shares"):
            trader._build_pair_orders(1, price_y=5000.0, price_x=50.0, hedge_ratio=1.0)

    def test_orders_are_whole_shares_and_opposite_sides(self, config, fake_alpaca):
        trader = self._trader(config, fake_alpaca)
        orders = trader._build_pair_orders(1, price_y=60.0, price_x=170.0, hedge_ratio=1.0)
        assert {o.side for o in orders} == {"BUY", "SELL"}
        for o in orders:
            assert isinstance(o.quantity, int) and o.quantity >= 1


class TestSetupDiagnostic:
    """The read-only `--check-alpaca` setup check."""

    def _diagnose(self, config, fake_alpaca, monkeypatch):
        from unittest.mock import patch

        from pairs_trading.execution_alpaca import diagnose

        monkeypatch.setenv("ALPACA_API_KEY", "PKTEST1234567890")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "sk_secret_1234567890")
        with patch("alpaca.trading.client.TradingClient", return_value=fake_alpaca):
            return diagnose(config)

    def _named(self, checks, name):
        return next(c for c in checks if c.name == name)

    def test_healthy_account_passes_every_check(self, config, fake_alpaca, monkeypatch):
        fake_alpaca.get_account.return_value.shorting_enabled = True
        fake_alpaca.get_account.return_value.multiplier = "2"
        fake_alpaca.get_account.return_value.account_number = "PA123456789"
        checks = self._diagnose(config, fake_alpaca, monkeypatch)
        blocking = [c for c in checks if not c.ok and c.fatal]
        assert blocking == [], [c.name for c in blocking]

    def test_missing_credentials_stops_early_with_a_fix(self, config, monkeypatch):
        from pairs_trading.execution_alpaca import diagnose

        for var in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                    "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("pairs_trading.config.load_dotenv", lambda *a, **k: None,
                            raising=False)
        checks = diagnose(config)
        cred = self._named(checks, "API credentials loaded")
        assert cred.ok is False
        assert "app.alpaca.markets" in cred.fix
        # Nothing after the credential check should have run.
        assert checks[-1] is cred

    def test_credentials_are_masked_never_printed(self, config, fake_alpaca, monkeypatch):
        checks = self._diagnose(config, fake_alpaca, monkeypatch)
        detail = self._named(checks, "API credentials loaded").detail
        assert "PKTEST1234567890" not in detail
        assert "sk_secret_1234567890" not in detail
        assert "*" in detail

    def test_shorting_disabled_is_flagged_as_blocking(self, config, fake_alpaca, monkeypatch):
        """A cash account cannot short, so no pairs trade can ever be placed."""
        fake_alpaca.get_account.return_value.shorting_enabled = False
        checks = self._diagnose(config, fake_alpaca, monkeypatch)
        shorting = self._named(checks, "Shorting enabled")
        assert shorting.ok is False and shorting.fatal
        assert "margin" in shorting.fix.lower()

    def test_cash_account_multiplier_is_flagged(self, config, fake_alpaca, monkeypatch):
        fake_alpaca.get_account.return_value.shorting_enabled = True
        fake_alpaca.get_account.return_value.multiplier = "1"
        checks = self._diagnose(config, fake_alpaca, monkeypatch)
        assert self._named(checks, "Margin account").ok is False

    def test_insufficient_buying_power_is_non_blocking(self, config, fake_alpaca, monkeypatch):
        fake_alpaca.get_account.return_value.buying_power = "10"
        checks = self._diagnose(config, fake_alpaca, monkeypatch)
        power = self._named(checks, "Buying power covers a trade")
        assert power.ok is False and power.fatal is False

    def test_halt_flag_is_surfaced(self, config, fake_alpaca, monkeypatch):
        from pairs_trading.risk_manager import RiskManager

        RiskManager(config).halt("earlier breach")
        checks = self._diagnose(config, fake_alpaca, monkeypatch)
        halted = self._named(checks, "Trading not halted")
        assert halted.ok is False
        assert "kill_switch --clear" in halted.fix

    def test_live_endpoint_is_refused_before_connecting(self, config, monkeypatch):
        """A live URL must fail the diagnostic without any network call."""
        from dataclasses import replace

        from pairs_trading.config import ExecutionConfig
        from pairs_trading.execution_alpaca import diagnose

        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        # Bypass ExecutionConfig validation to simulate a tampered config.
        bad = ExecutionConfig.__new__(ExecutionConfig)
        object.__setattr__(bad, "base_url", "https://api.alpaca.markets")
        for field_name, value in (
            ("notional_per_leg", 1000.0), ("time_in_force", "day"),
            ("require_market_open", True), ("verify_paper_account", True),
        ):
            object.__setattr__(bad, field_name, value)
        checks = diagnose(replace(config, execution=bad))
        endpoint = self._named(checks, "Endpoint is PAPER")
        assert endpoint.ok is False
        assert checks[-1] is endpoint

    def test_report_renders_next_steps_on_success(self, config, fake_alpaca, monkeypatch):
        from pairs_trading.execution_alpaca import render_diagnosis

        fake_alpaca.get_account.return_value.shorting_enabled = True
        fake_alpaca.get_account.return_value.multiplier = "2"
        report = render_diagnosis(self._diagnose(config, fake_alpaca, monkeypatch), config)
        assert "All checks passed" in report
        assert "--paper-trade" in report and "--dry-run" in report

    def test_report_names_the_blocking_count(self, config, fake_alpaca, monkeypatch):
        from pairs_trading.execution_alpaca import render_diagnosis

        fake_alpaca.get_account.return_value.shorting_enabled = False
        report = render_diagnosis(self._diagnose(config, fake_alpaca, monkeypatch), config)
        assert "blocking problem" in report

    def test_diagnostic_places_no_orders(self, config, fake_alpaca, monkeypatch):
        self._diagnose(config, fake_alpaca, monkeypatch)
        fake_alpaca.submit_order.assert_not_called()
        fake_alpaca.close_all_positions.assert_not_called()
        fake_alpaca.cancel_orders.assert_not_called()

    def test_cli_exposes_it_as_a_mode(self):
        from pairs_trading.main import build_parser

        assert build_parser().parse_args(["--check-alpaca"]).check_alpaca is True


# --------------------------------------------------------------------------
# strategy: spread and z-score
# --------------------------------------------------------------------------

class TestSpread:
    def test_price_diff_spread(self, coint_data, config):
        frame = compute_spread(coint_data, config.with_overrides(spread={"method": "price_diff"}))
        expected = coint_data.y - coint_data.x
        pd.testing.assert_series_equal(frame["spread"], expected.rename("spread"))

    def test_log_ratio_spread(self, coint_data, config):
        frame = compute_spread(coint_data, config.with_overrides(spread={"method": "log_ratio"}))
        expected = np.log(coint_data.y / coint_data.x)
        np.testing.assert_allclose(frame["spread"].values, expected.values, rtol=1e-12)

    def test_ols_hedge_ratio_recovers_the_true_beta(self, config):
        prices = make_cointegrated(beta=2.0, seed=11)
        data = PairData(prices=prices, pair=config.pair, source="synthetic")
        frame = compute_spread(data, config.with_overrides(spread={"hedge_window": 250}))
        assert frame["hedge_ratio"].dropna().mean() == pytest.approx(2.0, abs=0.1)

    def test_rolling_hedge_ratio_has_no_lookahead(self, coint_data, config):
        """beta_t must not change when data after t is altered."""
        from pairs_trading.strategy import rolling_hedge_ratio

        beta_full, _ = rolling_hedge_ratio(coint_data.y, coint_data.x, 60)
        cut = 800
        beta_truncated, _ = rolling_hedge_ratio(
            coint_data.y.iloc[:cut], coint_data.x.iloc[:cut], 60
        )
        np.testing.assert_allclose(
            beta_full.iloc[:cut].values, beta_truncated.values, rtol=1e-10
        )

    def test_ols_spread_warmup_is_nan(self, coint_data, config):
        frame = compute_spread(coint_data, config.with_overrides(spread={"hedge_window": 60}))
        assert frame["spread"].iloc[:59].isna().all()
        assert frame["spread"].iloc[59:].notna().all()


class TestZScore:
    def test_matches_manual_computation(self):
        rng = np.random.default_rng(3)
        s = pd.Series(rng.normal(0, 1, 200), index=pd.bdate_range("2020-01-01", periods=200))
        out = rolling_zscore(s, window=20)
        # Check one bar by hand.
        window = s.iloc[80:100]
        expected = (s.iloc[99] - window.mean()) / window.std(ddof=1)
        assert out["zscore"].iloc[99] == pytest.approx(expected)

    def test_warmup_bars_are_nan(self):
        s = pd.Series(np.arange(100.0))
        out = rolling_zscore(s, window=20)
        assert out["zscore"].iloc[:19].isna().all()
        assert out["zscore"].iloc[19:].notna().all()

    def test_constant_spread_yields_no_signal_not_infinity(self):
        """A flat spread means stale prices, not an infinitely strong signal."""
        s = pd.Series([5.0] * 60)
        out = rolling_zscore(s, window=20)
        assert out["zscore"].iloc[19:].isna().all()
        assert not np.isinf(out["zscore"].fillna(0)).any()

    def test_uses_trailing_not_centred_window(self):
        """Values after bar t must not influence z_t."""
        base = pd.Series(np.concatenate([np.zeros(30), np.ones(30)]))
        altered = base.copy()
        altered.iloc[40:] = 99.0
        z_base = rolling_zscore(base, 20)["zscore"]
        z_altered = rolling_zscore(altered, 20)["zscore"]
        pd.testing.assert_series_equal(z_base.iloc[:40], z_altered.iloc[:40])


# --------------------------------------------------------------------------
# strategy: the state machine
# --------------------------------------------------------------------------

class TestStateMachine:
    def _run(self, z_values, config, **overrides):
        cfg = config.with_overrides(signal={"execution_lag": 0, **overrides})
        z = pd.Series(z_values, index=pd.bdate_range("2020-01-01", periods=len(z_values)))
        return generate_positions(z, cfg)

    def test_enters_short_spread_when_zscore_is_high(self, config):
        out = self._run([0.0, 1.0, 2.5, 2.4], config)
        assert out["target"].tolist() == [0, 0, -1, -1]
        assert out["reason"].iloc[2] == ENTRY_SHORT

    def test_enters_long_spread_when_zscore_is_low(self, config):
        out = self._run([0.0, -1.0, -2.5, -2.4], config)
        assert out["target"].tolist() == [0, 0, 1, 1]
        assert out["reason"].iloc[2] == ENTRY_LONG

    def test_exits_when_spread_reverts_to_the_mean(self, config):
        out = self._run([0.0, 2.5, 1.5, 0.1], config)
        assert out["target"].tolist() == [0, -1, -1, 0]
        assert out["reason"].iloc[3] == "exit_mean_reversion"

    def test_stop_loss_fires_beyond_the_stop_threshold(self, config):
        out = self._run([0.0, 2.5, 3.0, 3.6], config)
        assert out["target"].tolist() == [0, -1, -1, 0]
        assert out["reason"].iloc[3] == EXIT_STOP

    def test_stop_loss_does_not_immediately_re_enter(self, config):
        """After stopping out at z=3.6, z=3.7 must not trigger a fresh entry.

        Without the lock the state machine would re-enter the position it just
        stopped out of, and keep doing so all the way up -- turning one bounded
        loss into an unbounded sequence of them.
        """
        out = self._run([0.0, 2.5, 3.6, 3.7, 3.8, 2.9], config)
        assert out["target"].tolist() == [0, -1, 0, 0, 0, 0]

    def test_lock_clears_once_spread_returns_to_the_mean(self, config):
        out = self._run([0.0, 2.5, 3.6, 3.0, 0.1, 2.5], config)
        assert out["target"].tolist() == [0, -1, 0, 0, 0, -1]
        assert out["reason"].iloc[5] == ENTRY_SHORT

    def test_exits_on_overshoot_through_the_mean(self, config):
        """A gap from -2.5 straight to +2.5 must still close the long position.

        The symmetric ``|z| <= exit_z`` form would miss the exit band entirely
        and leave the position on, wrong-sided.
        """
        out = self._run([0.0, -2.5, 2.5], config)
        assert out["target"].iloc[1] == 1
        assert out["target"].iloc[2] != 1

    def test_time_stop_closes_the_position(self, config):
        out = self._run([0.0, 2.5, 2.4, 2.3, 2.2], config, max_holding_days=3)
        assert out["reason"].iloc[3] == "exit_time_stop"
        assert out["target"].iloc[3] == 0

    def test_holds_through_bars_without_a_zscore(self, config):
        out = self._run([0.0, 2.5, np.nan, 0.1], config)
        assert out["target"].tolist() == [0, -1, -1, 0]

    def test_no_position_taken_during_warmup(self, config):
        out = self._run([np.nan, np.nan, 2.5], config)
        assert out["target"].tolist() == [0, 0, -1]

    def test_execution_lag_shifts_the_tradeable_position(self, config):
        cfg = config.with_overrides(signal={"execution_lag": 1})
        z = pd.Series([0.0, 2.5, 2.4, 0.1], index=pd.bdate_range("2020-01-01", periods=4))
        out = generate_positions(z, cfg)
        assert out["target"].tolist() == [0, -1, -1, 0]
        # The fill happens one bar after the signal.
        assert out["position"].tolist() == [0, 0, -1, -1]

    def test_zero_lag_means_same_bar_fill(self, config):
        out = self._run([0.0, 2.5], config)
        assert out["position"].tolist() == out["target"].tolist()


# --------------------------------------------------------------------------
# cointegration
# --------------------------------------------------------------------------

class TestCointegration:
    def test_cointegrated_pair_passes(self, coint_data, config):
        result = run_cointegration_tests(coint_data, config)
        assert result.is_cointegrated, result.failures
        assert result.engle_granger.pvalue < 0.05
        assert result.johansen.rejects_r0_trace

    def test_independent_random_walks_fail(self, random_data, config):
        result = run_cointegration_tests(random_data, config)
        assert not result.is_cointegrated
        assert result.failures

    def test_failure_blocks_the_run(self, random_data, config):
        result = run_cointegration_tests(random_data, config)
        with pytest.raises(NotCointegratedError) as exc:
            assert_tradeable(result, config)
        assert exc.value.result is result

    def test_enforce_false_downgrades_failure_to_a_log(self, random_data, config):
        cfg = config.with_overrides(cointegration={"enforce": False})
        result = run_cointegration_tests(random_data, cfg)
        assert_tradeable(result, cfg)  # must not raise

    # Tolerance widens with phi: OLS estimates of an autoregressive coefficient
    # are biased downward near the unit root (the Hurwicz/Kendall bias), so the
    # estimated half-life is systematically short for very persistent spreads.
    # That is a property of the estimator, not a defect here -- but it is worth
    # knowing that a reported half-life understates a slow-reverting spread.
    @pytest.mark.parametrize(
        "phi,expected,tol", [(0.9, 6.58, 0.10), (0.94, 11.21, 0.10), (0.98, 34.31, 0.20)]
    )
    def test_half_life_recovers_the_true_reversion_speed(self, phi, expected, tol):
        """Half-life of an AR(1) with coefficient phi is -ln(2)/ln(phi)."""
        rng = np.random.default_rng(5)
        n = 6000
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = phi * s[i - 1] + rng.normal(0, 1.0)
        estimate = half_life(pd.Series(s))
        assert estimate == pytest.approx(expected, rel=tol)

    def test_half_life_is_none_for_a_random_walk(self):
        rng = np.random.default_rng(9)
        walk = pd.Series(np.cumsum(rng.normal(0, 1, 3000)))
        hl = half_life(walk)
        # A random walk either has no reversion at all, or a spuriously huge one.
        assert hl is None or hl > 200

    def test_hurst_separates_mean_reversion_from_trending(self):
        from pairs_trading.cointegration import hurst_exponent

        rng = np.random.default_rng(13)
        n = 4000
        ou = np.zeros(n)
        for i in range(1, n):
            ou[i] = 0.9 * ou[i - 1] + rng.normal(0, 1)
        assert hurst_exponent(pd.Series(ou)) < 0.45
        walk = np.cumsum(rng.normal(0, 1, n))
        assert hurst_exponent(pd.Series(walk)) > 0.45

    def test_engle_granger_recovers_the_hedge_ratio(self, config):
        prices = make_cointegrated(beta=1.5, seed=21)
        res = engle_granger(prices["AAA"], prices["BBB"])
        assert res.hedge_ratio == pytest.approx(1.5, abs=0.05)

    def test_johansen_hedge_ratio_agrees_with_engle_granger(self, coint_data):
        eg = engle_granger(coint_data.y, coint_data.x)
        jo = johansen(coint_data.prices)
        assert jo.hedge_ratio == pytest.approx(eg.hedge_ratio, abs=0.15)

    def test_report_names_the_failures(self, random_data, config):
        report = run_cointegration_tests(random_data, config).report()
        assert "VERDICT: FAIL" in report
        assert "do NOT trade" in report


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------

class TestBacktest:
    def test_trade_pnl_reconciles_with_the_equity_curve(self, coint_data, config):
        result = run_backtest(coint_data, generate_signals(coint_data, config), config)
        total = sum(t.net_pnl for t in result.trades)
        delta = result.equity.iloc[-1] - result.equity.iloc[0]
        assert total == pytest.approx(delta, abs=1e-6)

    def test_no_position_is_left_open_at_the_end(self, coint_data, config):
        result = run_backtest(coint_data, generate_signals(coint_data, config), config)
        assert result.signals.frame["position"].iloc[-1] == 0 or result.trades[-1].exit_date == \
            result.equity.index[-1]

    def test_dollar_neutral_sizing_is_balanced_at_entry(self, coint_data, config):
        result = run_backtest(coint_data, generate_signals(coint_data, config), config)
        trade = result.trades[0]
        notional_y = abs(trade.quantity_y * trade.entry_price_y)
        notional_x = abs(trade.quantity_x * trade.entry_price_x)
        assert notional_y == pytest.approx(notional_x, rel=1e-9)
        assert notional_y == pytest.approx(config.backtest.gross_exposure_per_leg, rel=1e-9)

    def test_legs_have_opposite_signs(self, coint_data, config):
        result = run_backtest(coint_data, generate_signals(coint_data, config), config)
        for trade in result.trades:
            assert trade.quantity_y * trade.quantity_x < 0

    def test_costs_reduce_returns(self, coint_data, config):
        signals = generate_signals(coint_data, config)
        free = config.with_overrides(backtest={"commission_bps": 0.0, "slippage_bps": 0.0})
        expensive = config.with_overrides(backtest={"commission_bps": 10.0, "slippage_bps": 20.0})
        r_free = run_backtest(coint_data, signals, free)
        r_costly = run_backtest(coint_data, signals, expensive)
        assert r_costly.metrics.net_pnl < r_free.metrics.net_pnl
        assert r_free.metrics.total_costs == pytest.approx(0.0, abs=1e-9)

    def test_metrics_are_internally_consistent(self, coint_data, config):
        m = run_backtest(coint_data, generate_signals(coint_data, config), config).metrics
        assert m.n_wins + m.n_losses == m.n_trades
        assert m.win_rate == pytest.approx(m.n_wins / m.n_trades)
        assert m.max_drawdown >= 0
        assert m.net_pnl == pytest.approx(m.gross_pnl - m.total_costs, abs=1e-6)

    def test_a_flat_strategy_leaves_capital_untouched(self, coint_data, config):
        """Entry thresholds no z-score can reach must produce zero trades."""
        cfg = config.with_overrides(signal={"entry_z": 50.0, "exit_z": 0.5, "stop_z": 99.0})
        result = run_backtest(coint_data, generate_signals(coint_data, cfg), cfg)
        assert result.trades == []
        assert result.equity.nunique() == 1
        assert result.equity.iloc[-1] == pytest.approx(cfg.backtest.initial_capital)

    def test_writes_two_legs_per_fill_to_the_trade_log(self, coint_data, config, tmp_path):
        logger = TradeLogger(tmp_path / "trades.csv")
        result = run_backtest(coint_data, generate_signals(coint_data, config), config, logger)
        frame = logger.read()
        # Each round trip writes two legs on entry and two on exit.
        assert len(frame) == 4 * len(result.trades)
        assert set(frame["ticker"].unique()) == {"AAA", "BBB"}
        assert set(frame["action"].unique()) <= {"BUY", "SELL"}
        assert frame["zscore"].notna().all()

    def test_execution_lag_changes_results(self, coint_data, config):
        """A zero-lag run should beat a one-bar-lag run: it is look-ahead."""
        lagged = run_backtest(coint_data, generate_signals(coint_data, config), config)
        cfg0 = config.with_overrides(signal={"execution_lag": 0})
        instant = run_backtest(coint_data, generate_signals(coint_data, cfg0), cfg0)
        assert instant.metrics.net_pnl != lagged.metrics.net_pnl


# --------------------------------------------------------------------------
# trade log
# --------------------------------------------------------------------------

class TestTradeLog:
    def test_required_columns_come_first(self):
        from pairs_trading.trade_log import CSV_COLUMNS

        assert CSV_COLUMNS[:6] == [
            "timestamp", "action", "ticker", "quantity", "price", "zscore",
        ]

    def test_round_trip(self, tmp_path):
        logger = TradeLogger(tmp_path / "t.csv")
        logger.log(
            TradeRecord(
                timestamp="2024-01-02T00:00:00", action="BUY", ticker="KO",
                quantity=10, price=60.5, zscore=-2.3,
            )
        )
        frame = logger.read()
        assert len(frame) == 1
        assert frame.iloc[0]["ticker"] == "KO"
        assert frame.iloc[0]["zscore"] == pytest.approx(-2.3)
        assert frame.iloc[0]["notional"] == pytest.approx(605.0)

    def test_appends_across_sessions(self, tmp_path):
        path = tmp_path / "t.csv"
        for i in range(3):
            TradeLogger(path).log(
                TradeRecord(timestamp=f"2024-01-0{i+1}", action="SELL",
                            ticker="PEP", quantity=1, price=100.0, zscore=2.0)
            )
        assert len(TradeLogger(path).read()) == 3

    def test_rejects_a_negative_quantity(self):
        with pytest.raises(ValueError, match="non-negative"):
            TradeRecord(timestamp="t", action="BUY", ticker="KO",
                        quantity=-5, price=1.0, zscore=0.0)

    def test_rejects_an_unknown_action(self):
        with pytest.raises(ValueError, match="BUY or SELL"):
            TradeRecord(timestamp="t", action="HOLD", ticker="KO",
                        quantity=1, price=1.0, zscore=0.0)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class TestData:
    def test_extracts_price_from_a_flat_column_index(self):
        frame = pd.DataFrame({"Close": [1.0, 2.0], "Volume": [10, 20]})
        out = _extract_price_column(frame, "KO", "close")
        assert out.name == "KO"
        assert out.tolist() == [1.0, 2.0]

    def test_extracts_price_from_a_multiindex(self):
        cols = pd.MultiIndex.from_tuples([("Close", "KO"), ("Volume", "KO")])
        frame = pd.DataFrame([[1.0, 10], [2.0, 20]], columns=cols)
        out = _extract_price_column(frame, "KO", "close")
        assert out.tolist() == [1.0, 2.0]

    def test_missing_price_column_raises(self):
        frame = pd.DataFrame({"Open": [1.0]})
        with pytest.raises(DataError, match="no 'close' column"):
            _extract_price_column(frame, "KO", "close")

    def test_pair_data_exposes_legs_in_order(self, coint_data):
        assert coint_data.y.name == "AAA"
        assert coint_data.x.name == "BBB"
        assert len(coint_data) == len(coint_data.prices)

    def test_csv_loader_round_trips(self, tmp_path, config):
        from pairs_trading.data import load_prices_csv

        prices = make_cointegrated(n=300)
        path = tmp_path / "p.csv"
        prices.to_csv(path)
        loaded = load_prices_csv(path, config.pair)
        assert len(loaded) == 300
        assert list(loaded.prices.columns) == ["AAA", "BBB"]


# --------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline_on_a_cointegrated_pair(self, coint_data, config, tmp_path):
        result = run_cointegration_tests(coint_data, config)
        assert result.is_cointegrated
        assert_tradeable(result, config)

        signals = generate_signals(coint_data, config)
        logger = TradeLogger(tmp_path / "trades.csv")
        backtest = run_backtest(coint_data, signals, config, logger)

        assert backtest.metrics.n_trades > 0
        assert "BACKTEST RESULTS" in backtest.summary()
        assert logger.read().shape[0] > 0

    def test_chart_is_written(self, coint_data, config, tmp_path):
        from pairs_trading.backtest import plot_results

        backtest = run_backtest(coint_data, generate_signals(coint_data, config), config)
        out = tmp_path / "chart.png"
        plot_results(backtest, out, show=False)
        assert out.exists() and out.stat().st_size > 10_000

    def test_cli_backtest_via_csv(self, tmp_path, capsys):
        from pairs_trading.main import main

        path = tmp_path / "prices.csv"
        make_cointegrated().to_csv(path)
        code = main([
            "--backtest", "--pair", "AAA/BBB", "--csv", str(path),
            "--trade-log", str(tmp_path / "trades.csv"),
            "--plot-output", str(tmp_path / "plot.png"), "--quiet",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "VERDICT: PASS" in out
        assert "BACKTEST RESULTS" in out
        assert (tmp_path / "trades.csv").exists()

    def test_cli_blocks_a_non_cointegrated_pair(self, tmp_path, capsys):
        from pairs_trading.main import main

        path = tmp_path / "prices.csv"
        make_independent().to_csv(path)
        code = main([
            "--backtest", "--pair", "AAA/BBB", "--csv", str(path),
            "--trade-log", str(tmp_path / "trades.csv"), "--no-plot", "--quiet",
        ])
        assert code == 1
        captured = capsys.readouterr()
        assert "VERDICT: FAIL" in captured.out
        assert "NotCointegratedError" in captured.err

    def test_cli_force_overrides_the_gate(self, tmp_path, capsys):
        from pairs_trading.main import main

        path = tmp_path / "prices.csv"
        make_independent().to_csv(path)
        code = main([
            "--backtest", "--pair", "AAA/BBB", "--csv", str(path), "--force",
            "--trade-log", str(tmp_path / "trades.csv"), "--no-plot", "--quiet",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "--force is set" in out
        assert "BACKTEST RESULTS" in out

    def test_cli_rejects_a_malformed_pair(self):
        from pairs_trading.main import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--backtest", "--pair", "KOPEP"])

    @pytest.mark.parametrize("method", ["price_diff", "log_ratio", "ols"])
    def test_every_spread_method_runs(self, coint_data, config, method):
        cfg = config.with_overrides(spread={"method": method})
        result = run_backtest(coint_data, generate_signals(coint_data, cfg), cfg)
        assert np.isfinite(result.metrics.sharpe)
