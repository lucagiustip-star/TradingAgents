"""Tests for the HTML dashboard generator.

The dashboard is a reporting surface, so the properties worth pinning are:
it renders at all from every combination of missing inputs, it stays a single
self-contained file, and it never leaks user data into an external request.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pairs_trading.backtest import run_backtest
from pairs_trading.config import load_config
from pairs_trading.dashboard import build_html, generate
from pairs_trading.data import PairData
from pairs_trading.risk_manager import RiskManager
from pairs_trading.strategy import generate_signals

pytestmark = pytest.mark.unit


def make_prices(n: int = 700, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = 100 + np.cumsum(rng.normal(0, 1.0, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.94 * ou[i - 1] + rng.normal(0, 1.0)
    frame = pd.DataFrame({"AAA": 50 + x + ou, "BBB": x}, index=idx)
    frame.index.name = "date"
    return frame


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
        logging=replace(cfg.logging, trade_log=str(tmp_path / "trades.csv")),
    )


@pytest.fixture
def result(config):
    data = PairData(prices=make_prices(), pair=config.pair, source="synthetic")
    return run_backtest(data, generate_signals(data, config), config)


class TestRendering:
    def test_renders_with_no_data_at_all(self, config):
        """A fresh install, nothing run yet, must still produce a usable page."""
        page = build_html(config)
        assert "<title>" in page
        assert "AAA/BBB" in page
        assert "No trades logged yet" in page
        assert "No risk events yet" in page

    def test_renders_with_a_backtest(self, config, result):
        page = build_html(config, result)
        assert "Cumulative P&amp;L" in page or "Cumulative P&L" in page
        assert "chart-pnl" in page and "chart-z" in page and "chart-dd" in page
        assert "Spread z-score" in page

    def test_shows_active_when_not_halted(self, config):
        assert "ACTIVE" in build_html(config)

    def test_shows_halt_banner_and_reason(self, config):
        RiskManager(config).halt("spread relationship broke down")
        page = build_html(config)
        assert "TRADING HALTED" in page
        assert "spread relationship broke down" in page
        assert "kill_switch --clear" in page

    def test_halt_reason_is_the_line_not_the_whole_file(self, config):
        """The banner shows the reason, not the flag file's own boilerplate.

        `halt_reason()` parses out the reason line; dumping the whole file would
        paste its "Clear this file to resume trading" instructions into every
        rejection message and alert, burying the actual cause.
        """
        risk = RiskManager(config)
        risk.halt("a specific reason")
        assert risk.halt_reason() == "a specific reason"
        assert "Clear this file to resume trading" in risk.halt_details()

        page = build_html(config)
        assert "a specific reason" in page
        assert "Clear this file to resume trading" not in page

    def test_writes_a_file(self, config, tmp_path, result):
        path = generate(config, tmp_path / "dash.html", result)
        assert path.exists()
        assert path.stat().st_size > 20_000

    def test_metrics_reach_the_page(self, config, result):
        page = build_html(config, result)
        assert f"{result.metrics.n_trades}" in page
        assert "Sharpe" in page and "Win rate" in page


class TestSelfContained:
    """A dashboard that phoned home would leak what you trade."""

    def test_no_external_urls(self, config, result):
        page = build_html(config, result)
        external = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', page)
        assert external == [], f"external resource(s) referenced: {external}"

    def test_no_network_calls_in_script(self, config, result):
        page = build_html(config, result)
        for token in ("fetch(", "XMLHttpRequest", "WebSocket", "import(", "cdn."):
            assert token not in page, f"page performs a network call: {token}"

    def test_styles_and_scripts_are_inline(self, config, result):
        page = build_html(config, result)
        assert "<style>" in page and "<script>" in page
        assert "stylesheet" not in page


class TestTheming:
    def test_defines_both_light_and_dark_palettes(self, config):
        page = build_html(config)
        assert "prefers-color-scheme:dark" in page
        assert '[data-theme="dark"]' in page

    def test_body_background_is_explicit(self, config):
        """A transparent body borrows the host page's colour and can go unreadable."""
        assert "background:var(--plane)" in build_html(config)


class TestCharts:
    def test_zscore_panel_is_windowed(self, config, result):
        """The z-score panel shows a recent window, not the whole history.

        Compressing 1000+ bars into one frame renders the line as a solid block
        and stacks every trade marker, which misrepresents how often the signal
        actually fires.
        """
        page = build_html(config, result)
        assert re.search(r"Last \d+ bars of \d+", page)

    def test_marker_meaning_is_not_colour_alone(self, config, result):
        page = build_html(config, result)
        for label in ("Long-spread entry", "Short-spread entry", "Exit"):
            assert label in page

    def test_thresholds_are_drawn(self, config, result):
        page = build_html(config, result)
        assert f"entry +{config.signal.entry_z:g}" in page
        assert f"stop +{config.signal.stop_z:g}" in page

    def test_chart_payloads_are_valid_json(self, config, result):
        import json

        page = build_html(config, result)
        for block in re.findall(
            r'<script type="application/json" id="[^"]+">(.*?)</script>', page, re.DOTALL
        ):
            data = json.loads(block)
            assert len(data["values"]) == data["n"]
            assert len(data["labels"]) == data["n"]
            # The hover layer needs both viewBox extents to place the tooltip.
            assert data["vbW"] > 0 and data["vbH"] > 0

    def test_no_dual_axis_in_the_backtest_chart(self):
        """Two y-scales on one frame invite false readings of the crossings."""
        import inspect

        from pairs_trading import backtest

        source = inspect.getsource(backtest.plot_results)
        assert "twinx" not in source and "twiny" not in source


class TestEscaping:
    def test_halt_reason_is_escaped(self, config):
        RiskManager(config).halt('<script>alert("x")</script>')
        page = build_html(config)
        assert '<script>alert("x")</script>' not in page
        assert "&lt;script&gt;" in page


class TestCli:
    def test_dashboard_via_csv(self, tmp_path, capsys):
        from pairs_trading.dashboard import main

        prices = tmp_path / "p.csv"
        make_prices().to_csv(prices)
        out = tmp_path / "d.html"
        code = main([
            "--pair", "AAA/BBB", "--csv", str(prices), "--output", str(out),
        ])
        assert code == 0
        assert out.exists()
        assert "Dashboard:" in capsys.readouterr().out

    def test_dashboard_renders_without_price_data(self, tmp_path, capsys):
        """An unreachable data vendor must degrade, not abort."""
        from pairs_trading.dashboard import main

        out = tmp_path / "d.html"
        code = main([
            "--pair", "AAA/BBB", "--csv", str(tmp_path / "missing.csv"),
            "--output", str(out),
        ])
        assert code == 0
        assert out.exists()
        assert "No backtest has been run" in out.read_text()

    def test_main_dashboard_flag_is_a_mode(self):
        from pairs_trading.main import build_parser

        args = build_parser().parse_args(["--dashboard"])
        assert args.dashboard is True

    def test_dashboard_is_exclusive_with_backtest(self):
        from pairs_trading.main import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--dashboard", "--backtest"])
