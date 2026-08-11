"""Self-contained HTML dashboard for the pairs-trading system.

Renders one page from whatever the system has actually produced: the risk/halt
state, the backtest equity curve and z-score, the trade log and the risk-event
log, plus live paper-account state when asked for it.

    python -m pairs_trading.dashboard --pair KO/PEP
    python -m pairs_trading.dashboard --pair KO/PEP --live
    python -m pairs_trading.dashboard --csv prices.csv --open

DESIGN NOTES
------------
* **No external requests.** All CSS, JS and chart geometry are inlined, so the
  file opens from disk, survives being emailed, and leaks nothing about what you
  trade to a CDN.
* **Charts are hand-built SVG**, not a plotting library. The page has to be one
  self-contained file, and the three charts here (two line charts and a
  drawdown area) do not justify shipping a charting bundle inline.
* **One y-axis per chart, always.** Equity, z-score and drawdown each get their
  own panel rather than being twinned onto shared axes: two scales on one frame
  make the reader infer a relationship from crossings that are an artefact of
  the scaling.
* **Degrades gracefully.** Every section renders an explanatory empty state when
  its data source is missing, so the page is useful on a fresh install before
  anything has been run.
* **Theme-aware.** Light and dark palettes are both defined explicitly, under
  the OS media query and an explicit ``data-theme`` attribute.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, ConfigError, load_config, resolve_path

logger = logging.getLogger("pairs_trading.dashboard")

# Palette slots actually used, from the validated reference palette.
SERIES_BLUE_LIGHT, SERIES_BLUE_DARK = "#2a78d6", "#3987e5"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

MAX_CHART_POINTS = 1500
# Bars shown in the z-score panel. About a trading year -- enough context to see
# the current dislocation in perspective, few enough that individual fills stay
# distinguishable.
Z_WINDOW_BARS = 250


# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------

def _money(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def _pct(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:+.{decimals}%}"


def _num(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:,.{decimals}f}"


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# page pieces
# ---------------------------------------------------------------------------

@dataclass
class Tile:
    """One stat tile. ``tone`` picks a status colour for the value."""

    label: str
    value: str
    sub: str = ""
    tone: str = "neutral"      # neutral | good | warning | critical

    def render(self) -> str:
        sub = f'<div class="tile-sub">{_esc(self.sub)}</div>' if self.sub else ""
        return (
            f'<div class="tile"><div class="tile-label">{_esc(self.label)}</div>'
            f'<div class="tile-value tone-{self.tone}">{_esc(self.value)}</div>{sub}</div>'
        )


def _tone_for(value: float | None, good_when_positive: bool = True) -> str:
    if value is None or pd.isna(value):
        return "neutral"
    if value == 0:
        return "neutral"
    positive = value > 0
    return "good" if positive == good_when_positive else "critical"


def _meter(used: float, limit: float | None, label: str) -> str:
    """A usage meter for a risk limit: how much of the cap is consumed.

    Colour is a *status*, not a series: green under half, amber past 75%, red at
    the cap. It ships with the numeric label beside it so the state never rests
    on colour alone.
    """
    if limit is None or limit <= 0:
        return (
            f'<div class="meter"><div class="meter-head"><span>{_esc(label)}</span>'
            f'<span class="muted">no limit set</span></div></div>'
        )
    frac = max(0.0, min(used / limit, 1.0))
    tone = "good" if frac < 0.5 else "warning" if frac < 0.75 else "critical"
    return (
        f'<div class="meter"><div class="meter-head"><span>{_esc(label)}</span>'
        f'<span class="nums">{_money(used)} / {_money(limit)}</span></div>'
        f'<div class="meter-track"><div class="meter-fill tone-{tone}" '
        f'style="width:{frac * 100:.1f}%"></div></div></div>'
    )


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------

def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values spanning [low, high]."""
    if high <= low:
        return [low]
    raw = (high - low) / max(count - 1, 1)
    magnitude = 10 ** (len(f"{int(abs(raw))}") - 1) if abs(raw) >= 1 else 10 ** -2
    for mult in (1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        step = magnitude * mult
        if step >= raw:
            break
    start = (low // step) * step
    ticks, value = [], start
    while value <= high + step * 0.5:
        if value >= low - step * 0.5:
            ticks.append(round(value, 6))
        value += step
    return ticks or [low, high]


def _downsample(frame: pd.DataFrame, limit: int = MAX_CHART_POINTS) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    step = len(frame) // limit + 1
    return frame.iloc[::step]


def _line_chart(
    chart_id: str,
    x_labels: list[str],
    values: list[float],
    *,
    title: str,
    y_fmt: str = "money",
    height: int = 240,
    reference_lines: list[tuple[float, str, str]] | None = None,
    markers: list[tuple[int, float, str, str]] | None = None,
    fill_to_zero: bool = False,
    note: str = "",
    legend: list[tuple[str, str]] | None = None,
) -> str:
    """Render one single-series line chart as inline SVG.

    Args:
        chart_id: Unique id, used to wire the hover layer.
        x_labels: One label per point (dates), shown in the tooltip.
        values: The series.
        title: Chart heading; it names the series, so no legend box is needed.
        y_fmt: ``money`` | ``plain`` | ``pct`` -- controls tick and tooltip text.
        reference_lines: ``(y, label, tone)`` horizontal rules (thresholds).
        markers: ``(index, y, tone, label)`` points to emphasise (trade fills).
        fill_to_zero: Shade the area between the line and zero.
        note: Sub-heading, e.g. the window a truncated chart is showing.
        legend: ``(tone, label)`` swatches. Required whenever markers carry
            meaning, so identity never rests on colour alone.
    """
    if not values:
        return _empty_panel(title, "No data yet.")

    width, pad_l, pad_r, pad_t, pad_b = 1000, 62, 16, 14, 26
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    lo, hi = min(values), max(values)
    for ref in reference_lines or []:
        lo, hi = min(lo, ref[0]), max(hi, ref[0])
    if fill_to_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        hi, lo = hi + 1, lo - 1
    span = hi - lo
    lo -= span * 0.06
    hi += span * 0.06

    n = len(values)

    def sx(i: int) -> float:
        return pad_l + (plot_w * i / max(n - 1, 1))

    def sy(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / (hi - lo))

    def fmt(v: float) -> str:
        if y_fmt == "money":
            return _money(v, 0)
        if y_fmt == "pct":
            return f"{v:.1f}%"
        return f"{v:,.2f}"

    parts: list[str] = []

    # gridlines + y ticks
    for tick in _nice_ticks(lo, hi):
        y = sy(tick)
        if not (pad_t - 1 <= y <= pad_t + plot_h + 1):
            continue
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{pad_l - 8}" y="{y + 3.5:.1f}" text-anchor="end">{fmt(tick)}</text>'
        )

    # x ticks
    for i in [int(round(k * (n - 1) / 4)) for k in range(5)] if n > 1 else [0]:
        parts.append(
            f'<text class="tick" x="{sx(i):.1f}" y="{height - 8}" text-anchor="middle">'
            f"{_esc(x_labels[i] if i < len(x_labels) else '')}</text>"
        )

    # reference lines (thresholds) -- chrome, not series
    for value, label, tone in reference_lines or []:
        y = sy(value)
        parts.append(
            f'<line class="ref ref-{tone}" x1="{pad_l}" y1="{y:.1f}" '
            f'x2="{pad_l + plot_w}" y2="{y:.1f}"/>'
        )
        # Labels sit just inside the LEFT edge: the right edge carries the most
        # recent bars, which is exactly where a reader looks and exactly where
        # a label would sit on top of live data. An empty label draws the rule
        # without text, so a symmetric pair is annotated once rather than twice
        # on top of itself.
        if label:
            parts.append(
                f'<text class="ref-label ref-{tone}" x="{pad_l + 5}" y="{y - 4:.1f}" '
                f'text-anchor="start">{_esc(label)}</text>'
            )

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values)
    )

    if fill_to_zero:
        zero_y = sy(0.0)
        area = f"M{sx(0):.1f},{zero_y:.1f} " + " ".join(
            f"L{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values)
        ) + f" L{sx(n - 1):.1f},{zero_y:.1f} Z"
        parts.append(f'<path class="area" d="{area}"/>')
        parts.append(
            f'<line class="zero" x1="{pad_l}" y1="{zero_y:.1f}" '
            f'x2="{pad_l + plot_w}" y2="{zero_y:.1f}"/>'
        )

    parts.append(f'<path class="series" d="{path}"/>')

    # trade markers, drawn with a surface ring so overlaps stay legible
    for index, value, tone, label in markers or []:
        if 0 <= index < n:
            parts.append(
                f'<circle class="marker marker-{tone}" cx="{sx(index):.1f}" cy="{sy(value):.1f}" '
                f'r="4.5"><title>{_esc(label)}</title></circle>'
            )

    # hover layer
    parts.append(f'<line class="crosshair" id="{chart_id}-cross" x1="0" y1="{pad_t}" x2="0" y2="{pad_t + plot_h}" style="opacity:0"/>')
    parts.append(f'<circle class="hover-dot" id="{chart_id}-dot" r="4.5" style="opacity:0"/>')
    parts.append(
        f'<rect id="{chart_id}-hit" x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="transparent"/>'
    )

    payload = json.dumps(
        {
            "padL": pad_l, "padT": pad_t, "plotW": plot_w, "plotH": plot_h,
            "vbW": width, "vbH": height,
            "lo": lo, "hi": hi, "n": n, "fmt": y_fmt,
            "labels": x_labels, "values": [round(float(v), 6) for v in values],
        }
    )

    note_html = f'<p class="panel-note">{_esc(note)}</p>' if note else ""
    legend_html = ""
    if legend:
        swatches = "".join(
            f'<span class="legend-item"><span class="swatch swatch-{tone}"></span>{_esc(label)}</span>'
            for tone, label in legend
        )
        legend_html = f'<div class="legend">{swatches}</div>'

    return f"""<section class="panel">
  <h3 class="panel-title">{_esc(title)}</h3>
  {note_html}{legend_html}
  <div class="chart-wrap">
    <svg class="chart" id="{chart_id}" viewBox="0 0 {width} {height}"
         preserveAspectRatio="none" role="img" aria-label="{_esc(title)}">
      {''.join(parts)}
    </svg>
    <div class="tooltip" id="{chart_id}-tip"></div>
  </div>
  <script type="application/json" id="{chart_id}-data">{payload}</script>
</section>"""


def _empty_panel(title: str, message: str, hint: str = "") -> str:
    hint_html = f'<div class="empty-hint"><code>{_esc(hint)}</code></div>' if hint else ""
    return (
        f'<section class="panel"><h3 class="panel-title">{_esc(title)}</h3>'
        f'<div class="empty"><p>{_esc(message)}</p>{hint_html}</div></section>'
    )


def _table(title: str, frame: pd.DataFrame, columns: list[tuple[str, str]],
           empty_message: str, hint: str = "", limit: int = 25) -> str:
    """Render a DataFrame as a table panel. ``columns`` is ``[(key, header)]``."""
    if frame is None or frame.empty:
        return _empty_panel(title, empty_message, hint)

    present = [(k, h) for k, h in columns if k in frame.columns]
    rows = frame.tail(limit).iloc[::-1]

    head = "".join(f"<th>{_esc(h)}</th>" for _, h in present)
    body = []
    for _, row in rows.iterrows():
        cells = []
        for key, _ in present:
            value = row[key]
            cls = ""
            if key in ("action", "event", "severity"):
                text = str(value)
                tone = {
                    "BUY": "good", "SELL": "critical",
                    "order_rejected": "warning", "daily_loss_breach": "critical",
                    "trading_halted": "critical", "kill_switch": "critical",
                    "order_approved": "good", "halt_cleared": "warning",
                    "critical": "critical", "warning": "warning", "info": "neutral",
                }.get(text, "neutral")
                cells.append(f'<td><span class="pill tone-{tone}">{_esc(text)}</span></td>')
                continue
            if key == "timestamp":
                # Backtest fills are dated bars, so "T00:00:00" is noise; live
                # fills carry a real time worth keeping, to the minute.
                text = str(value).replace("T00:00:00", "")
                if "T" in text:
                    text = text.split("+")[0].replace("T", " ")[:16]
                cls = ' class="nums"'
            elif isinstance(value, float):
                text = f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"
                cls = ' class="nums"'
            else:
                text = str(value)
            cells.append(f"<td{cls}>{_esc(text)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")

    note = (
        f'<p class="panel-note">Showing the {min(limit, len(frame))} most recent '
        f"of {len(frame)} rows, newest first.</p>"
    )
    return f"""<section class="panel">
  <h3 class="panel-title">{_esc(title)}</h3>
  {note}
  <div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


# ---------------------------------------------------------------------------
# data gathering
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def _live_account(config: Config) -> dict[str, Any] | None:
    """Fetch paper-account state, or ``None`` if unreachable.

    Never raises: the dashboard must still render when the broker is down,
    since "the broker is down" is exactly when you want to look at it.
    """
    try:
        from .execution_alpaca import AlpacaPaperTrader

        trader = AlpacaPaperTrader(config, trade_logger=None, dry_run=True)
        snapshot = trader.account_snapshot()
        position = trader.get_pair_position()
        return {
            "equity": snapshot.equity,
            "buying_power": snapshot.buying_power,
            "daily_pnl": snapshot.daily_pnl,
            "daily_pnl_pct": snapshot.daily_pnl_pct,
            "exposure": snapshot.open_exposure,
            "positions": snapshot.position_count,
            "market_open": snapshot.market_open,
            "shares_y": position.shares_y,
            "shares_x": position.shares_x,
            "direction": position.direction,
            "base_url": trader.base_url,
        }
    except Exception as exc:
        logger.warning("Live account unavailable: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

_CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  color-scheme:light;
  --plane:#f9f9f7; --surface:#fcfcfb;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series:#2a78d6; --series-soft:rgba(42,120,214,0.14);
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
  --radius:10px;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --plane:#0d0d0d; --surface:#1a1a19;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series:#3987e5; --series-soft:rgba(57,135,229,0.18);
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --plane:#0d0d0d; --surface:#1a1a19;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series:#3987e5; --series-soft:rgba(57,135,229,0.18);
}
body{
  margin:0; padding:24px 20px 64px;
  background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:14px; line-height:1.5;
}
.wrap{max-width:1140px;margin:0 auto}
header.page{margin-bottom:20px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-0.01em}
.sub{color:var(--ink-2);font-size:13px;margin:0}
.nums{font-variant-numeric:tabular-nums}
.muted{color:var(--muted)}
.notice{
  padding:9px 14px;border-radius:8px;margin-bottom:16px;
  background:var(--warning);color:#1a1a19;font-weight:640;font-size:12.5px;
  letter-spacing:.02em;
}
.banner{
  display:flex;gap:12px;align-items:flex-start;
  padding:14px 16px;border-radius:var(--radius);margin:16px 0 22px;
  border:1px solid var(--border);background:var(--surface);
}
.banner.halted{border-left:4px solid var(--critical)}
.banner.active{border-left:4px solid var(--good)}
.banner-icon{font-size:17px;line-height:1.3}
.banner-title{font-weight:650;margin-bottom:2px}
.banner-body{color:var(--ink-2);font-size:13px}
.banner code{background:var(--plane);padding:1px 6px;border-radius:5px;font-size:12px}
.grid-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px;margin-bottom:22px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:13px 15px}
.tile-label{font-size:11.5px;text-transform:uppercase;letter-spacing:.045em;color:var(--muted);margin-bottom:5px}
.tile-value{font-size:21px;font-weight:640;letter-spacing:-0.015em}
.tile-sub{font-size:12px;color:var(--ink-2);margin-top:3px}
.tone-good{color:var(--good)} .tone-critical{color:var(--critical)}
.tone-warning{color:var(--warning)} .tone-neutral{color:var(--ink)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;margin-bottom:18px}
.panel-title{font-size:14.5px;font-weight:640;margin:0 0 3px}
.panel-note{font-size:12px;color:var(--muted);margin:0 0 11px}
.meters{display:grid;gap:14px}
.meter-head{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:5px;gap:12px}
.meter-track{height:7px;border-radius:4px;background:var(--grid);overflow:hidden}
.meter-fill{height:100%;border-radius:4px}
.meter-fill.tone-good{background:var(--good)}
.meter-fill.tone-warning{background:var(--warning)}
.meter-fill.tone-critical{background:var(--critical)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 2px;font-size:12px;color:var(--ink-2)}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.swatch{width:9px;height:9px;border-radius:50%;display:inline-block;border:1px solid var(--surface)}
.swatch-long{background:var(--good)} .swatch-short{background:var(--critical)}
.swatch-exit{background:var(--muted)}
.chart-wrap{position:relative;margin-top:10px}
.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.zero{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.series{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}
.area{fill:var(--series-soft);stroke:none}
.ref{stroke-width:1;stroke-dasharray:5 4;vector-effect:non-scaling-stroke}
.ref-entry{stroke:var(--good)} .ref-exit{stroke:var(--muted)} .ref-stop{stroke:var(--critical)}
.ref-label{font-size:10.5px}
.ref-label.ref-entry{fill:var(--good)} .ref-label.ref-exit{fill:var(--muted)} .ref-label.ref-stop{fill:var(--critical)}
.marker{stroke:var(--surface);stroke-width:2}
.marker-long{fill:var(--good)} .marker-short{fill:var(--critical)} .marker-exit{fill:var(--muted)}
.crosshair{stroke:var(--axis);stroke-width:1;pointer-events:none}
.hover-dot{fill:var(--series);stroke:var(--surface);stroke-width:2;pointer-events:none}
.tooltip{
  position:absolute;pointer-events:none;opacity:0;transform:translate(-50%,-115%);
  background:var(--ink);color:var(--surface);padding:5px 9px;border-radius:6px;
  font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums;z-index:5;
}
.table-wrap{overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
td.nums{font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11.5px;font-weight:600;border:1px solid currentColor}
.empty{padding:22px 2px;color:var(--ink-2);font-size:13px}
.empty p{margin:0 0 9px}
.empty-hint code{background:var(--plane);border:1px solid var(--border);padding:6px 10px;border-radius:6px;display:inline-block;font-size:12px}
footer.page{margin-top:26px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);padding-top:14px}
"""

_JS = """
(function(){
  document.querySelectorAll('svg.chart').forEach(function(svg){
    var id = svg.id;
    var node = document.getElementById(id + '-data');
    if(!node) return;
    var d = JSON.parse(node.textContent);
    var hit = document.getElementById(id + '-hit');
    var cross = document.getElementById(id + '-cross');
    var dot = document.getElementById(id + '-dot');
    var tip = document.getElementById(id + '-tip');
    if(!hit) return;

    function fmt(v){
      if(d.fmt === 'money'){
        var s = v < 0 ? '-' : '';
        return s + '$' + Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0});
      }
      if(d.fmt === 'pct') return v.toFixed(2) + '%';
      return v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
    }
    function show(evt){
      var r = svg.getBoundingClientRect();
      // preserveAspectRatio="none": x and y scale independently, so map each
      // axis by its own viewBox extent. Scaling y by the width puts the
      // tooltip in the wrong place on every non-square chart.
      var vx = (evt.clientX - r.left) / r.width * d.vbW;
      var i = Math.round((vx - d.padL) / d.plotW * (d.n - 1));
      i = Math.max(0, Math.min(d.n - 1, i));
      var x = d.padL + d.plotW * i / Math.max(d.n - 1, 1);
      var y = d.padT + d.plotH * (1 - (d.values[i] - d.lo) / (d.hi - d.lo));
      cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.style.opacity = 1;
      dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.style.opacity = 1;
      tip.style.opacity = 1;
      tip.style.left = (x / d.vbW * r.width) + 'px';
      tip.style.top  = (y / d.vbH * r.height) + 'px';
      tip.textContent = d.labels[i] + '  ·  ' + fmt(d.values[i]);
    }
    function hide(){ cross.style.opacity=0; dot.style.opacity=0; tip.style.opacity=0; }
    hit.addEventListener('mousemove', show);
    hit.addEventListener('mouseleave', hide);
    hit.addEventListener('touchmove', function(e){ if(e.touches[0]) show(e.touches[0]); });
  });
})();
"""


def build_html(
    config: Config,
    backtest_result: Any | None = None,
    live: dict[str, Any] | None = None,
    notice: str = "",
    standalone: bool = True,
) -> str:
    """Assemble the whole dashboard page.

    Args:
        config: Loaded configuration.
        backtest_result: Optional completed backtest, for the charts.
        live: Optional live account state from :func:`_live_account`.
        notice: Banner text shown above everything else. Use it to label a page
            built from anything other than this account's real activity --
            a demo, a replay, a shared snapshot -- so a reader can never mistake
            it for the live book.
        standalone: Emit a complete HTML document. Set False to get just the
            page content, for embedding in a host that supplies its own
            ``<!doctype>``/``<head>``/``<body>`` wrapper.
    """
    from .risk_manager import RiskManager

    risk = RiskManager(config)
    pair = str(config.pair)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    trades = _read_csv(resolve_path(config.logging.trade_log))
    events = _read_csv(resolve_path(config.risk.risk_log))

    # ---- status banner -------------------------------------------------
    if risk.is_halted():
        banner = (
            '<div class="banner halted"><div class="banner-icon">&#9940;</div><div>'
            '<div class="banner-title tone-critical">TRADING HALTED</div>'
            f'<div class="banner-body">{_esc(risk.halt_reason())}<br>'
            "No new positions will be opened. Clear with "
            "<code>python -m pairs_trading.kill_switch --clear</code></div></div></div>"
        )
    else:
        banner = (
            '<div class="banner active"><div class="banner-icon">&#9989;</div><div>'
            '<div class="banner-title tone-good">ACTIVE</div>'
            '<div class="banner-body">No halt flag. Risk checks run on every order.</div>'
            "</div></div>"
        )

    # ---- headline tiles ------------------------------------------------
    tiles: list[Tile] = []
    if live and "error" not in live:
        direction = {1: "Long spread", -1: "Short spread", 0: "Flat"}.get(live["direction"], "?")
        tiles += [
            Tile("Account equity", _money(live["equity"])),
            Tile("Today's P&L", _money(live["daily_pnl"]),
                 _pct(live["daily_pnl_pct"]), _tone_for(live["daily_pnl"])),
            Tile("Buying power", _money(live["buying_power"])),
            Tile("Gross exposure", _money(live["exposure"]), f"{live['positions']} position(s)"),
            Tile("Pair position", direction,
                 f"{config.pair.y} {live['shares_y']:+,.0f} / {config.pair.x} {live['shares_x']:+,.0f}"),
            Tile("Market", "Open" if live["market_open"] else "Closed",
                 tone="good" if live["market_open"] else "neutral"),
        ]
    elif live and "error" in live:
        tiles.append(Tile("Live account", "unavailable", "see footer", "warning"))

    if backtest_result is not None:
        m = backtest_result.metrics
        tiles += [
            Tile("Backtest net P&L", _money(m.net_pnl), _pct(m.total_return), _tone_for(m.net_pnl)),
            Tile("Sharpe", _num(m.sharpe), "annualised"),
            Tile("Max drawdown", _pct(-m.max_drawdown), tone="critical" if m.max_drawdown > 0 else "neutral"),
            Tile("Win rate", f"{m.win_rate:.0%}", f"{m.n_wins}W / {m.n_losses}L"),
            Tile("Round trips", f"{m.n_trades}", f"avg {m.avg_bars_held:.1f} bars"),
            Tile("Costs paid", _money(m.total_costs), "commission + slippage"),
        ]

    tiles_html = (
        f'<div class="grid-tiles">{"".join(t.render() for t in tiles)}</div>' if tiles else ""
    )

    # ---- risk limit meters ---------------------------------------------
    used_exposure = live["exposure"] if live and "error" not in live else 0.0
    daily_loss = -(live["daily_pnl"] if live and "error" not in live else 0.0)
    meters = "".join([
        _meter(max(used_exposure, 0.0), config.risk.max_total_exposure_usd, "Total exposure"),
        _meter(max(daily_loss, 0.0), config.risk.max_daily_loss_usd, "Daily loss vs limit"),
        _meter(2 * config.execution.notional_per_leg, config.risk.max_position_size_usd,
               "Next trade size vs per-trade cap"),
    ])
    risk_panel = (
        '<section class="panel"><h3 class="panel-title">Risk limits</h3>'
        '<p class="panel-note">Every order is checked against these before it is sent. '
        "A breach rejects the order; it is never resized.</p>"
        f'<div class="meters">{meters}</div></section>'
    )

    # ---- charts ---------------------------------------------------------
    charts: list[str] = []
    if backtest_result is not None:
        equity = _downsample(backtest_result.equity.to_frame("equity"))
        labels = [d.strftime("%Y-%m-%d") for d in equity.index]
        pnl = (equity["equity"] - float(backtest_result.equity.iloc[0])).tolist()
        charts.append(_line_chart(
            "chart-pnl", labels, pnl,
            title=f"Cumulative P&L — {pair} backtest",
            y_fmt="money", height=250, fill_to_zero=True,
        ))

        drawdown = (backtest_result.equity / backtest_result.equity.cummax() - 1.0) * 100.0
        dd = _downsample(drawdown.to_frame("dd"))
        charts.append(_line_chart(
            "chart-dd", [d.strftime("%Y-%m-%d") for d in dd.index], dd["dd"].tolist(),
            title="Drawdown", y_fmt="pct", height=170, fill_to_zero=True,
        ))

        # The z-score panel answers "where is the spread right now", so it shows
        # a recent window at full resolution. Downsampling the whole history
        # into this frame turns the line into a solid block and stacks the
        # trade markers on top of each other -- unreadable, and misleading
        # about how often signals actually fire.
        frame = backtest_result.signals.frame
        valid = frame[frame["zscore"].notna()]
        zf = valid.tail(Z_WINDOW_BARS)
        sig = config.signal
        index_of = {ts: i for i, ts in enumerate(zf.index)}
        markers: list[tuple[int, float, str, str]] = []
        for trade in backtest_result.trades:
            for ts, value, tone, what in (
                (trade.entry_date, trade.entry_zscore,
                 "long" if trade.direction > 0 else "short", "entry"),
                (trade.exit_date, trade.exit_zscore, "exit", trade.exit_reason),
            ):
                if ts in index_of:
                    markers.append((
                        index_of[ts], value, tone,
                        f"{ts.date()} {what} z={value:+.2f}",
                    ))
        window_note = (
            f"Last {len(zf)} bars of {len(valid)} "
            f"({zf.index[0].date()} to {zf.index[-1].date()}). "
            f"{len(markers)} fills in this window."
        )
        charts.append(_line_chart(
            "chart-z", [d.strftime("%Y-%m-%d") for d in zf.index], zf["zscore"].tolist(),
            title="Spread z-score, with entries and exits",
            y_fmt="plain", height=250, note=window_note,
            legend=[("long", "Long-spread entry"), ("short", "Short-spread entry"),
                    ("exit", "Exit")],
            reference_lines=[
                (sig.entry_z, f"entry +{sig.entry_z:g}", "entry"),
                (-sig.entry_z, f"entry -{sig.entry_z:g}", "entry"),
                (sig.exit_z, f"exit band +/-{sig.exit_z:g}", "exit"), (-sig.exit_z, "", "exit"),
                (sig.stop_z, f"stop +{sig.stop_z:g}", "stop"),
                (-sig.stop_z, f"stop -{sig.stop_z:g}", "stop"),
            ],
            markers=markers,
        ))
    else:
        charts.append(_empty_panel(
            "Performance charts",
            "No backtest has been run for this dashboard, so there is no equity curve "
            "or z-score to plot.",
            f"python -m pairs_trading.dashboard --pair {pair}",
        ))

    # ---- tables ---------------------------------------------------------
    trades_table = _table(
        "Trade log", trades,
        [("timestamp", "Time"), ("action", "Action"), ("ticker", "Ticker"),
         ("quantity", "Qty"), ("price", "Price"), ("zscore", "Z-score"),
         ("side", "Side"), ("reason", "Reason"), ("mode", "Mode")],
        "No trades logged yet.",
        "python -m pairs_trading.main --backtest --pair " + pair,
    )
    events_table = _table(
        "Risk events", events,
        [("timestamp", "Time"), ("event", "Event"), ("severity", "Severity"),
         ("symbol", "Symbol"), ("notional", "Notional"), ("equity", "Equity"),
         ("daily_pnl", "Daily P&L"), ("reason", "Reason")],
        "No risk events yet. Rejections, breaches and kill-switch activations appear here.",
    )

    footer_bits = [f"Generated {generated}", f"config: {config.source_path}"]
    if live and "error" in live:
        footer_bits.append(f"live account unavailable: {live['error'][:160]}")
    elif live:
        footer_bits.append(f"paper endpoint: {live.get('base_url', '')}")

    notice_html = (
        f'<div class="notice">{_esc(notice)}</div>' if notice else ""
    )

    body = f"""<title>Pairs trading — {_esc(pair)}</title>
<style>{_CSS}</style>
<div class="wrap">
  {notice_html}
  <header class="page">
    <h1>Pairs trading &mdash; {_esc(pair)}</h1>
    <p class="sub">Statistical arbitrage &middot; Alpaca <strong>paper</strong> trading &middot; {_esc(generated)}</p>
  </header>
  {banner}
  {tiles_html}
  {risk_panel}
  {''.join(charts)}
  {trades_table}
  {events_table}
  <footer class="page">{_esc(' · '.join(footer_bits))}</footer>
</div>
<script>{_JS}</script>"""

    if not standalone:
        return body
    # A file opened straight from disk needs a doctype, or the browser falls
    # back to quirks mode and the layout shifts under you.
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{body}\n</html>"
    )


def generate(
    config: Config,
    output: str | Path | None = None,
    backtest_result: Any | None = None,
    live: bool = False,
    notice: str = "",
    standalone: bool = True,
) -> Path:
    """Write the dashboard HTML and return its path."""
    live_data = _live_account(config) if live else None
    html_text = build_html(config, backtest_result, live_data, notice, standalone)
    path = resolve_path(output or "logs/dashboard.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    logger.info("Dashboard written to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairs_trading.dashboard",
        description="Generate a self-contained HTML dashboard for the pairs-trading system.",
    )
    parser.add_argument("--pair", help="Ticker pair, e.g. KO/PEP.")
    parser.add_argument("--config", help="Path to config.yaml.")
    parser.add_argument("--csv", help="Load prices from a CSV instead of Yahoo Finance.")
    parser.add_argument("--output", help="Output HTML path (default logs/dashboard.html).")
    parser.add_argument("--live", action="store_true",
                        help="Include live paper-account state (needs Alpaca keys).")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Skip the backtest; render logs and risk state only.")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="Open the dashboard in a browser when done.")
    parser.add_argument("--notice",
                        help="Banner text above the page. Use it to label a page built from "
                             "anything other than this account's real activity.")
    parser.add_argument("--embed", action="store_true",
                        help="Emit page content only, without the <!doctype>/<head> wrapper, "
                             "for embedding in a host that supplies its own.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    try:
        config = load_config(args.config)
        if args.pair:
            from .main import _parse_pair

            y, x = _parse_pair(args.pair)
            config = config.with_overrides(pair={"y": y, "x": x})
    except (ConfigError, argparse.ArgumentTypeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    result = None
    if not args.no_backtest:
        try:
            from .backtest import run_backtest
            from .data import fetch_pair, load_prices_csv
            from .strategy import generate_signals

            data = (
                load_prices_csv(args.csv, config.pair) if args.csv else fetch_pair(config)
            )
            result = run_backtest(data, generate_signals(data, config), config)
        except Exception as exc:
            # A dashboard that refuses to render because prices are unavailable
            # is useless exactly when you most want to look at it.
            logger.warning("Backtest unavailable (%s); rendering logs only.", exc)

    path = generate(
        config, args.output, result, live=args.live,
        notice=args.notice or "", standalone=not args.embed,
    )
    print(f"Dashboard: {path}")
    if args.open_browser:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
