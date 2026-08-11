"""News-driven veto: halt trading when a headline breaks the pair's relationship.

WHAT THIS IS, AND WHAT IT IS DELIBERATELY NOT
---------------------------------------------
This is a **veto**, not a signal. It can stop the strategy trading; it can never
tell it what to trade. That asymmetry is the whole design.

The strategy trades daily bars with a one-bar execution lag, so a news feed
cannot usefully inform the *direction* of a position -- there is no path from a
headline to a better z-score. What news can do is tell you the spread's
equilibrium has structurally changed, which the statistics only discover later,
after the loss. Cointegration is a claim about a stable long-run relationship;
a merger, a spin-off or a delisting ends that relationship outright, and the
z-score keeps producing confident entry signals into a spread that will never
revert.

THE DISTINCTION THAT MATTERS
----------------------------
Most news about a stock is *not* a reason to stop. Earnings, guidance changes,
analyst moves and price-target updates move one leg against the other -- and
that divergence is precisely what this strategy exists to trade. A guard that
halted on every earnings headline would be an off-switch, not a risk control.

Only **structural** events halt trading:

* the pair stops being two independent companies (merger, acquisition,
  take-private, spin-off),
* one leg stops being continuously tradeable (bankruptcy, delisting, an
  indefinite halt),
* the reported fundamentals stop being trustworthy (restatement, auditor
  resignation, an SEC accounting investigation).

Each of those permanently changes the price relationship. Everything else is
recorded and reported, never acted on.

FAIL-OPEN, LOUDLY
-----------------
When the news feed is unreachable the guard reports the failure and lets trading
continue. A veto that halts on its own outage would hand the news vendor an
off-switch for your strategy, and unavailable news is not evidence of a problem.
Set ``news.fail_closed`` to invert this if you would rather stop when blind.

The classifier is deliberately rule-based: deterministic, inspectable, and
testable without a model in the loop. :meth:`NewsGuard.classify` is the single
extension point if you later want an LLM to read the ambiguous cases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import Config, resolve_path
from .risk_manager import Alert, AlertDispatcher, RiskEvent, RiskEventLogger, RiskManager

logger = logging.getLogger(__name__)

EVENT_NEWS_HALT = "news_halt"
EVENT_NEWS_SCAN = "news_scan"

# Severity ordering, least to most serious.
INFO, WARNING, BLOCKING = "info", "warning", "blocking"


class NewsUnavailableError(RuntimeError):
    """Raised when the news feed cannot be read."""


# ---------------------------------------------------------------------------
# classification rules
# ---------------------------------------------------------------------------
#
# Each rule is (category, severity, compiled pattern). Patterns are matched
# case-insensitively against "headline. summary".
#
# Word boundaries matter more than they look. "merger" must not fire on
# "emerges"; "halt" must not fire on "Halton". Every pattern below is anchored
# on \b for that reason.

def _p(*alternatives: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


BLOCKING_RULES: list[tuple[str, re.Pattern[str]]] = [
    # The pair stops being two separate companies.
    # `acquired` and `acquisition` are matched bare, not only in longer phrases
    # like "acquired by". "AAA to be acquired" must fire, and requiring a
    # trailing preposition silently misses it.
    #
    # This over-triggers on minority-stake and bolt-on deals that do not really
    # end the relationship. That trade is deliberate and runs throughout this
    # module: a false positive costs a pause you can clear in one command, a
    # false negative costs money for as long as the spread keeps not reverting.
    ("merger_acquisition", _p(
        r"merger", r"merges? with", r"to merge", r"acquisitions?",
        r"to acquire", r"acquires?", r"acquired", r"takeover", r"take[- ]private",
        r"buyout", r"tender offer", r"all[- ]stock deal", r"all[- ]cash deal",
    )),
    ("spin_off", _p(
        r"spin[- ]?off", r"spins? off", r"split[- ]off", r"carve[- ]out",
        r"separate into two", r"demerger",
    )),
    # One leg stops being continuously tradeable.
    ("distress", _p(
        r"bankrupt(?:cy)?", r"chapter 11", r"chapter 7", r"insolvency",
        r"going concern", r"liquidation", r"receivership", r"defaults? on",
    )),
    ("delisting", _p(
        r"delist(?:ed|ing)?", r"deregistration", r"removed from (?:the )?exchange",
        r"trading suspend(?:ed|ion)", r"suspension of trading",
    )),
    # The reported fundamentals stop being trustworthy.
    ("accounting", _p(
        r"restat(?:e|es|ed|ement)", r"accounting (?:irregularit|fraud|error)\w*",
        r"auditor resign\w*", r"material weakness", r"sec investigation",
        r"sec probe", r"accounting scandal",
    )),
]

WARNING_RULES: list[tuple[str, re.Pattern[str]]] = [
    # Real flow effects on one leg, but the relationship survives them.
    ("index_change", _p(
        r"added to the s&p", r"removed from the s&p", r"index inclusion",
        r"index removal", r"joins? the s&p", r"index rebalanc\w*",
    )),
    ("halt", _p(r"trading halt(?:ed)?", r"volatility halt")),
    ("leadership", _p(r"ceo (?:resign|step|depart)\w*", r"cfo (?:resign|step|depart)\w*")),
    ("regulatory", _p(r"antitrust", r"doj (?:sues|lawsuit|investigation)", r"ftc (?:sues|challenge)")),
]

# Recorded for context, never acted on. This is the ordinary divergence the
# strategy is built to trade -- halting on it would defeat the strategy.
INFO_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("earnings", _p(r"earnings", r"quarterly results", r"eps", r"revenue",
                    r"guidance", r"outlook", r"beats?", r"misses?")),
    ("analyst", _p(r"price target", r"upgrade[sd]?", r"downgrade[sd]?",
                   r"initiated coverage", r"reiterates?")),
    ("dividend", _p(r"dividend", r"buyback", r"share repurchase")),
]


@dataclass
class NewsItem:
    """One headline from the feed."""

    id: str
    headline: str
    summary: str
    symbols: list[str]
    created_at: datetime
    source: str = ""
    url: str = ""

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.summary}".strip()


@dataclass
class Finding:
    """A classified headline."""

    item: NewsItem
    category: str
    severity: str
    matched: str

    @property
    def is_blocking(self) -> bool:
        return self.severity == BLOCKING

    def describe(self) -> str:
        return (
            f"[{self.severity}/{self.category}] "
            f"{', '.join(self.item.symbols) or '?'}: {self.item.headline}"
        )


@dataclass
class ScanResult:
    """Outcome of one scan over the pair's recent news."""

    pair: str
    scanned: int
    findings: list[Finding] = field(default_factory=list)
    lookback_hours: float = 24.0
    error: str = ""
    halted: bool = False

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.is_blocking]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing structural was found (an error is not "ok")."""
        return not self.blocking and not self.error

    def report(self) -> str:
        width = 74
        lines = ["=" * width, f" NEWS GUARD -- {self.pair}", "=" * width]
        lines.append(f" Scanned {self.scanned} item(s) from the last {self.lookback_hours:g}h")

        if self.error:
            lines += ["", f" FEED ERROR: {self.error}"]

        if self.blocking:
            lines += ["", " BLOCKING -- these break the pair's relationship:"]
            lines += [f"   x {f.describe()}" for f in self.blocking]
        if self.warnings:
            lines += ["", " Worth knowing:"]
            lines += [f"   ! {f.describe()}" for f in self.warnings]

        informational = [f for f in self.findings if f.severity == INFO]
        if informational:
            lines += ["", f" Ordinary news ({len(informational)} item(s)) -- not acted on;"]
            lines += ["   earnings and analyst moves are the divergence this strategy trades."]
            lines += [f"   . {f.item.headline[:90]}" for f in informational[:5]]

        lines.append("")
        if self.halted:
            lines.append(" VERDICT: TRADING HALTED. Review, then clear manually.")
        elif self.blocking:
            lines.append(" VERDICT: structural event found (halt not enabled).")
        elif self.error:
            lines.append(" VERDICT: could not check. Trading continues (fail-open).")
        else:
            lines.append(" VERDICT: clear -- no structural break detected.")
        lines.append("=" * width)
        return "\n".join(lines)


class NewsGuard:
    """Scans a pair's recent news for events that end its cointegration.

    Uses Alpaca's news API, which authenticates with the same paper keys the
    trader already needs -- no additional credentials, no new secret to manage.
    """

    def __init__(
        self,
        config: Config,
        risk: RiskManager | None = None,
        alerts: AlertDispatcher | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.news = config.news
        self.risk = risk or RiskManager(config)
        self.alerts = alerts if alerts is not None else AlertDispatcher.from_env()
        self.events = RiskEventLogger(resolve_path(config.risk.risk_log))
        self._client = client

    # -- feed ------------------------------------------------------------

    def _build_client(self) -> Any:
        from alpaca.data.historical.news import NewsClient

        from .config import alpaca_credentials

        api_key, secret_key = alpaca_credentials()
        return NewsClient(api_key=api_key, secret_key=secret_key)

    def fetch(self, lookback_hours: float | None = None) -> list[NewsItem]:
        """Pull recent headlines for both legs.

        Raises:
            NewsUnavailableError: if the feed cannot be read.
        """
        hours = lookback_hours if lookback_hours is not None else self.news.lookback_hours
        start = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            client = self._client or self._build_client()
            from alpaca.data.requests import NewsRequest

            response = client.get_news(
                NewsRequest(
                    symbols=",".join(self.config.pair.tickers),
                    start=start,
                    limit=self.news.max_items,
                    include_content=False,
                    exclude_contentless=False,
                )
            )
        except Exception as exc:
            raise NewsUnavailableError(str(exc)) from exc

        raw = getattr(response, "data", response)
        if isinstance(raw, dict):
            raw = raw.get("news", [])

        items: list[NewsItem] = []
        for entry in raw or []:
            created = getattr(entry, "created_at", None) or datetime.now(timezone.utc)
            items.append(
                NewsItem(
                    id=str(getattr(entry, "id", "")),
                    headline=str(getattr(entry, "headline", "") or ""),
                    summary=str(getattr(entry, "summary", "") or ""),
                    symbols=[str(s).upper() for s in (getattr(entry, "symbols", None) or [])],
                    created_at=created,
                    source=str(getattr(entry, "source", "") or ""),
                    url=str(getattr(entry, "url", "") or ""),
                )
            )
        return items

    # -- classification ---------------------------------------------------

    def classify(self, item: NewsItem) -> Finding | None:
        """Categorise one headline. Override this to plug in a model.

        Rules are checked most-serious first and the first match wins, so a
        headline reading "Q3 earnings released ahead of the merger vote"
        classifies as a merger rather than as earnings. Under-reacting to a
        structural event is the expensive error; over-reacting costs a pause.
        """
        text = item.text
        if not text.strip():
            return None

        for rules, severity in (
            (BLOCKING_RULES, BLOCKING),
            (WARNING_RULES, WARNING),
            (INFO_RULES, INFO),
        ):
            for category, pattern in rules:
                match = pattern.search(text)
                if match:
                    return Finding(item, category, severity, match.group(0))
        return None

    def _relevant(self, item: NewsItem) -> bool:
        """Whether a headline actually concerns one of the pair's legs.

        The feed returns items tagged with many symbols; a market-wrap article
        mentioning forty tickers is not news *about* your pair. Items tagged
        with more than ``max_symbols`` are treated as roundups and skipped.
        """
        tickers = set(self.config.pair.tickers)
        if not item.symbols:
            return False
        if not tickers & set(item.symbols):
            return False
        return len(item.symbols) <= self.news.max_symbols

    # -- scan and enforce -------------------------------------------------

    def scan(self, lookback_hours: float | None = None) -> ScanResult:
        """Fetch and classify. Does not halt -- see :meth:`enforce`."""
        hours = lookback_hours if lookback_hours is not None else self.news.lookback_hours
        result = ScanResult(pair=str(self.config.pair), scanned=0, lookback_hours=hours)

        try:
            items = self.fetch(hours)
        except NewsUnavailableError as exc:
            result.error = str(exc)
            logger.warning("News feed unavailable: %s", exc)
            return result

        relevant = [i for i in items if self._relevant(i)]
        result.scanned = len(relevant)
        for item in relevant:
            finding = self.classify(item)
            if finding:
                result.findings.append(finding)

        logger.info(
            "News scan for %s: %d relevant item(s), %d blocking, %d warning",
            result.pair, result.scanned, len(result.blocking), len(result.warnings),
        )
        return result

    def enforce(self, result: ScanResult) -> ScanResult:
        """Act on a scan: halt on a structural event, alert either way.

        A blocking finding writes the same ``TRADING_HALTED`` flag the circuit
        breaker and kill switch use, so clearing it is the same deliberate
        manual act -- there is one way to resume trading, not three.
        """
        for finding in result.findings:
            self.events.log(
                RiskEvent(
                    timestamp=finding.item.created_at.isoformat(),
                    event=EVENT_NEWS_SCAN,
                    reason=finding.item.headline[:300],
                    severity=finding.severity,
                    pair=result.pair,
                    symbol=",".join(finding.item.symbols[:4]),
                    detail=f"{finding.category} (matched {finding.matched!r}) {finding.item.url}",
                )
            )

        if result.error and self.news.fail_closed:
            reason = f"News feed unavailable and news.fail_closed is set: {result.error}"
            self.risk.halt(reason)
            result.halted = True
            return result

        if not result.blocking:
            return result

        summary = "; ".join(f.describe() for f in result.blocking[:3])
        reason = f"NEWS GUARD: structural event on {result.pair} -- {summary}"

        self.alerts.send(
            Alert(
                severity="critical",
                title=f"News guard: structural event on {result.pair}",
                message=(
                    "A headline indicates the pair's price relationship has structurally "
                    "changed. Cointegration assumes a stable long-run relationship; these "
                    "events end it, and the z-score will keep signalling entries into a "
                    "spread that will not revert.\n\n"
                    + "\n".join(f"- {f.describe()}\n  {f.item.url}" for f in result.blocking)
                ),
                context={
                    "pair": result.pair,
                    "items_scanned": result.scanned,
                    "action": "halted" if self.news.halt_on_blocking else "reported only",
                },
            )
        )

        if self.news.halt_on_blocking:
            self.risk.halt(reason, event=EVENT_NEWS_HALT)
            result.halted = True
        else:
            logger.error("%s (news.halt_on_blocking is false -- not halting.)", reason)
        return result

    def run(self, lookback_hours: float | None = None) -> ScanResult:
        """Scan and enforce in one call. The scheduled entry point."""
        return self.enforce(self.scan(lookback_hours))
