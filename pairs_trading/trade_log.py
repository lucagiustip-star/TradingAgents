"""CSV trade logging, shared by the backtester and the paper-trading client.

Every fill -- simulated or submitted to Alpaca -- lands in the same file with the
same schema, so a paper-trading session can be diffed against the backtest that
justified it. The required columns are, in order::

    timestamp, action, ticker, quantity, price, zscore

followed by context columns (mode, pair, spread, hedge ratio, reason, order id)
that make a post-mortem possible without re-deriving state.

The writer appends and is crash-safe in the only sense that matters here: each
row is flushed as it is written, so a session killed mid-run still leaves a
complete record of the fills that happened before the interruption.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """One leg of one fill.

    A pairs trade always writes two rows -- one per leg -- sharing a
    ``group_id``, because the two legs are a single economic decision and
    reconstructing that from timestamps alone is guesswork.

    Attributes:
        timestamp: ISO-8601. UTC for live fills, the bar's date for backtests.
        action: ``BUY`` or ``SELL``.
        ticker: The symbol traded.
        quantity: Shares. Always positive; direction is carried by ``action``.
        price: Fill price, inclusive of modelled slippage in a backtest.
        zscore: The spread z-score that triggered this trade -- the whole point
            of the log, since it lets you check that live entries fired at the
            same thresholds the backtest assumed.
        mode: ``backtest`` or ``paper``.
        pair: e.g. ``KO/PEP``.
        side: ``entry`` or ``exit``.
        reason: Strategy reason code (``entry_long_spread``, ``exit_stop_loss``, ...).
        spread: The spread value at signal time.
        hedge_ratio: Beta used for sizing this trade.
        notional: ``quantity * price``.
        commission: Modelled or actual commission for this leg.
        group_id: Shared identifier for the two legs of one pairs trade.
        order_id: Broker order id, empty for backtests.
    """

    timestamp: str
    action: str
    ticker: str
    quantity: float
    price: float
    zscore: float
    mode: str = "backtest"
    pair: str = ""
    side: str = ""
    reason: str = ""
    spread: float = float("nan")
    hedge_ratio: float = float("nan")
    notional: float = float("nan")
    commission: float = 0.0
    group_id: str = ""
    order_id: str = ""

    def __post_init__(self) -> None:
        if self.action not in ("BUY", "SELL"):
            raise ValueError(f"action must be BUY or SELL, got {self.action!r}")
        if self.quantity < 0:
            raise ValueError(
                f"quantity must be non-negative (direction lives in `action`), got {self.quantity}"
            )
        if pd.isna(self.notional):
            self.notional = round(self.quantity * self.price, 4)


CSV_COLUMNS = [f.name for f in fields(TradeRecord)]


class TradeLogger:
    """Appends :class:`TradeRecord` rows to a CSV file.

    The header is written once, when the file is created. Reopening an existing
    log appends to it, so a day's paper trading accumulates across runs rather
    than overwriting yesterday's record.
    """

    def __init__(self, path: str | Path, mode: str = "backtest", echo: bool = False) -> None:
        """
        Args:
            path: Destination CSV. Parent directories are created as needed.
            mode: Default ``mode`` stamped on records that do not set their own.
            echo: Also log each row at INFO level, useful when paper trading
                interactively.
        """
        self.path = Path(path)
        self.mode = mode
        self.echo = echo
        self._count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
            logger.debug("Created trade log %s", self.path)

    def log(self, record: TradeRecord) -> None:
        """Append one leg to the CSV and flush it to disk."""
        if not record.mode:
            record.mode = self.mode
        row = asdict(record)
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
        self._count += 1
        if self.echo:
            logger.info(
                "%s %s %.4f @ %.4f (z=%+.2f, %s)",
                record.action, record.ticker, record.quantity, record.price,
                record.zscore, record.reason or record.side,
            )

    def log_many(self, records: list[TradeRecord]) -> None:
        """Append several legs -- typically the two legs of one pairs trade."""
        for record in records:
            self.log(record)

    @property
    def count(self) -> int:
        """Number of legs written by this logger instance."""
        return self._count

    def read(self) -> pd.DataFrame:
        """Read the log back as a DataFrame, with ``timestamp`` parsed."""
        if not self.path.exists():
            return pd.DataFrame(columns=CSV_COLUMNS)
        frame = pd.read_csv(self.path)
        if "timestamp" in frame.columns and not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", utc=False)
        return frame


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (used for live fills)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_group_id(pair: str, timestamp: str) -> str:
    """Build the identifier shared by the two legs of one pairs trade."""
    stamp = timestamp.replace(":", "").replace("-", "").replace("+", "")
    return f"{pair.replace('/', '')}-{stamp}"
