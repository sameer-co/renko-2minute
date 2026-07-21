"""SOL/USDT Renko Forward-Testing Bot
====================================
Exchange  : Binance (public REST + WebSocket — no API key needed for data)
Timeframe : 2-minute candles (resampled from 1-minute)
Box Size  : ATR-14 (recalculated every new completed candle)
Buy Signal: First GREEN brick after a trend reversal (red → green)
Stop-Loss : 1.5 × ATR below entry
Take-Profit: 3 × SL distance above entry  (i.e. 4.5 × ATR)
Alerts    : Telegram
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit only this section
# ─────────────────────────────────────────────────────────────────────────────
TG_TOKEN   = "8392707199:AAHjWHGLoZ3Udm4rS5JlgSaPLez1qZbHMOo"
TG_CHAT_ID = "1950462171"

SYMBOL           = "SOLUSDT"
ATR_PERIOD       = 14
SEED_CANDLES     = 400        # 400 × 1m = 200 × 2m candles for warm-up
ATR_MULTIPLIER   = 1.0
SL_ATR_MULT      = 1.5
TP_SL_MULT       = 3.0
RESAMPLE_MINUTES = 2          # resample 1m → 2m

BINANCE_REST_URL = "https://api.binance.com"
BINANCE_WS_URL   = "wss://stream.binance.com:9443/ws"
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("renko_bot")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════
async def tg_send(session: aiohttp.ClientSession, text: str) -> None:
    url     = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    for attempt in range(2):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    return
                body = await r.text()
                log.warning("TG HTTP %s: %s", r.status, body)
        except Exception as exc:
            log.warning("TG send error (attempt %d): %s", attempt + 1, exc)
        await asyncio.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2-MINUTE RESAMPLER
# ══════════════════════════════════════════════════════════════════════════════
class CandleResampler:
    """
    Accumulates 1-min closed candles and emits a completed N-min candle
    every RESAMPLE_MINUTES bars based on timestamp bucketing.
    """

    def __init__(self, minutes: int = RESAMPLE_MINUTES):
        self.minutes = minutes
        self._bucket: Optional[int]   = None
        self._open:   Optional[float] = None
        self._high:   Optional[float] = None
        self._low:    Optional[float] = None
        self._close:  Optional[float] = None
        self._ts:     Optional[datetime] = None
        self._count:  int = 0

    def _bucket_id(self, ts: datetime) -> int:
        epoch_min = int(ts.timestamp()) // 60
        return epoch_min // self.minutes

    def feed(self, candle: dict) -> Optional[dict]:
        """
        Feed one closed 1-min candle.
        Returns a completed N-min candle dict, or None if not yet complete.
        """
        bucket = self._bucket_id(candle["ts"])

        if self._bucket is None or bucket != self._bucket:
            self._bucket = bucket
            self._open   = candle["open"]
            self._high   = candle["high"]
            self._low    = candle["low"]
            self._close  = candle["close"]
            self._ts     = candle["ts"]
            self._count  = 1
        else:
            self._high  = max(self._high,  candle["high"])
            self._low   = min(self._low,   candle["low"])
            self._close = candle["close"]
            self._count += 1

        if self._count >= self.minutes:
            result = {
                "open":  self._open,
                "high":  self._high,
                "low":   self._low,
                "close": self._close,
                "ts":    self._ts,
            }
            self._bucket = None
            self._count  = 0
            return result

        return None


def resample_candles(candles_1m: list, minutes: int = RESAMPLE_MINUTES) -> list:
    """Batch-resample a list of 1-min candles into N-min candles."""
    resampler = CandleResampler(minutes)
    result    = []
    for c in candles_1m:
        out = resampler.feed(c)
        if out:
            result.append(out)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ATR CALCULATOR  (Wilder / RMA smoothing)
# ══════════════════════════════════════════════════════════════════════════════
class ATR:
    def __init__(self, period: int = 14):
        self.period       = period
        self._prev_close: Optional[float] = None
        self._rma:        Optional[float] = None
        self._count       = 0
        self._warm        = False
        self._sum_tr      = 0.0

    @property
    def value(self) -> Optional[float]:
        return self._rma if self._warm else None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low  - self._prev_close))

        self._prev_close = close
        self._count += 1

        if not self._warm:
            self._sum_tr += tr
            if self._count >= self.period:
                self._rma  = self._sum_tr / self.period
                self._warm = True
        else:
            alpha     = 1.0 / self.period
            self._rma = self._rma * (1 - alpha) + tr * alpha

        return self._rma if self._warm else None


# ══════════════════════════════════════════════════════════════════════════════
# RENKO ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RenkoBrick:
    direction:   int
    open_price:  float
    close_price: float
    formed_at:   datetime


@dataclass
class RenkoState:
    bricks:       list  = field(default_factory=list)
    last_close:   Optional[float]    = None
    current_dir:  Optional[int]      = None
    box_size:     Optional[float]    = None
    pending_open: Optional[float]    = None

    def set_box(self, box: float) -> None:
        self.box_size = round(box, 4)

    def _snap(self, price: float, box: float) -> float:
        return math.floor(price / box) * box

    def seed_price(self, price: float) -> None:
        if self.last_close is None and self.box_size:
            self.last_close   = self._snap(price, self.box_size)
            self.pending_open = self.last_close

    def feed(self, price: float, ts: datetime) -> list:
        if self.box_size is None or self.last_close is None:
            return []

        box    = self.box_size
        new_bx = []

        while True:
            up_target   = self.last_close + box
            down_target = self.last_close - box

            if price >= up_target:
                open_p = self.last_close
                brick  = RenkoBrick(+1, open_p, open_p + box, ts)
                new_bx.append(brick)
                self.bricks.append(brick)
                self.last_close  = open_p + box
                self.current_dir = +1

            elif price <= down_target:
                open_p = self.last_close
                brick  = RenkoBrick(-1, open_p, open_p - box, ts)
                new_bx.append(brick)
                self.bricks.append(brick)
                self.last_close  = open_p - box
                self.current_dir = -1
            else:
                break

        return new_bx


# ══════════════════════════════════════════════════════════════════════════════
# TRADE TRACKER
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    entry_price:  float
    sl:           float
    tp:           float
    atr_at_entry: float
    entered_at:   datetime
    status:       str = "OPEN"
    exit_price:   Optional[float]    = None
    exited_at:    Optional[datetime] = None

    @property
    def pnl_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price * 100


class TradeManager:
    def __init__(self):
        self.open_trade: Optional[Trade] = None
        self.history:    list[Trade]     = []

    def has_open(self) -> bool:
        return self.open_trade is not None

    def open(self, price: float, sl: float, tp: float, atr: float, ts: datetime) -> Trade:
        t = Trade(price, sl, tp, atr, ts)
        self.open_trade = t
        return t

    def check(self, price: float, ts: datetime) -> Optional[Trade]:
        t = self.open_trade
        if t is None:
            return None
        if price <= t.sl:
            t.status     = "HIT_SL"
            t.exit_price = t.sl
            t.exited_at  = ts
            self.history.append(t)
            self.open_trade = None
            return t
        if price >= t.tp:
            t.status     = "HIT_TP"
            t.exit_price = t.tp
            t.exited_at  = ts
            self.history.append(t)
            self.open_trade = None
            return t
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BINANCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
async def fetch_klines(session: aiohttp.ClientSession, limit: int = 400) -> list:
    """Fetch recent 1-min klines from Binance REST."""
    url    = f"{BINANCE_REST_URL}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": "1m", "limit": limit}
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
        r.raise_for_status()
        raw = await r.json()
    candles = []
    for k in raw:
        candles.append({
            "open":  float(k[1]),
            "high":  float(k[2]),
            "low":   float(k[3]),
            "close": float(k[4]),
            "ts":    datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
        })
    return candles


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BOT
# ══════════════════════════════════════════════════════════════════════════════
class RenkoBot:
    def __init__(self):
        self.atr       = ATR(ATR_PERIOD)
        self.renko     = RenkoState()
        self.trades    = TradeManager()
        self.resampler = CandleResampler(RESAMPLE_MINUTES)
        self.session:          Optional[aiohttp.ClientSession] = None
        self._last_candle_ts:  Optional[int] = None
        self._prev_brick_dir:  Optional[int] = None

    # ── seeding ───────────────────────────────────────────────────────────────
    async def seed(self) -> None:
        log.info("Fetching %d × 1-min seed candles for %s …", SEED_CANDLES, SYMBOL)
        candles_1m = await fetch_klines(self.session, limit=SEED_CANDLES)

        # Exclude the still-open last 1m candle before resampling
        candles_2m = resample_candles(candles_1m[:-1])
        log.info("Resampled into %d × %d-min candles", len(candles_2m), RESAMPLE_MINUTES)

        for c in candles_2m:
            atr_val = self.atr.update(c["high"], c["low"], c["close"])
            if atr_val and self.renko.box_size is None:
                self.renko.set_box(atr_val * ATR_MULTIPLIER)
                self.renko.seed_price(c["close"])
                log.info("Box size initialised: %.4f USDT  (ATR=%.4f)",
                         self.renko.box_size, atr_val)

            if self.renko.box_size:
                self.renko.feed(c["close"], c["ts"])

        self._last_candle_ts = int(candles_1m[-1]["ts"].timestamp() * 1000)
        log.info("Seed complete. Renko bricks built: %d", len(self.renko.bricks))
        if self.renko.bricks:
            last = self.renko.bricks[-1]
            log.info("Last brick: %s @ %.4f → %.4f",
                     "GREEN" if last.direction == 1 else "RED",
                     last.open_price, last.close_price)

    # ── WebSocket listener ────────────────────────────────────────────────────
    async def _ws_listen(self) -> None:
        stream = f"{SYMBOL.lower()}@kline_1m"
        url    = f"{BINANCE_WS_URL}/{stream}"
        log.info("Connecting to Binance WebSocket: %s", url)

        while True:
            try:
                async with self.session.ws_connect(
                    url,
                    heartbeat=20,
                    receive_timeout=90,
                ) as ws:
                    await tg_send(
                        self.session,
                        f"🤖 <b>Renko Bot LIVE</b>\n"
                        f"Symbol    : <code>{SYMBOL}</code>\n"
                        f"Timeframe : <code>{RESAMPLE_MINUTES}-min (resampled from 1m)</code>\n"
                        f"Box (ATR-14): <code>{self.renko.box_size:.4f} USDT</code>\n"
                        f"Signal    : 1st green brick after red→green reversal\n"
                        f"SL: 1.5×ATR | TP: 3×SL distance\n"
                        f"Started   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                    )
                    log.info("WebSocket connected ✓")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_msg(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            log.warning("WS closed/error — reconnecting …")
                            break

            except Exception as exc:
                log.error("WS error: %s — reconnecting in 5 s …", exc)
                await asyncio.sleep(5)

    # ── Handle each incoming 1-min WS message ────────────────────────────────
    async def _handle_ws_msg(self, data: dict) -> None:
        k         = data.get("k", {})
        is_closed = k.get("x", False)

        price = float(k["c"])
        ts_ms = int(k["t"])
        ts    = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        # Always check open trade on every tick
        if self.trades.has_open():
            closed = self.trades.check(price, ts)
            if closed:
                await self._on_trade_closed(closed)

        # Only process fully closed 1-min candles for resampling
        if not is_closed:
            return
        if ts_ms == self._last_candle_ts:
            return   # duplicate guard
        self._last_candle_ts = ts_ms

        candle_1m = {
            "open":  float(k["o"]),
            "high":  float(k["h"]),
            "low":   float(k["l"]),
            "close": float(k["c"]),
            "ts":    ts,
        }

        # Feed into resampler — only proceed if a 2-min candle is complete
        candle_2m = self.resampler.feed(candle_1m)
        if candle_2m is None:
            return   # 2-min candle not yet complete

        high  = candle_2m["high"]
        low   = candle_2m["low"]
        close = candle_2m["close"]
        ts_2m = candle_2m["ts"]

        log.info("2m candle closed  O=%.4f H=%.4f L=%.4f C=%.4f  @ %s UTC",
                 candle_2m["open"], high, low, close,
                 ts_2m.strftime("%H:%M:%S"))

        # Update ATR with the completed 2-min candle
        atr_val = self.atr.update(high, low, close)
        if atr_val is None:
            return   # still warming up

        # Adaptive box update (only if change > 5%)
        new_box = round(atr_val * ATR_MULTIPLIER, 4)
        if self.renko.box_size is None:
            self.renko.set_box(new_box)
            self.renko.seed_price(close)
        else:
            if abs(new_box - self.renko.box_size) / self.renko.box_size > 0.05:
                log.info("Box updated: %.4f → %.4f", self.renko.box_size, new_box)
                self.renko.set_box(new_box)

        # Save direction BEFORE feeding new price
        dir_before = self.renko.current_dir

        # Feed close of 2-min candle into Renko
        new_bricks = self.renko.feed(close, ts_2m)

        for brick in new_bricks:
            log.info(
                "%s brick @ %.4f→%.4f  |  ATR=%.4f  |  Box=%.4f  |  %s UTC",
                "🟢 GREEN" if brick.direction == 1 else "🔴 RED",
                brick.open_price, brick.close_price,
                atr_val, self.renko.box_size,
                ts_2m.strftime("%H:%M:%S"),
            )

            # Signal: first green brick after red→green reversal
            if (
                brick.direction == +1
                and dir_before   == -1
                and not self.trades.has_open()
            ):
                await self._on_buy_signal(brick, atr_val, ts_2m)

            dir_before = brick.direction

    # ── Trade entry ───────────────────────────────────────────────────────────
    async def _on_buy_signal(self, brick: RenkoBrick, atr: float, ts: datetime) -> None:
        entry   = brick.close_price
        sl      = round(entry - SL_ATR_MULT * atr, 4)
        sl_dist = entry - sl
        tp      = round(entry + TP_SL_MULT * sl_dist, 4)

        self.trades.open(entry, sl, tp, atr, ts)
        log.info("BUY SIGNAL  entry=%.4f  SL=%.4f  TP=%.4f  ATR=%.4f",
                 entry, sl, tp, atr)

        msg = (
            f"🟢 <b>BUY SIGNAL — {SYMBOL}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 Entry  : <code>{entry:.4f} USDT</code>\n"
            f"🛑 SL     : <code>{sl:.4f} USDT</code>  (−{SL_ATR_MULT}×ATR)\n"
            f"🎯 TP     : <code>{tp:.4f} USDT</code>  (+{TP_SL_MULT}×SL dist)\n"
            f"📊 ATR-14 : <code>{atr:.4f} USDT</code>\n"
            f"📦 Box    : <code>{self.renko.box_size:.4f} USDT</code>\n"
            f"🕐 Time   : <code>{ts.strftime('%Y-%m-%d %H:%M:%S UTC')}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<i>Signal: Trend reversal — first green brick after red run</i>"
        )
        await tg_send(self.session, msg)

    # ── Trade exit ────────────────────────────────────────────────────────────
    async def _on_trade_closed(self, trade: Trade) -> None:
        emoji  = "✅" if trade.status == "HIT_TP" else "❌"
        result = "TAKE PROFIT" if trade.status == "HIT_TP" else "STOP LOSS"
        pnl    = trade.pnl_pct

        log.info("TRADE CLOSED  %s  entry=%.4f  exit=%.4f  PnL=%.2f%%",
                 trade.status, trade.entry_price, trade.exit_price, pnl)

        msg = (
            f"{emoji} <b>{result} HIT — {SYMBOL}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 Entry  : <code>{trade.entry_price:.4f} USDT</code>\n"
            f"🚪 Exit   : <code>{trade.exit_price:.4f} USDT</code>\n"
            f"💰 PnL    : <code>{pnl:+.2f}%</code>\n"
            f"🕐 In     : <code>{trade.entered_at.strftime('%H:%M:%S UTC')}</code>\n"
            f"🕐 Out    : <code>{trade.exited_at.strftime('%H:%M:%S UTC')}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<b>Total trades: {len(self.trades.history)}</b>\n"
            f"{self._summary()}"
        )
        await tg_send(self.session, msg)

    def _summary(self) -> str:
        if not self.trades.history:
            return ""
        wins    = [t for t in self.trades.history if t.status == "HIT_TP"]
        losses  = [t for t in self.trades.history if t.status == "HIT_SL"]
        total   = len(self.trades.history)
        win_r   = len(wins) / total * 100 if total else 0
        avg_pnl = sum(t.pnl_pct for t in self.trades.history) / total if total else 0
        return (
            f"📈 Wins: {len(wins)}  |  📉 Losses: {len(losses)}\n"
            f"🏆 Win Rate: {win_r:.1f}%  |  Avg PnL: {avg_pnl:+.2f}%"
        )

    # ── Entry point ───────────────────────────────────────────────────────────
    async def run(self) -> None:
        connector = aiohttp.TCPConnector(limit=10, ssl=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            await self.seed()
            await self._ws_listen()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    bot = RenkoBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
