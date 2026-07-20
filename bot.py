import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TG_TOKEN   = "8392707199:AAHjWHGLoZ3Udm4rS5JlgSaPLez1qZbHMOo"
TG_CHAT_ID = "1950462171"

SYMBOL          = "SOLUSDT"
TIMEFRAME_MINS  = 2           # Synthesized timeframe block (e.g. 2 minutes)
ATR_PERIOD      = 14          
SEED_CANDLES    = 200         # Fetch 200 × 2 = 400 1m candles for seeding
ATR_MULTIPLIER  = 1.0         
SL_ATR_MULT     = 1.5         
TP_SL_MULT      = 3.0         

BINANCE_REST_URL = "https://api.binance.com"
BINANCE_WS_URL   = "wss://stream.binance.com:9443/ws"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("renko")

# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINES
# ─────────────────────────────────────────────────────────────────────────────

class ATR:
    def __init__(self, period: int):
        self.period = period
        self.tr_history = deque(maxlen=period)
        self.prev_close: Optional[float] = None
        self.current_atr: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        
        self.tr_history.append(tr)
        self.prev_close = close

        if len(self.tr_history) == self.period:
            self.current_atr = sum(self.tr_history) / self.period
            return self.current_atr
        return None

class RenkoState:
    def __init__(self):
        self.box_size: Optional[float] = None
        self.bricks: list[float] = []
        self.direction: int = 0
        
        # Temporary states for unconfirmed bricks
        self.current_price: float = 0.0
        self.potential_bricks: int = 0
        self.potential_direction: int = 0
        self.potential_top: float = 0.0
        self.potential_bottom: float = 0.0

    def set_box(self, size: float):
        self.box_size = size

    def seed_price(self, price: float):
        if not self.bricks:
            self.bricks.append(price)

    def feed(self, price: float, ts: datetime) -> list[dict]:
        if not self.box_size or not self.bricks:
            return []

        new_bricks = []
        last_brick = self.bricks[-1]
        
        diff = price - last_brick
        if abs(diff) >= self.box_size:
            brick_count = int(abs(diff) / self.box_size)
            step_dir = 1 if diff > 0 else -1

            if self.direction != 0 and step_dir != self.direction:
                if brick_count >= 2:
                    brick_count -= 1
                    last_brick += (self.box_size * step_dir)
                    self.bricks.append(last_brick)
                    new_bricks.append({"price": last_brick, "dir": step_dir, "ts": ts})
                    self.direction = step_dir
                else:
                    brick_count = 0

            for _ in range(brick_count):
                last_brick += (self.box_size * step_dir)
                self.bricks.append(last_brick)
                new_bricks.append({"price": last_brick, "dir": step_dir, "ts": ts})
                self.direction = step_dir

        return new_bricks

    def update_live(self, price: float) -> None:
        if not self.box_size or not self.bricks:
            return
            
        self.current_price = price
        last_brick = self.bricks[-1]
        diff = price - last_brick
        
        self.potential_bricks = 0
        self.potential_direction = 0
        
        if abs(diff) >= self.box_size:
            p_count = int(abs(diff) / self.box_size)
            p_dir = 1 if diff > 0 else -1
            
            if self.direction != 0 and p_dir != self.direction:
                if p_count >= 2:
                    p_count -= 1
                    self.potential_direction = p_dir
                    self.potential_bricks = p_count
                else:
                    p_count = 0
            else:
                self.potential_direction = p_dir
                self.potential_bricks = p_count
                
            if self.potential_bricks > 0:
                p_offset = self.potential_bricks * self.box_size * self.potential_direction
                if self.direction != 0 and p_dir != self.direction:
                    p_offset += (self.box_size * p_dir)
                    
                final_p_brick = last_brick + p_offset
                if self.potential_direction == 1:
                    self.potential_top = final_p_brick
                    self.potential_bottom = final_p_brick - self.box_size
                else:
                    self.potential_top = final_p_brick + self.box_size
                    self.potential_bottom = final_p_brick

@dataclass
class Trade:
    id: int
    direction: str
    entry_price: float
    sl: float
    tp: float
    status: str = "OPEN"
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    open_ts: float = field(default_factory=time.time)

class TradeManager:
    def __init__(self):
        self.active_trade: Optional[Trade] = None
        self.trade_counter = 0

    def open_trade(self, direction: str, price: float, atr: float) -> Trade:
        self.trade_counter += 1
        sl_dist = atr * SL_ATR_MULT
        tp_dist = sl_dist * TP_SL_MULT
        
        if direction == "LONG":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
            
        self.active_trade = Trade(self.trade_counter, direction, price, sl, tp)
        return self.active_trade

    def check_exit(self, current_price: float) -> Optional[Trade]:
        t = self.active_trade
        if not t: return None
        
        closed = False
        if t.direction == "LONG":
            if current_price <= t.sl or current_price >= t.tp:
                closed = True
        else:
            if current_price >= t.sl or current_price <= t.tp:
                closed = True
                
        if closed:
            t.status = "CLOSED"
            t.exit_price = current_price
            if t.direction == "LONG":
                t.pnl = current_price - t.entry_price
            else:
                t.pnl = t.entry_price - current_price
            self.active_trade = None
            return t
        return None

# ─────────────────────────────────────────────────────────────────────────────
# BOT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def tg_send(session: aiohttp.ClientSession, text: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                log.warning("Telegram alert failed: %s", await resp.text())
    except Exception as e:
        log.error("Telegram request error: %s", e)

async def fetch_klines(session: aiohttp.ClientSession, limit: int = 200) -> list[dict]:
    url = f"{BINANCE_REST_URL}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": "1m", "limit": limit}
    async with session.get(url, params=params) as resp:
        raw = await resp.json()
        
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

class RenkoBot:
    def __init__(self):
        self.atr     = ATR(ATR_PERIOD)
        self.renko   = RenkoState()
        self.trades  = TradeManager()
        self.session: Optional[aiohttp.ClientSession] = None
        
        self._last_candle_ts: Optional[int] = None  
        self._prev_brick_dir: Optional[int] = None  
        
        # Trackers for the synthetic 2m timeframe
        self._cur_synth_high = -math.inf
        self._cur_synth_low  = math.inf

    async def seed(self) -> None:
        fetch_limit = SEED_CANDLES * TIMEFRAME_MINS
        log.info("Fetching %d 1m candles to build %d synthetic %dm candles...", fetch_limit, SEED_CANDLES, TIMEFRAME_MINS)
        candles_1m = await fetch_klines(self.session, limit=fetch_limit)
        
        candles_synth = []
        c_high = -math.inf
        c_low = math.inf

        for i, c in enumerate(candles_1m):
            c_high = max(c_high, c["high"])
            c_low = min(c_low, c["low"])
            
            # End of a synthetic block (e.g., minute 1, 3, 5 for a 2m block)
            if c["ts"].minute % TIMEFRAME_MINS == (TIMEFRAME_MINS - 1):
                is_last_candle = (i == len(candles_1m) - 1)
                
                # Only process fully completed blocks into historical bricks
                if not is_last_candle:
                    candles_synth.append({
                        "high": c_high,
                        "low": c_low,
                        "close": c["close"],
                        "ts": c["ts"]
                    })
                    # Reset tracker for the next block
                    c_high = -math.inf
                    c_low = math.inf
                    
        # Carry over the remaining high/low to the live websocket trackers
        self._cur_synth_high = c_high
        self._cur_synth_low  = c_low

        for c in candles_synth:
            atr_val = self.atr.update(c["high"], c["low"], c["close"])
            if atr_val and self.renko.box_size is None:
                self.renko.set_box(atr_val * ATR_MULTIPLIER)
                self.renko.seed_price(c["close"])
                log.info("Box size initialised: %.4f USDT  (ATR=%.4f)", self.renko.box_size, atr_val)

            if self.renko.box_size:
                self.renko.feed(c["close"], c["ts"])

        # Set the previous brick direction for signals
        self._prev_brick_dir = self.renko.direction

    async def _handle_ws_msg(self, data: dict) -> None:
        k = data.get("k", {})
        if not k: return

        is_closed = k.get("x", False)
        ts_ms     = int(k["t"])
        ts        = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        
        high  = float(k["h"])
        low   = float(k["l"])
        close = float(k["c"])

        # Constantly update live Renko and Check active trades
        self.renko.update_live(close)
        
        closed_trade = self.trades.check_exit(close)
        if closed_trade:
            icon = "🟢" if closed_trade.pnl > 0 else "🔴"
            msg = (f"{icon} <b>TRADE CLOSED</b> {icon}\n"
                   f"Dir: {closed_trade.direction}\n"
                   f"Entry: {closed_trade.entry_price:.4f}\n"
                   f"Exit:  {closed_trade.exit_price:.4f}\n"
                   f"PnL:   <b>{closed_trade.pnl:.4f} USDT</b>")
            log.info(f"Trade Closed: {closed_trade.direction} PnL={closed_trade.pnl:.4f}")
            await tg_send(self.session, msg)

        # Update synthetic timeframe boundaries
        self._cur_synth_high = max(self._cur_synth_high, high)
        self._cur_synth_low  = min(self._cur_synth_low, low)

        # ── Only process a CLOSED 1m candle for ATR / Renko ────────────────
        if not is_closed:
            return
            
        if ts_ms == self._last_candle_ts:
            return   # duplicate
        self._last_candle_ts = ts_ms

        # Check if this 1m candle finishes our synthetic timeframe block
        if ts.minute % TIMEFRAME_MINS != (TIMEFRAME_MINS - 1):
            return
            
        # ── SYNTHETIC CANDLE IS FINISHED ──
        final_high = self._cur_synth_high
        final_low  = self._cur_synth_low
        
        # Reset synth trackers for next block
        self._cur_synth_high = -math.inf
        self._cur_synth_low  = math.inf

        atr_val = self.atr.update(final_high, final_low, close)
        if not atr_val or not self.renko.box_size:
            return

        new_bricks = self.renko.feed(close, ts)
        for b in new_bricks:
            bdir = b["dir"]
            
            log.info(f"🧱 New Brick: {b['price']:.4f} | Dir: {bdir}")

            # Trading Logic
            if self._prev_brick_dir is not None and bdir != self._prev_brick_dir:
                if self.trades.active_trade:
                    # Close opposite trade
                    ct = self.trades.active_trade
                    ct.status = "CLOSED"
                    ct.exit_price = close
                    ct.pnl = (close - ct.entry_price) if ct.direction == "LONG" else (ct.entry_price - close)
                    self.trades.active_trade = None
                    log.info(f"Reversal closed trade {ct.id}, PnL={ct.pnl:.4f}")
                
                # Open New Trade
                t_dir = "LONG" if bdir == 1 else "SHORT"
                t = self.trades.open_trade(t_dir, close, atr_val)
                msg = (f"⚡ <b>NEW {t_dir}</b> ⚡\n"
                       f"Symbol: {SYMBOL}\n"
                       f"Entry: <code>{t.entry_price:.4f}</code>\n"
                       f"SL: <code>{t.sl:.4f}</code>\n"
                       f"TP: <code>{t.tp:.4f}</code>\n\n"
                       f"Box Size: <code>{self.renko.box_size:.4f}</code>")
                log.info(f"Opened {t_dir} at {t.entry_price:.4f} (SL:{t.sl:.4f} TP:{t.tp:.4f})")
                asyncio.create_task(tg_send(self.session, msg))

            self._prev_brick_dir = bdir

    async def _ws_listen(self) -> None:
        stream = f"{SYMBOL.lower()}@kline_1m"
        url = f"{BINANCE_WS_URL}/{stream}"
        
        log.info(f"Connecting to Binance WS: {url}")
        
        async with self.session.ws_connect(url) as ws:
            log.info("WebSocket connected. Bot is live.")
            
            startup_msg = (f"🤖 <b>Bot Started</b> 🤖\n"
                           f"Pair: {SYMBOL}\n"
                           f"Timeframe: <code>Synthetic {TIMEFRAME_MINS}m</code>\n"
                           f"Box (ATR-14): <code>{self.renko.box_size:.4f} USDT</code>")
            await tg_send(self.session, startup_msg)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_ws_msg(data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    log.warning("WebSocket disconnected.")
                    break

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.seed()
            
            while True:
                try:
                    await self._ws_listen()
                except Exception as e:
                    log.error(f"WS error: {e}")
                log.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = RenkoBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
