"""
Triagem barata de premissas de entrada, em pandas puro.

Responde uma pergunta só: "depois deste sinal, o preço sobe?"
Sem stop, sem parcial, sem trailing, sem proteção -- essas coisas só importam DEPOIS
que a entrada provou ter sinal. Entrada no open do candle seguinte, saída no open
H candles depois, líquido de taxa.

Regras de leitura:
  - A MEDIANA importa mais que a média: média positiva com mediana negativa = você
    depende de poucas caudas gordas, o que não sobrevive fora da amostra.
  - Compare sempre com a linha `baseline` (todos os candles). Um sinal que rende o
    mesmo que "comprar em qualquer momento" não é um sinal.
  - Só interessa o que se mantém nos QUATRO períodos, principalmente oos e fwd.

Uso: python user_data/lab/probe_entry.py [--horizons 6,12,24,48]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.strategy import merge_informative_pair


DATA = Path(__file__).resolve().parents[1] / "data" / "binance"
PAIRS = ["BTC", "ETH", "SOL", "XRP", "BNB", "LINK", "AVAX", "DOGE", "SUI", "ADA"]
PERIODS = {
    "train": ("2022-01-01", "2024-01-01"),
    "valid": ("2024-01-01", "2025-01-01"),
    "oos": ("2025-01-01", "2026-01-01"),
    "fwd": ("2026-01-01", "2027-01-01"),
}
FEE = 0.002  # ida e volta na Binance spot


def load(pair: str, tf: str) -> pd.DataFrame:
    df = pd.read_feather(DATA / f"{pair}_USDT-{tf}.feather")
    return df.sort_values("date").reset_index(drop=True)


def btc_regime() -> pd.DataFrame:
    df = load("BTC", "4h")
    df["btc_ema50"] = ta.EMA(df, timeperiod=50)
    df["btc_ema200"] = ta.EMA(df, timeperiod=200)
    df["btc_adx"] = ta.ADX(df, timeperiod=14)
    df["btc_atrp"] = ta.ATR(df, timeperiod=14) / df["close"]
    df["btc_bull"] = (
        (df["btc_ema50"] > df["btc_ema200"]) & (df["btc_adx"] > 20) & (df["btc_atrp"] < 0.07)
    ).astype(int)
    return df[["date", "btc_bull"]]


def build(pair: str, btc: pd.DataFrame) -> pd.DataFrame:
    df = load(pair, "30m")
    inf = load(pair, "4h")
    inf["ema20"] = ta.EMA(inf, timeperiod=20)
    inf["ema50"] = ta.EMA(inf, timeperiod=50)
    inf["ema200"] = ta.EMA(inf, timeperiod=200)
    inf["up"] = (inf["ema20"] > inf["ema50"]).astype(int)
    inf["above200"] = (inf["close"] > inf["ema200"]).astype(int)
    # merge_informative_pair desloca o candle 4h em 1 -> sem lookahead.
    df = merge_informative_pair(df, inf[["date", "up", "above200"]], "30m", "4h", ffill=True)
    df = merge_informative_pair(df, btc, "30m", "4h", ffill=True)

    df["sma20"] = ta.SMA(df, timeperiod=20)
    df["sma50"] = ta.SMA(df, timeperiod=50)
    df["sma200"] = ta.SMA(df, timeperiod=200)
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["rsi2"] = ta.RSI(df, timeperiod=2)
    df["rsi14"] = ta.RSI(df, timeperiod=14)
    df["z"] = (df["close"] - df["sma20"]) / df["atr"]
    df["bb_low"] = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)["lowerband"]
    df["down3"] = (
        (df["close"] < df["close"].shift(1))
        & (df["close"].shift(1) < df["close"].shift(2))
        & (df["close"].shift(2) < df["close"].shift(3))
    )
    df["hh20"] = df["high"].rolling(20).max().shift(1)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    df["pair"] = pair
    return df


# --------------------------------------------------------------- premissas testadas
# up_4h / above200_4h / btc_bull_4h vêm do timeframe superior (já deslocados).
SIGNALS = {
    "baseline (todos os candles)": lambda d: pd.Series(True, index=d.index),
    # --- família mean-reversion, filtro de tendência crescente
    "MR rsi2<10": lambda d: d.rsi2 < 10,
    "MR rsi2<10 + close>sma200": lambda d: (d.rsi2 < 10) & (d.close > d.sma200),
    "MR rsi2<5  + close>sma200": lambda d: (d.rsi2 < 5) & (d.close > d.sma200),
    "MR rsi2<10 + 4h up": lambda d: (d.rsi2 < 10) & (d.up_4h == 1),
    "MR rsi2<10 + 4h>ema200": lambda d: (d.rsi2 < 10) & (d.above200_4h == 1),
    "MR rsi2<10 + 4h>ema200 + btc": lambda d: (d.rsi2 < 10) & (d.above200_4h == 1) & (d.btc_bull_4h == 1),
    # --- família z-score (distância da média em ATRs)
    "MR z<-1.5 + close>sma200": lambda d: (d.z < -1.5) & (d.close > d.sma200),
    "MR z<-2.5 + close>sma200": lambda d: (d.z < -2.5) & (d.close > d.sma200),
    "MR z<-2.5 + 4h>ema200": lambda d: (d.z < -2.5) & (d.above200_4h == 1),
    "MR z<-3.5": lambda d: d.z < -3.5,
    # --- bandas e sequência
    "MR close<BBlow + close>sma200": lambda d: (d.close < d.bb_low) & (d.close > d.sma200),
    "MR 3 quedas + close>sma200": lambda d: d.down3 & (d.close > d.sma200),
    "MR 3 quedas + rsi14<35 + 4h up": lambda d: d.down3 & (d.rsi14 < 35) & (d.up_4h == 1),
    # --- controle: o breakout reprovado no round 1
    "BREAKOUT (round 1, controle)": lambda d: (
        (d.close > d.hh20) & (d.rsi14 > 50) & (d.rsi14 < 70)
        & (d.volume > d.vol_sma * 1.2) & (d.up_4h == 1) & (d.btc_bull_4h == 1)
    ),
}


def fwd_returns(df: pd.DataFrame, mask: np.ndarray, h: int) -> np.ndarray:
    """Entra no open do candle seguinte, sai no open h candles depois. Dedupe: um
    sinal por vez por par (o próximo só conta depois que o anterior fecharia),
    senão as amostras se sobrepõem e a estatística fica inflada."""
    op = df["open"].to_numpy()
    idx = np.flatnonzero(mask)
    idx = idx[(idx + 1 + h) < len(op)]
    kept, last = [], -10**9
    for i in idx:
        if i > last + h:
            kept.append(i)
            last = i
    if not kept:
        return np.array([])
    kept = np.array(kept)
    return op[kept + 1 + h] / op[kept + 1] - 1 - FEE


def revert_returns(
    df: pd.DataFrame, mask: np.ndarray, target: str, max_hold: int, stop: float = 0.0
) -> np.ndarray:
    """Saída por REVERSÃO, não por tempo: ordem limite de venda em `target`, cancelada
    no candle max_hold (sai no open). `target` = "sma20" (volta à média) ou "pct:0.01".
    Preenchimento quando high >= alvo -- é assim que uma limite resting seria executada.
    Com `stop`, sai também se low <= entry*(1-stop). Quando alvo e stop caem no mesmo
    candle não dá para saber a ordem intrabar: assume o STOP primeiro (conservador)."""
    op = df["open"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    sma = df["sma20"].to_numpy()
    idx = np.flatnonzero(mask)
    idx = idx[(idx + 1 + max_hold) < len(op)]
    out, last = [], -10**9
    for i in idx:
        if i <= last + max_hold:
            continue
        entry = op[i + 1]
        tgt = sma[i] if target == "sma20" else entry * (1 + float(target.split(":")[1]))
        if not np.isfinite(tgt) or tgt <= entry:
            continue  # já está acima da média: não é um trade de reversão
        sl = i + 1 + max_hold
        hit = np.flatnonzero(hi[i + 1 : sl] >= tgt)
        t_hit = hit[0] if len(hit) else 10**9
        if stop > 0:
            barrier = entry * (1 - stop)
            sh = np.flatnonzero(lo[i + 1 : sl] <= barrier)
            s_hit = sh[0] if len(sh) else 10**9
        else:
            s_hit = 10**9
        if s_hit <= t_hit and s_hit < 10**9:  # empate vai para o stop
            out.append(-stop - FEE)
            last = i + s_hit
        elif t_hit < 10**9:
            out.append(tgt / entry - 1 - FEE)
            last = i + t_hit
        else:
            out.append(op[sl] / entry - 1 - FEE)
            last = i + max_hold
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="6,12,24,48")
    ap.add_argument("--mode", default="time", choices=["time", "revert"])
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    btc = btc_regime()
    frames = {p: build(p, btc) for p in PAIRS}

    if args.mode == "revert":
        run_revert(frames)
        return

    for name, fn in SIGNALS.items():
        print("=" * 96)
        print(name)
        print(f"  {'horiz':>5} " + "".join(f"{p:>23}" for p in PERIODS))
        for h in horizons:
            cells = []
            for period, (start, end) in PERIODS.items():
                rets = []
                for df in frames.values():
                    sub = df[(df.date >= start) & (df.date < end)].reset_index(drop=True)
                    if len(sub) < 300:
                        continue
                    m = fn(sub).fillna(False).to_numpy()
                    rets.append(fwd_returns(sub, m, h))
                r = np.concatenate(rets) if rets else np.array([])
                if len(r) < 30:
                    cells.append(f"{'n<30':>23}")
                    continue
                med = np.median(r) * 100
                mean = r.mean() * 100
                win = (r > 0).mean() * 100
                cells.append(f"  n={len(r):>4} md={med:>+5.2f} mn={mean:>+5.2f} w={win:>4.1f}")
            print(f"  {h * 0.5:>4.0f}h " + "".join(cells))
    print()
    print("md = mediana %, mn = média %, w = % de acerto. Tudo líquido de 0.2% de taxa.")


REVERT_SIGNALS = {
    "MR z<-2.5 + 4h>ema200": SIGNALS["MR z<-2.5 + 4h>ema200"],
    "MR z<-1.5 + close>sma200": SIGNALS["MR z<-1.5 + close>sma200"],
    "MR rsi2<10 + 4h>ema200": SIGNALS["MR rsi2<10 + 4h>ema200"],
    "MR 3 quedas + rsi14<35 + 4h up": SIGNALS["MR 3 quedas + rsi14<35 + 4h up"],
    "BREAKOUT (controle)": SIGNALS["BREAKOUT (round 1, controle)"],
}


def run_revert(frames: dict) -> None:
    """Mesmas entradas, mas saindo na reversão à média ou num alvo fixo."""
    print("SAIDA POR REVERSAO (ordem limite no alvo, cancelada no max_hold)")
    for name, fn in REVERT_SIGNALS.items():
        print("=" * 96)
        print(name)
        print(f"  {'alvo':>10} {'stop':>5} " + "".join(f"{p:>23}" for p in PERIODS))
        for target in ["sma20", "pct:0.010", "pct:0.020"]:
          for stop in [0.0, 0.02, 0.04, 0.08]:
            for max_hold in [48]:
                cells = []
                for period, (start, end) in PERIODS.items():
                    rets = []
                    for df in frames.values():
                        sub = df[(df.date >= start) & (df.date < end)].reset_index(drop=True)
                        if len(sub) < 300:
                            continue
                        m = fn(sub).fillna(False).to_numpy()
                        rets.append(revert_returns(sub, m, target, max_hold, stop))
                    r = np.concatenate(rets) if rets else np.array([])
                    if len(r) < 30:
                        cells.append(f"{'n<30':>23}")
                        continue
                    cells.append(
                        f"  n={len(r):>4} md={np.median(r) * 100:>+5.2f} "
                        f"mn={r.mean() * 100:>+5.2f} w={(r > 0).mean() * 100:>4.1f}"
                    )
                lbl = "sem" if stop == 0 else f"{stop:.0%}"
                print(f"  {target:>10} {lbl:>5} " + "".join(cells))


if __name__ == "__main__":
    main()
