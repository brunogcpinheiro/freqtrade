"""
Funding como POSICIONAMENTO, não como carry: quando o lado comprado está lotado e pagando caro,
o que acontece com o preço nos dias seguintes? Mesmo harness do probe_entry.py (4 períodos,
dedupe, baseline), sobre candles 1d de futuros com features de funding.

Sinais (long-only para comparar com os experimentos anteriores):
  crowded SHORT (funding muito negativo / no fundo do percentil) -> tese: squeeze para cima
  crowded LONG  (funding muito positivo / no topo)               -> tese: cai (long seria perder)

Uso: python user_data/lab/probe_funding.py [--universe lista.txt] [--horizons 1,3,7]
"""

import argparse

import numpy as np
import pandas as pd

import probe_xsec as px
from probe_entry import FEE, PERIODS, fwd_returns


def build(pair: str) -> pd.DataFrame | None:
    f = px.FUT / f"{pair}_USDT_USDT-1d-futures.feather"
    if not f.exists():
        return None
    df = pd.read_feather(f).sort_values("date").reset_index(drop=True)
    fund = px.funding_daily()[pair] if pair in px.funding_daily().columns else None
    if fund is None:
        return None
    df = df.merge(fund.rename("fund1").reset_index(), on="date", how="left")
    df["fund1"] = df["fund1"].shift(1)  # funding do dia ANTERIOR ao sinal (conhecido no close)
    df["fund7"] = df["fund1"].rolling(7, min_periods=5).mean()
    df["fund_pct"] = df["fund7"].rolling(90, min_periods=60).rank(pct=True)
    df["ret1"] = df["close"].pct_change()
    df["sma200"] = df["close"].rolling(200).mean()
    return df


SIGNALS = {
    "baseline (todos os candles)": lambda d: pd.Series(True, index=d.index),
    "crowded SHORT: fund7 < 0": lambda d: d.fund7 < 0,
    "crowded SHORT: fund7 < -0.03%/d": lambda d: d.fund7 < -0.0003,
    "crowded SHORT: fund_pct < 0.10": lambda d: d.fund_pct < 0.10,
    "crowded SHORT: pct<0.10 + close>sma200": lambda d: (d.fund_pct < 0.10) & (d.close > d.sma200),
    "crowded SHORT: fund1 < -0.05%/d (1 dia)": lambda d: d.fund1 < -0.0005,
    "crowded LONG: fund7 > 0.05%/d": lambda d: d.fund7 > 0.0005,
    "crowded LONG: fund7 > 0.10%/d": lambda d: d.fund7 > 0.0010,
    "crowded LONG: fund_pct > 0.90": lambda d: d.fund_pct > 0.90,
    "crowded LONG: pct>0.90 + queda ontem": lambda d: (d.fund_pct > 0.90) & (d.ret1 < 0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe")
    ap.add_argument("--horizons", default="1,3,7")
    a = ap.parse_args()
    if a.universe:
        px.UNIVERSE = [ln.strip().split("/")[0] for ln in open(a.universe) if ln.strip()]
    horizons = [int(h) for h in a.horizons.split(",")]
    frames = [f for f in (build(p) for p in px.UNIVERSE) if f is not None]
    print(f"FUNDING COMO POSICIONAMENTO  |  {len(frames)} perps 1d  |  long-only, líquido de {FEE:.1%}")
    for name, fn in SIGNALS.items():
        print("=" * 96)
        print(name)
        print(f"  {'horiz':>5} " + "".join(f"{p:>23}" for p in PERIODS))
        for h in horizons:
            cells = []
            for period, (start, end) in PERIODS.items():
                rets = []
                for df in frames:
                    sub = df[(df.date >= start) & (df.date < end)].reset_index(drop=True)
                    if len(sub) < 30:
                        continue
                    rets.append(fwd_returns(sub, fn(sub).fillna(False).to_numpy(), h))
                r = np.concatenate(rets) if rets else np.array([])
                if len(r) < 30:
                    cells.append(f"{'n<30':>23}")
                    continue
                cells.append(f"  n={len(r):>4} md={np.median(r)*100:>+5.2f} mn={r.mean()*100:>+5.2f} w={(r>0).mean()*100:>4.1f}")
            print(f"  {h:>4}d " + "".join(cells))
    print("\nmd = mediana %, mn = média %, w = % acerto. Leia contra a linha baseline do MESMO período.")


if __name__ == "__main__":
    main()
