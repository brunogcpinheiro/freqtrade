"""
Triagem CROSS-SECTIONAL: em vez de "este par sobe?", "qual dos N sobe mais que os outros?".

A cada data de rebalance, ranqueia os pares pelo retorno passado de L candles e mede o
retorno forward de H candles de:
  top-k    : os k de maior retorno passado   (momentum)
  bottom-k : os k de menor retorno passado   (reversão, se bottom > top)
  universo : média igual de todos os pares disponíveis na data

Métricas por período:
  spread  = top - bottom   -> sinal puro; o beta de mercado entra nas duas pernas e se cancela
  excesso = top - universo -> o que a perna comprada rende ACIMA de "comprar tudo igual"
  long    = top líquido de taxa -> o que você de fato embolsa em spot long-only
  t       = média/desvio*sqrt(n) do spread; |t| > 2 em oos E fwd é o mínimo para interessar

Rebalances não se sobrepõem (passo = H), então n é o número de apostas independentes.
Spread e excesso são brutos (a taxa é igual nas duas pernas); `long` desconta FEE.

Uso: python user_data/lab/probe_xsec.py [--tf 1d|4h] [--k 2]
"""

import argparse

import numpy as np
import pandas as pd

from probe_entry import FEE, PAIRS, PERIODS, load


# lookback L e horizonte H em candles do timeframe
GRID = {
    "1d": {"L": [1, 3, 7, 14, 30, 60], "H": [1, 3, 7, 14]},
    "4h": {"L": [6, 18, 42, 84, 180], "H": [6, 18, 42]},
}
HOURS = {"1d": 24, "4h": 4}
MIN_PAIRS = 6  # SUI só existe desde 2023-05; ranqueia entre os que existem na data


def panel(tf: str, col: str) -> pd.DataFrame:
    return pd.DataFrame({p: load(p, tf).set_index("date")[col] for p in PAIRS}).sort_index()


def run(tf: str, k: int) -> None:
    close = panel(tf, "close")
    opn = panel(tf, "open")
    print(f"TF = {tf}  |  k = {k} de {len(PAIRS)} pares  |  spread/excesso brutos, long líquido de {FEE:.1%}")
    hdr = f"  {'L':>4} {'H':>4} " + "".join(f"{p:>34}" for p in PERIODS)
    for L in GRID[tf]["L"]:
        print("=" * len(hdr))
        print(f"lookback {L * HOURS[tf] / 24:.0f}d")
        print(hdr)
        past = close / close.shift(L) - 1
        for H in GRID[tf]["H"]:
            # entra no open do candle seguinte, sai no open H candles depois
            fwd = opn.shift(-(1 + H)) / opn.shift(-1) - 1
            cells = []
            for period, (start, end) in PERIODS.items():
                dates = past.index[(past.index >= start) & (past.index < end)][::H]
                spread, excess, long = [], [], []
                for t in dates:
                    rp = past.loc[t].dropna()
                    rf = fwd.loc[t].reindex(rp.index).dropna()
                    rp = rp.reindex(rf.index)
                    if len(rp) < MIN_PAIRS:
                        continue
                    order = rp.sort_values().index
                    top, bot = rf[order[-k:]].mean(), rf[order[:k]].mean()
                    spread.append(top - bot)
                    excess.append(top - rf.mean())
                    long.append(top - FEE)
                if len(spread) < 20:
                    cells.append(f"{'n<20':>34}")
                    continue
                s = np.array(spread)
                t_stat = s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if s.std(ddof=1) > 0 else 0.0
                cells.append(
                    f"  n={len(s):>3} spr={s.mean() * 100:>+5.2f} t={t_stat:>+4.1f} "
                    f"exc={np.mean(excess) * 100:>+5.2f} long={np.mean(long) * 100:>+5.2f}"
                )
            print(f"  {L:>4} {H:>4} " + "".join(cells))
    print()
    print("spr>0 = momentum (top-k continua subindo); spr<0 = reversão (bottom-k recupera).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d", choices=list(GRID))
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    run(a.tf, a.k)
