"""
Triagem de CARRY DE FUNDING (cash-and-carry): short no perpétuo + long no spot do mesmo ativo.

O retorno não vem do preço -- as duas pernas se cancelam -- e sim do funding que o short recebe
a cada 8h quando a taxa é positiva (e paga quando é negativa). É a única premissa do lab que
não exige prever direção.

Simplificações (todas contra a estratégia ou neutras, exceto a última):
  - hedge perfeito: P&L de preço = 0. Na prática há basis perp-spot; na Binance o mark price é
    ancorado no índice e o drift diário do basis tem média ~0.
  - seleção: a cada rebalance (H dias) escolhe os k pares de maior funding MÉDIO nos 7 dias
    anteriores (conhecido em t). Baseline: todos os pares igualmente.
  - taxa: 0,10% spot + 0,05% perp por lado => 0,30% ida e volta por nome trocado.
  - NÃO modela: margem/liquidação do short (capital ~1,5x o notional), risco de exchange,
    dias em que o funding vai muito negativo e você paga (esses ESTÃO nos dados).

Retornos em % do notional de uma perna, por dia. Anualizado = média diária x 365.

Uso: python user_data/lab/probe_carry.py [--k 5] [--hold 7] [--universe lista.txt]
"""

import argparse

import numpy as np
import pandas as pd

import probe_xsec as px
from probe_entry import PERIODS


FEE_RT = 0.0010 * 2 + 0.0005 * 2  # spot ida+volta + perp ida+volta por nome trocado


def carry_series(k: int | None, hold: int, start: str, end: str) -> pd.DataFrame:
    fund = px.funding_daily()
    fund = fund.loc[(fund.index >= "2021-06-01")]
    signal = fund.rolling(7, min_periods=5).mean().shift(1)  # média dos 7d ANTERIORES
    rows, prev = [], set()
    dates = fund.index[(fund.index >= start) & (fund.index < end)]
    held: set = set()
    for i, t in enumerate(dates):
        if i % hold == 0:  # rebalance
            sig = signal.loc[t].dropna() if t in signal.index else pd.Series(dtype=float)
            held = set(sig.nlargest(k).index) if k else set(sig.index)
            changed = len(held ^ prev)
            fee = changed / max(len(held), 1) * FEE_RT
            prev = held
        else:
            fee = 0.0
        if not held:
            continue
        f = fund.loc[t].reindex(list(held)).fillna(0)
        rows.append({"date": t, "funding": f.mean(), "fee": fee})
    df = pd.DataFrame(rows).set_index("date")
    df["net"] = df["funding"] - df["fee"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=7, help="dias entre rebalances")
    ap.add_argument("--universe", help="arquivo com um par por linha (BTC/USDT:USDT)")
    a = ap.parse_args()
    if a.universe:
        px.UNIVERSE = [ln.strip().split("/")[0] for ln in open(a.universe) if ln.strip()]
    print(f"CARRY DE FUNDING  |  universo {len(px.UNIVERSE)} perps  |  rebalance a cada {a.hold}d  |  % do notional/dia")
    hdr = f"{'seleção':>14} " + "".join(f"{p:>24}" for p in PERIODS) + f" | {'média%/d':>8} {'anual%':>7} {'pior dia%':>9} {'maxDD%':>7} {'anos+':>5}"
    print(hdr)
    for k in [None, 3, 5, 10]:
        if k and k > len(px.UNIVERSE):
            continue
        cells = []
        for period, (s, e) in PERIODS.items():
            n = carry_series(k, a.hold, s, e).net
            cells.append(f"  {n.mean()*100:>+6.3f} (fund {carry_series(k, a.hold, s, e).funding.mean()*100:>+5.3f})")
        n = carry_series(k, a.hold, "2022-01-01", "2027-01-01").net
        eq = (1 + n).cumprod(); dd = eq / eq.cummax() - 1
        yrs = n.groupby(n.index.year).sum()
        label = f"top-{k} funding" if k else "todos (baseline)"
        print(f"{label:>14} " + "".join(cells) +
              f" | {n.mean()*100:>8.3f} {n.mean()*365*100:>7.1f} {n.min()*100:>9.2f} {dd.min()*100:>7.2f} {(yrs>0).sum()}/{len(yrs)}")
    print()
    print("fund = funding bruto recebido; net desconta 0,30% por nome trocado no rebalance. Hedge perfeito assumido.")


if __name__ == "__main__":
    main()
