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

Modo --futures (só 1d): long-short de verdade em perpétuos USDT-M.
  - taxa de futuros por perna (--fee, default 0.05% taker por lado), cobrada só nos nomes que
    ENTRAM ou SAEM de cada perna a cada rebalance (turnover real, não 100%/dia)
  - funding: a perna comprada paga a taxa, a vendida recebe (soma das 3 cobranças diárias)
  - net = spread bruto - taxas - funding líquido

Universo (--futures): default são os 10 pares do lab. --universe arquivo.txt (um par por
linha, formato BTC/USDT:USDT) usa todos os que tiverem dados; --top N seleciona a cada
rebalance os N de maior volume em USDT nos 30 dias ANTERIORES (ponto-a-ponto, sem olhar o
futuro). Nada disso remove o viés de sobrevivência: só existem os pares vivos hoje.

Uso: python user_data/lab/probe_xsec.py [--tf 1d|4h] [--k 2] [--futures] [--fee 0.0005]
                                        [--universe lista.txt] [--top 40]
"""

import argparse
import pathlib

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


FUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "binance" / "futures"
UNIVERSE: list[str] = list(PAIRS)  # símbolos base (BTC, ETH...); sobrescrito por --universe
TOP_N: int | None = None  # --top: seleção ponto-a-ponto por volume dos 30d anteriores


def fut_panel(tf: str, col: str) -> pd.DataFrame:
    cols = {}
    for p in UNIVERSE:
        f = FUT / f"{p}_USDT_USDT-{tf}-futures.feather"
        if f.exists():
            cols[p] = pd.read_feather(f).set_index("date")[col]
    return pd.DataFrame(cols).sort_index()


def funding_daily() -> pd.DataFrame:
    """Funding pago por dia (soma das cobranças de 8h). Positivo = long paga, short recebe."""
    cols = {}
    for p in UNIVERSE:
        f = FUT / f"{p}_USDT_USDT-1h-funding_rate.feather"
        if f.exists():
            cols[p] = pd.read_feather(f).set_index("date")["funding_rate"].resample("1D").sum()
    return pd.DataFrame(cols).sort_index()


def spread_series(k: int, fee_side: float, L: int, start: str, end: str) -> pd.DataFrame:
    """Série diária do long-short em futuros: colunas gross, fee, fund, net (fração do
    notional de uma perna). Rebalance diário; entra no open de t+1, sai no open de t+2."""
    close = fut_panel("1d", "close")
    opn = fut_panel("1d", "open")
    past = close / close.shift(L) - 1
    fwd = opn.shift(-2) / opn.shift(-1) - 1
    # funding pago durante o dia t+1 (a posição é carregada de open t+1 a open t+2)
    fund_next = funding_daily().reindex(past.index).shift(-1).reindex(columns=past.columns)
    # volume em USDT dos 30 dias ANTERIORES (shift(1): não inclui o dia do rebalance)
    dvol = (fut_panel("1d", "volume") * close).rolling(30, min_periods=20).mean().shift(1)
    rows, prev_top, prev_bot = [], set(), set()
    for t in past.index[(past.index >= start) & (past.index < end)]:
        rp = past.loc[t].dropna()
        if TOP_N:
            liquid = dvol.loc[t].reindex(rp.index).dropna().nlargest(TOP_N).index
            rp = rp.reindex(liquid)
        rf = fwd.loc[t].reindex(rp.index).dropna()
        rp = rp.reindex(rf.index)
        if len(rp) < MIN_PAIRS:
            continue
        order = rp.sort_values().index
        top, bot = set(order[-k:]), set(order[:k])
        # cada nome trocado = 1 saída + 1 entrada; custo por perna = trocados/k * 2 * fee
        changed = len(top ^ prev_top) / 2 + len(bot ^ prev_bot) / 2
        fr = fund_next.loc[t].fillna(0)
        rows.append({
            "date": t,
            "gross": rf[list(top)].mean() - rf[list(bot)].mean(),
            "fee": changed / k * 2 * fee_side,
            "fund": fr[list(top)].mean() - fr[list(bot)].mean(),
        })
        prev_top, prev_bot = top, bot
    df = pd.DataFrame(rows).set_index("date")
    df["net"] = df["gross"] - df["fee"] - df["fund"]
    return df


def run_futures(k: int, fee_side: float) -> None:
    print(f"FUTUROS 1d long-short  |  k = {k}  |  taxa {fee_side:.3%} por lado  |  H = 1 dia")
    hdr = f"  {'L':>3} " + "".join(f"{p:>42}" for p in PERIODS)
    print(hdr)
    for L in [1, 3, 7]:
        cells = []
        for period, (start, end) in PERIODS.items():
            df = spread_series(k, fee_side, L, start, end)
            if len(df) < 20:
                cells.append(f"{'n<20':>42}")
                continue
            g, f_, fn, net = (df[c].to_numpy() for c in ["gross", "fee", "fund", "net"])
            t_stat = net.mean() / net.std(ddof=1) * np.sqrt(len(net))
            cells.append(
                f"  n={len(net):>3} bruto={g.mean() * 100:>+5.2f} taxa={f_.mean() * 100:>5.2f} "
                f"fund={fn.mean() * 100:>+5.2f} NET={net.mean() * 100:>+5.2f} t={t_stat:>+4.1f}"
            )
        print(f"  {L:>3} " + "".join(cells))
    print()
    print("Tudo em % do notional de UMA perna, por dia. fund>0 = a perna comprada pagou mais do que a vendida recebeu.")


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
    ap.add_argument("--futures", action="store_true", help="long-short em perpétuos (só 1d)")
    ap.add_argument("--fee", type=float, default=0.0005, help="taxa de futuros por lado")
    ap.add_argument("--universe", help="arquivo com um par por linha (BTC/USDT:USDT)")
    ap.add_argument("--top", type=int, help="N mais líquidos nos 30d anteriores, a cada rebalance")
    a = ap.parse_args()
    if a.universe:
        UNIVERSE = [ln.strip().split("/")[0] for ln in open(a.universe) if ln.strip()]
    TOP_N = a.top
    if a.futures:
        run_futures(a.k, a.fee)
    else:
        run(a.tf, a.k)
