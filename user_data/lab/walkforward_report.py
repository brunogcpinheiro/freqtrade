"""
Lê os resultados de backtest gerados por 02_walkforward.sh e monta a tabela
train / valid / oos / fwd com as métricas que importam para decidir se há edge.

Critério de aceite (v1): Profit Factor > 1.3 e Max Drawdown < 20% em TODOS os períodos,
sem alterar parâmetros entre eles. PF > 1.5 fora da amostra = interessante de verdade.
"""

import argparse
import sys
from pathlib import Path

from freqtrade.data.btanalysis import get_latest_backtest_filename, load_backtest_stats


PERIODS = ["train", "valid", "oos", "fwd"]
PF_MIN = 1.3
DD_MAX = 0.20


def find_result(results_dir: Path, period: str) -> Path | None:
    """Cada período tem seu próprio diretório (--backtest-directory); usa o mais recente."""
    d = results_dir / period
    if not d.is_dir():
        return None
    try:
        return d / get_latest_backtest_filename(d)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", required=True, help="diretório do TAG (contém train/ valid/ oos/ fwd/)"
    )
    args = ap.parse_args()
    results_dir = Path(args.results)

    rows = []
    for period in PERIODS:
        f = find_result(results_dir, period)
        if not f:
            rows.append((period, None))
            continue
        stats = load_backtest_stats(f)
        _, s = next(iter(stats["strategy"].items()))
        rows.append((period, s))

    header = (
        f"{'período':8} {'range':23} {'trades':>6} {'win%':>6} {'lucro%':>8} "
        f"{'maxDD%':>7} {'PF':>5} {'sharpe':>7} {'expect.':>8} ok"
    )
    print(header)
    print("-" * len(header))
    all_ok = True
    any_data = False
    for period, s in rows:
        if s is None:
            print(f"{period:8} {'(sem resultado)':23}")
            all_ok = False
            continue
        any_data = True
        pf = s.get("profit_factor") or 0.0
        dd = s.get("max_drawdown_account") or 0.0
        ok = pf > PF_MIN and dd < DD_MAX and s["total_trades"] > 0
        all_ok &= ok
        rng = f"{s['backtest_start'][:10]}..{s['backtest_end'][:10]}"
        print(
            f"{period:8} {rng:23} {s['total_trades']:>6} {s['winrate'] * 100:>6.1f} "
            f"{s['profit_total'] * 100:>8.2f} {dd * 100:>7.2f} {pf:>5.2f} "
            f"{s.get('sharpe', 0):>7.2f} {s.get('expectancy', 0):>8.3f} {'✅' if ok else '❌'}"
        )
    print()
    if not any_data:
        print("Nenhum resultado encontrado. Rode 01_download_data.sh e 02_walkforward.sh primeiro.")
        return 1
    if all_ok:
        print(
            f"VEREDITO: passou em todos os períodos (PF > {PF_MIN}, DD < {DD_MAX:.0%}). "
            "Próximo passo: 05_bias_checks.sh e depois 04_dryrun.sh por 4-8 semanas."
        )
    else:
        print(
            "VEREDITO: NÃO passou em todos os períodos. Não vá para live com isso. "
            "Revise a lógica (não só os parâmetros) e repita."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
