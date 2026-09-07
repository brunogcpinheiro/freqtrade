#!/usr/bin/env bash
# Roda o backtest da estratégia em cada período do protocolo e gera a tabela comparativa.
# Uso: ./02_walkforward.sh            (usa os parâmetros default da estratégia ou o .json ao lado dela)
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

TAG="${TAG:-wf-$(date +%Y%m%d-%H%M%S)}"
for period in "train:$TRAIN" "valid:$VALID" "oos:$OOS" "fwd:$FWD"; do
  name="${period%%:*}"; range="${period#*:}"
  echo "=============================== $name  $range"
  mkdir -p "$USER_DATA/backtest_results/$TAG/$name"
  $FT backtesting \
    --config "$CONFIG" \
    --userdir "$USER_DATA" \
    --strategy "$STRATEGY" \
    --timerange "$range" \
    --enable-protections \
    --cache none \
    --export trades \
    --backtest-directory "$USER_DATA/backtest_results/$TAG/$name" \
    || echo "!! backtest $name falhou (faltam dados para esse período?)"
done

echo
"${PYTHON:-python}" "$LAB_DIR/walkforward_report.py" --results "$USER_DATA/backtest_results/$TAG" ${REPORT_FLAGS:-}
