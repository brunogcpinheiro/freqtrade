#!/usr/bin/env bash
# Checagens de viés antes de confiar em qualquer backtest:
#  - lookahead-analysis: a estratégia "vê o futuro"?
#  - recursive-analysis: indicadores mudam conforme a quantidade de candles de startup?
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
RANGE="${RANGE:-20240101-20240401}"

$FT lookahead-analysis --config "$CONFIG" --userdir "$USER_DATA" --strategy "$STRATEGY" \
  --timerange "$RANGE" --max-open-trades 3
$FT recursive-analysis --config "$CONFIG" --userdir "$USER_DATA" --strategy "$STRATEGY" \
  --timerange "$RANGE" --pairs BTC/USDT
