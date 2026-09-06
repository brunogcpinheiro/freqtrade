#!/usr/bin/env bash
# Hyperopt APENAS no período de treino, com a loss que procura edge robusto.
# Depois: revise user_data/strategies/TrendRegimeMomentum.json (gerado pelo hyperopt)
# e rode ./02_walkforward.sh para ver se os parâmetros sobrevivem fora da amostra.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

EPOCHS="${EPOCHS:-300}"
SPACES="${SPACES:-buy sell stoploss}"

$FT hyperopt \
  --config "$CONFIG" \
  --userdir "$USER_DATA" \
  --strategy "$STRATEGY" \
  --hyperopt-loss RobustEdgeHyperOptLoss \
  --spaces $SPACES \
  --timerange "$TRAIN" \
  --enable-protections \
  --epochs "$EPOCHS" \
  --min-trades 60 \
  --random-state 42 \
  -j -1

echo
echo "Melhores épocas: $FT hyperopt-list --userdir $USER_DATA --best --profitable"
echo "Aplicar época N:  $FT hyperopt-show --userdir $USER_DATA -n N   (o resultado é gravado em TrendRegimeMomentum.json)"
