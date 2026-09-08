#!/usr/bin/env bash
# Dry-run (paper trading) em tempo real com a Binance. Não usa dinheiro real.
# WebUI: http://127.0.0.1:8080 (usuário/senha na config usada).
# Ex.: CONFIG=user_data/lab/config_futures.json STRATEGY=FundingFactor user_data/lab/04_dryrun.sh
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
DB="${DB:-$USER_DATA/tradesv3.dryrun.$STRATEGY.sqlite}"
# credenciais da WebUI (e, no futuro, chaves da exchange) ficam fora do git:
SECRETS="$LAB_DIR/secrets.local.json"
EXTRA=(); [ -f "$SECRETS" ] && EXTRA=(--config "$SECRETS")

$FT trade \
  --config "$CONFIG" "${EXTRA[@]}" \
  --userdir "$USER_DATA" \
  --strategy "$STRATEGY" \
  --db-url "sqlite:///$DB"
