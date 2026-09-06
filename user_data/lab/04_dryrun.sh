#!/usr/bin/env bash
# Dry-run (paper trading) em tempo real com a Binance. Não usa dinheiro real.
# WebUI: http://127.0.0.1:8080 (usuário/senha em config_lab.json -> troque-os antes).
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

$FT trade \
  --config "$CONFIG" \
  --userdir "$USER_DATA" \
  --strategy "$STRATEGY" \
  --db-url "sqlite:///$USER_DATA/tradesv3.dryrun.sqlite"
