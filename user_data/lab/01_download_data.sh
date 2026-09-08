#!/usr/bin/env bash
# Baixa candles 30m (estratégia) e 4h (regime/alinhamento) da Binance para todos os pares.
# Precisa de acesso à internet à Binance. ~5 anos x 10 pares de 30m leva alguns minutos.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

$FT download-data \
  --config "$CONFIG" \
  --userdir "$USER_DATA" \
  --exchange binance \
  --pairs $PAIRS \
  --timeframes 30m 4h \
  --timerange 20211101- \
  --prepend

# BTC/USDT em 4h é obrigatório para o regime, mesmo se BTC sair da whitelist.
$FT download-data --config "$CONFIG" --userdir "$USER_DATA" --exchange binance \
  --pairs BTC/USDT --timeframes 4h --timerange 20211101- --prepend
