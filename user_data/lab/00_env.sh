# Carregado pelos outros scripts. Ajuste se necessário.
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA="$(cd "$LAB_DIR/.." && pwd)"
CONFIG="${CONFIG:-$LAB_DIR/config_lab.json}"
STRATEGY="${STRATEGY:-TrendRegimeMomentum}"
PAIRS="${PAIRS:-BTC/USDT ETH/USDT SOL/USDT XRP/USDT BNB/USDT LINK/USDT AVAX/USDT DOGE/USDT SUI/USDT ADA/USDT}"
FT="${FT:-freqtrade}"

# Períodos do protocolo anti-overfitting (walk-forward).
TRAIN="20220101-20240101"   # Hyperopt SÓ aqui
VALID="20240101-20250101"   # validação (parâmetros congelados)
OOS="20250101-20260101"     # out-of-sample (parâmetros congelados)
FWD="20260101-"             # forward / ano corrente (parâmetros congelados)
