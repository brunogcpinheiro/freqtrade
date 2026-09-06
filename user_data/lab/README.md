# Laboratório de estratégias (v1)

Objetivo: usar o Freqtrade como **laboratório** para descobrir se existe uma estratégia com
edge estatisticamente robusto, antes de colocar qualquer dinheiro real. Não é "ligar um bot e
torcer": é `estratégia → backtest → hyperopt → walk-forward → dry-run → live`.

## O que existe aqui

| Arquivo | O que é |
|---|---|
| `../strategies/TrendRegimeMomentum.py` | Estratégia v1: regime BTC 4h + alinhamento 4h + momentum/breakout 30m, saídas parciais, stop dinâmico. |
| `../hyperopts/RobustEdgeHyperOptLoss.py` | Loss de Hyperopt que premia Profit Factor e penaliza drawdown e poucos trades (não otimiza "lucro máximo"). |
| `config_lab.json` | Config de **dry-run**: Binance spot, USDT, carteira 200, `max_open_trades=3`, 60 USDT por posição, 10 pares. |
| `00_env.sh` | Períodos do protocolo e variáveis compartilhadas. |
| `01_download_data.sh` | Baixa candles 30m e 4h da Binance (precisa de internet). |
| `02_walkforward.sh` | Backtest nos 4 períodos + tabela comparativa (`walkforward_report.py`). |
| `03_hyperopt.sh` | Hyperopt **só no período de treino**. |
| `04_dryrun.sh` | Paper trading em tempo real com WebUI. |
| `05_bias_checks.sh` | `lookahead-analysis` e `recursive-analysis`. |

## Como a estratégia decide

```
BTC/USDT 4h : EMA50 > EMA200  e  ADX > 20  e  ATR/close < 7%        -> regime OK
Par 4h      : EMA20 > EMA50                                          -> alinhamento OK
Par 30m     : EMA20 > EMA50, close > EMA20, RSI 50-70, ADX > 20 e subindo,
              volume > 1.2x média(20), close > máxima dos últimos 20 candles -> ENTRA

Gestão      : stop inicial -6%
              +2%  vende 25%  -> stop vai para o break-even
              +4%  vende 25%  -> stop passa a seguir o preço (3% abaixo, nunca < +1%)
Saída total : EMA20 < EMA50 ("trend_lost") ou RSI < 45 com close < EMA20 ("momentum_lost")
Proteções   : CooldownPeriod, StoplossGuard, MaxDrawdown, LowProfitPairs
```

Os limiares do regime de BTC são fixos de propósito. Só os parâmetros de entrada/saída do par
são otimizáveis. Quanto menos parâmetros o Hyperopt mexe, menor o risco de overfitting.

## Protocolo anti-overfitting

| Período | Range | Uso |
|---|---|---|
| train | 2022-01 → 2024-01 | Único período onde o Hyperopt roda |
| valid | 2024 | Parâmetros congelados |
| oos | 2025 | Parâmetros congelados |
| fwd | 2026 → hoje | Parâmetros congelados |

Critério de aceite v1 (o `walkforward_report.py` já aplica):
**Profit Factor > 1.3 e Max Drawdown < 20% em todos os períodos.** PF > 1.5 fora da amostra
merece atenção séria. Se o treino brilha e o resto morre, a estratégia vai para o lixo, não os
parâmetros.

## Passo a passo

```bash
# 0. ambiente (uma vez)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-hyperopt.txt -e .

# 1. dados
user_data/lab/01_download_data.sh

# 2. baseline com os parâmetros default (sem hyperopt)
user_data/lab/02_walkforward.sh

# 3. hyperopt SÓ no treino, depois repita o passo 2 para ver se sobrevive fora da amostra
EPOCHS=300 user_data/lab/03_hyperopt.sh
freqtrade hyperopt-list --userdir user_data --best --profitable
freqtrade hyperopt-show --userdir user_data -n <época>   # grava TrendRegimeMomentum.json
user_data/lab/02_walkforward.sh

# 4. checagens de viés antes de confiar em qualquer número
user_data/lab/05_bias_checks.sh

# 5. paper trading real (4 a 8 semanas), WebUI em http://127.0.0.1:8080
user_data/lab/04_dryrun.sh
```

Antes do passo 5 troque `username`, `password` e `jwt_secret_key` em `config_lab.json`.
Para live, além das chaves da exchange, mude `dry_run` para `false` e comece com o mesmo
capital do dry-run.

## O que foi verificado no ambiente do repositório

- A estratégia carrega, gera entradas, faz as duas saídas parciais e o stop dinâmico funciona
  (backtest em dados **sintéticos**, apenas para validar mecânica).
- O Hyperopt roda com a `RobustEdgeHyperOptLoss` e grava os parâmetros.
- `lookahead-analysis`: **sem viés** de lookahead.
- Nenhum backtest com dados reais foi executado ainda: o ambiente onde isto foi construído não
  tem acesso às APIs de exchanges. **Os números de dados sintéticos não dizem nada sobre
  lucro real.** O primeiro resultado que importa é o do passo 2 na sua máquina.

## Ideias para a v2 (só depois de a v1 ter um veredito)

- Trocar o stop fixo por stop baseado em ATR.
- Testar 1h como timeframe principal (menos ruído, menos taxas).
- Adicionar filtro de dominância/força relativa versus BTC.
- FreqAI só quando existir uma versão "burra" que já sobrevive fora da amostra.
