# Laboratório de estratégias

Objetivo: usar o Freqtrade como **laboratório** para descobrir se existe uma estratégia com edge
estatisticamente robusto, antes de colocar qualquer dinheiro real.

**Estado atual: nenhuma estratégia aprovada.** Três experimentos rodaram e os três foram
reprovados com evidência (ver [Registro de experimentos](#registro-de-experimentos)). O que está
pronto e vale reaproveitar é o *método*, não o alfa.

## Ordem certa de trabalho

A v1 foi construída na ordem errada: 261 linhas de estratégia — regime, parciais, trailing,
proteções — antes de alguém perguntar se a entrada tinha sinal. Não tinha. Todo o resto foi
tempo perdido em cima de uma moeda sendo jogada.

A ordem certa é a inversa, do teste mais barato para o mais caro:

| # | Pergunta | Ferramenta | Custo |
|---|---|---|---|
| 1 | Depois deste sinal, o preço sobe? | `probe_entry.py` | segundos |
| 2 | O sinal sobrevive à execução real (fill, timing de candle)? | `TRM_RawEntry.py` + backtest | minutos |
| 3 | A gestão de posição melhora ou piora o edge? | estratégia completa + `02_walkforward.sh` | minutos |
| 4 | Os parâmetros aguentam fora da amostra? | `03_hyperopt.sh` no treino + repetir o passo 3 | horas |

**Não passe de um degrau enquanto o anterior não tiver passado.** Hyperopt em cima de uma
entrada sem sinal só encontra parâmetros que fazem o treino brilhar e morrem em 2025.

## O que existe aqui

| Arquivo | O que é |
|---|---|
| `probe_entry.py` | Triagem de premissas de entrada em pandas puro. Mede o retorno forward depois do sinal, líquido de taxa, nos 4 períodos. Dezenas de variantes em segundos. |
| `../strategies/TRM_RawEntry.py` | Mede o edge bruto de uma entrada dentro do Freqtrade: sem stop, sem parcial, sem trailing, saída só por tempo. Passo de confirmação do `probe_entry.py`. |
| `../strategies/TrendRegimeMomentum.py` | **Experimento 1, reprovado.** Mantido como registro: regime BTC 4h + breakout 30m. |
| `../hyperopts/RobustEdgeHyperOptLoss.py` | Loss de Hyperopt que premia Profit Factor e penaliza drawdown e poucos trades. |
| `config_lab.json` | Config de **dry-run**: Binance spot, USDT, carteira 200, `max_open_trades=3`, 60 USDT por posição, 10 pares. |
| `00_env.sh` | Períodos do protocolo e variáveis compartilhadas. |
| `01_download_data.sh` | Baixa candles 30m e 4h da Binance. |
| `02_walkforward.sh` | Backtest nos 4 períodos + tabela comparativa (`walkforward_report.py`). |
| `03_hyperopt.sh` | Hyperopt **só no período de treino**. |
| `04_dryrun.sh` | Paper trading em tempo real com WebUI. |
| `05_bias_checks.sh` | `lookahead-analysis` e `recursive-analysis`. |

## Protocolo anti-overfitting

| Período | Range | Uso |
|---|---|---|
| train | 2022-01 → 2024-01 | Único período onde o Hyperopt roda |
| valid | 2024 | Parâmetros congelados |
| oos | 2025 | Parâmetros congelados |
| fwd | 2026 → hoje | Parâmetros congelados |

Critério de aceite (aplicado por `walkforward_report.py`): **Profit Factor > 1.3 e Max Drawdown
< 20% em todos os períodos.** Se o treino brilha e o resto morre, a estratégia vai para o lixo,
não os parâmetros.

### Como ler o `probe_entry.py`

```bash
.venv/bin/python user_data/lab/probe_entry.py                          # 30m, saída por tempo fixo
.venv/bin/python user_data/lab/probe_entry.py --mode revert            # saída por reversão, com stop
.venv/bin/python user_data/lab/probe_entry.py --tf 4h --horizons 3,6,12,24   # outro timeframe base
.venv/bin/python user_data/lab/probe_entry.py --tf 1d --horizons 2,3,5,10
```

`--tf` escolhe o timeframe base; o filtro de tendência vem sempre do timeframe acima
(30m→4h, 4h→1d, 1d→1w). `--horizons` é em candles do `--tf`.

- A linha `baseline` (todos os candles) dá **mediana −0,20% em todo período e horizonte** — que é
  exatamente a taxa de 0,2%. O harness se auto-valida, e qualquer célula da tabela se lê como
  **edge bruto = mediana + 0,20**.
- A **mediana importa mais que a média**. Média positiva com mediana negativa significa depender
  de poucas caudas gordas, e isso não sobrevive fora da amostra.
- Acerto alto com média negativa = cauda esquerda comendo tudo. É o modo de falha do
  mean-reversion.
- Só interessa o que se mantém nos **quatro** períodos, principalmente `oos` e `fwd`.

## Registro de experimentos

### Experimento 1 — Breakout com filtro de regime (`TrendRegimeMomentum`) — ❌ REPROVADO

Walk-forward com os parâmetros default, sem hyperopt:

```
período  range                   trades   win%   lucro%  maxDD%    PF  sharpe  expect.
train    2022-01-01..2024-01-01     217   45.6    -0.88   10.74  0.99   -0.03   -0.008
valid    2024-01-01..2025-01-01     183   41.5   -10.16   17.90  0.85   -0.60   -0.111
oos      2025-01-01..2026-01-01     117   47.0    10.09    8.94  1.27    0.59    0.172
fwd      2026-01-01..2026-09-06      59   42.4    -3.41   10.20  0.79   -0.45   -0.116
```

Nenhum período tem significância estatística (p entre 0,30 e 0,95). Perde 10% em 2024 com o
mercado subindo 137%, e ganha 10% em 2025 com o mercado caindo 33% — uma seguidora de tendência
long-only fazendo o oposto disso não tem sinal, tem ruído.

Causa raiz, em dois testes:

1. `momentum_lost` parecia ser o vilão (60% dos trades, 87-93% perdedores, −55% no treino).
   Removida a saída, as perdas apenas migraram para `trend_lost` (−53,6% no valid). Não era a
   saída.
2. Medida a entrada crua (`TRM_RawEntry.py`, sem stop/parcial/trailing, saída por tempo): a
   **mediana do retorno forward é negativa em 11 dos 12 casos** testados. O que parecia edge em
   48h (PF 1,48 no train, 1,41 no valid) evapora em 2025-2026 (0,99 e 0,87).

Confirmado depois pelo `probe_entry.py`, que mede o breakout como controle: edge **bruto**
−0,04 / −0,14 / −0,01 / −0,15 pp nos quatro períodos. A entrada era pior que aleatória — estava
comprando topos locais.

### Experimento 2 — Mean-reversion (compra de pullback) — ❌ REPROVADO

Premissa oposta à do experimento 1. Testadas 13 variantes no `probe_entry.py`: RSI(2) estilo
Connors, z-score `(close − SMA20)/ATR`, banda de Bollinger inferior e sequências de quedas, cada
uma com e sem filtro de tendência superior (`close > SMA200`, EMA20>EMA50 em 4h, regime de BTC).

**A reversão existe e é mensurável.** Toda variante bate o baseline na mediana, e o sinal do
efeito é o oposto do breakout. A melhor (`z < −2,5` + par acima da EMA200 em 4h, horizonte 3h)
tem edge bruto de **+0,35 / +0,27 / +0,16 / +0,11 pp** nos quatro períodos.

**Só que não paga a taxa, e está morrendo.** 0,2% de ida e volta na Binance spot; mesmo com
desconto BNB o piso é ~0,15%. Nenhum período fora da amostra sobra positivo — e a sequência
0,35 → 0,27 → 0,16 → 0,11 é decaimento de alfa: o efeito está sendo arbitrado.

Saída por reversão à média em vez de tempo fixo (`--mode revert`) não resolve, e expõe a
armadilha clássica:

```
alvo +0,5%, hold 24h    train             valid             oos               fwd
                        md +0.30 mn -0.07 md +0.30 mn -0.06 md +0.30 mn -0.23 md +0.30 mn -0.27
                        acerto 88.7%      acerto 91.2%      acerto 86.9%      acerto 77.3%
```

**Acerto de 88% com média negativa em todos os períodos.** Você ganha pouco quase sempre e os
10% de perdas levam tudo.

E o stop não salva: varrendo 2%, 4% e 8% contra alvos de reversão, **todo nível de stop piora ou
empata a média** (melhor caso, alvo SMA20: sem stop +0,28 no train → +0,20 com stop de 8%,
−0,04 com 4%, −0,07 com 2%; `oos` e `fwd` seguem negativos em todos). Cortar a cauda esquerda
corta junto a reversão que se foi buscar. É a tensão fundamental do mean-reversion, e aqui ela
não tem solução dentro desta premissa.

### Experimento 3 — As mesmas premissas em 4h e 1d — ❌ REPROVADO

Hipótese: os experimentos 1 e 2 morrem porque em 30m o edge bruto (0,1-0,3 pp) é da ordem da
taxa (0,2 pp). Em timeframe maior o movimento alvo cresce e a taxa vira ruído. Rodadas as
mesmas 13 premissas em 4h (filtro 1d) e 1d (filtro 1w), horizontes de 12h a 20 dias.

**O edge não escala com o timeframe. O ruído escala.** Os números ficam maiores nas duas
direções, mas não mais consistentes:

- Em 4h, **nenhuma configuração** tem média positiva nos quatro períodos, nem com saída por
  tempo nem por reversão. A única linha com média positiva em `oos` e `fwd` ao mesmo tempo
  (3 quedas + RSI14<35 + tendência superior, alvo SMA20) é negativa em `train` e `valid`.
- Em 1d a amostra colapsa: a maioria das premissas filtradas cai abaixo de 30 sinais por
  período; o breakout e o z<−3,5 não têm amostra em período nenhum. Não dá para concluir nada,
  e isso é a conclusão — 10 pares × 4 períodos não sustentam triagem diária.
- O que aparece de "bom" em 4h/1d é beta de mercado, não sinal. O `baseline` em 4h a 192h dá
  mediana **+1,40% em 2024 e −0,92/−1,29/−0,96% nos outros três** — o mercado subiu 137% em 2024
  e caiu nos outros anos. Qualquer sinal long parece brilhante em 2024 e ruim no resto, e a
  única leitura honesta é sinal *versus baseline do mesmo período*, onde nada sobra.
- O stop segue não ajudando em 4h: 2% e 4% destroem o acerto (20-30%), 8% empata com "sem stop".

Isso fecha a questão de escala de tempo: **o problema é a premissa, não o timeframe.** Nem
breakout nem reversão à média em indicadores de preço, em nenhuma escala de 30m a 1d, mostram
expectativa positiva estável fora da amostra neste universo.

## Problemas metodológicos a corrigir no próximo experimento

- **Viés de sobrevivência**: a whitelist são 10 moedas escolhidas por terem sobrevivido até 2026.
  SUI só tem dados desde maio/2023 e mesmo assim entra no período de treino. Precisa de uma
  whitelist reconstruída ponto-a-ponto no tempo (top N por volume *naquela data*).
- **Amostra pequena**: 40 a 220 trades por período não distingue edge de sorte. O
  `probe_entry.py` contorna isso medindo milhares de sinais em vez de trades.
- **Teste múltiplo**: varrer 13 variantes × 4 períodos gera falsos positivos por construção. A
  defesa usada aqui é exigir consistência nos quatro períodos, com peso em `oos` e `fwd`.

## O que a evidência sugere tentar em seguida

Três experimentos fecham a família inteira de "indicador de preço em um par, long-only": breakout
e reversão, de 30m a 1d, com e sem filtro de regime. Em nenhuma combinação existe expectativa
positiva estável fora da amostra. Continuar nessa família é procurar o 14º indicador.

O que resta são premissas de natureza diferente:

1. **Prêmio estrutural em vez de direcional.** Funding rate em perpétuos, basis spot-futuro,
   carry: edge que não depende de prever direção e não compete com todo mundo olhando o mesmo
   RSI. Exige dados de futuros (o `download-data` do Freqtrade baixa funding com
   `--trading-mode futures`).
2. **Cross-sectional em vez de time-series.** Em vez de "este par vai subir?", "qual dos 10 vai
   subir mais que os outros?" — força relativa, rotação. Neutraliza o beta de mercado que
   domina as tabelas de 4h/1d, e é o que o experimento 3 mostrou ser o grande contaminante.
3. **Só depois**: FreqAI. Um modelo em cima de features cujo edge bruto individual é negativo
   aprende ruído mais rápido, não menos.

## Passo a passo

```bash
# 0. ambiente (uma vez)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-hyperopt.txt -e .

# 1. dados
user_data/lab/01_download_data.sh

# 2. triagem da premissa de entrada  <-- comece SEMPRE aqui
.venv/bin/python user_data/lab/probe_entry.py

# 3. confirmação no Freqtrade (só se o passo 2 passar nos 4 períodos)
freqtrade backtesting --config user_data/lab/config_lab.json --userdir user_data \
  --strategy TRM_Raw24 --timerange 20250101-20260101 --cache none

# 4. estratégia completa nos 4 períodos
user_data/lab/02_walkforward.sh

# 5. hyperopt SÓ no treino, depois repita o passo 4
EPOCHS=300 user_data/lab/03_hyperopt.sh
freqtrade hyperopt-list --userdir user_data --best --profitable
freqtrade hyperopt-show --userdir user_data -n <época>

# 6. checagens de viés
user_data/lab/05_bias_checks.sh

# 7. paper trading (4 a 8 semanas), WebUI em http://127.0.0.1:8080
user_data/lab/04_dryrun.sh
```

Antes do passo 7 troque `username`, `password` e `jwt_secret_key` em `config_lab.json`.
Para live, além das chaves da exchange, mude `dry_run` para `false`.
