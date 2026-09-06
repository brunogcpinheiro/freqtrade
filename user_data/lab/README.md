# Laboratório de estratégias

Objetivo: usar o Freqtrade como **laboratório** para descobrir se existe uma estratégia com edge
estatisticamente robusto, antes de colocar qualquer dinheiro real.

**Estado atual: nenhuma estratégia aprovada.** Cinco experimentos rodaram; o quinto (long-short em
futuros) é o primeiro com retorno líquido positivo em todos os anos — e ainda assim inoperável
como está. Todos com evidência (ver [Registro de experimentos](#registro-de-experimentos)). O que está
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
| `probe_xsec.py` | Triagem **cross-sectional**: ranqueia os pares pelo retorno passado e mede o spread top-k − bottom-k. O beta de mercado se cancela por construção. |
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

### Experimento 4 — Momentum cross-sectional (`probe_xsec.py`) — ❌ REPROVADO, mas com sinal

Muda a pergunta de "este par sobe?" para "qual dos 10 sobe mais que os outros?". A cada
rebalance ranqueia os pares pelo retorno dos últimos L candles e mede o retorno forward de H
candles do top-k menos o do bottom-k (`spread`). O beta de mercado entra igual nas duas pernas
e se cancela — é o contaminante que dominou o experimento 3. Rebalances não se sobrepõem.

**É a primeira premissa do lab com sinal consistente nos quatro períodos.** Momentum de 1 dia
(L=1, H=1, 1d):

```
                     train        valid        oos          fwd
spread  k=2          +0.17        +0.27        +0.14        +0.26   (t: 1.5 / 1.4 / 1.0 / 2.3)
spread  k=3          +0.11        +0.15        +0.15        +0.20   (t: 1.2 / 1.0 / 1.3 / 2.1)
spread  k=2, em 4h   +0.17        +0.20        +0.26        +0.01
excesso k=2          +0.03        +0.16        +0.04        +0.13   (top-2 menos a média do universo)
long    k=2          -0.15        +0.26        -0.20        -0.15   (top-2 líquido de 0,2% de taxa)
```

Três leituras:

1. **O sinal existe.** Positivo em 4/4 períodos com k=2 e k=3, 3/4 em 4h, e sobrevive a
   diferentes k. Nada antes chegou perto disso.
2. **Está do lado errado para spot.** Decompondo o spread: o top-2 rende só +0,03 a +0,16 pp/dia
   acima do universo, mas o bottom-2 rende **−0,10 a −0,14 pp/dia abaixo**, em todos os
   períodos. Dois terços do efeito é "os piores de ontem continuam piores". Isso só se monetiza
   vendendo, e em spot long-only vira no máximo um filtro de exclusão de pairlist — que só vale
   alguma coisa em cima de uma entrada que ainda não existe.
3. **Não paga o custo.** Rebalance diário custa 0,2 pp/dia contra um excesso de 0,03-0,16. A
   perna comprada é negativa líquida em 3 de 4 períodos. Rebalancear menos (H=3, H=7) para
   diluir a taxa mata o sinal: em `oos` o spread vira negativo. O efeito é genuinamente de 1 dia.

Lookbacks longos (30-60d) invertem em 2026 (t de −2,2 a −2,6 em `fwd`): o momentum de médio
prazo virou reversão este ano. Instável demais para construir em cima.

### Experimento 5 — Long-short em perpétuos (`probe_xsec.py --futures`) — ⚠️ SINAL REAL, INOPERÁVEL COMO ESTÁ

Destrava o lado vendido do experimento 4. Dados de futuros USDT-M da Binance (candles 1d +
funding a cada 8h, desde 2021-06). Rebalance diário, entra no open de t+1, sai no open de t+2.
Taxa de futuros cobrada só nos nomes que **entram ou saem** de cada perna (turnover real, ≈77%
por perna/dia), funding pago pela perna comprada e recebido pela vendida.

**Com taxa maker (0,02% por lado), o spread de 1 dia (L=1, k=2) é líquido-positivo nos quatro
períodos e nos cinco anos:**

```
                     train    valid    oos      fwd
bruto                +0.14    +0.27    +0.15    +0.26     (pp/dia do notional de uma perna)
taxa (maker)          0.06     0.06     0.06     0.06
funding              +0.02    +0.00    +0.01    +0.01     (irrelevante)
NET                  +0.06    +0.21    +0.08    +0.19

por ano (maker):     2022 +2.9   2023 +42.6   2024 +76.7   2025 +28.9   2026 +46.2
```

Funding não importa (±0,02 pp/dia). L=3 e L=7 são negativos em `oos` — o efeito é de 1 dia.

**Por que ainda não é operável**, na série diária agregada 2022→hoje:

```
fee/lado   média%/d  std%/d   t    sharpe  pior dia   maxDD    dias em DD   anos+
0.020%       0.116    3.09   1.5   0.71    -30.6%    -64.6%      955        5/5
0.035%       0.069    3.09   0.9   0.43    -30.6%    -73.1%     1613        4/5
0.050%       0.023    3.09   0.3   0.14    -30.7%    -79.6%     1613        3/5
```

1. **Depende inteiramente de execução maker.** Com taxa taker o sharpe cai para 0,14. Com 77% de
   turnover diário em 4 nomes, cada ordem limite não executada perde o sinal do dia. A
   estratégia é uma aposta na qualidade dos fills, não no alfa.
2. **Cauda gorda.** Pior dia −30,6% (2022-11-09, colapso da FTX), cinco dias piores que −10%,
   drawdown máximo −64,6% do notional de uma perna, 955 dias (2,6 anos) abaixo do pico. Sharpe
   0,71 e t pooled de 1,5 com cinco anos de dados.
3. **Viés no lado vendido.** O universo são 10 moedas vivas em 2026. As que foram a zero e saíram
   da exchange — exatamente as que uma perna vendida mais lucraria *ou* mais explodiria num
   squeeze — não estão nos dados. O resultado do short está enviesado e não se sabe para que lado.

A causa dos itens 1 e 2 é a mesma: **k=2 de 10 é um spread de 4 nomes.** Um deles explodir é o
resultado do dia. Estratégias cross-sectional tiram o sharpe da diversificação — 20 nomes por
perna, não 2 — e isso é a próxima coisa a testar, não um indicador novo.

## Problemas metodológicos a corrigir no próximo experimento

- **Viés de sobrevivência**: a whitelist são 10 moedas escolhidas por terem sobrevivido até 2026.
  SUI só tem dados desde maio/2023 e mesmo assim entra no período de treino. Precisa de uma
  whitelist reconstruída ponto-a-ponto no tempo (top N por volume *naquela data*).
- **Amostra pequena**: 40 a 220 trades por período não distingue edge de sorte. O
  `probe_entry.py` contorna isso medindo milhares de sinais em vez de trades.
- **Teste múltiplo**: varrer 13 variantes × 4 períodos gera falsos positivos por construção. A
  defesa usada aqui é exigir consistência nos quatro períodos, com peso em `oos` e `fwd`.

## O que a evidência sugere tentar em seguida

Quatro experimentos fecham o que dá para extrair de **preço** neste universo (10 majors, spot,
long-only, taxa de varejo): time-series não tem sinal em escala nenhuma; cross-sectional tem um
sinal real de ~0,15 pp/dia que fica majoritariamente no lado vendido e não cobre 0,2 pp de taxa.

A conclusão honesta do lab até aqui: **não existe edge de preço operável em spot long-only com
taxa de 0,2% neste universo.** Isso não é fracasso do método — é o resultado que o método
existe para produzir antes de dinheiro real entrar.

O experimento 5 mostrou que futuros destravam o sinal, mas que com 10 pares ele é um spread de
4 nomes com cauda de −30% num dia. O que muda a equação agora:

1. **Universo maior.** Top 40-60 perpétuos por volume, k=8-10 por perna. Se o efeito de 1 dia
   é universal, a média se mantém e o desvio cai com a raiz do número de nomes — é de onde vem o
   sharpe de qualquer estratégia cross-sectional. Testável em minutos com os mesmos probes.
   Caveat: mesmo top-60 de hoje exclui as delistadas; o lado vendido continua enviesado.
2. **Só se (1) der sharpe > 1,5 com maxDD < 30%:** estratégia Freqtrade em futuros
   (`can_short=True`, `margin_mode=isolated`, ordens limite) e **dry-run para medir a taxa de
   fill maker** — que é a variável de que o edge depende, e a única coisa que paper trading mede
   melhor que backtest.
3. **FreqAI** continua depois de tudo isso.

## Passo a passo

```bash
# 0. ambiente (uma vez)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-hyperopt.txt -e .

# 1. dados
user_data/lab/01_download_data.sh

# 2. triagem da premissa de entrada  <-- comece SEMPRE aqui
.venv/bin/python user_data/lab/probe_entry.py            # time-series: "este par sobe?"
.venv/bin/python user_data/lab/probe_xsec.py --tf 1d     # cross-sectional: "qual sobe mais?"
.venv/bin/python user_data/lab/probe_xsec.py --futures --k 2 --fee 0.0002   # long-short em perpétuos

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
