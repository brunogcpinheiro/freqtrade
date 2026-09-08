# Laboratório de estratégias

Objetivo: usar o Freqtrade como **laboratório** para descobrir se existe uma estratégia com edge
estatisticamente robusto, antes de colocar qualquer dinheiro real.

**Estado atual: uma estratégia passou triagem E confirmação no Freqtrade.** Sete experimentos de
preço reprovados; o oitavo — funding como *posicionamento*, cross-sectional em perpétuos — é
positivo nos 4 períodos no probe e no backtest do Freqtrade (`FundingFactor`), e o dry-run já
roda com seleção idêntica à do backtest. Fase atual: paper trading. Tudo com evidência (ver [Registro de experimentos](#registro-de-experimentos)). O que está
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
| `probe_funding.py` | Funding como posicionamento, time-series por par (crowded long/short → retorno forward). Reprovado long-only. |
| `probe_carry.py` | Carry de funding (short perp + long spot, delta-neutro). Retorno = funding recebido − taxas de rebalance. |
| `universe_perps_pre2022.txt` | 60 perpétuos USDT-M cripto listados antes de 2022, ordenados por volume atual. Universo do experimento 6. |
| `../strategies/TRM_RawEntry.py` | Mede o edge bruto de uma entrada dentro do Freqtrade: sem stop, sem parcial, sem trailing, saída só por tempo. Passo de confirmação do `probe_entry.py`. |
| `../strategies/FundingFactor.py` | **Experimento 8 no Freqtrade.** Futuros, long-short por ranking de funding, 1d. |
| `config_futures.json` | Config de futuros para a `FundingFactor`: Binance USDT-M, isolada, 1×, 20 posições de 100 USDT, top-40 do universo. |
| `secrets.local.json` | **Não versionado.** Credenciais da WebUI (e chaves da exchange, quando houver); o `04_dryrun.sh` mescla via segundo `--config`. |
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

### Experimento 6 — Long-short em 61 perpétuos — ❌ REPROVADO (e reprova o 5)

Hipótese do experimento 5: k=2 de 10 é um spread de 4 nomes; com 60 nomes e k=8-10 a média se
mantém e o desvio cai. Universo: os 60 perps cripto USDT-M listados antes de 2022 mais líquidos
hoje (`universe_perps_pre2022.txt`), com e sem seleção ponto-a-ponto dos 40/20 mais líquidos
nos 30 dias anteriores. 18 configurações, L=1, H=1.

```
config              train        valid        oos          fwd     | sharpe  maxDD%  anos+
top-all k=5  maker  -0.05        +0.05        +0.14        +0.18   |  0.23   -76.4   3/5
top-all k=10 maker  -0.13        -0.02        -0.03        +0.22   | -0.29   -83.3   1/5
top-40  k=8  maker  -0.12        -0.12        -0.05        +0.22   | -0.43   -90.8   1/5
top-20  k=10 maker  -0.11        -0.03        -0.08        +0.14   | -0.52   -79.3   1/5
(taker: tudo pior; 17 das 18 configurações com sharpe negativo)
```

**A média não se manteve — inverteu.** Em 60 altcoins o efeito bruto de 1 dia é de *reversão*
(≈ −0,13 pp/dia para momentum no train, t até −2,8), o oposto dos 10 majors, e em 2026 vira
momentum de novo. Quando o sinal de um efeito troca entre sub-universos e entre anos, não é edge
com o sinal errado — é ruído. Nem momentum nem reversão pagam a taxa de forma estável em
nenhum recorte.

Leitura retroativa do experimento 5: 5/5 anos positivos com k=2 em 10 nomes, pooled t=1,5,
depois de dezenas de variantes testadas nos experimentos 4-5, era exatamente o que um falso
positivo por teste múltiplo parece. O experimento 6 foi o teste de robustez que ele precisava e
não passou.

Isso fecha **cross-sectional em preço**, spot ou futuros.

### Experimento 7 — Carry de funding, cash-and-carry (`probe_carry.py`) — ❌ REAL, MAS PEQUENO DEMAIS

Única premissa do lab que não prevê direção: short no perpétuo + long no spot, retorno = funding
recebido. Hedge perfeito assumido (basis diário na Binance tem média ~0), taxa de 0,30% por
nome trocado (spot + perp, ida e volta), funding negativo *está* nos dados.

```
                              train     valid     oos       fwd    | anual%  maxDD%  anos+
10 majors, todos, hold 7d    +0.001    +0.030    +0.007    +0.003  |   3.4    -5.9    4/5
61 perps, todos, hold 7d     +0.000    +0.032    +0.001   -0.024   |   1.5    -7.3    3/5
61 perps, top-3 fund, 7d     -0.036    -0.007    -0.035   -0.010   |  -7.7   -30.7    0/5
61 perps, top-3 fund, 30d    -0.004    +0.031    -0.002   +0.016   |   3.3    -6.3    4/5
(pp/dia do notional de uma perna)
```

O prêmio existe e tem o perfil esperado: positivo em 4 de 5 anos, drawdown de um dígito, e
**só paga de verdade em mercado de alta** (2024: +0,03 pp/dia ≈ 11%/ano; 2022 e 2025-26: ≈ 0).
Rebalancear semanalmente atrás do funding mais alto perde para a taxa; a cada 30 dias volta a
~3%/ano. Como o short exige margem, o capital empregado é ~1,5× o notional: **≈ 2-7%/ano no
capital**, dependendo do ano. É menos do que USDT rende em lending, com risco de exchange em cima.

Não é um bot. É uma linha de rendimento passivo que já foi arbitrada até o custo de capital.

### Experimento 8 — Fator de funding cross-sectional (`probe_xsec.py --signal funding`) — ✅ PASSOU NA TRIAGEM

Primeira premissa de **informação** em vez de preço: o funding revela quem está posicionado e
pagando para ficar. A cada dia, long nos k perpétuos de funding médio (7d) mais **baixo**
(short lotado, você recebe funding) e short nos k de funding mais **alto** (long lotado, você
recebe funding). Market-neutral por notional. Mesma série de custos do experimento 5.

```
config                   train        valid        oos          fwd     | sharpe pior%  maxDD%  por ano (% notional de 1 perna)
61 perps k=10 maker      +0.14 t2.2   +0.09 t0.9   +0.27 t3.0   +0.24 t2.0  | 1.85  -8.5   -39.5   22:+32 23:+68 24:+33 25:+98 26:+60
61 perps k=10 taker      +0.12 t1.9   +0.07 t0.8   +0.25 t2.8   +0.23 t1.9  | 1.67  -8.5   -42.0   22:+26 23:+62 24:+26 25:+91 26:+56
top-40 liq k=10 maker    +0.12 t1.9   +0.18 t1.7   +0.22 t2.1   +0.23 t2.0  | 1.76 -11.9   -37.9   22:+40 23:+48 24:+65 25:+79 26:+58
top-40 liq k=5  maker    +0.19 t1.9   +0.32 t2.3   +0.20 t1.5   +0.38 t1.9  | 1.73 -15.3   -35.8   22:+68 23:+68 24:+118 25:+71 26:+94
top-20 liq k=10 maker    +0.13 t2.2   +0.21 t2.5   +0.10 t1.2   +0.00 t0.0  | 1.47  -7.7   -26.9   22:+11 23:+84 24:+79 25:+35 26:+1
10 majors k=3 maker      +0.16 t2.0   +0.40 t3.1   +0.05 t0.5   +0.08 t1.1  | 1.61 -18.6   -30.0   22:+43 23:+73 24:+146 25:+18 26:+20
```

Por que isso é diferente dos experimentos 4-6:

- **Positivo em 4/4 períodos e 5/5 anos em toda configuração**, t ≥ 2 em 3 dos 4 períodos
  para k=10. Um sinal testado (não dezenas), com mecanismo econômico: você é pago para tomar o
  outro lado do posicionamento lotado.
- **Metade preço, metade funding recebido**, e as duas metades são positivas em todos os
  períodos separadamente. Não depende de uma só.
- **Turnover ≈ 5%/dia** (funding muda devagar): taxa custa 0,01 pp/dia, taker vs maker é
  irrelevante. O oposto do experimento 5, que vivia de fills maker em 77% de turnover.
- **Cauda controlada**: pior dia −8,5% (k=10) contra −30,6% do experimento 5. 126 de 1707 dias
  com |retorno| > 5%, dos dois lados.

O que ainda pesa contra:

- **Onde o edge mora em 2026**: top-20 por liquidez dá +0,00 no `fwd`; 61 perps e top-40 dão
  +0,23. O efeito hoje vive nos nomes 21-60 por volume — negociáveis na Binance, mas com
  slippage maior do que o modelo assume.
- **Sobrevivência**: só moedas vivas em 2026. Delistadas teriam caído na perna vendida (ganho)
  ou na comprada (perda) — a direção do viés é ambígua.
- **Drawdown de −38 a −44% do notional de uma perna** e 330-420 dias abaixo do pico. No capital
  (2 pernas a 1×) é metade disso: ~−20% e retorno ~15-50%/ano. Sharpe não muda.
- **Short em moeda que dobra num dia**: k=10 limita a −10% da perna. Margem isolada por posição
  limita a liquidação a uma posição.

Time-series long-only com o mesmo sinal (`probe_funding.py`): nada consistente. O edge só
aparece no cross-section, o que faz sentido — é relativo, não direcional.

Checagens extras antes de subir de degrau: sinal atrasado 1/2/3/7 dias extras degrada suave
(sharpe 1,76 → 1,47 → 1,62 → 1,36 → 1,00) — sem penhasco, logo sem lookahead; decomposição por
ano mostra preço e funding positivos nos 5 anos, funding recebido em 2026 no máximo da série.

#### Confirmação no Freqtrade (`FundingFactor`, `config_futures.json`)

Top-40 do universo, k=10, 1× isolada, 20 posições de 100 USDT, 1d. Sem hyperopt.

```
período  range                   trades   win%   lucro%  maxDD%    PF  sharpe
train    2022-01-01..2024-01-01    1725   49.8    25.01   10.16  1.08    1.03
valid    2024-01-01..2025-01-01     982   53.4    51.37   19.34  1.24    2.58
oos      2025-01-01..2026-01-01     962   53.8    24.38   13.74  1.14    2.10
fwd      2026-01-01..2026-09-05     633   52.6    12.04    8.46  1.12    1.61
```

- **Positivo nos 4 períodos, DD < 20% em todos, sharpe 1,0-2,6.** Retorno no capital ≈ 12-50%/ano,
  abaixo do probe (~30%) como esperado: carteira não 100% alocada, stake fixo, custos reais.
- **Funding aplicado pelo backtester** (`funding_fees` nos trades): 32-68% do lucro. Cada perna
  sozinha é beta — long perde em 2022-23/2025 e ganha em 2024, a short o inverso — e a **soma**
  é o que é estável. É o hedge funcionando.
- **Critério de aceite**: reprova no gate v1 (PF > 1,3). Num book neutro com ~1000 trades/ano e
  53% de acerto, PF ≈ 1,1-1,2 é o formato normal; a métrica certa é o sharpe. Foi adicionado um
  gate `--neutral` ao `walkforward_report.py` (lucro > 0, sharpe > 1, DD < 20%) — **depois** de
  ver o resultado, e por isso está explícito lá e aqui. Nesse gate passa nos 4 períodos.
- **`lookahead-analysis` acusa "bias" em `rank`/`n`: falso positivo estrutural.** A ferramenta
  roda o backtest truncado com `pair_whitelist = [o par do trade]`
  (`lookahead.py`, `prepare_data`), o que destrói qualquer ranking cross-pair. O teste correto —
  dois backtests completos terminando em 2025-07 e 2026-01 — dá **455 entradas idênticas** no
  período comum. Sem dependência do futuro.
#### Dry-run (paper trading)

```bash
CONFIG=user_data/lab/config_futures.json STRATEGY=FundingFactor user_data/lab/04_dryrun.sh
```

Em dry-run/live, `_load_funding` busca `fetch_funding_rate_history` da exchange (500 linhas
mais recentes, corta em 15 dias) e recalcula o ranking uma vez por dia UTC. Verificação feita
em 2026-09-07: os 20 pares escolhidos pelo bot ao vivo são **exatamente** os que o rank do
feather escolhe para o último candle fechado (10/10 long, 10/10 short).

Bug encontrado e corrigido nessa verificação, que vale registrar: a primeira versão usava
`since=15d, limit=100`. A Binance devolve as 100 linhas mais *antigas* a partir de `since`; para
pares com funding a cada 4h (COTI: 6 cobranças/dia) os dias recentes ficavam de fora, o par
virava NaN e sumia do ranking — 19/20 no primeiro teste. **Nenhum backtest pegaria isso.** É o
tipo de coisa que o dry-run existe para pegar.

O que o dry-run mede nas próximas semanas — e o que não mede:

- **Mede**: fills das ordens limite nos nomes menos líquidos, funding real vs. histórico,
  comportamento no rebalance diário, restart do bot. Compare periodicamente as posições
  abertas com o rank do feather (script acima, `probe_xsec.py`), não o P&L.
- **Não mede**: edge. 8 semanas ≈ 60 rebalances; o sharpe de 1,0-2,6 vem de 4 anos. Um mês
  ruim não reprova e um mês bom não aprova.

Antes de live, ainda: chaves da exchange em `secrets.local.json`, `dry_run: false`, e começar
com capital que você aceita perder inteiro — o backtest tem viés de sobrevivência no universo.

## Conclusão do lab (2026-09)

Sete experimentos, um universo de 10 majors + 60 altcoins, 2022 → hoje, sempre com o mesmo
protocolo (train / valid / oos / fwd, parâmetros congelados, custos reais):

| # | Premissa | Resultado |
|---|---|---|
| 1 | Breakout + regime BTC, 30m | Sem sinal; entrada pior que aleatória |
| 2 | Reversão à média, 30m | Sinal real (+0,1-0,3 pp), abaixo da taxa, decaindo |
| 3 | Os mesmos em 4h e 1d | Não escala; o que parece sinal é beta de mercado |
| 4 | Momentum cross-sectional, spot | Sinal em 4/4 períodos, mas no lado vendido e abaixo da taxa |
| 5 | Long-short em perpétuos, 10 majors | Positivo 5/5 anos com maker, sharpe 0,7, maxDD −65% |
| 6 | O mesmo em 61 perps | Inverte o sinal; reprova o 5 como artefato de universo |
| 7 | Carry de funding | Real, 2-7%/ano no capital |
| 8 | **Fator de funding cross-sectional** | **Passou triagem e Freqtrade: 4/4 períodos, sharpe 1,0-2,6, DD < 20%** |

**Não existe, neste universo e com taxa de varejo, uma estratégia de preço com expectativa
positiva estável fora da amostra.** O que passou não é preço — é informação sobre posicionamento. O único retorno estrutural mensurável (funding) rende menos
que renda fixa em stablecoin. Isso é o resultado do método funcionando, não falhando: ele
existia para impedir que dinheiro real entrasse numa dessas sete coisas — e impediu.

O que mudaria a conclusão, em ordem de plausibilidade: (a) custo de execução de market maker
(≤ 0,01%), que transforma os experimentos 2 e 5 — mas isso é infraestrutura, não estratégia;
(b) dados que ninguém mais tem (fluxo, order book, on-chain) — o lab só olhou preço e funding;
(c) FreqAI em cima de (b), nunca em cima de preço puro.

**Paper trading não entra nessa lista.** Dry-run mede execução, não edge; sem uma estratégia com
expectativa positiva, ele só mostra ruído mais devagar.

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

Seis experimentos fecham tudo que é **previsão de preço** neste lab: time-series (breakout,
reversão, 30m a 1d) e cross-sectional (momentum/reversão de 1 a 60 dias, 10 a 60 nomes, spot e
futuros). Nada tem expectativa positiva estável fora da amostra depois de custos.

Ver [Conclusão do lab](#conclusão-do-lab-2026-09).

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
.venv/bin/python user_data/lab/probe_xsec.py --futures --k 8 --fee 0.0002 --universe user_data/lab/universe_perps_pre2022.txt --top 40
.venv/bin/python user_data/lab/probe_carry.py --hold 30 --universe user_data/lab/universe_perps_pre2022.txt
.venv/bin/python user_data/lab/probe_xsec.py --futures --signal funding --k 10 --fee 0.0002 --universe user_data/lab/universe_perps_pre2022.txt --top 40

# 3. confirmação no Freqtrade (só se o passo 2 passar nos 4 períodos)
freqtrade backtesting --config user_data/lab/config_lab.json --userdir user_data \
  --strategy TRM_Raw24 --timerange 20250101-20260101 --cache none

# 4. estratégia completa nos 4 períodos
user_data/lab/02_walkforward.sh
CONFIG=user_data/lab/config_futures.json STRATEGY=FundingFactor REPORT_FLAGS=--neutral user_data/lab/02_walkforward.sh

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
