"""
TrendRegimeMomentum - estratégia v1 do laboratório quantitativo.

Ideia (spot, apenas long):
  1. Regime  : BTC/USDT em 4h precisa estar em tendência de alta saudável
               (EMA50 > EMA200, ADX acima do mínimo, volatilidade dentro do aceitável).
  2. Alinhamento: o próprio par em 4h também precisa estar em tendência (EMA20 > EMA50).
  3. Entrada 30m: EMA rápida > EMA lenta, RSI na faixa de momentum (50-70), ADX subindo,
               volume acima da média e rompimento da máxima recente.
  4. Gestão   : stop-loss inicial fixo; +2% vende 25%; +4% vende mais 25%; depois disso o
               stop vai para o break-even e passa a seguir o preço (trailing).
  5. Saída total: EMA rápida cruza abaixo da lenta ou RSI perde momentum.

Todos os parâmetros com `optimize=True` são elegíveis ao Hyperopt.
O regime de BTC é deliberadamente fixo (não otimizável) para reduzir overfitting.
"""

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    informative,
    stoploss_from_open,
)


class TrendRegimeMomentum(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "30m"
    informative_timeframe = "4h"
    can_short = False
    process_only_new_candles = True

    # EMA200 em 4h = 200 * 8 candles de 30m = 1600 candles. Margem extra para ADX/ATR.
    startup_candle_count = 1700

    # ROI desativado na prática: a saída é feita por sinal, parciais e stop dinâmico.
    minimal_roi = {"0": 1.0}

    # Stop-loss "duro" inicial (relativo ao preço de entrada). Otimizável no space `stoploss`.
    stoploss = -0.06
    use_custom_stoploss = True
    trailing_stop = False  # trailing é feito em custom_stoploss

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Saídas parciais via adjust_trade_position (apenas reduções, nunca aumentos).
    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ---------------------------------------------------------------- parâmetros (buy)
    ema_fast = IntParameter(10, 30, default=20, space="buy", optimize=True)
    ema_slow = IntParameter(40, 80, default=50, space="buy", optimize=True)
    rsi_low = IntParameter(45, 58, default=50, space="buy", optimize=True)
    rsi_high = IntParameter(62, 80, default=70, space="buy", optimize=True)
    adx_min = IntParameter(15, 35, default=20, space="buy", optimize=True)
    vol_mult = DecimalParameter(1.0, 2.5, decimals=1, default=1.2, space="buy", optimize=True)
    breakout_lookback = IntParameter(10, 40, default=20, space="buy", optimize=True)

    # ---------------------------------------------------------------- parâmetros (sell)
    tp1_pct = DecimalParameter(0.010, 0.040, decimals=3, default=0.020, space="sell", optimize=True)
    tp2_pct = DecimalParameter(0.030, 0.080, decimals=3, default=0.040, space="sell", optimize=True)
    trail_pct = DecimalParameter(
        0.015, 0.060, decimals=3, default=0.030, space="sell", optimize=True
    )
    exit_rsi = IntParameter(35, 50, default=45, space="sell", optimize=True)

    # ---------------------------------------------------------------- regime BTC (fixo)
    btc_adx_min = 20
    btc_atr_pct_max = 0.07  # ATR(14)/close em 4h; acima disso o mercado está "caótico"

    # ---------------------------------------------------------------- proteções
    @property
    def protections(self):
        return [
            # Espera 2 candles (1h) depois de fechar um trade antes de reentrar no mesmo par.
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            # 3 stops em 24h (48 candles de 30m) => pausa geral de 12h.
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 3,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
            # Drawdown de 15% nos últimos 10 dias (com >= 10 trades) => pausa geral de 1 dia.
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 480,
                "trade_limit": 10,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.15,
            },
            # Par com >= 4 trades e lucro <= 0 nos últimos 10 dias => pausa de 2 dias no par.
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 480,
                "trade_limit": 4,
                "stop_duration_candles": 96,
                "required_profit": 0.0,
            },
        ]

    # ---------------------------------------------------------------- indicadores
    @informative("4h", "BTC/{stake}")
    def populate_indicators_btc_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Regime de mercado medido no BTC em 4h. Colunas viram btc_usdt_<col>_4h."""
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr_pct"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]
        dataframe["regime_ok"] = (
            (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["adx"] > self.btc_adx_min)
            & (dataframe["atr_pct"] < self.btc_atr_pct_max)
        ).astype(int)
        return dataframe

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Alinhamento de tendência do próprio par em 4h. Colunas viram <col>_4h."""
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["trend_ok"] = (dataframe["ema20"] > dataframe["ema50"]).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs pré-calculadas para todo o range dos parâmetros: assim o Hyperopt não precisa
        # recalcular indicadores a cada época (padrão recomendado pela documentação).
        for val in self.ema_fast.range:
            dataframe[f"ema_{val}"] = ta.EMA(dataframe, timeperiod=val)
        for val in self.ema_slow.range:
            dataframe[f"ema_{val}"] = ta.EMA(dataframe, timeperiod=val)
        for val in self.breakout_lookback.range:
            # shift(1): máxima dos N candles ANTERIORES (evita lookahead).
            dataframe[f"hh_{val}"] = dataframe["high"].rolling(val).max().shift(1)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["adx_rising"] = (dataframe["adx"] > dataframe["adx"].shift(3)).astype(int)
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    # ---------------------------------------------------------------- entradas
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = dataframe[f"ema_{self.ema_fast.value}"]
        ema_slow = dataframe[f"ema_{self.ema_slow.value}"]
        hh = dataframe[f"hh_{self.breakout_lookback.value}"]

        conditions = (
            # regime e alinhamento nos timeframes superiores
            (dataframe["btc_usdt_regime_ok_4h"] == 1)
            & (dataframe["trend_ok_4h"] == 1)
            # tendência local
            & (ema_fast > ema_slow)
            & (dataframe["close"] > ema_fast)
            # momentum sem exaustão
            & (dataframe["rsi"] > self.rsi_low.value)
            & (dataframe["rsi"] < self.rsi_high.value)
            & (dataframe["adx"] > self.adx_min.value)
            & (dataframe["adx_rising"] == 1)
            # participação
            & (dataframe["volume"] > dataframe["vol_sma"] * self.vol_mult.value)
            # rompimento da máxima recente
            & (dataframe["close"] > hh)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[conditions, ["enter_long", "enter_tag"]] = (1, "trend_breakout")
        return dataframe

    # ---------------------------------------------------------------- saídas por sinal
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = dataframe[f"ema_{self.ema_fast.value}"]
        ema_slow = dataframe[f"ema_{self.ema_slow.value}"]

        trend_lost = ema_fast < ema_slow
        momentum_lost = (dataframe["rsi"] < self.exit_rsi.value) & (dataframe["close"] < ema_fast)

        dataframe.loc[trend_lost, ["exit_long", "exit_tag"]] = (1, "trend_lost")
        dataframe.loc[momentum_lost & ~trend_lost, ["exit_long", "exit_tag"]] = (1, "momentum_lost")
        return dataframe

    # ---------------------------------------------------------------- saídas parciais
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ):
        exits_done = trade.nr_of_successful_exits
        if trade.has_open_orders:
            return None

        # +tp1 -> vende 25% da posição original.
        if exits_done == 0 and current_exit_profit >= self.tp1_pct.value:
            amount = trade.stake_amount * 0.25
            reason = "tp1_25pct"
        # +tp2 -> vende mais 25% da posição original (= 1/3 do que sobrou).
        elif exits_done == 1 and current_exit_profit >= self.tp2_pct.value:
            amount = trade.stake_amount / 3
            reason = "tp2_25pct"
        else:
            return None

        # Respeita o mínimo da exchange tanto para a parcela vendida quanto para o restante.
        remaining = trade.stake_amount - amount
        if min_stake is not None and (amount < min_stake or remaining < min_stake):
            return None
        return -amount, reason

    # ---------------------------------------------------------------- stop dinâmico
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        exits_done = trade.nr_of_successful_exits

        if exits_done == 0:
            # Antes da primeira parcial: stop duro inicial (self.stoploss).
            return None

        if exits_done == 1:
            # Após a 1ª parcial: protege o break-even (+0.3% cobre as taxas de ida e volta).
            return stoploss_from_open(0.003, current_profit, is_short=trade.is_short)

        # Após a 2ª parcial: trailing. Nunca abaixo de +1% e sempre `trail_pct` abaixo do preço.
        # O freqtrade só move o stop a favor do trade, então isso funciona como trailing.
        trail = max(0.01, current_profit - self.trail_pct.value)
        return stoploss_from_open(trail, current_profit, is_short=trade.is_short)
