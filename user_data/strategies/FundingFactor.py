"""
FundingFactor - experimento 8 do laboratório, no Freqtrade.

Fator de funding cross-sectional em perpétuos USDT-M, market-neutral por notional:
  a cada candle 1d, ranqueia a whitelist pelo funding médio dos últimos 7 dias;
  long  nos k de funding mais BAIXO  (short lotado -> o short paga funding para você)
  short nos k de funding mais ALTO   (long lotado  -> o long paga funding para você)
  sai quando o par deixa o seu bucket. Sem ROI, sem trailing; stop largo só como guarda.

Timing: candle 1d de data t fecha em t+1 00:00; as 3 cobranças de funding do dia t já são
conhecidas nesse momento. O Freqtrade entra no open do candle seguinte — igual ao probe.

ponytail: o funding vem do feather baixado pelo `download-data --trading-mode futures`.
Serve para backtest/hyperopt. Para dry-run/live é preciso trocar `_load_funding` por uma
chamada à exchange (fetch_funding_rate_history) — único trecho que muda.
"""

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter


class FundingFactor(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 10

    minimal_roi = {"0": 100}
    stoploss = -0.50  # guarda contra liquidação; a saída de verdade é por sinal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    k = IntParameter(3, 15, default=10, space="buy", optimize=False)
    fund_window = 7

    _rank: DataFrame | None = None  # data x par -> posição no ranking (1 = menor funding)
    _n: pd.Series | None = None  # data -> quantos pares tinham funding naquele dia

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float,
                 max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        return 1.0

    # ------------------------------------------------------------------ ranking cross-pair
    def _load_funding(self, pairs: list[str]) -> DataFrame:
        """Funding pago por dia (soma das cobranças de 8h), uma coluna por par."""
        datadir = Path(self.config["datadir"]) / "futures"
        cols = {}
        for pair in pairs:
            stem = pair.replace("/", "_").replace(":", "_")
            files = sorted(datadir.glob(f"{stem}-*-funding_rate.feather"))
            if not files:
                continue
            f = pd.read_feather(files[-1]).set_index("date")["funding_rate"]
            cols[pair] = f.resample("1D").sum(min_count=1)
        return DataFrame(cols).sort_index()

    def _ensure_ranks(self) -> None:
        if self._rank is not None:
            return
        fund = self._load_funding(self.dp.current_whitelist())
        f7 = fund.rolling(self.fund_window, min_periods=5).mean()
        self._rank = f7.rank(axis=1, method="first")
        self._n = f7.notna().sum(axis=1)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._ensure_ranks()
        pair = metadata["pair"]
        if pair in self._rank.columns:
            dataframe["rank"] = dataframe["date"].map(self._rank[pair])
        else:
            dataframe["rank"] = float("nan")
        dataframe["n"] = dataframe["date"].map(self._n)
        return dataframe

    # ------------------------------------------------------------------ sinais
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        k = self.k.value
        ok = dataframe["rank"].notna() & (dataframe["n"] >= 2 * k)
        dataframe.loc[ok & (dataframe["rank"] <= k), ["enter_long", "enter_tag"]] = (1, "low_funding")
        dataframe.loc[ok & (dataframe["rank"] > dataframe["n"] - k), ["enter_short", "enter_tag"]] = (1, "high_funding")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        k = self.k.value
        left_low = dataframe["rank"].isna() | (dataframe["rank"] > k)
        left_high = dataframe["rank"].isna() | (dataframe["rank"] <= dataframe["n"] - k)
        dataframe.loc[left_low, ["exit_long", "exit_tag"]] = (1, "left_low_bucket")
        dataframe.loc[left_high, ["exit_short", "exit_tag"]] = (1, "left_high_bucket")
        return dataframe
