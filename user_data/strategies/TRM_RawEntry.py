"""Mede o edge BRUTO da entrada: sem saída por sinal, sem parciais, sem trailing.
Só entra e sai depois de N horas. Se a entrada não tem edge, nada aqui salva."""
from TrendRegimeMomentum import TrendRegimeMomentum
from pandas import DataFrame


class _Raw(TrendRegimeMomentum):
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    stoploss = -0.99
    HOURS = 12

    @property
    def protections(self):
        return []

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def __init__(self, config):
        self.minimal_roi = {"0": 100, str(self.HOURS * 60): -1}
        super().__init__(config)


class TRM_Raw12(_Raw):
    HOURS = 12


class TRM_Raw24(_Raw):
    HOURS = 24


class TRM_Raw48(_Raw):
    HOURS = 48
