"""
RobustEdgeHyperOptLoss

Loss function para Hyperopt que procura um edge ROBUSTO em vez do maior lucro possível.
Combina:
  - Profit Factor (peso maior)      -> qualidade do retorno
  - Max drawdown relativo (penaliza) -> estabilidade
  - Número de trades (penaliza <)    -> significância estatística

Uso:
  freqtrade hyperopt --hyperopt-loss RobustEdgeHyperOptLoss ...
"""

from datetime import datetime
from typing import Any

import numpy as np
from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss


MIN_TRADES = 60  # abaixo disso o resultado não tem significância
TARGET_MAX_DD = 0.20  # drawdown de conta acima disso é fortemente penalizado
PF_CAP = 3.0  # PF acima disso não ganha mais pontos (evita "um trade sortudo")


class RobustEdgeHyperOptLoss(IHyperOptLoss):
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        backtest_stats: dict[str, Any],
        starting_balance: float,
        **kwargs,
    ) -> float:
        if trade_count < 5 or results.empty:
            return 100.0

        wins = results.loc[results["profit_abs"] > 0, "profit_abs"].sum()
        losses = abs(results.loc[results["profit_abs"] < 0, "profit_abs"].sum())
        profit_factor = wins / losses if losses > 0 else PF_CAP
        profit_factor = min(profit_factor, PF_CAP)

        try:
            dd = calculate_max_drawdown(
                results, value_col="profit_abs", starting_balance=starting_balance, relative=True
            )
            max_dd = dd.relative_account_drawdown
        except ValueError:
            max_dd = 0.0

        total_profit_ratio = results["profit_abs"].sum() / starting_balance

        # --- componentes ---
        # PF de 1.0 => 0 ; PF de 3.0 => 2.0
        pf_score = profit_factor - 1.0
        # drawdown acima do alvo penaliza de forma quadrática
        dd_penalty = (max(0.0, max_dd - TARGET_MAX_DD) / TARGET_MAX_DD) ** 2 * 2.0 + max_dd
        # poucos trades penaliza
        trade_penalty = max(0.0, (MIN_TRADES - trade_count) / MIN_TRADES) * 2.0
        # prejuízo líquido nunca pode ser "bom"
        loss_penalty = 2.0 if total_profit_ratio <= 0 else 0.0

        score = pf_score + np.tanh(total_profit_ratio) - dd_penalty - trade_penalty - loss_penalty
        return -float(score)
