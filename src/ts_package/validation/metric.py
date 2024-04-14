import numpy as np


class BusinessSimulation():
    """
    Симулятор для управления инвестиционными процессами.
    """
    def __init__(self, base_rate=0.07):
        self.base_rate = base_rate
        self.daytime_yield = base_rate + 0.5 / 100
        self.overnight_yield = base_rate - 0.9 / 100
        self.borrow_rate = base_rate + 1 / 100

    def calculate_returns(self, actuals, forecasts):
        actual_values = actuals.copy()
        returns = np.zeros(actual_values.shape)

        # Начало дня
        # Определяем, где согласно прогнозам выгодно инвестировать
        invest_decision = (forecasts > 0)
        # Рассчитываем доходность от инвестиций
        returns[invest_decision] += self.daytime_yield * forecasts[invest_decision]
        # Корректируем остатки средств после инвестиций
        actual_values[invest_decision] -= forecasts[invest_decision]
        # Компенсируем недостаток средств по отрицательным прогнозам
        actual_values[~invest_decision] -= forecasts[~invest_decision]

        # Завершение дня - итоговые расчеты баланса
        positive_liquidity = (actual_values > 0)
        # Обработка ситуаций с положительным балансом
        returns[positive_liquidity] += self.overnight_yield * actual_values[positive_liquidity]
        # Рассмотрение случаев заимствования средств
        returns[~positive_liquidity] += self.borrow_rate * actual_values[~positive_liquidity]

        return returns.sum()

    def evaluate_performance(self, actuals, forecasts):
        total_returns = self.calculate_returns(actuals, forecasts)
        return total_returns