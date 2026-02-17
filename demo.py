import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio

sample = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/example_test_files/sample_submission.csv")
sample.nunique()
stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])
tmpdf = stock_prices[stock_prices["SecuritiesCode"]==1301].reset_index(drop=True)
tmpdf.head(3)
tmpdf["Close_shift1"] = tmpdf["Close"].shift(-1)
tmpdf["Close_shift2"] = tmpdf["Close"].shift(-2)

tmpdf["rate"] = (tmpdf["Close_shift2"] - tmpdf["Close_shift1"]) / tmpdf["Close_shift1"]
stock_prices.head(3)
tmpdf2 = stock_prices[stock_prices["Date"]=="2021-12-02"].reset_index(drop=True)
tmpdf2["rank"] = tmpdf2["Target"].rank(ascending=False,method="first") -1 
tmpdf2 = tmpdf2.sort_values("rank").reset_index(drop=True)

tmpdf2_top200 = tmpdf2.iloc[:200,:]
weights = np.linspace(start=2, stop=1, num=200)
tmpdf2_top200["weights"] = weights
tmpdf2_top200.head(3)
tmpdf2_top200["calc_weights"] = tmpdf2_top200["Target"] * tmpdf2_top200["weights"]
tmpdf2_top200.head(3)

Sup = tmpdf2_top200["calc_weights"].sum()/np.mean(weights)
tmpdf2_bottom200 = tmpdf2.iloc[-200:,:]
tmpdf2_bottom200 = tmpdf2_bottom200.sort_values("rank",ascending = False).reset_index(drop=True)

tmpdf2_bottom200["weights"] = weights
tmpdf2_bottom200.head(3)

tmpdf2_bottom200["calc_weights"] = tmpdf2_bottom200["Target"] * tmpdf2_bottom200["weights"]
tmpdf2_bottom200.head(3)

Sdown = tmpdf2_bottom200["calc_weights"].sum()/np.mean(weights)

daily_spread_return = Sup - Sdown

idcount = stock_prices.groupby("Date")["SecuritiesCode"].count().reset_index()


plt.plot(idcount["Date"],idcount["SecuritiesCode"])
idcount.loc[idcount["SecuritiesCode"]==2000,:]
stock_prices2 = stock_prices.loc[stock_prices["Date"]>= "2021-01-01"].reset_index(drop=True)
stock_prices2["Rank"] = stock_prices2.groupby("Date")["Target"].rank(ascending=False,method="first") -1 
stock_prices2["Rank"] =stock_prices2["Rank"].astype("int") # floatだとエラー

score = calc_spread_return_sharpe(stock_prices2, portfolio_size= 200, toprank_weight_ratio= 2)
stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_prices.head(3)

trades_spec = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/data_specifications/trades_spec.csv")
trades = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/trades.csv")
trades.head(3)

trades_spec = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/data_specifications/trades_spec.csv")
secondary_stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
secondary_stock_prices.head(3)
options = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/options.csv")
options.head(3)

options_spec = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/data_specifications/options_spec.csv")
financials = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv")
stock_fin_spec = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_fin_spec.csv")
stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
stock_list_spec = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_list_spec.csv")

import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    break
supplemental_stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
supplemental_stock_prices["Rank"] = supplemental_stock_prices.groupby("Date")["Target"].rank(ascending=False,method="first") -1
finday = supplemental_stock_prices[supplemental_stock_prices["Date"]=="2022-02-28"].reset_index(drop=True)
finday[finday["Rank"]==finday["Rank"].iloc[0]]
finday["Rank"] = finday["Rank"].astype("int")
findaydict = dict(zip(finday["SecuritiesCode"],finday["Rank"]))

sample_prediction.head(3)
sample_prediction["Rank"]  = sample_prediction["SecuritiesCode"].map(findaydict)
env.predict(sample_prediction)

for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    sample_prediction['Rank'] = sample_prediction["SecuritiesCode"].map(findaydict)
    env.predict(sample_prediction)

sample_prediction