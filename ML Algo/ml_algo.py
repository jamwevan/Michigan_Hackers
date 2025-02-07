import yfinance as yf
import pandas as pd
import ta
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

def fetch_stock_data(ticker: str, start_date: str, end_date: str):
    """
    Fetch historical market data for a given equity from Yahoo Finance and compute technical indicators.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL')
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    - pd.DataFrame: Stock data including Open, High, Low, Close, Volume, Adjusted Close, and technical indicators.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # Compute rolling statistics
    data['Rolling_Mean'] = data['Close'].rolling(window=20).mean()
    data['Rolling_Std'] = data['Close'].rolling(window=20).std()
    data['Z_Score'] = (data['Close'] - data['Rolling_Mean']) / data['Rolling_Std']

    # Compute Bollinger Bands
    data['Bollinger_High'] = data['Rolling_Mean'] + (data['Rolling_Std'] * 2)
    data['Bollinger_Low'] = data['Rolling_Mean'] - (data['Rolling_Std'] * 2)

    # Compute technical indicators
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()

    return data


def adf_test(series):
    """Performs Augmented Dickey-Fuller (ADF) test with a relaxed threshold for stationarity."""
    result = adfuller(series, autolag='AIC')
    print(f"\nADF Test Statistic: {result[0]}")
    print(f"ADF Test p-value: {result[1]}")
    print(f"ADF Critical Values: {result[4]}")
    return result[1]  # Return p-value

def phillips_perron_test(series):
    """Performs the Phillips-Perron (PP) test for stationarity."""
    result = PhillipsPerron(series)
    print(f"\nPhillips-Perron Test Statistic: {result.stat}")
    print(f"Phillips-Perron Test p-value: {result.pvalue}")
    return result.pvalue

if __name__ == "__main__":
    '''
    # Fetch and save stock data for a single ticker
    ticker = input("Enter stock ticker symbol: ").upper()
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print("\nFetched Data:")
    print(stock_data.head())  # Display first few rows

    # Save to CSV
    stock_data.to_csv(f"{ticker}_market_data.csv")
    print(f"Data saved to {ticker}_market_data.csv")
    '''
    # Input stock tickers and date range
    ticker1 = input("Enter first stock ticker for pair testing: ").upper()
    ticker2 = input("Enter second stock ticker for pair testing: ").upper()
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # Fetch closing prices
    stock1 = fetch_stock_data(ticker1, start_date, end_date)['Close']
    stock2 = fetch_stock_data(ticker2, start_date, end_date)['Close']

    # Compute raw spread
    spread = stock1 - stock2
    print("\nSpread Summary:")
    print(spread.describe())

    # Run ADF Test with a relaxed threshold (p < 0.3)
    adf_pvalue = adf_test(spread)
    if adf_pvalue < 0.3:
        print(f"\n{ticker1} and {ticker2} may be cointegrated (ADF Test).")
    else:
        print(f"\n{ticker1} and {ticker2} do not show strong cointegration (ADF Test).")

    # Run Phillips-Perron Test
    pp_pvalue = phillips_perron_test(spread)
    if pp_pvalue < 0.3:
        print(f"\n{ticker1} and {ticker2} may be cointegrated (Phillips-Perron Test).")
    else:
        print(f"\n{ticker1} and {ticker2} do not show strong cointegration (Phillips-Perron Test).")
