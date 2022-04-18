import yfinance as yf
import pandas as pd
from os.path import exists

# Download yf data
# ================================

def get_history(symbol, period= "max"):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period)
    return hist

def get_returns(history):
    rets = history['Close'].pct_change().rename("Returns")
    return rets

def output_csv(df, path, symbol, rets_or_hist="rets"):
    with open(f'{path}/{symbol}_{rets_or_hist}.csv', 'w') as f:
        df.to_csv(f, index = True, header = True)

def download_data(symbol, path="data/yahoo_data", returns=True, history=False):
    hist = get_history(symbol)
    if history :
        output_csv(hist, path, symbol, rets_or_hist="hist")
    if returns :
        rets = get_returns(hist)
        output_csv(rets, path, symbol, rets_or_hist="rets")
        
def download_portfolio(symbols, path="data/yahoo_data", returns=True, history=False):
    for symbol in symbols:
        print(f'Downloading symbol {symbol}')
        download_data(symbol, path="data/yahoo_data", returns=True, history=False)
        
# Read ff data
# ================================

def read(symbol, path="data/yahoo_data", rets_or_hist="rets"):
    file_path = f'{path}/{symbol}_{rets_or_hist}.csv'
    if exists(file_path):
        column_name = "Returns" if rets_or_hist == "rets" else "Close"
        df = pd.read_csv(file_path, index_col=0, parse_dates=True).rename(columns={column_name: symbol})
        return df
    else:
        print("Doesnt exist, going to download...")
        download_data(symbol, path="data/yahoo_data", returns=True, history=False)
        return read(symbol, path, rets_or_hist)
    
def read_portfolio(symbols, path="data/yahoo_data", rets_or_hist="rets"):
    portfolio = read(symbols[0], path, rets_or_hist)
    for symbol in symbols[1:]:
        df = read(symbol, path, rets_or_hist)
        portfolio = pd.concat([portfolio, df], axis=1, join="inner")
    return portfolio