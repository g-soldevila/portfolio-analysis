import yfinance as yf
import pandas as pd
from os.path import exists

# Download yf data
# ================================

# Data
# --------------------------------

def get_history(symbol, period= "max"):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period)
    return hist

def get_returns(history):
    rets = history['Close'].pct_change().rename("Returns")
    return rets

def save_data(df, path, symbol, rets_or_hist="rets"):
    with open(f'{path}/{symbol}_{rets_or_hist}.csv', 'w') as f:
        df.to_csv(f, index = True, header = True)

def download_data(symbol, path="data/yahoo_data", returns=True, history=False):
    hist = get_history(symbol)
    if history :
        save_data(hist, path, symbol, rets_or_hist="hist")
    if returns :
        rets = get_returns(hist)
        save_data(rets, path, symbol, rets_or_hist="rets")
        
def download_portfolio_data(symbols, path="data/yahoo_data", returns=True, history=False):
    for symbol in symbols:
        print(f'Downloading symbol {symbol}')
        download_data(symbol, path="data/yahoo_data", returns=True, history=False)
        
# Portfolio Config
# --------------------------------

def save_portfolio(df, portfolio_name, path="data/yahoo_data/portfolio"):
    with open(f'{path}/{portfolio_name}.csv', 'w') as f:
        df.to_csv(f, index = True, header = True)

# Ticker Info
# --------------------------------
        
def download_portfolio_info(symbols, properties, portfolio_name):
    data = {}
    for prop in properties:
        data[prop] = []
        
    for symbol in symbols:
        print(f"downloading symbol {symbol}")
        ticker = yf.Ticker(symbol)
        for prop in properties:
            prop_value = ticker.info[prop]
            data[prop].append(prop_value)
            
    df = pd.DataFrame.from_dict(data)
    df.index = symbols
    df.head()
    
    save_portfolio_info(df, portfolio_name)
    
def save_portfolio_info(df, portfolio_name, path="data/yahoo_data/portfolio"):
    with open(f'{path}/{portfolio_name}_info.csv', 'w') as f:
        df.to_csv(f, index = True, header = True)
    
        
# Read ff data
# ================================

def read_data(symbol, path="data/yahoo_data", rets_or_hist="rets"):
    file_path = f'{path}/{symbol}_{rets_or_hist}.csv'
    if exists(file_path):
        column_name = "Returns" if rets_or_hist == "rets" else "Close"
        df = pd.read_csv(file_path, index_col=0, parse_dates=True).rename(columns={column_name: symbol})
        return df
    else:
        print("Doesnt exist, going to download...")
        download_data(symbol, path="data/yahoo_data", returns=True, history=False)
        return read(symbol, path, rets_or_hist)
    
def read_portfolio_data(portfolio, path="data/yahoo_data", rets_or_hist="rets"):
    if isinstance(portfolio, pd.DataFrame):
        symbols = portfolio.index
    else:
        symbols = portfolio
        
    portfolio_data = read_data(symbols[0], path, rets_or_hist)
    for symbol in symbols[1:]:
        symbol_data = read_data(symbol, path, rets_or_hist)
        portfolio_data = pd.concat([portfolio_data, symbol_data], axis=1, join="inner")
        
    return portfolio_data

def read_portfolio(portfolio_name, path="data/yahoo_data/portfolio"):
    file_path = f'{path}/{portfolio_name}.csv'
    df = pd.read_csv(file_path, index_col=0)
    
    info_file_path = f'{path}/{portfolio_name}_info.csv'
    if exists(file_path):
        info = read_portfolio_info(portfolio_name, path=path)
        df = pd.concat([df, info], axis=1, join="inner")
        
    return df
    
def read_portfolio_info(portfolio_name, path="data/yahoo_data/portfolio"):
    file_path = f'{path}/{portfolio_name}_info.csv'
    df = pd.read_csv(file_path, index_col=0)
    return df
