import pandas as pd
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
                          
# Download and update ff library
# ================================

def format_ff_url(path, name, daily=False, ext="csv"):
    return f'{path}{name}_Daily_{ext.upper()}.zip' if daily else f'{path}{name}_{ext.upper()}.zip'

def get_ff_url(name, daily=False, ext="csv"):
    httpPath = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    return format_ff_url(httpPath, name, daily, "csv")
                          
def get_url_list(names, monthly= True, daily=False, ext="csv"): 
    monthly_urls = [get_ff_url(name, daily=False, ext=ext) for name in names] if monthly else []
    daily_urls = [get_ff_url(name, daily=True, ext=ext) for name in names] if daily else []
    return monthly_urls + daily_urls                         

def download_and_unzip(url, extract_to="data/fama_french/tmp/"):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    
def update_files(urls):
    for url in urls:
        print(f'Downloading file: {url}')
        download_and_unzip(url)
                          
# Read FF CSV
# ================================

def format_ff_path(path, name, daily=False, ext="csv"):
    return f'{path}{name}_Daily.{ext}' if daily else f'{path}{name}.{ext}'

def get_ff_path(name, daily=False):
    filePath = "data/fama_french/"
    return format_ff_path(filePath, name, daily, "csv")
                          
def get_ff_factors(name, daily=False):
    return pd.read_csv(get_ff_path(name, daily), index_col=0, parse_dates=True, na_values=-99.99) / 100
