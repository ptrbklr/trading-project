import pandas as pd
from datetime import timedelta

def trades_to_df(trades):
    cols = ['price', 'volume', 'time', 'side', 'type', 'misc']
    df = pd.DataFrame(trades, columns=cols)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['price'] = df['price'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

def build_candles(df_trades, interval_minutes=15):
    df = df_trades.copy()
    df = df.set_index('time').sort_index()

    rule = f'{interval_minutes}min'
    ohlc = df['price'].resample(rule).ohlc()
    vol = df['volume'].resample(rule).sum()
    count = df['price'].resample(rule).count()

    candles = ohlc.join(vol.rename('volume')).join(count.rename('trades'))
    candles = candles.dropna(subset=['open', 'high', 'low', 'close'])

    candles.reset_index(inplace=True)
    candles.rename(columns={'time': 'timestamp'}, inplace=True)
    return candles
