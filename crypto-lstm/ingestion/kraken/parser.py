import pandas as pd

def parse_trades(raw_trades):
    cols_7 = ['price', 'volume', 'time', 'side', 'type', 'misc', 'trade_id']
    cols_6 = ['price', 'volume', 'time', 'side', 'type', 'misc']

    if not raw_trades:
        return pd.DataFrame(columns=cols_6)

    cols = cols_7 if len(raw_trades[0]) == 7 else cols_6
    df = pd.DataFrame(raw_trades, columns=cols)

    if 'trade_id' in df.columns:
        df = df.drop(columns=['trade_id'])

    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df
