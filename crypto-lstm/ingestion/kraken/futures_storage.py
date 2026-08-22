import os
import pandas as pd

class FuturesSnapshotStorage:
    COLUMNS = ["timestamp", "mark_price", "index_price", "funding_rate", "open_interest", "volume_24h"]

    def __init__(self, base_dir, pair_name):
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, f"{pair_name}_futures.csv")

    def append(self, row: dict):
        df_row = pd.DataFrame([row], columns=self.COLUMNS)
        write_header = not os.path.exists(self.path)
        df_row.to_csv(self.path, mode="a", header=write_header, index=False)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.read_csv(self.path, parse_dates=["timestamp"])
