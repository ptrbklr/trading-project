import os
import pickle
from datetime import datetime

class TradeStorage:
    def __init__(self, base_dir, pair_name):
        self.path = os.path.join(base_dir, f"{pair_name}_raw.pkl")
        os.makedirs(base_dir, exist_ok=True)

    def load(self):
        if not os.path.exists(self.path):
            return None
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def save(self, trades):
        with open(self.path, "wb") as f:
            pickle.dump(trades, f)

    def metadata(self, trades):
        if not trades:
            return None
        timestamps = [t[2] for t in trades]
        return {
            "count": len(trades),
            "start": datetime.utcfromtimestamp(min(timestamps)),
            "end": datetime.utcfromtimestamp(max(timestamps)),
        }
