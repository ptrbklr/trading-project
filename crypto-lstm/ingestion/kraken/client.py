import time
import requests

class KrakenClient:
    BASE_URL = "https://api.kraken.com/0/public/Trades"

    def __init__(self, min_interval=1.0, max_retries=5):
        self.last_request = 0
        self.min_interval = min_interval
        self.max_retries = max_retries

    def _rate_limit(self):
        delta = time.time() - self.last_request
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last_request = time.time()

    def fetch_trades(self, pair, since=None, count=1000):
        params = {"pair": pair, "count": min(count, 1000)}
        if since is not None:
            params["since"] = int(since * 1e9) if since < 1e12 else int(since)

        retries = 0
        while retries < self.max_retries:
            try:
                self._rate_limit()
                r = requests.get(self.BASE_URL, params=params)
                r.raise_for_status()
                data = r.json()

                if data.get("error"):
                    msg = data["error"][0]
                    if "Too many requests" in msg:
                        time.sleep(5 * (retries + 1))
                        retries += 1
                        continue
                    raise RuntimeError(msg)

                # Kraken may rename the pair in the response (e.g. XRPEUR -> XXRPZEUR)
                result_key = next(k for k in data["result"] if k != "last")
                return data["result"][result_key], data["result"].get("last")

            except Exception:
                retries += 1
                time.sleep(2 * retries)

        raise RuntimeError("Max retries exceeded")
