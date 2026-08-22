import time
import requests

class KrakenFuturesClient:
    # https://futures.kraken.com/derivatives/api/v3 (public, no auth required for tickers)
    BASE_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"

    def __init__(self, min_interval=1.0, max_retries=5):
        self.last_request = 0
        self.min_interval = min_interval
        self.max_retries = max_retries

    def _rate_limit(self):
        delta = time.time() - self.last_request
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last_request = time.time()

    def fetch_ticker(self, symbol):
        retries = 0
        while retries < self.max_retries:
            try:
                self._rate_limit()
                r = requests.get(self.BASE_URL)
                r.raise_for_status()
                data = r.json()

                if data.get("result") != "success":
                    raise RuntimeError(f"Kraken Futures API error: {data}")

                for ticker in data.get("tickers", []):
                    if ticker.get("symbol", "").upper() == symbol.upper():
                        return ticker

                raise RuntimeError(f"Symbol {symbol} not found in tickers response")

            except Exception:
                retries += 1
                time.sleep(2 * retries)

        raise RuntimeError("Max retries exceeded")
