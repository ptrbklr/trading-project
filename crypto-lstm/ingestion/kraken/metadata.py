# ingestion/kraken/metadata.py

def resolve_kraken_pair(pair: str) -> str:
    """Resolve configured asset names to Kraken API pair names."""
    pair = pair.upper().replace("/", "")
    return {
        "BTC": "XXBTZEUR",
    }.get(pair, pair)


def normalize_pair_name(pair: str) -> str:
    """
    Normalize Kraken pair names into clean base-asset names for filenames.
    Examples:
        BTCEUR → BTC
        XXBTZEUR → BTC
        XETHZUSD → ETH
        XRPUSD → XRP
    """

    p = pair.upper().replace("/", "_")

    # Kraken base-asset prefixes
    prefix_map = {
        "XXBT": "BTC",
        "XETH": "ETH",
        "ETH":  "ETH",
        "XXRP": "XRP",
        "XRP":  "XRP",
        "XLTC": "LTC",
        "LTC":  "LTC",
        "XXLM": "XLM",
        "XLM":  "XLM",
        "XXMR": "XMR",
        "XMR":  "XMR",
        "XXDG": "DOGE",
        "DOGE": "DOGE",
        "ADA":  "ADA",
        "DOT":  "DOT",
        "LINK": "LINK",
        "SOL":  "SOL",
    }

    # Kraken quote-asset suffixes
    suffixes = [
        "ZUSD", "ZEUR", "ZGBP", "ZJPY", "ZCAD", "ZAUD", "ZCHF",
        "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"
    ]

    # Try to match prefix
    for kraken_prefix, clean in prefix_map.items():
        if p.startswith(kraken_prefix):
            return clean

    # Try to match suffix-based pairs (e.g., BTCEUR)
    for suffix in suffixes:
        if p.endswith(suffix):
            base = p.replace(suffix, "")
            # If base matches known assets, return it
            if base in prefix_map.values():
                return base

    # Fallback: return raw pair without suffixes
    for suffix in suffixes:
        if p.endswith(suffix):
            return p.replace(suffix, "")

    return p
