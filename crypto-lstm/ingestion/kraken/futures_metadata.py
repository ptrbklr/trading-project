def resolve_futures_symbol(pair: str) -> str:
    """Resolve configured asset names to Kraken Futures perpetual ticker symbols."""
    pair = pair.upper().replace("/", "")
    return {
        "BTC": "PF_XBTUSD",
        "ETH": "PF_ETHUSD",
    }.get(pair, f"PF_{pair}USD")
