import requests
import pandas as pd
from datetime import datetime, timezone
import time


def main():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum,solana",  # TODO: comma-separated coin names
        "vs_currencies": "usd",  # TODO: currency
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()  # TODO: raise_for_statu
        time.sleep(0.3)  # TODO: polite delay
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error recup de data: {e}")  # TODO: print error and return
        return
    rows = [
        {
            "coin": coin,
            "price_usd": values["usd"],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        for coin, values in data.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv("prices.csv", index=False)
    print(f"Saved {len(df)} rows to prices.csv")


if __name__ == "__main__":
    main()
