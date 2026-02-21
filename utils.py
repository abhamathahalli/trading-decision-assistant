import pandas as pd

REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names (strip spaces)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    # Handle common "date" column name variations
    if "Date" not in df.columns:
        for c in df.columns:
            if c.lower() == "date":
                df.rename(columns={c: "Date"}, inplace=True)
                break

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Need: {REQUIRED_COLS}")

    # Parse date and sort
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")

    # Ensure numeric columns
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df.reset_index(drop=True)