import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fetch_ohlcv(start: str, end: str, ticker: str = "005930") -> pd.DataFrame:
    try:
        import FinanceDataReader as fdr

        df = fdr.DataReader(ticker, start, end)
        if df.empty:
            raise ValueError("empty OHLCV from FinanceDataReader")
        df = df.rename(
            columns={"Close": "close_price", "Volume": "vol", "Open": "open", "High": "high", "Low": "low"}
        )
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        from pykrx import stock

        raw = stock.get_market_ohlcv_by_date(start.replace("-", ""), end.replace("-", ""), ticker)
        if raw.empty:
            raise ValueError("empty OHLCV from pykrx")
        raw.index = pd.to_datetime(raw.index)
        return raw.rename(columns={"종가": "close_price", "거래량": "vol", "시가": "open", "고가": "high", "저가": "low"})


def _fetch_fundamental(start: str, end: str, ticker: str = "005930") -> pd.DataFrame:
    try:
        from pykrx import stock

        f = stock.get_market_fundamental_by_date(start.replace("-", ""), end.replace("-", ""), ticker)
        f.index = pd.to_datetime(f.index)
        return f.rename(columns={"PER": "pe_ttm", "PBR": "pb", "DIV": "dv_ttm"})
    except Exception:
        return pd.DataFrame()


def _fetch_net_amount(start: str, end: str, ticker: str = "005930") -> pd.Series:
    try:
        from pykrx import stock

        n = stock.get_market_trading_value_by_date(start.replace("-", ""), end.replace("-", ""), ticker)
        n.index = pd.to_datetime(n.index)
        col_map = {c: c for c in n.columns}
        inst_col = next((c for c in col_map if "기관" in c), None)
        foreign_col = next((c for c in col_map if "외국" in c), None)
        if inst_col and foreign_col:
            return n[inst_col].fillna(0) + n[foreign_col].fillna(0)
    except Exception:
        pass
    return pd.Series(dtype=float)


def build_stock_data(start: str, end: str, out_path: Path) -> pd.DataFrame:
    ohlcv = _fetch_ohlcv(start, end)
    df = pd.DataFrame(index=ohlcv.index)
    df["stock_id"] = "005930"
    df["date"] = df.index.strftime("%Y-%m-%d")
    df["close_price"] = pd.to_numeric(ohlcv["close_price"], errors="coerce")
    df["pre_close"] = df["close_price"].shift(1)
    df["change"] = df["close_price"] - df["pre_close"]
    df["pct_chg"] = np.where(df["pre_close"] > 0, (df["change"] / df["pre_close"]) * 100.0, 0.0)

    fund = _fetch_fundamental(start, end)
    df["pe_ttm"] = fund.get("pe_ttm", pd.Series(index=df.index, dtype=float))
    df["pb"] = fund.get("pb", pd.Series(index=df.index, dtype=float))
    df["dv_ttm"] = fund.get("dv_ttm", pd.Series(index=df.index, dtype=float))
    df["ps_ttm"] = 0.0

    df["vol"] = pd.to_numeric(ohlcv["vol"], errors="coerce").fillna(0.0)
    for w in (5, 10, 30):
        df[f"vol_{w}"] = df["vol"].rolling(w, min_periods=1).mean()
        df[f"ma_hfq_{w}"] = df["close_price"].rolling(w, min_periods=1).mean()

    net = _fetch_net_amount(start, end)
    df["elg_amount_net"] = net.reindex(df.index).fillna(0.0)

    for c in ["pe_ttm", "pb", "ps_ttm", "dv_ttm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    cols = [
        "stock_id", "date", "close_price", "pre_close", "change", "pct_chg", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
        "vol", "vol_5", "vol_10", "vol_30", "ma_hfq_5", "ma_hfq_10", "ma_hfq_30", "elg_amount_net",
    ]
    out = df[cols].copy().fillna(0)
    out.to_csv(out_path, index=False)
    return out


def build_trading_days(stock_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    cal = stock_df[["date"]].rename(columns={"date": "cal_date"}).copy()
    cal["is_open"] = 1
    cal["pretrade_date"] = cal["cal_date"].shift(1).fillna("")
    cal.to_csv(out_path, index=False)
    return cal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-12-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--out_stock", default="data/stock_data_kr.csv")
    parser.add_argument("--out_calendar", default="data/trading_days_kr.csv")
    args = parser.parse_args()

    stock_path = Path(args.out_stock)
    cal_path = Path(args.out_calendar)
    stock_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.parent.mkdir(parents=True, exist_ok=True)

    stock_df = build_stock_data(args.start, args.end, stock_path)
    build_trading_days(stock_df, cal_path)
    print(f"saved {stock_path} rows={len(stock_df)}")
    print(f"saved {cal_path} rows={len(stock_df)}")


if __name__ == "__main__":
    main()
