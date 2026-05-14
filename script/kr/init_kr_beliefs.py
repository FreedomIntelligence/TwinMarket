import argparse
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "삼성전자와 한국 주식시장에 대해 중립적인 관점을 유지하며, "
    "현재 정보가 부족하여 추가 뉴스와 가격 흐름을 관찰한다."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", default="util/belief/belief_100.csv")
    p.add_argument("--out_csv", default="util/belief/belief_100_kr.csv")
    args = p.parse_args()

    src = Path(args.in_csv)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if "belief" in df.columns:
        df["belief"] = SYSTEM_PROMPT
    if "attitude" in df.columns:
        # 기존 분포/필드는 유지
        df["attitude"] = df["attitude"].fillna("중립")

    df.to_csv(out, index=False)
    print(f"saved {out} rows={len(df)}")


if __name__ == "__main__":
    main()
