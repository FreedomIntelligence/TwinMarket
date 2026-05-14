import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _normalize_news_item(item: Any) -> List[str]:
    if isinstance(item, list):
        if all(isinstance(x, str) for x in item):
            return [x.strip() for x in item if x and x.strip()]
        out = []
        for obj in item:
            if isinstance(obj, dict):
                title = str(obj.get("title", "")).strip()
                body = str(obj.get("content", obj.get("body", ""))).strip()
                txt = " ".join([x for x in [title, body] if x]).strip()
                if txt:
                    out.append(txt)
        return out
    if isinstance(item, str):
        txt = item.strip()
        return [txt] if txt else []
    return []


def build_news(input_pkl: Path, output_pkl: Path, trading_days_csv: Path) -> pd.DataFrame:
    df = pd.read_pickle(input_pkl)
    if "cal_date" not in df.columns or "news" not in df.columns:
        raise ValueError("input pkl must include cal_date and news columns")

    norm = pd.DataFrame()
    norm["cal_date"] = pd.to_datetime(df["cal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    norm = norm.dropna(subset=["cal_date"])
    norm["news"] = df.loc[norm.index, "news"].map(_normalize_news_item)

    if trading_days_csv.exists():
        cal = pd.read_csv(trading_days_csv)
        cal_dates = pd.to_datetime(cal["cal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        base = pd.DataFrame({"cal_date": cal_dates.dropna().unique()})
        merged = base.merge(norm, on="cal_date", how="left")
        merged["news"] = merged["news"].apply(lambda x: x if isinstance(x, list) else [])
        norm = merged.sort_values("cal_date").reset_index(drop=True)
    else:
        norm = norm.sort_values("cal_date").reset_index(drop=True)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    norm.to_pickle(output_pkl)
    return norm


def build_information_db(news_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    docs: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for _, row in news_df.iterrows():
        cal_date = row["cal_date"]
        for text in row["news"]:
            docs.append(text)
            metadata.append({"datetime": cal_date, "title": "", "content": text, "type": "kr_news", "source": "samsung"})

    if not docs:
        index = faiss.IndexFlatIP(384)
    else:
        emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

    with open(output_dir / "faiss_index.pkl", "wb") as f:
        pickle.dump(index, f)
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_pkl", default="data/samsung_mk_news_20260201_20260331.pkl")
    p.add_argument("--output_pkl", default="data/samsung_news.pkl")
    p.add_argument("--trading_days", default="data/trading_days_kr.csv")
    p.add_argument("--output_db", default="data/InformationDB_samsung")
    args = p.parse_args()

    news_df = build_news(Path(args.input_pkl), Path(args.output_pkl), Path(args.trading_days))
    build_information_db(news_df, Path(args.output_db))
    print(f"saved news rows={len(news_df)} -> {args.output_pkl}")
    print(f"saved information db -> {args.output_db}")


if __name__ == "__main__":
    main()
