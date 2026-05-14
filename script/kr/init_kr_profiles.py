import argparse
import shutil
import sqlite3
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_db", default="data/sys_1000.db")
    p.add_argument("--dst_db", default="data/sys_100_kr.db")
    args = p.parse_args()

    src = Path(args.src_db)
    dst = Path(args.dst_db)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)

    conn = sqlite3.connect(dst)
    cur = conn.cursor()
    cur.execute("DELETE FROM TradingDetails")

    latest_sql = """
    SELECT p.*
    FROM Profiles p
    JOIN (
      SELECT user_id, MAX(created_at) AS latest_created_at
      FROM Profiles
      GROUP BY user_id
    ) t
      ON p.user_id = t.user_id AND p.created_at = t.latest_created_at
    """
    rows = cur.execute(latest_sql).fetchall()
    cols = [d[0] for d in cur.description]
    idx = {c: i for i, c in enumerate(cols)}

    for r in rows:
        user_id = r[idx["user_id"]]
        created_at = r[idx["created_at"]]
        original_ini_cash = float(r[idx.get("ini_cash")])
        new_ini_cash = 100_000_000 if original_ini_cash <= 10_000_000 else 1_000_000_000

        cur.execute(
            """
            UPDATE Profiles
            SET ini_cash=?, current_cash=?, total_value=?, total_return=?, return_rate=?,
                cur_positions=?, initial_positions=?, stock_returns=?, yest_returns=?, fol_ind=?
            WHERE user_id=? AND created_at=?
            """,
            (
                new_ini_cash,
                new_ini_cash,
                new_ini_cash,
                0,
                0.0,
                "{}",
                '{"005930": 0}',
                "{}",
                "{}",
                '["전기전자", "반도체"]',
                user_id,
                created_at,
            ),
        )

    conn.commit()
    conn.close()
    print(f"initialized KR db: {dst}")


if __name__ == "__main__":
    main()
