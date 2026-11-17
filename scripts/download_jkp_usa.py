#!/usr/bin/env python3
"""Download US Jensen-Kelly-Pedersen (JKP) stock characteristics from WRDS.

Produces data/raw/usa.csv in the format the MFD replication code expects
(programs/00_preparation/process_stock_characteristics_returns.py):
  * one row per stock-month, excntry = 'USA'
  * date and eom as integer YYYYMMDD (e.g. 19660731)

Source: WRDS table contrib_global_factor.global_factor (JKP Global Factor Data).

How it is fast:
  * one query per calendar year, run on several connections in parallel
  * each query uses Postgres COPY ... TO STDOUT (CSV), much faster than
    row-by-row fetching through pandas.read_sql
  * filters (US, sample period, CRSP common stock, NYSE/AMEX/Nasdaq) are
    applied on the server, so only needed rows cross the network

Resumable: each year is saved to data/raw/jkp_usa_parts/year=YYYY.parquet;
finished years are skipped on rerun.

Login: uses ~/.pgpass (host wrds-pgdata.wharton.upenn.edu, port 9737).
Connects with psycopg2 directly; the wrds package's raw_sql breaks on '%' with
pandas >= 2.3.

Usage (from the project root, in the mfd conda env):
  python scripts/download_jkp_usa.py                     # 1966-2025, paper columns
  python scripts/download_jkp_usa.py --all-columns       # all 444 JKP columns
  python scripts/download_jkp_usa.py --start 2020 --end 2020 --out /tmp/test
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TABLE = "contrib_global_factor.global_factor"
HOST, PORT, DBNAME = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"

# Identifier / filter / return columns used anywhere in the authors' programs
# (usa.csv readers in 00_preparation, 01_run_mfd and 02_analysis).
BASE_COLS = [
    "id", "date", "eom", "permno", "permco", "gvkey", "iid", "excntry",
    "source_crsp", "common", "primary_sec", "obs_main", "exch_main",
    "crsp_shrcd", "crsp_exchcd", "sic", "ff49", "size_grp",
    "prc", "me", "market_equity", "assets", "dolvol",
    "ret", "ret_exc", "ret_exc_lead1m",
]


def connect(user: str | None):
    return psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=user,
                            sslmode="require", connect_timeout=60)


def table_columns(con) -> list[str]:
    with con.cursor() as cur:
        cur.execute(
            "select column_name from information_schema.columns "
            "where table_schema = %s and table_name = %s order by ordinal_position",
            TABLE.split("."))
        return [r[0] for r in cur.fetchall()]


def jkp_characteristics(factor_details: Path) -> list[str]:
    return pd.read_excel(factor_details)["abr_jkp"].dropna().tolist()


def build_query(cols: list[str], year: int, start: str, end: str, universe_filter: bool) -> str:
    where = [
        "excntry = 'USA'",
        f"eom >= DATE '{max(start, f'{year}-01-01')}'",
        f"eom <= DATE '{min(end, f'{year}-12-31')}'",
    ]
    if universe_filter:  # same filters as the authors' prep script
        where += ["crsp_shrcd in (10, 11)", "crsp_exchcd in (1, 2, 3)"]
    sel = ", ".join(f'"{c}"' for c in cols)
    return (f"COPY (SELECT {sel} FROM {TABLE} WHERE {' AND '.join(where)} "
            f"ORDER BY eom, permno) TO STDOUT WITH (FORMAT csv, HEADER true)")


def fetch_year(year: int, cols: list[str], args, parts_dir: Path) -> tuple[int, int, float]:
    out = parts_dir / f"year={year}.parquet"
    if out.exists():
        return year, pq.ParquetFile(out).metadata.num_rows, 0.0
    t0 = time.time()
    for attempt in range(5):
        try:
            con = connect(args.user)
            try:
                buf = io.BytesIO()
                with con.cursor() as cur:
                    cur.copy_expert(build_query(cols, year, args.start_date, args.end_date,
                                                not args.no_universe_filter), buf)
            finally:
                con.close()
            break
        except psycopg2.OperationalError as e:
            wait = 2 ** attempt * 5
            print(f"  {year}: connection error ({e.__class__.__name__}), retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"year {year} failed after retries")

    buf.seek(0)
    str_cols = [c for c in ("gvkey", "iid", "excntry", "id") if c in cols]
    tbl = pacsv.read_csv(buf, convert_options=pacsv.ConvertOptions(
        column_types={c: pa.string() for c in str_cols}))
    df = tbl.to_pandas()
    for c in ("date", "eom"):  # 'YYYY-MM-DD' -> int YYYYMMDD, as in JKP's usa.csv
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y%m%d").astype("Int64")
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return year, len(df), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1966, help="first year (default 1966)")
    ap.add_argument("--end", type=int, default=2025, help="last year (default 2025)")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"), help="output folder for usa.csv")
    ap.add_argument("--user", default=os.environ.get("WRDS_USER"), help="WRDS username (else from ~/.pgpass)")
    ap.add_argument("--workers", type=int, default=4, help="parallel WRDS connections (WRDS caps per-user sessions)")
    ap.add_argument("--all-columns", action="store_true", help="download all JKP columns, not just those used")
    ap.add_argument("--no-universe-filter", action="store_true",
                    help="keep all share/exchange codes (default keeps shrcd 10/11, exchcd 1/2/3)")
    ap.add_argument("--factor-details", default=None,
                    help="Factor Details.xlsx listing the 153 characteristics (default: data/raw, then the replication package)")
    ap.add_argument("--no-csv", action="store_true", help="only write usa.parquet, skip usa.csv")
    args = ap.parse_args()
    args.start_date, args.end_date = f"{args.start}-01-01", f"{args.end}-12-31"

    out_dir = Path(args.out)
    parts_dir = out_dir / "jkp_usa_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    con = connect(args.user)
    available = table_columns(con)
    con.close()

    if args.all_columns:
        cols = available
    else:
        fd = Path(args.factor_details) if args.factor_details else out_dir / "Factor Details.xlsx"
        if not fd.exists():
            fd = ROOT / "data" / "pkg_extracted" / "data" / "raw" / "Factor Details.xlsx"
        chars = jkp_characteristics(fd)
        cols = list(dict.fromkeys(BASE_COLS + chars))
        missing = [c for c in cols if c not in available]
        if missing:
            sys.exit(f"Columns not in {TABLE}: {missing}")
        print(f"{len(chars)} characteristics + {len(BASE_COLS)} id/return columns from {fd.name}")

    years = list(range(args.start, args.end + 1))
    print(f"Downloading {TABLE}, USA, {args.start}-{args.end}, {len(cols)} columns, "
          f"{args.workers} connections -> {parts_dir}")
    t0, total = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_year, y, cols, args, parts_dir) for y in years]
        for f in as_completed(futs):
            y, n, dt = f.result()
            total += n
            print(f"  {y}: {n:>7,} rows" + (f"  ({dt:.0f}s)" if dt else "  (already downloaded)"))
    print(f"Fetched {total:,} rows in {time.time() - t0:.0f}s")

    # combine years, in order
    tables = [pq.read_table(parts_dir / f"year={y}.parquet") for y in years]
    tbl = pa.concat_tables(tables, promote_options="default")
    pq.write_table(tbl, out_dir / "usa.parquet")
    if not args.no_csv:
        pacsv.write_csv(tbl, out_dir / "usa.csv")

    df = tbl.select(["eom", "permno"]).to_pandas()
    per_month = df.groupby("eom").size()
    print(f"\nWrote {out_dir/'usa.parquet'}" + ("" if args.no_csv else f" and {out_dir/'usa.csv'}"))
    print(f"  rows {len(df):,}  months {per_month.size}  ({per_month.index.min()} .. {per_month.index.max()})")
    print(f"  stocks/month: median {int(per_month.median()):,}, min {per_month.min():,}, max {per_month.max():,}")
    print(f"  unique permnos {df.permno.nunique():,}")


if __name__ == "__main__":
    main()
