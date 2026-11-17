#!/usr/bin/env python3
"""Fast parallel downloader for the MFD replication package on Harvard Dataverse.

Bali, Kelly, Moerke, Rahman - Machine Forecast Disagreement
doi:10.7910/DVN/BO8SQ8

How it is fast:
  * Each file is split into byte-range chunks (default 64 MB).
  * All chunks of all files go into one thread pool, so big files use many
    connections at once and small files don't wait in line.
  * Dataverse 303-redirects to an S3 presigned URL; we resolve it once per file
    and re-resolve when it expires (403), avoiding a redirect hop per chunk.

Robustness:
  * Resumable: finished chunks are recorded in <file>.progress; rerun to resume.
  * Retries with backoff per chunk.
  * MD5 of every completed file is checked against Dataverse's checksum.

Usage:
  python scripts/download_replication.py --out data/replication_package
  python scripts/download_replication.py --only programs.7z ReadMe.pdf   # small files first
  python scripts/download_replication.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/BO8SQ8"
UA = "dataverse-parallel-downloader/1.0"


def list_files(session: requests.Session) -> list[dict]:
    r = session.get(f"{BASE}/api/datasets/:persistentId/", params={"persistentId": DOI}, timeout=60)
    r.raise_for_status()
    out = []
    for f in r.json()["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        out.append({
            "id": df["id"],
            "name": df["filename"],
            "dir": f.get("directoryLabel", ""),
            "size": df["filesize"],
            "md5": (df.get("checksum") or {}).get("value") if (df.get("checksum") or {}).get("type") == "MD5" else df.get("md5"),
        })
    return out


class FileJob:
    """State for one file: presigned URL cache, chunk bookkeeping, progress file."""

    def __init__(self, meta: dict, out_dir: str, chunk_size: int):
        self.meta = meta
        self.path = os.path.join(out_dir, meta["dir"], meta["name"])
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.part = self.path + ".part"
        self.progress_path = self.path + ".progress"
        self.size = meta["size"]
        self.chunks = [(s, min(s + chunk_size, self.size) - 1) for s in range(0, self.size, chunk_size)] or [(0, -1)]
        self.lock = threading.Lock()
        self.url: str | None = None
        self.url_time = 0.0
        self.done: set[int] = set()
        if os.path.exists(self.progress_path) and os.path.exists(self.part):
            with open(self.progress_path) as fh:
                saved = json.load(fh)
            if saved.get("size") == self.size and saved.get("chunk_size") == chunk_size:
                self.done = set(saved["done"])
        self.chunk_size = chunk_size
        if not os.path.exists(self.part):
            with open(self.part, "wb") as fh:
                fh.truncate(self.size)  # sparse preallocation

    def pending(self) -> list[int]:
        return [i for i in range(len(self.chunks)) if i not in self.done]

    def get_url(self, session: requests.Session, force: bool = False) -> str:
        with self.lock:
            # presigned URLs last 3600 s; refresh well before that
            if force or self.url is None or time.time() - self.url_time > 2400:
                r = session.get(f"{BASE}/api/access/datafile/{self.meta['id']}",
                                allow_redirects=False, timeout=60)
                if r.status_code in (301, 302, 303, 307, 308):
                    self.url = r.headers["Location"]
                elif r.status_code == 200:  # served directly, no redirect
                    self.url = f"{BASE}/api/access/datafile/{self.meta['id']}"
                else:
                    r.raise_for_status()
                self.url_time = time.time()
            return self.url

    def mark_done(self, idx: int) -> None:
        with self.lock:
            self.done.add(idx)
            tmp = self.progress_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"size": self.size, "chunk_size": self.chunk_size, "done": sorted(self.done)}, fh)
            os.replace(tmp, self.progress_path)

    def complete(self) -> bool:
        return len(self.done) == len(self.chunks)


class Progress:
    def __init__(self, total: int):
        self.total, self.got, self.t0 = total, 0, time.time()
        self.lock = threading.Lock()
        self.last = 0.0

    def add(self, n: int) -> None:
        with self.lock:
            self.got += n
            now = time.time()
            if now - self.last > 1 or self.got >= self.total:
                self.last = now
                rate = self.got / max(now - self.t0, 1e-9)
                eta = (self.total - self.got) / rate if rate > 0 else float("inf")
                sys.stderr.write(f"\r  {self.got/1e9:7.2f} / {self.total/1e9:.2f} GB  "
                                 f"{rate/1e6:7.1f} MB/s  ETA {eta/60:5.1f} min   ")
                sys.stderr.flush()


thread_local = threading.local()


def session() -> requests.Session:
    if not hasattr(thread_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = UA
        thread_local.s = s
    return thread_local.s


def fetch_chunk(job: FileJob, idx: int, prog: Progress, retries: int = 8) -> None:
    start, end = job.chunks[idx]
    if end < start:  # empty file
        job.mark_done(idx)
        return
    for attempt in range(retries):
        s = session()
        written = 0
        try:
            url = job.get_url(s, force=attempt > 0)
            with s.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 120)) as r:
                if r.status_code == 200 and (start != 0 or end != job.size - 1):
                    raise IOError("server ignored Range header")
                if r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code}")
                fd = os.open(job.part, os.O_WRONLY)
                try:
                    pos = start
                    for buf in r.iter_content(1 << 20):
                        os.pwrite(fd, buf, pos)
                        pos += len(buf)
                        written += len(buf)
                        prog.add(len(buf))
                finally:
                    os.close(fd)
            if written != end - start + 1:
                raise IOError(f"short read {written} != {end - start + 1}")
            job.mark_done(idx)
            return
        except Exception as e:  # noqa: BLE001
            prog.add(-written)
            wait = min(2 ** attempt, 60)
            sys.stderr.write(f"\n  [retry {attempt+1}/{retries}] {job.meta['name']} chunk {idx}: {e}; sleeping {wait}s\n")
            time.sleep(wait)
    raise RuntimeError(f"failed {job.meta['name']} chunk {idx}")


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(8 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def finalize(job: FileJob) -> str:
    if job.meta["md5"]:
        got = md5sum(job.part)
        if got != job.meta["md5"]:
            os.remove(job.progress_path)  # force full re-download next run
            return f"MD5 MISMATCH {job.meta['name']}: {got} != {job.meta['md5']}"
    os.replace(job.part, job.path)
    if os.path.exists(job.progress_path):
        os.remove(job.progress_path)
    return f"OK  {job.meta['name']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/replication_package")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--only", nargs="*", help="download only these filenames")
    ap.add_argument("--list", action="store_true", help="list files and exit")
    args = ap.parse_args()

    files = list_files(session())
    if args.list:
        for f in files:
            print(f"{f['size']/1e6:10.1f} MB  {f['name']}")
        print(f"{sum(f['size'] for f in files)/1e9:10.2f} GB total")
        return
    if args.only:
        files = [f for f in files if f["name"] in set(args.only)]

    os.makedirs(args.out, exist_ok=True)
    jobs, skipped = [], []
    for f in files:
        dest = os.path.join(args.out, f["dir"], f["name"])
        if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
            skipped.append(f["name"])
            continue
        jobs.append(FileJob(f, args.out, args.chunk_mb << 20))
    if skipped:
        print(f"Already present: {', '.join(skipped)}")

    total = sum(j.chunks[i][1] - j.chunks[i][0] + 1 for j in jobs for i in j.pending())
    print(f"Downloading {len(jobs)} files, {total/1e9:.2f} GB remaining, {args.workers} threads")
    prog = Progress(total)

    # small files first so README/programs arrive immediately
    tasks = [(j, i) for j in sorted(jobs, key=lambda j: j.size) for i in j.pending()]
    failed = False
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_chunk, j, i, prog): j for j, i in tasks}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                failed = True
                sys.stderr.write(f"\n  ERROR {e}\n")
                continue
            if j.complete() and not getattr(j, "finalized", False):
                j.finalized = True
                sys.stderr.write("\n")
                print(finalize(j))
    # jobs with zero pending chunks (fully downloaded on a prior run but not finalized)
    for j in jobs:
        if j.complete() and not getattr(j, "finalized", False):
            print(finalize(j))
    print("\nDone." if not failed else "\nSome chunks failed; rerun to resume.")


if __name__ == "__main__":
    main()
