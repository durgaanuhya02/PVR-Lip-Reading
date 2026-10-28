"""Download a list of archive URLs (e.g. the personal LRS-VoxMM links from the KAIST form),
resume interrupted downloads, unpack them and delete each archive after unpacking.

The URLs are read from an environment variable (one per line or comma separated) so that
personal access links never appear in code or notebook output; on Kaggle store them as a
secret named VOXMM_URLS.

Usage:
    VOXMM_URLS="https://...\nhttps://..." python -m src.preprocess.download_urls --env VOXMM_URLS --out /tmp/lrs_voxmm
"""
import argparse
import os
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path


def filename_for(url, i):
    name = Path(urllib.parse.urlparse(url).path).name
    return name or f"part_{i:02d}"


def download(url, dest):
    # curl: follow redirects, resume (-C -), retry; --fail so HTTP errors are not saved as files
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "-C", "-",
           "-sS", "-o", str(dest), url]
    subprocess.run(cmd, check=True)


def unpack(path, out_dir):
    name = path.name.lower()
    if name.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        shutil.unpack_archive(str(path), str(out_dir))
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="VOXMM_URLS", help="environment variable holding the URLs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    urls = [u.strip() for u in re.split(r"[\n,]+", os.environ.get(args.env, "")) if u.strip()]
    if not urls:
        raise SystemExit(f"no URLs found in ${args.env}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archives = out / "_archives"
    archives.mkdir(exist_ok=True)

    for i, url in enumerate(urls):
        dest = archives / filename_for(url, i)
        print(f"[{i + 1}/{len(urls)}] downloading {dest.name}", flush=True)  # never print the URL
        download(url, dest)
        size = dest.stat().st_size / 1e9
        if unpack(dest, out):
            print(f"    unpacked {dest.name} ({size:.2f} GB)", flush=True)
            if not args.keep_archives:
                dest.unlink()
        else:
            print(f"    kept {dest.name} ({size:.2f} GB) - not a recognised archive", flush=True)
    print("done; top-level contents:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
