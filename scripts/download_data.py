"""下载并解压 FETAL_PLANES_DB 数据集（Zenodo 记录 3904280）。

数据规模约 2.1 GB，下载完成后会自动解压到 data/ 目录。

用法:
    python scripts/download_data.py            # 下载到项目 data/ 目录
    python scripts/download_data.py --out /path/to/data
"""
from __future__ import annotations

import argparse
import os
import zipfile

import requests
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1"
EXPECTED_MD5 = "2a5fcc2cefb789bcc0f6c1f73e0ea43f"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def download(url: str, dest: str) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                     desc="下载 FETAL_PLANES_ZENODO.zip") as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))


def verify_md5(path: str) -> bool:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f, tqdm(desc="校验 MD5", unit="B", unit_scale=True) as bar:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
            bar.update(len(chunk))
    ok = h.hexdigest() == EXPECTED_MD5
    print(f"MD5: {h.hexdigest()}  期望: {EXPECTED_MD5}  -> {'通过' if ok else '不一致'}")
    return ok


def extract(zip_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print("正在解压...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"解压完成 -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="下载 FETAL_PLANES_DB 数据集")
    parser.add_argument("--out", default=None, help="输出目录（默认项目 data/）")
    parser.add_argument("--skip-verify", action="store_true", help="跳过 MD5 校验")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(project_root, "data")
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "FETAL_PLANES_ZENODO.zip")

    if not os.path.exists(zip_path):
        download(ZENODO_URL, zip_path)
    else:
        print(f"已存在 {zip_path}，跳过下载")

    if not args.skip_verify:
        verify_md5(zip_path)

    extract(zip_path, out_dir)
    print("完成。数据集应位于 data/FETAL_PLANES_ZENODO/ 下。")


if __name__ == "__main__":
    main()
