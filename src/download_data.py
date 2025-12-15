import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from .config import CFG
from .utils import safe_makedirs


def download_file(url: str, out_path: Path) -> None:
    safe_makedirs(out_path.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main():
    zip_path = CFG.DATA_DIR.parent / "dataset-resized.zip"
    print(f"Downloading TrashNet zip to: {zip_path}")
    download_file(CFG.HF_DATASET_ZIP_URL, zip_path)

    print(f"Extracting to: {CFG.DATA_DIR}")
    safe_makedirs(CFG.DATA_DIR)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CFG.DATA_DIR)

    # The zip usually contains a folder named "dataset-resized" inside
    print("Done.")
    print("Tip: dataset images should be under:")
    print("  data/trashnet/dataset-resized/<class_name>/*.jpg")


if __name__ == "__main__":
    main()
