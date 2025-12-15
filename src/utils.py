import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    files = []
    for name in os.listdir(folder):
        p = folder / name
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def safe_makedirs(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_label(label: str) -> str:
    # Normalize HF labels like "card_board", "Cardboard", "paper " → "cardboard"
    s = label.strip().lower()
    s = s.replace(" ", "").replace("_", "")
    return s


def map_to_classes(label: str, classes: List[str]) -> str:
    nl = normalize_label(label)
    for c in classes:
        if normalize_label(c) == nl:
            return c
    # fallback
    return "trash"
