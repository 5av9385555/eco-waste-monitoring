from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # HF model id (can be swapped)
    MODEL_ID: str = "prithivMLmods/Trash-Net"

    # Data locations
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data" / "trashnet"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    # Dataset source (HF mirror) - downloads dataset-resized.zip
    HF_DATASET_ZIP_URL: str = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"

    # Class names (TrashNet standard order)
    CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CFG = Config()
