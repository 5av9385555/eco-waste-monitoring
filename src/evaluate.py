import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import pipeline

from .config import CFG
from .utils import list_images, load_rgb, map_to_classes, safe_makedirs


def collect_dataset(root: Path, classes: List[str]) -> Tuple[List[Path], List[str]]:
    paths, labels = [], []
    for cls in classes:
        folder = root / cls
        if not folder.exists():
            continue
        for p in list_images(folder):
            paths.append(p)
            labels.append(cls)
    return paths, labels


def main():
    # Locate dataset folder
    # After extraction, typical path: data/trashnet/dataset-resized/<class>/*
    dataset_root = CFG.DATA_DIR / "dataset-resized"
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. Run: python -m src.download_data"
        )

    safe_makedirs(CFG.REPORTS_DIR)

    print(f"Loading pretrained model: {CFG.MODEL_ID}")
    clf = pipeline("image-classification", model=CFG.MODEL_ID)

    img_paths, y_true = collect_dataset(dataset_root, CFG.CLASSES)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found under {dataset_root}")

    y_pred = []
    for p in tqdm(img_paths, desc="Inference"):
        img = load_rgb(p)
        out = clf(img, top_k=1)[0]
        pred = map_to_classes(out["label"], CFG.CLASSES)
        y_pred.append(pred)

    acc = float(accuracy_score(y_true, y_pred))
    report_text = classification_report(y_true, y_pred, labels=CFG.CLASSES, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=CFG.CLASSES)

    # Save outputs
    metrics = {"model_id": CFG.MODEL_ID, "num_images": len(img_paths), "accuracy": acc}
    (CFG.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (CFG.REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    # Plot confusion matrix
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CFG.CLASSES))
    plt.xticks(ticks, CFG.CLASSES, rotation=45, ha="right")
    plt.yticks(ticks, CFG.CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(CFG.REPORTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print("Saved:")
    print(" - reports/metrics.json")
    print(" - reports/classification_report.txt")
    print(" - reports/confusion_matrix.png")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
