import argparse
from pathlib import Path

from tqdm import tqdm
from transformers import pipeline

from .config import CFG
from .utils import list_images, load_rgb, map_to_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with images to classify")
    ap.add_argument("--topk", type=int, default=1, help="Top-k predictions to show")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    clf = pipeline("image-classification", model=CFG.MODEL_ID)
    imgs = list_images(folder)

    for p in tqdm(imgs, desc="Predicting"):
        img = load_rgb(p)
        outs = clf(img, top_k=args.topk)
        if args.topk == 1:
            pred = map_to_classes(outs[0]["label"], CFG.CLASSES)
            score = outs[0]["score"]
            print(f"{p.name:30} -> {pred:9} ({score*100:6.2f}%)")
        else:
            pretty = ", ".join([f"{map_to_classes(o['label'], CFG.CLASSES)}:{o['score']*100:.1f}%" for o in outs])
            print(f"{p.name:30} -> {pretty}")


if __name__ == "__main__":
    main()
