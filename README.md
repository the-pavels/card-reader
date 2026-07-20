# Card reader

Real-time playing card recognition from a webcam. A YOLO11s detector, fine-tuned on a fully synthetic dataset generated from photos of one physical deck, reads the corner index (rank + suit) of cards on the table and emits JSON events.

![Live demo: cards recognized in real time](showcase.gif)

No hand-labeled data: bounding boxes come for free because the dataset generator composites card images with known geometry.

## How it works

```
capture.py  →  extract_cards.py  →  generate_dataset.py  →  train.py  →  reader.py
 (webcam        (contour detect,      (composite cards        (fine-tune    (live
  footage of     warp each card       onto backgrounds        YOLO11s)      recognition)
  each card)     to 500×700 PNG)      with analytic labels)
```

1. **`capture.py`** — film each of the 52 cards and a few background shots
   (SPACE/S to save card frames, B for backgrounds, Q to quit).
2. **`extract_cards.py`** — finds the card contour in each frame and warps it
   to a flat 500×700 PNG in `cards/<CARD>/`.
3. **`generate_dataset.py`** — builds a YOLO detection dataset (4,000 train /
   400 val, 640×640): random cards composited onto random background crops with
   rotation, scale, perspective and lighting jitter. Corner index zones are
   projected through the same transform, so labels are computed, not annotated.
   A random patch sometimes covers a corner (simulated finger) and that label
   is dropped. Check `dataset/preview/` before training — boxes must sit on
   the corner glyphs.
4. **`train.py`** — fine-tunes COCO-pretrained YOLO11s for 60 epochs. Runs
   locally (Apple Silicon MPS, overnight) or on a free Colab T4 (~1–2 h, recipe
   in the docstring). Result on the synthetic val set: **mAP50 0.992,
   mAP50-95 0.911** across 52 classes.
5. **`reader.py`** — webcam loop. Both corners of a card vote (confidence-
   weighted), a rolling window of frames must agree before a
   `CARD_RECOGNIZED` JSON event is printed, with a cooldown against repeats.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python reader.py   # needs models/playing_cards_custom.pt, see below
```

With several cameras attached, the reader window gets a "Camera" dropdown (top-right) — click it to switch feeds live. `--camera 2` sets the starting camera. `capture.py` uses an interactive picker instead (SPACE = next, ENTER = select), same `--camera` flag to skip it.

## Dataset & weights

- **Dataset** (`dataset.zip`, ~370 MB): attached to the GitHub Release —
  unzip and point `data=dataset/data.yaml` at it. Note: `path:` inside
  `data.yaml` is absolute; fix it for your machine (e.g. `/content/dataset`
  on Colab).
- **Weights**: `models/playing_cards_custom.pt` (YOLO11s, 52 classes) is in
  the repo. The raw card assets (`cards/`) and background shots
  (`backgrounds/`) are included, so you can regenerate the dataset and retrain
  from scratch.

## Training on Colab

```
!pip install ultralytics
!unzip -q dataset.zip
!sed -i 's|^path: .*|path: /content/dataset|' dataset/data.yaml
!yolo detect train model=yolo11s.pt data=dataset/data.yaml epochs=60 imgsz=640
```

Download `runs/detect/train/weights/best.pt`, copy it to `models/playing_cards_custom.pt`. Tip: mount Google Drive and add `project=/content/drive/MyDrive/card-reader` so checkpoints survive runtime disconnects (resume with `yolo detect train resume model=.../weights/last.pt`).

## Known limitations

The model is only as general as the synthetic data. It performs well in conditions matching the training distribution (this deck, dark table surface, roughly top-down camera, cards 25–65 % of frame height) and degrades on unseen surfaces, shallow/oblique camera angles, and small/distant cards. Tens are the weakest rank (two-glyph index). To adapt: capture new backgrounds and sharper card frames, widen the augmentation ranges in `generate_dataset.py`, regenerate, retrain.

## Credits

- Detection stack: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
  (AGPL-3.0 — applies to the library and arguably to trained weights).
- Earlier off-the-shelf models this project compared against:
  [sroot/yolo11s-playing-cards-detector](https://huggingface.co/sroot/yolo11s-playing-cards-detector),
  [mustafakemal0146/playing-cards-yolov8](https://huggingface.co/mustafakemal0146/playing-cards-yolov8).

Code and dataset: MIT (see LICENSE).
