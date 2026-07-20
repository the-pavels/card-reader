"""Fine-tune YOLO11s on the generated dataset.

Local (Apple Silicon GPU via MPS, slow but works overnight):
    .venv/bin/python train.py

Colab (free T4 GPU, ~1-2h): zip the dataset, upload, then:
    !pip install ultralytics
    !unzip dataset.zip
    !yolo detect train model=yolo11s.pt data=dataset/data.yaml epochs=60 imgsz=640
    then download runs/detect/train/weights/best.pt

After training, copy the weights:
    cp runs/detect/train/weights/best.pt models/playing_cards_custom.pt
and point MODEL_PATH in reader.py at it.
"""

import torch
from ultralytics import YOLO


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    print(f"Training on device: {device}")

    model = YOLO("yolo11s.pt")  # COCO-pretrained base, auto-downloads
    results = model.train(
        data="dataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        device=device,
        patience=15,
    )
    print(f"Done. Best weights: {results.save_dir}/weights/best.pt")
    print("Copy to models/ and update MODEL_PATH in reader.py.")


if __name__ == "__main__":
    main()
