"""Generate a labeled YOLO dataset from extracted cards + background photos.

Composites random cards from cards/<CARD>/ onto random crops of
backgrounds/*.jpg with random rotation, scale, perspective and lighting.
Corner index zones are computed analytically through the same transform,
so bounding-box labels come for free. Sometimes a corner is deliberately
covered by a random patch (simulates a finger) and its label is dropped.

Output: dataset/{images,labels}/{train,val}/ + dataset/data.yaml
Preview images with drawn boxes: dataset/preview/
"""

import random
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 640
N_TRAIN = 4000
N_VAL = 400
MAX_CARDS_PER_IMAGE = 3
CARD_MIN_HEIGHT = 160
CARD_MAX_HEIGHT = 420
CORNER_ZONE_W = 0.19  # fraction of card width covered by the index zone
CORNER_ZONE_H = 0.26
OCCLUDER_PROB = 0.35
ZONE_VISIBLE_FRACTION = 0.6

CARDS_DIR = Path("cards")
BACKGROUNDS_DIR = Path("backgrounds")
DATASET_DIR = Path("dataset")


def load_assets():
    card_images = {}

    for card_dir in sorted(CARDS_DIR.iterdir()):
        if card_dir.is_dir():
            images = [cv2.imread(str(p)) for p in sorted(card_dir.glob("*.png"))]
            images = [img for img in images if img is not None]

            if images:
                card_images[card_dir.name] = images

    backgrounds = [cv2.imread(str(p)) for p in sorted(BACKGROUNDS_DIR.glob("*.jpg"))]
    backgrounds = [bg for bg in backgrounds if bg is not None]

    if not card_images:
        raise RuntimeError("No card images. Run extract_cards.py first.")

    if not backgrounds:
        raise RuntimeError("No backgrounds/*.jpg. Capture some with capture.py (B key).")

    return card_images, backgrounds


def random_background_crop(backgrounds):
    bg = random.choice(backgrounds)
    h, w = bg.shape[:2]

    if h < IMG_SIZE or w < IMG_SIZE:
        bg = cv2.resize(bg, (max(w, IMG_SIZE), max(h, IMG_SIZE)))
        h, w = bg.shape[:2]

    x = random.randint(0, w - IMG_SIZE)
    y = random.randint(0, h - IMG_SIZE)
    return bg[y : y + IMG_SIZE, x : x + IMG_SIZE].copy()


def corner_zone_polygons(card_w, card_h):
    """Both index zones in card coordinates: top-left and bottom-right."""
    zw, zh = card_w * CORNER_ZONE_W, card_h * CORNER_ZONE_H
    top_left = np.float32([[0, 0], [zw, 0], [zw, zh], [0, zh]])
    bottom_right = np.float32(
        [[card_w - zw, card_h - zh], [card_w, card_h - zh], [card_w, card_h], [card_w - zw, card_h]]
    )
    return [top_left, bottom_right]


def random_card_homography(card_w, card_h):
    """Random placement: returns 3x3 homography from card coords to canvas."""
    target_h = random.uniform(CARD_MIN_HEIGHT, CARD_MAX_HEIGHT)
    scale = target_h / card_h
    half_w, half_h = card_w * scale / 2, card_h * scale / 2
    angle = random.uniform(0, 2 * np.pi)
    cos, sin = np.cos(angle), np.sin(angle)

    margin = max(half_w, half_h)
    cx = random.uniform(margin * 0.6, IMG_SIZE - margin * 0.6)
    cy = random.uniform(margin * 0.6, IMG_SIZE - margin * 0.6)

    local = np.float32([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])
    jitter = np.random.uniform(-0.06, 0.06, local.shape).astype(np.float32) * target_h
    rotated = np.float32([[x * cos - y * sin, x * sin + y * cos] for x, y in local])
    destination = rotated + jitter + np.float32([cx, cy])

    source = np.float32([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]])
    return cv2.getPerspectiveTransform(source, destination), destination


def adjust_lighting(image):
    gain = random.uniform(0.7, 1.25)
    bias = random.uniform(-25, 20)
    out = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    if random.random() < 0.25:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out


def compose_image(card_images, backgrounds, class_ids):
    canvas = random_background_crop(backgrounds)
    owner = np.full((IMG_SIZE, IMG_SIZE), -1, dtype=np.int16)
    zones = []  # (card_name, zone polygon in canvas coords, owner index)

    n_cards = random.randint(1, MAX_CARDS_PER_IMAGE)
    placed = 0

    for _ in range(n_cards):
        card_name = random.choice(list(card_images))
        card_img = adjust_lighting(random.choice(card_images[card_name]))
        card_h, card_w = card_img.shape[:2]

        for _attempt in range(5):
            homography, quad = random_card_homography(card_w, card_h)
            quad_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            cv2.fillConvexPoly(quad_mask, quad.astype(np.int32), 255)
            quad_area = int(np.count_nonzero(quad_mask))

            if quad_area == 0:
                continue

            overlap = np.count_nonzero(quad_mask[owner >= 0]) / quad_area

            if overlap > 0.25:
                continue

            warped = cv2.warpPerspective(card_img, homography, (IMG_SIZE, IMG_SIZE))
            inside = quad_mask > 0
            canvas[inside] = warped[inside]
            owner[inside] = placed

            for zone in corner_zone_polygons(card_w, card_h):
                projected = cv2.perspectiveTransform(zone.reshape(1, 4, 2), homography)
                zones.append((card_name, projected.reshape(4, 2), placed))

            placed += 1
            break

    # Simulated finger: paint a patch over one random zone, drop its label.
    if zones and random.random() < OCCLUDER_PROB:
        _, zone_poly, _ = random.choice(zones)
        cover = zone_poly.mean(axis=0)
        size = random.uniform(0.7, 1.4) * np.ptp(zone_poly, axis=0).max()
        color = tuple(int(c) for c in np.random.randint(60, 200, 3))
        x1, y1 = (cover - size / 2).astype(int)
        x2, y2 = (cover + size / 2).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        occluded = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.rectangle(occluded, (x1, y1), (x2, y2), 255, -1)
        owner[occluded > 0] = -2

    labels = []

    for card_name, zone_poly, owner_index in zones:
        zone_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.fillConvexPoly(zone_mask, zone_poly.astype(np.int32), 255)
        zone_area = int(np.count_nonzero(zone_mask))

        if zone_area == 0:
            continue

        visible = np.count_nonzero((zone_mask > 0) & (owner == owner_index)) / zone_area

        if visible < ZONE_VISIBLE_FRACTION:
            continue

        xs, ys = zone_poly[:, 0], zone_poly[:, 1]
        x1, x2 = np.clip([xs.min(), xs.max()], 0, IMG_SIZE - 1)
        y1, y2 = np.clip([ys.min(), ys.max()], 0, IMG_SIZE - 1)

        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        cx, cy = (x1 + x2) / 2 / IMG_SIZE, (y1 + y2) / 2 / IMG_SIZE
        w, h = (x2 - x1) / IMG_SIZE, (y2 - y1) / IMG_SIZE
        labels.append(f"{class_ids[card_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return canvas, labels


def main():
    card_images, backgrounds = load_assets()
    class_names = sorted(card_images)

    if len(class_names) < 52:
        print(f"WARNING: only {len(class_names)} card classes found, expected 52.")

    class_ids = {name: i for i, name in enumerate(class_names)}

    for split, count in [("train", N_TRAIN), ("val", N_VAL)]:
        images_dir = DATASET_DIR / "images" / split
        labels_dir = DATASET_DIR / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            image, labels = compose_image(card_images, backgrounds, class_ids)
            cv2.imwrite(str(images_dir / f"{split}_{i:05d}.jpg"), image)
            (labels_dir / f"{split}_{i:05d}.txt").write_text("\n".join(labels))

            if (i + 1) % 500 == 0:
                print(f"{split}: {i + 1}/{count}")

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    (DATASET_DIR / "data.yaml").write_text(
        f"path: {DATASET_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names_yaml}\n"
    )

    preview_dir = DATASET_DIR / "preview"
    preview_dir.mkdir(exist_ok=True)

    for i in range(8):
        image, labels = compose_image(card_images, backgrounds, class_ids)

        for line in labels:
            cls, cx, cy, w, h = line.split()
            cx, cy, w, h = (float(v) * IMG_SIZE for v in (cx, cy, w, h))
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                class_names[int(cls)],
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imwrite(str(preview_dir / f"preview_{i}.jpg"), image)

    print(f"Done. Check {preview_dir}/ - boxes must sit on card corners.")


if __name__ == "__main__":
    main()
