"""Extract flattened card images from captured frames.

For every captures/<CARD>/*.jpg: find the white card quadrilateral,
perspective-warp it to a canonical upright 500x700 image, save to
cards/<CARD>/. Bad frames (card not found) are skipped and reported.
"""

from pathlib import Path

import cv2
import numpy as np

CARD_WIDTH = 500
CARD_HEIGHT = 700
MIN_AREA_FRACTION = 0.03

CAPTURES_DIR = Path("captures")
CARDS_DIR = Path("cards")


def order_corners(points):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    points = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


SKIN_LOW = (0, 135, 85)
SKIN_HIGH = (255, 180, 135)


def skin_mask(frame, min_area_fraction=0.005):
    """Mask of large skin-colored regions (the hand). Small red blobs
    (diamond/heart pips also fall in the skin color range) are ignored."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, SKIN_LOW, SKIN_HIGH)
    frame_area = frame.shape[0] * frame.shape[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    hand = np.zeros_like(skin)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= frame_area * min_area_fraction:
            hand[labels == i] = 255

    return hand


def find_card_quad(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, white = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hand = cv2.dilate(skin_mask(frame), np.ones((9, 9), np.uint8))
    mask = cv2.bitwise_and(white, cv2.bitwise_not(hand))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    best = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < frame_area * MIN_AREA_FRACTION or area < best_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx
        else:
            # Hand clipped a corner: fall back to the min-area rectangle if
            # the contour still fills it like a clean card would.
            rect = cv2.minAreaRect(contour)
            box_area = rect[1][0] * rect[1][1]

            if box_area == 0 or area / box_area < 0.80:
                continue

            quad = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

        width, height = sorted(cv2.minAreaRect(quad)[1])

        if height == 0 or not (0.5 <= width / height <= 0.95):
            continue

        best = quad
        best_area = area

    return best


def crop_is_clean(warped, max_skin_fraction=0.45):
    """Reject crops where a finger covers a corner index zone.

    Only the two index corners matter for training labels; court-card art
    or red pips elsewhere are fine. Red corner glyphs occupy ~10-20% of the
    zone (up to ~0.4 with motion blur), a finger fully covering it 50%+.
    """
    h, w = warped.shape[:2]
    zone_w, zone_h = int(w * 0.19), int(h * 0.26)
    ycrcb = cv2.cvtColor(warped, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, SKIN_LOW, SKIN_HIGH)

    for zone in (skin[0:zone_h, 0:zone_w], skin[h - zone_h : h, w - zone_w : w]):
        if np.count_nonzero(zone) / zone.size > max_skin_fraction:
            return False

    return True


def warp_card(frame, quad):
    corners = order_corners(quad)
    top = np.linalg.norm(corners[0] - corners[1])
    side = np.linalg.norm(corners[1] - corners[2])

    if top > side:
        # Long edge is horizontal: rotate corner order so output is upright.
        corners = np.roll(corners, 1, axis=0)

    target = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(frame, matrix, (CARD_WIDTH, CARD_HEIGHT))


def main():
    if not CAPTURES_DIR.is_dir():
        raise RuntimeError("No captures directory. Run capture.py first.")

    total_saved = 0

    for card_dir in sorted(CAPTURES_DIR.iterdir()):
        if not card_dir.is_dir():
            continue

        out_dir = CARDS_DIR / card_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        skipped = 0

        for frame_path in sorted(card_dir.glob("*.jpg")):
            frame = cv2.imread(str(frame_path))

            if frame is None:
                skipped += 1
                continue

            quad = find_card_quad(frame)

            if quad is None:
                skipped += 1
                continue

            warped = warp_card(frame, quad)

            if not crop_is_clean(warped):
                skipped += 1
                continue

            cv2.imwrite(str(out_dir / f"{frame_path.stem}.png"), warped)
            saved += 1

        total_saved += saved
        status = "WARN: none extracted!" if saved == 0 else ""
        print(f"{card_dir.name}: {saved} extracted, {skipped} skipped {status}")

    print(f"Total: {total_saved} card images in {CARDS_DIR}/")


if __name__ == "__main__":
    main()
