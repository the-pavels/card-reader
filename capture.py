"""Capture training frames of your own deck.

Shows a prompt for each of the 52 cards. Hold the card in view, press SPACE,
then slowly tilt/rotate it while a burst of frames is saved to
captures/<CARD>/. Press B anytime to snapshot an empty-table background
into backgrounds/ (grab 5-10 of those too - move the camera around).

Keys: SPACE = burst for current card, S = skip card, B = background shot,
Q = quit (progress is kept, rerun to continue).
"""

import argparse
import time
from pathlib import Path

import cv2

from camera_picker import pick_camera

FRAMES_PER_BURST = 20
BURST_SECONDS = 6.0

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["C", "D", "H", "S"]
CARDS = [f"{rank}{suit}" for suit in SUITS for rank in RANKS]

CAPTURES_DIR = Path("captures")
BACKGROUNDS_DIR = Path("backgrounds")


def main():
    parser = argparse.ArgumentParser(description="Capture training frames of a deck.")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="camera index; omit to pick interactively",
    )
    args = parser.parse_args()

    CAPTURES_DIR.mkdir(exist_ok=True)
    BACKGROUNDS_DIR.mkdir(exist_ok=True)

    todo = [c for c in CARDS if len(list((CAPTURES_DIR / c).glob("*.jpg"))) < FRAMES_PER_BURST]
    print(f"{len(CARDS) - len(todo)} cards already captured, {len(todo)} to go.")

    camera = cv2.VideoCapture(pick_camera(args.camera))

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    card_index = 0
    background_count = len(list(BACKGROUNDS_DIR.glob("*.jpg")))

    while card_index < len(todo):
        card = todo[card_index]

        ok, frame = camera.read()

        if not ok:
            print("Could not read camera frame.")
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"Show: {card}  ({card_index + 1}/{len(todo)})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            display,
            "SPACE=capture  S=skip  B=background  Q=quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            card_index += 1
        elif key == ord("b"):
            background_count += 1
            path = BACKGROUNDS_DIR / f"bg_{background_count:03d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")
        elif key == ord(" "):
            card_dir = CAPTURES_DIR / card
            card_dir.mkdir(exist_ok=True)
            interval = BURST_SECONDS / FRAMES_PER_BURST

            for i in range(FRAMES_PER_BURST):
                burst_start = time.time()
                ok, frame = camera.read()

                if not ok:
                    break

                cv2.imwrite(str(card_dir / f"{card}_{i:02d}.jpg"), frame)

                display = frame.copy()
                cv2.putText(
                    display,
                    f"{card}: {i + 1}/{FRAMES_PER_BURST} - tilt the card slowly",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                cv2.imshow("Capture", display)
                cv2.waitKey(max(1, int((interval - (time.time() - burst_start)) * 1000)))

            print(f"Captured {card}")
            card_index += 1

    camera.release()
    cv2.destroyAllWindows()

    remaining = [c for c in CARDS if len(list((CAPTURES_DIR / c).glob("*.jpg"))) == 0]
    print(f"Done. Backgrounds: {background_count}. Cards still missing: {remaining or 'none'}")


if __name__ == "__main__":
    main()
