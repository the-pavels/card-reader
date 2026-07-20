import json
import time
from collections import Counter, deque

import cv2
import torch
from ultralytics import YOLO


CAMERA_INDEX = 2
MODEL_PATH = "models/playing_cards_custom.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CONF_THRESHOLD = 0.40
VOTE_WINDOW = 6
VOTES_REQUIRED = 4
COOLDOWN_SECONDS = 2.0

model = YOLO(MODEL_PATH)


def recognize_cards(frame):
    """Return (best_card, best_confidence, detections).

    detections is a list of (card, confidence, (x1, y1, x2, y2)) for every
    box above CONF_THRESHOLD. The same card usually appears twice because
    the model detects both corner glyphs, so the winner is picked by
    summing confidence per card — two honest corners outvote one ghost box.
    """
    result = model(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)[0]

    votes = Counter()
    confidence_by_card = {}
    detections = []

    for box in result.boxes:
        card = model.names[int(box.cls)]
        confidence = float(box.conf)
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        detections.append((card, confidence, (x1, y1, x2, y2)))
        votes[card] += confidence
        confidence_by_card[card] = max(confidence, confidence_by_card.get(card, 0.0))

    if not votes:
        return None, 0.0, detections

    best_card = votes.most_common(1)[0][0]
    return best_card, confidence_by_card[best_card], detections


def emit_card_event(card, confidence):
    event = {
        "type": "CARD_RECOGNIZED",
        "card": card,
        "confidence": round(confidence, 3),
        "timestamp": time.time(),
    }

    print(json.dumps(event))


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera. Try CAMERA_INDEX values 0, 1, 2, or 3."
        )

    history = deque(maxlen=VOTE_WINDOW)
    latest_confidence = {}
    last_emitted_card = None
    last_emitted_at = 0.0

    while True:
        ok, frame = camera.read()

        if not ok:
            print("Could not read camera frame.")
            break

        card, confidence, detections = recognize_cards(frame)
        history.append(card)

        if card is not None:
            latest_confidence[card] = confidence

        frame_votes = Counter(c for c in history if c is not None)
        winner, winner_votes = (
            frame_votes.most_common(1)[0] if frame_votes else (None, 0)
        )

        now = time.time()

        if winner is not None and winner_votes >= VOTES_REQUIRED:
            can_emit = (
                winner != last_emitted_card
                or now - last_emitted_at >= COOLDOWN_SECONDS
            )

            if can_emit:
                emit_card_event(winner, latest_confidence.get(winner, 0.0))
                last_emitted_card = winner
                last_emitted_at = now

        for det_card, det_conf, (x1, y1, x2, y2) in detections:
            color = (0, 255, 0) if det_card == card else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{det_card} {det_conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        label = f"{card or 'unknown'} {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Card reader", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
