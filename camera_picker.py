"""Camera selection shared by capture.py and reader.py.

Probes camera indices 0-5. capture.py uses pick_camera(): a live preview
where SPACE cycles cameras, ENTER confirms, Q aborts. reader.py uses
CameraDropdown: a click-to-open dropdown drawn onto its own window.
Pass an explicit index (--camera N) to skip either.
"""

import cv2
import numpy as np

MAX_PROBE_INDEX = 6
WINDOW = "Pick camera"

DROPDOWN_W = 190
DROPDOWN_ROW_H = 36
DROPDOWN_MARGIN = 12


def probe_cameras():
    """Indices of cameras that open and deliver a frame."""
    found = []

    for index in range(MAX_PROBE_INDEX):
        camera = cv2.VideoCapture(index)
        ok = camera.isOpened() and camera.read()[0]
        camera.release()

        if ok:
            found.append(index)

    return found


def pick_camera(preferred=None):
    """Return a working camera index, interactively if more than one exists."""
    if preferred is not None:
        return preferred

    cameras = probe_cameras()

    if not cameras:
        raise RuntimeError(
            f"No working cameras found (tried indices 0-{MAX_PROBE_INDEX - 1})."
        )

    if len(cameras) == 1:
        print(f"Using the only camera found: index {cameras[0]}")
        return cameras[0]

    print(f"Cameras found at indices {cameras}. SPACE=next, ENTER=select, Q=abort.")
    position = 0
    camera = cv2.VideoCapture(cameras[position])

    while True:
        ok, frame = camera.read()

        if not ok:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.putText(
            frame,
            f"Camera index {cameras[position]}  ({position + 1}/{len(cameras)})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            frame,
            "SPACE=next  ENTER=select  Q=abort",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            camera.release()
            position = (position + 1) % len(cameras)
            camera = cv2.VideoCapture(cameras[position])
        elif key in (13, 10):  # ENTER
            camera.release()
            cv2.destroyWindow(WINDOW)
            return cameras[position]
        elif key == ord("q"):
            camera.release()
            cv2.destroyWindow(WINDOW)
            raise SystemExit("Camera selection aborted.")


class CameraDropdown:
    """Dropdown picker rendered onto an OpenCV window, top-right corner.

    Click the header to open the list, click a row to switch camera.
    Read .position each frame to see the current choice.
    """

    def __init__(self, window, cameras, position=0):
        self.cameras = cameras
        self.position = position
        self.is_open = False
        self._rects = {}  # "header" / row index -> (x1, y1, x2, y2)
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for key, (x1, y1, x2, y2) in self._rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "header":
                    self.is_open = not self.is_open
                else:
                    self.position = key
                    self.is_open = False
                return

        self.is_open = False

    def _row(self, frame, rect, text, highlight):
        x1, y1, x2, y2 = rect
        fill = (70, 120, 70) if highlight else (45, 45, 45)
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(
            frame,
            text,
            (x1 + 10, y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    def draw(self, frame):
        width = frame.shape[1]
        x1 = width - DROPDOWN_W - DROPDOWN_MARGIN
        x2 = width - DROPDOWN_MARGIN
        self._rects = {}

        header = (x1, DROPDOWN_MARGIN, x2, DROPDOWN_MARGIN + DROPDOWN_ROW_H)
        self._rects["header"] = header
        arrow = "^" if self.is_open else "v"
        self._row(frame, header, f"Camera {self.cameras[self.position]}  {arrow}", False)

        if self.is_open:
            for i, cam in enumerate(self.cameras):
                y1 = DROPDOWN_MARGIN + DROPDOWN_ROW_H * (i + 1)
                rect = (x1, y1, x2, y1 + DROPDOWN_ROW_H)
                self._rects[i] = rect
                self._row(frame, rect, f"Camera {cam}", i == self.position)
