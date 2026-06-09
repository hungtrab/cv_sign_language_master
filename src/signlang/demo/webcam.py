"""OpenCV-based real-time webcam demo.

Keys while the window is focused:
* ``q`` — quit
* ``c`` — clear text buffer
* ``s`` — save buffer to ``output.txt``
* ``d`` — dump the current frame + crop + top-5 prediction under ``debug_crops/``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from ..classification.inference import LetterClassifier
from ..detection.detector import HandDetector
from ..pipeline.two_stage import TwoStagePipeline
from ..utils import draw_bbox, draw_label
from ..utils.visualize import draw_text_buffer


def _dump_debug(frame_bgr, out, *, root: Path) -> Path | None:
    """Save (frame, crop, prediction metadata) for offline analysis."""
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    label = out.prediction.label if out.prediction else "no-pred"
    conf = int(round((out.prediction.confidence if out.prediction else 0.0) * 100))
    stem = f"{ts}_{label}_conf{conf:03d}"

    frame_path = root / f"{stem}_frame.jpg"
    cv2.imwrite(str(frame_path), frame_bgr)

    crop_path: str | None = None
    if out.detections:
        crop = out.detections[0].crop(frame_bgr)
        crop_path = str(root / f"{stem}_crop.jpg")
        cv2.imwrite(crop_path, crop)

    meta = {
        "timestamp": ts,
        "fps": round(out.fps, 2),
        "elapsed_ms": round(out.elapsed_ms, 2),
        "detection": (
            {
                "bbox_xyxy": list(out.detections[0].bbox_xyxy),
                "confidence": out.detections[0].confidence,
            }
            if out.detections else None
        ),
        "prediction": (
            {
                "label": out.prediction.label,
                "confidence": out.prediction.confidence,
                "top5": out.prediction.top5,
            }
            if out.prediction else None
        ),
        "accepted_letter": out.accepted_letter,
        "frame_path": str(frame_path),
        "crop_path": crop_path,
    }
    json_path = root / f"{stem}.json"
    json_path.write_text(json.dumps(meta, indent=2))
    return json_path


def run_cv2_demo(cfg) -> None:
    window_name = "Sign Language - q quit | c clear | s save | d dump"
    debug_root = Path("debug_crops")
    detector = HandDetector(
        cfg.paths.detector_weights,
        conf_threshold=float(cfg.inference.detection_conf_threshold),
    )
    classifier = LetterClassifier(cfg.paths.classifier_weights)
    pipeline = TwoStagePipeline(
        detector, classifier,
        cls_conf_threshold=float(cfg.inference.classification_conf_threshold),
        cls_margin_threshold=float(cfg.inference.get("classification_margin_threshold", 0.15)),
        smoothing_window=int(cfg.inference.smoothing_window),
        letter_repeat_cooldown_ms=int(cfg.inference.letter_repeat_cooldown_ms),
        require_neutral_between_letters=bool(cfg.inference.get("require_neutral_between_letters", True)),
        neutral_frames_to_reset=int(cfg.inference.get("neutral_frames_to_reset", 3)),
    )

    cap = cv2.VideoCapture(int(cfg.camera.index))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cfg.camera.index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.camera.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.camera.height))
    cap.set(cv2.CAP_PROP_FPS, int(cfg.camera.fps))

    bbox_color = tuple(int(c) for c in cfg.ui.bbox_color)
    text_color = tuple(int(c) for c in cfg.ui.text_color)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, int(cfg.camera.width), int(cfg.camera.height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera frame read failed; stopping demo")
                break
            if bool(cfg.camera.get("mirror", False)):
                frame = cv2.flip(frame, 1)
            out = pipeline.process(frame)
            # Keep a clean copy so 'd' dumps the raw frame (and a faithful crop)
            # rather than the version we're about to draw overlays on.
            raw_frame = frame.copy()

            for det in out.detections:
                if cfg.ui.show_bbox:
                    draw_bbox(frame, det.bbox_xyxy, color=bbox_color)
                label_parts = ["hand"]
                if cfg.ui.show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")
                draw_label(frame, " ".join(label_parts),
                           origin=(int(det.bbox_xyxy[0]), int(det.bbox_xyxy[1]) - 4),
                           color=bbox_color, bg=(0, 0, 0))

            if out.prediction is not None:
                draw_label(frame,
                           f"{out.prediction.label} ({out.prediction.confidence * 100:.0f}%)",
                           origin=(12, 32), color=text_color, bg=(0, 0, 0),
                           font_scale=float(cfg.ui.font_scale), thickness=2)
            draw_label(frame, f"{out.fps:.1f} FPS  {out.elapsed_ms:.0f} ms",
                       origin=(12, 60), color=(220, 220, 220), bg=(0, 0, 0))

            if cfg.ui.show_text_buffer:
                draw_text_buffer(frame, out.text_buffer, color=text_color)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                pipeline.reset_text_buffer()
            if key == ord("s"):
                Path("output.txt").write_text(pipeline.text_buffer)
                print("buffer saved -> output.txt")
            if key == ord("d"):
                dumped = _dump_debug(raw_frame, out, root=debug_root)
                if dumped is not None:
                    print(f"debug snapshot saved -> {dumped}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
