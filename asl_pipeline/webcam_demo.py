"""Real-time ASL alphabet recognition demo.

Usage:
    python webcam_demo.py --pipeline mediapipe_resnet18
    python webcam_demo.py --pipeline landmark_mlp --camera 0

Keys:
    q / ESC : quit
    c       : clear text buffer
    s       : save text buffer to output.txt
    d       : dump debug frame + prediction to debug_crops/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from asl.pipelines.registry import build_pipeline, list_pipelines, register_defaults
from asl.utils.config import load_config
from asl.utils.smoothing import PredictionSmoother
from asl.utils.visualization import draw_bbox, draw_hud, draw_label, draw_text_buffer


def _dump_debug(frame_rgb, frame_out, *, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    label = frame_out.prediction.label if frame_out.prediction else "none"
    conf = int((frame_out.prediction.confidence if frame_out.prediction else 0) * 100)
    stem = f"{ts}_{label}_{conf:03d}"

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(root / f"{stem}_frame.jpg"), frame_bgr)

    meta = {
        "timestamp": ts,
        "fps": round(frame_out.fps, 2),
        "elapsed_ms": round(frame_out.elapsed_ms, 2),
        "prediction": {
            "label": frame_out.prediction.label,
            "confidence": frame_out.prediction.confidence,
            "top_k": frame_out.prediction.top_k,
        } if frame_out.prediction else None,
        "accepted_letter": frame_out.accepted_letter,
    }
    json_path = root / f"{stem}.json"
    json_path.write_text(json.dumps(meta, indent=2))
    return json_path


def main():
    parser = argparse.ArgumentParser(description="ASL webcam demo")
    parser.add_argument("--pipeline", default="mediapipe_resnet18",
                        help="Pipeline name from registry")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--list-pipelines", action="store_true",
                        help="List available pipelines and exit")
    args = parser.parse_args()

    register_defaults()

    if args.list_pipelines:
        print("Available pipelines:")
        for name in list_pipelines():
            print(f"  - {name}")
        return

    cfg = load_config(args.config)

    smoother = PredictionSmoother(
        smoothing_window=int(cfg.inference.get("smoothing_window", 7)),
        commit_frames=int(cfg.inference.get("commit_frames", 10)),
        cooldown_ms=int(cfg.inference.get("letter_repeat_cooldown_ms", 600)),
        require_neutral=bool(cfg.inference.get("require_neutral_between_letters", True)),
        neutral_frames_to_reset=int(cfg.inference.get("neutral_frames_to_reset", 3)),
    )

    pipeline_name = args.pipeline or cfg.pipeline.get("name", "mediapipe_resnet18")
    print(f"Building pipeline: {pipeline_name}")
    pipeline = build_pipeline(pipeline_name, smoother=smoother)
    print(f"Pipeline ready: {pipeline.name}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    window = f"ASL Demo — {pipeline.name} — q quit | c clear | s save"
    debug_root = Path("debug_crops")
    mirror = bool(cfg.camera.get("mirror", True))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = pipeline.predict_frame(frame_rgb)

            # Draw overlays
            if out.representation_output:
                pipeline.representation.visualize(frame, out.representation_output)

            letter = out.prediction.label if out.prediction else None
            conf = out.prediction.confidence if out.prediction else 0.0
            draw_hud(frame, pipeline_name=pipeline.name, letter=letter,
                     confidence=conf, fps=out.fps, elapsed_ms=out.elapsed_ms)

            if out.text_buffer:
                draw_text_buffer(frame, out.text_buffer)

            if out.accepted_letter:
                draw_label(frame, f"ACCEPTED: {out.accepted_letter}",
                           origin=(10, 80), color=(0, 255, 0), font_scale=0.6)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("c"):
                pipeline.reset()
            elif key == ord("s"):
                Path("output.txt").write_text(smoother.text_buffer)
                print(f"Saved -> output.txt")
            elif key == ord("d"):
                path = _dump_debug(frame_rgb, out, root=debug_root)
                print(f"Debug saved -> {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
