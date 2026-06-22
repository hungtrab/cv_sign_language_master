"""Real-time ASL alphabet recognition demo.

Usage:
    # Select components directly:
    python webcam_demo.py --enhancement raw --representation mediapipe --classifier mlp
    python webcam_demo.py --enhancement clahe --representation yolo --classifier resnet18
    python webcam_demo.py --enhancement zero_dce --representation mmpose --classifier xgboost

    # Or use pipeline name shortcut:
    python webcam_demo.py --pipeline raw_mp_mlp

    # List all options:
    python webcam_demo.py --list

Keys:
    q / ESC   : quit
    c         : clear text buffer
    s         : save text buffer to output.txt
    d         : dump debug frame + prediction to debug_crops/
    e         : cycle enhancement method
    r         : cycle representation method
    1-3       : switch classifier (1=MLP, 2=XGBoost, 3=ResNet18)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from asl.pipelines.registry import register_defaults, build_pipeline, list_pipelines
from asl.pipelines.base import Pipeline
from asl.utils.config import load_config
from asl.utils.smoothing import PredictionSmoother
from asl.utils.visualization import draw_bbox, draw_hud, draw_label, draw_text_buffer

ENHANCEMENTS = ["raw", "clahe", "gamma", "sharpening", "zero_dce"]
REPRESENTATIONS = ["mediapipe", "mmpose", "yolo"]
CLASSIFIERS_FOR_LANDMARKS = ["mlp", "xgboost"]
CLASSIFIERS_FOR_IMAGE = ["resnet18"]

# Map short names to registry pipeline names
def _make_pipeline_name(enh: str, rep: str, clf: str) -> str:
    rep_short = {"mediapipe": "mp", "mmpose": "mmpose", "yolo": "yolo"}[rep]
    clf_short = {"mlp": "mlp", "xgboost": "xgb", "resnet18": "resnet18"}[clf]
    return f"{enh}_{rep_short}_{clf_short}"


def _get_valid_classifiers(rep: str) -> list[str]:
    if rep in ("mediapipe", "mmpose"):
        return CLASSIFIERS_FOR_LANDMARKS
    return CLASSIFIERS_FOR_IMAGE


def _build(enh: str, rep: str, clf: str, smoother) -> Pipeline:
    name = _make_pipeline_name(enh, rep, clf)
    return build_pipeline(name, smoother=smoother)


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
    parser = argparse.ArgumentParser(
        description="ASL webcam demo — select enhancement, representation, and classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --enhancement raw --representation mediapipe --classifier mlp
  %(prog)s -e clahe -r yolo -c resnet18
  %(prog)s -e zero_dce -r mmpose -c xgboost
  %(prog)s --pipeline raw_mp_mlp
        """)

    parser.add_argument("-e", "--enhancement", default="raw",
                        choices=ENHANCEMENTS,
                        help="Enhancement method (default: raw)")
    parser.add_argument("-r", "--representation", default="mediapipe",
                        choices=REPRESENTATIONS,
                        help="Representation / pose estimator (default: mediapipe)")
    parser.add_argument("-c", "--classifier", default=None,
                        help="Classifier: mlp, xgboost (landmarks) or resnet18 (image crop)")
    parser.add_argument("--pipeline", default=None,
                        help="Pipeline name shortcut (overrides -e/-r/-c)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--list", action="store_true",
                        help="List all options and exit")
    args = parser.parse_args()

    register_defaults()

    if args.list:
        print("Enhancement options:", ", ".join(ENHANCEMENTS))
        print("Representation options:", ", ".join(REPRESENTATIONS))
        print("Classifier options:")
        print("  mediapipe/mmpose → mlp, xgboost")
        print("  yolo             → resnet18")
        print()
        print("All 25 pipeline names:")
        for name in list_pipelines():
            print(f"  {name}")
        return

    cfg = load_config(args.config)

    smoother = PredictionSmoother(
        smoothing_window=int(cfg.inference.get("smoothing_window", 7)),
        commit_frames=int(cfg.inference.get("commit_frames", 10)),
        cooldown_ms=int(cfg.inference.get("letter_repeat_cooldown_ms", 600)),
        require_neutral=bool(cfg.inference.get("require_neutral_between_letters", True)),
        neutral_frames_to_reset=int(cfg.inference.get("neutral_frames_to_reset", 3)),
    )

    # Current selections
    enh = args.enhancement
    rep = args.representation
    if args.classifier:
        clf = args.classifier
    else:
        clf = _get_valid_classifiers(rep)[0]

    if args.pipeline:
        pipeline_name = args.pipeline
        pipeline = build_pipeline(pipeline_name, smoother=smoother)
    else:
        pipeline = _build(enh, rep, clf, smoother)
        pipeline_name = _make_pipeline_name(enh, rep, clf)

    print(f"Pipeline: {pipeline_name}")
    print(f"  Enhancement:    {enh}")
    print(f"  Representation: {rep}")
    print(f"  Classifier:     {clf}")
    print()
    print("Keys: q=quit  c=clear  s=save  d=debug  e=cycle enhance  r=cycle repr  1-3=switch clf")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    window = "ASL Demo"
    debug_root = Path("debug_crops")
    mirror = bool(cfg.camera.get("mirror", True))

    def rebuild():
        nonlocal pipeline, pipeline_name
        try:
            pipeline.close()
        except Exception:
            pass
        smoother.reset()
        pipeline_name = _make_pipeline_name(enh, rep, clf)
        pipeline = _build(enh, rep, clf, smoother)
        print(f"Switched → {pipeline_name}  (e={enh} r={rep} c={clf})")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = pipeline.predict_frame(frame_rgb)

            if out.representation_output:
                pipeline.representation.visualize(frame, out.representation_output)

            letter = out.prediction.label if out.prediction else None
            conf = out.prediction.confidence if out.prediction else 0.0

            # HUD
            h_frame, w_frame = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w_frame, 70), (20, 20, 20), -1)
            cv2.putText(frame, f"E:{enh}  R:{rep}  C:{clf}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
            if letter:
                cv2.putText(frame, f"{letter} ({conf*100:.0f}%)",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"{out.fps:.0f} FPS  {out.elapsed_ms:.0f}ms",
                        (w_frame - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            if out.text_buffer:
                draw_text_buffer(frame, out.text_buffer)

            if out.accepted_letter:
                draw_label(frame, f">> {out.accepted_letter}",
                           origin=(10, 90), color=(0, 255, 0), font_scale=0.7)

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
                print(f"Debug -> {path}")
            elif key == ord("e"):
                idx = (ENHANCEMENTS.index(enh) + 1) % len(ENHANCEMENTS)
                enh = ENHANCEMENTS[idx]
                rebuild()
            elif key == ord("r"):
                idx = (REPRESENTATIONS.index(rep) + 1) % len(REPRESENTATIONS)
                rep = REPRESENTATIONS[idx]
                clf = _get_valid_classifiers(rep)[0]
                rebuild()
            elif key == ord("1"):
                valid = _get_valid_classifiers(rep)
                if len(valid) > 0:
                    clf = valid[0]
                    rebuild()
            elif key == ord("2"):
                valid = _get_valid_classifiers(rep)
                if len(valid) > 1:
                    clf = valid[1]
                    rebuild()
            elif key == ord("3"):
                if rep == "yolo":
                    clf = "resnet18"
                    rebuild()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
