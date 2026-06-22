"""Real-time ASL alphabet recognition demo (OpenCV, styled like cv2/web).

Usage:
    python webcam_demo.py -e raw -r mediapipe -c mlp
    python webcam_demo.py -e zero_dce -r yolo -c resnet18
    python webcam_demo.py --list

Keys:
    q / ESC   : quit
    c         : clear word
    BACKSPACE : delete last char
    SPACE     : add space to word
    s         : save word to output.txt
    d         : dump debug snapshot
    e         : cycle enhancement
    r         : cycle representation
    1-3       : switch classifier
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

ENHANCEMENTS = ["raw", "clahe", "gamma", "sharpening", "zero_dce"]
REPRESENTATIONS = ["mediapipe", "mmpose", "yolo"]
CLASSIFIERS_FOR_LANDMARKS = ["mlp", "xgboost"]
CLASSIFIERS_FOR_IMAGE = ["resnet18"]

# Colors (BGR)
BG_DARK = (20, 15, 11)
BG_CARD = (41, 30, 22)
ACCENT_GREEN = (153, 211, 52)
ACCENT_BLUE = (248, 189, 56)
TEXT_WHITE = (243, 237, 230)
TEXT_MUTED = (176, 155, 139)
DANGER_RED = (113, 113, 248)
BAR_BG = (64, 49, 36)
BAR_FILL = (153, 211, 52)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _make_pipeline_name(enh, rep, clf):
    rep_short = {"mediapipe": "mp", "mmpose": "mmpose", "yolo": "yolo"}[rep]
    clf_short = {"mlp": "mlp", "xgboost": "xgb", "resnet18": "resnet18"}[clf]
    return f"{enh}_{rep_short}_{clf_short}"


def _get_valid_classifiers(rep):
    return CLASSIFIERS_FOR_LANDMARKS if rep in ("mediapipe", "mmpose") else CLASSIFIERS_FOR_IMAGE


def _build(enh, rep, clf, smoother):
    name = _make_pipeline_name(enh, rep, clf)
    return build_pipeline(name, smoother=smoother)


def _draw_hand_skeleton(frame, landmarks, color=ACCENT_GREEN):
    if not landmarks:
        return
    for a, b in HAND_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            pt1 = (int(landmarks[a][0]), int(landmarks[a][1]))
            pt2 = (int(landmarks[b][0]), int(landmarks[b][1]))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), 4, TEXT_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def _draw_big_letter(frame, letter, conf, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Shadow
    cv2.putText(frame, letter, (x+2, y+2), font, 2.5, (0, 0, 0), 6, cv2.LINE_AA)
    # Letter
    cv2.putText(frame, letter, (x, y), font, 2.5, TEXT_WHITE, 5, cv2.LINE_AA)
    # Confidence below
    cv2.putText(frame, f"{conf*100:.0f}%", (x+5, y+35), font, 0.7, ACCENT_GREEN, 2, cv2.LINE_AA)


def _draw_score_bars(frame, top_k, x, y, bar_w=150, bar_h=14, spacing=22):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (label, score) in enumerate(top_k[:5]):
        cy = y + i * spacing
        # Label
        cv2.putText(frame, label, (x, cy + 10), font, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)
        # Bar background
        bx = x + 25
        cv2.rectangle(frame, (bx, cy), (bx + bar_w, cy + bar_h), BAR_BG, -1)
        # Bar fill
        fill_w = int(bar_w * score)
        cv2.rectangle(frame, (bx, cy), (bx + fill_w, cy + bar_h), BAR_FILL, -1)
        # Percent
        cv2.putText(frame, f"{score*100:.0f}%", (bx + bar_w + 6, cy + 11),
                     font, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)


def _draw_word_bar(frame, word, h, w):
    bar_y = h - 50
    cv2.rectangle(frame, (0, bar_y), (w, h), BG_DARK, -1)
    cv2.line(frame, (0, bar_y), (w, bar_y), BAR_BG, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Word", (12, bar_y + 20), font, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
    display = word if word else "..."
    cv2.putText(frame, display, (12, bar_y + 42), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


def _draw_top_bar(frame, enh, rep, clf, fps, elapsed, hand_present, w):
    cv2.rectangle(frame, (0, 0), (w, 55), BG_DARK, -1)
    cv2.line(frame, (0, 55), (w, 55), BAR_BG, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(frame, "ASL Recognizer", (12, 22), font, 0.55, TEXT_WHITE, 1, cv2.LINE_AA)

    # Component info
    cfg_text = f"E:{enh}  R:{rep}  C:{clf}"
    cv2.putText(frame, cfg_text, (12, 44), font, 0.4, ACCENT_BLUE, 1, cv2.LINE_AA)

    # Status badge
    if hand_present:
        badge_text = "Hand detected"
        badge_color = ACCENT_GREEN
    else:
        badge_text = "Show your hand"
        badge_color = TEXT_MUTED

    (tw, th), _ = cv2.getTextSize(badge_text, font, 0.42, 1)
    bx = w - tw - 24
    cv2.rectangle(frame, (bx - 8, 10), (bx + tw + 8, 10 + th + 12),
                  BAR_BG, -1, cv2.LINE_AA)
    cv2.putText(frame, badge_text, (bx, 10 + th + 4), font, 0.42, badge_color, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.0f} FPS  {elapsed:.0f}ms",
                (w - 160, 48), font, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)


def _draw_controls_hint(frame, h, w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    hints = "e:enhance  r:repr  1-3:clf  c:clear  q:quit"
    cv2.putText(frame, hints, (w - 380, h - 58), font, 0.33, TEXT_MUTED, 1, cv2.LINE_AA)


def _dump_debug(frame_rgb, frame_out, *, root):
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    label = frame_out.prediction.label if frame_out.prediction else "none"
    conf = int((frame_out.prediction.confidence if frame_out.prediction else 0) * 100)
    stem = f"{ts}_{label}_{conf:03d}"
    cv2.imwrite(str(root / f"{stem}_frame.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    meta = {
        "timestamp": ts,
        "prediction": {
            "label": frame_out.prediction.label,
            "confidence": frame_out.prediction.confidence,
            "top_k": frame_out.prediction.top_k,
        } if frame_out.prediction else None,
    }
    p = root / f"{stem}.json"
    p.write_text(json.dumps(meta, indent=2))
    return p


def main():
    parser = argparse.ArgumentParser(
        description="ASL webcam demo",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--enhancement", default="raw", choices=ENHANCEMENTS)
    parser.add_argument("-r", "--representation", default="mediapipe", choices=REPRESENTATIONS)
    parser.add_argument("-c", "--classifier", default=None)
    parser.add_argument("--pipeline", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    register_defaults()

    if args.list:
        print("Enhancement:", ", ".join(ENHANCEMENTS))
        print("Representation:", ", ".join(REPRESENTATIONS))
        print("Classifier: mlp/xgboost (landmarks) | resnet18 (yolo crop)")
        print("\nPipelines:")
        for n in list_pipelines():
            print(f"  {n}")
        return

    cfg = load_config(args.config)
    smoother = PredictionSmoother(
        smoothing_window=int(cfg.inference.get("smoothing_window", 7)),
        commit_frames=int(cfg.inference.get("commit_frames", 10)),
        cooldown_ms=int(cfg.inference.get("letter_repeat_cooldown_ms", 600)),
        require_neutral=bool(cfg.inference.get("require_neutral_between_letters", True)),
        neutral_frames_to_reset=int(cfg.inference.get("neutral_frames_to_reset", 3)),
    )

    enh = args.enhancement
    rep = args.representation
    clf = args.classifier or _get_valid_classifiers(rep)[0]

    if args.pipeline:
        pipeline = build_pipeline(args.pipeline, smoother=smoother)
    else:
        pipeline = _build(enh, rep, clf, smoother)

    print(f"Pipeline: {_make_pipeline_name(enh, rep, clf)}  (E:{enh} R:{rep} C:{clf})")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    mirror = bool(cfg.camera.get("mirror", True))
    debug_root = Path("debug_crops")

    def rebuild():
        nonlocal pipeline
        try:
            pipeline.close()
        except Exception:
            pass
        smoother.reset()
        pipeline = _build(enh, rep, clf, smoother)
        print(f"Switched -> {_make_pipeline_name(enh, rep, clf)}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = pipeline.predict_frame(frame_rgb)

            hand_present = out.representation_output is not None

            # Draw hand skeleton
            if out.representation_output and out.representation_output.landmarks:
                _draw_hand_skeleton(frame, out.representation_output.landmarks)
            elif out.representation_output and out.representation_output.bbox:
                x1, y1, x2, y2 = (int(v) for v in out.representation_output.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_GREEN, 2)

            # Big letter overlay (bottom-right of video area)
            if out.prediction and hand_present:
                _draw_big_letter(frame, out.prediction.label, out.prediction.confidence,
                                 w - 120, h - 100)

            # Top-5 score bars (right side)
            if out.prediction and hand_present:
                _draw_score_bars(frame, out.prediction.top_k, w - 210, 70)

            # Top bar
            _draw_top_bar(frame, enh, rep, clf, out.fps, out.elapsed_ms, hand_present, w)

            # Word bar
            _draw_word_bar(frame, smoother.text_buffer, h, w)

            # Controls hint
            _draw_controls_hint(frame, h, w)

            # Accepted flash
            if out.accepted_letter:
                cv2.putText(frame, f">> {out.accepted_letter}", (w // 2 - 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, ACCENT_GREEN, 3, cv2.LINE_AA)

            cv2.imshow("ASL Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord("c"):
                smoother.reset()
            elif key == 8:  # backspace
                smoother.text_buffer = smoother.text_buffer[:-1]
            elif key == ord(" "):
                smoother.text_buffer += " "
            elif key == ord("s"):
                Path("output.txt").write_text(smoother.text_buffer)
                print(f"Saved -> output.txt")
            elif key == ord("d"):
                print(f"Debug -> {_dump_debug(frame_rgb, out, root=debug_root)}")
            elif key == ord("e"):
                enh = ENHANCEMENTS[(ENHANCEMENTS.index(enh) + 1) % len(ENHANCEMENTS)]
                rebuild()
            elif key == ord("r"):
                rep = REPRESENTATIONS[(REPRESENTATIONS.index(rep) + 1) % len(REPRESENTATIONS)]
                clf = _get_valid_classifiers(rep)[0]
                rebuild()
            elif key == ord("1"):
                v = _get_valid_classifiers(rep)
                if len(v) > 0:
                    clf = v[0]; rebuild()
            elif key == ord("2"):
                v = _get_valid_classifiers(rep)
                if len(v) > 1:
                    clf = v[1]; rebuild()
            elif key == ord("3"):
                if rep == "yolo":
                    clf = "resnet18"; rebuild()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
