"""Gradio web demo. Same pipeline, browser-friendly UI.

Two tabs:
1. **Webcam stream** — live recognition.
2. **Image upload** — single-image prediction (handy for screenshots).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..classification.inference import LetterClassifier
from ..detection.detector import HandDetector
from ..pipeline.two_stage import TwoStagePipeline
from ..utils import draw_bbox, draw_label


def _annotate(frame_bgr: np.ndarray, out, *, bbox_color, text_color) -> np.ndarray:
    for det in out.detections:
        draw_bbox(frame_bgr, det.bbox_xyxy, color=bbox_color)
        draw_label(frame_bgr, f"hand {det.confidence:.2f}",
                   origin=(int(det.bbox_xyxy[0]), int(det.bbox_xyxy[1]) - 4),
                   color=bbox_color, bg=(0, 0, 0))
    if out.prediction is not None:
        draw_label(frame_bgr,
                   f"{out.prediction.label} ({out.prediction.confidence * 100:.0f}%)",
                   origin=(12, 32), color=text_color, bg=(0, 0, 0))
    return frame_bgr


def run_gradio_demo(cfg) -> None:
    import gradio as gr

    detector = HandDetector(cfg.paths.detector_weights,
                            conf_threshold=float(cfg.inference.detection_conf_threshold))
    classifier = LetterClassifier(cfg.paths.classifier_weights)
    pipeline = TwoStagePipeline(
        detector, classifier,
        cls_conf_threshold=float(cfg.inference.classification_conf_threshold),
        smoothing_window=int(cfg.inference.smoothing_window),
        letter_repeat_cooldown_ms=int(cfg.inference.letter_repeat_cooldown_ms),
    )
    bbox_color = tuple(int(c) for c in cfg.ui.bbox_color)
    text_color = tuple(int(c) for c in cfg.ui.text_color)

    def stream_fn(rgb_frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
        if rgb_frame is None:
            return rgb_frame, "—", pipeline.text_buffer
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        out = pipeline.process(bgr)
        annotated = _annotate(bgr, out, bbox_color=bbox_color, text_color=text_color)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        letter = (out.prediction.label if out.prediction else "—")
        return annotated_rgb, letter, pipeline.text_buffer

    def reset_fn():
        pipeline.reset_text_buffer()
        return ""

    with gr.Blocks(title="Sign Language Recognition") as demo:
        gr.Markdown("# Sign Language — real-time ASL recognition")
        with gr.Tab("Webcam"):
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
                annotated = gr.Image(type="numpy", label="Annotated")
            letter = gr.Textbox(label="Predicted letter")
            buffer_box = gr.Textbox(label="Text buffer", lines=2, interactive=False)
            cam.stream(stream_fn, inputs=[cam], outputs=[annotated, letter, buffer_box])
            clear_btn = gr.Button("Clear buffer")
            clear_btn.click(reset_fn, outputs=[buffer_box])
        with gr.Tab("Upload an image"):
            inp = gr.Image(type="numpy", label="Image")
            out_img = gr.Image(type="numpy", label="Annotated")
            out_lbl = gr.Textbox(label="Top-5 predictions", lines=6)

            def single_image_fn(rgb):
                # Stateless path: detect → classify the top crop → render top-5.
                # We bypass the streaming pipeline so a single still image is
                # not gated by the smoothing window / cooldown / re-arm logic.
                if rgb is None:
                    return None, "—"
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                detections = detector.detect(bgr, top_k=1)
                annotated = bgr.copy()
                if not detections:
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    return annotated_rgb, "no hand detected"
                det = detections[0]
                draw_bbox(annotated, det.bbox_xyxy, color=bbox_color)
                crop_rgb = cv2.cvtColor(det.crop(bgr), cv2.COLOR_BGR2RGB)
                pred = classifier.predict(crop_rgb)
                draw_label(
                    annotated,
                    f"{pred.label} ({pred.confidence * 100:.0f}%)",
                    origin=(int(det.bbox_xyxy[0]), max(20, int(det.bbox_xyxy[1]) - 4)),
                    color=text_color, bg=(0, 0, 0),
                )
                top5_str = "\n".join(f"{lbl}: {p * 100:.1f}%" for lbl, p in pred.top5)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                return annotated_rgb, top5_str

            inp.change(single_image_fn, inputs=[inp], outputs=[out_img, out_lbl])
    demo.launch()
