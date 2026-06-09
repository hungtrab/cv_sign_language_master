"""Run the live demo (OpenCV window or Gradio web app)."""

import argparse

from signlang.demo import run_cv2_demo, run_gradio_demo
from signlang.utils import load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--backend", choices=("cv2", "gradio"), default="cv2")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend == "cv2":
        run_cv2_demo(cfg)
    else:
        run_gradio_demo(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
