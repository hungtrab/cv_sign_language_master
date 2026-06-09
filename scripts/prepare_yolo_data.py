"""Thin re-export so users can call ``python -m scripts.prepare_yolo_data ...``."""

from signlang.data.prepare_yolo_data import main


if __name__ == "__main__":
    raise SystemExit(main())
