"""Train the second-stage letter classifier."""

import argparse

from signlang.classification import train_classifier
from signlang.utils import get_logger, load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("signlang.train_cls")
    log.info(f"experiment={cfg.experiment_name}")
    best = train_classifier(cfg)
    log.info(f"best checkpoint -> {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
