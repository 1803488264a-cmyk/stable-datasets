"""Reproduce the fixed DeFungi SimCLR protocol, including Windows diagnostics support."""

import argparse
import importlib
import logging
import os
import runpy
import signal
import sys
from pathlib import Path


def portable_signal_info(label="current"):
    """Log only signal names supported by the current platform."""
    handlers = {
        name: str(signal.getsignal(getattr(signal, name)))
        for name in ("SIGUSR1", "SIGUSR2", "SIGCONT", "SIGTERM")
        if hasattr(signal, name)
    }
    logging.getLogger(__name__).info("Signal handlers (%s): %s", label, handlers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    os.environ["STABLE_DATASETS_ROOT"] = data_root.as_posix()
    os.environ["PROJECT_ROOT"] = output_root.as_posix()
    if os.name == "nt":
        # Upstream ab836bf69's diagnostic assumes POSIX signals exist.
        # Keep all Manager training/checkpoint behavior; replace logging only.
        manager = importlib.import_module("stable_pretraining.manager")
        manager.print_signal_info = portable_signal_info
    checkpoint_dir = output_root / ("smoke_checkpoints" if args.smoke_test else "formal_checkpoints")
    sys.argv = [
        "benchmarks.run",
        "model=simclr",
        "dataset=defungi",
        "backbone=vit_tiny_patch16_224",
        f"seed={args.seed}",
        "training.batch_size=32",
        "training.max_epochs=50",
        "training.num_workers=0",
        "training.precision=32-true",
        "model.projector.hidden_dim=256",
        f"smoke_test={str(args.smoke_test).lower()}",
        "wandb.enabled=false",
        "checkpoint.every_n_epochs=1",
        f"checkpoint.dir={checkpoint_dir.as_posix()}",
    ]
    runpy.run_module("benchmarks.run", run_name="__main__")


if __name__ == "__main__":
    main()
