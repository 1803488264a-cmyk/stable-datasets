"""Evaluate CMU checkpoints and baselines without updating model weights."""

import argparse
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from benchmarks.dataset import get_config
from benchmarks.models import create_backbone, val_transform
from stable_datasets.images import CMUFaces


def metrics(targets, predictions):
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1, 2, 3]).tolist(),
    }


def predict(backbone, head, images, device):
    outputs = []
    with torch.inference_mode():
        for batch in images.split(32):
            outputs.append(head(backbone(batch.to(device))).argmax(dim=1).cpu())
    return torch.cat(outputs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()

    config = get_config("cmufaces")
    transform = val_transform(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation device:", device)

    dataset = CMUFaces(
        download_dir=args.data_root / "downloads",
        processed_cache_dir=args.data_root / "processed",
    )
    splits = ["train", "validation"]
    if args.include_test:
        splits.append("test")

    images, targets, pixels = {}, {}, {}
    for split in splits:
        records = [dataset[split][i] for i in range(len(dataset[split]))]
        targets[split] = np.array([int(row["label"]) for row in records])
        pixels[split] = np.stack([np.asarray(row["image"], dtype=np.float32).reshape(-1) / 255.0 for row in records])
        images[split] = torch.stack([transform(row)["image"] for row in records])
        print(split, "samples:", len(records))

    evaluations = []
    models = []

    for seed in (42, 43, 44):
        run_dir = args.checkpoint_root / f"simclr_vit_tiny_patch16_224_cmufaces_seed{seed}"
        last = torch.load(run_dir / "last.ckpt", map_location="cpu", weights_only=False)
        candidates = [
            value
            for value in last.get("callbacks", {}).values()
            if isinstance(value, dict) and value.get("best_model_path")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one best-model record for seed {seed}.")

        selection = candidates[0]
        recorded_path = selection["best_model_path"]
        # Prefer the supplied run directory so copied Windows checkpoints work on Linux.
        best_path = run_dir / PureWindowsPath(recorded_path).name
        if not best_path.is_file():
            best_path = Path(recorded_path)
        if not best_path.is_file():
            raise FileNotFoundError(best_path)

        expected = float(selection["best_model_score"])
        del last
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]

        backbone = create_backbone("vit_tiny_patch16_224", config)
        backbone.load_state_dict(
            {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")},
            strict=True,
        )
        head = nn.Linear(192, 4)
        prefix = "callbacks_modules.linear_probe."
        head.load_state_dict(
            {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)},
            strict=True,
        )
        backbone = backbone.to(device).eval()
        head = head.to(device).eval()

        predictions = predict(backbone, head, images["validation"], device)
        validation = metrics(targets["validation"], predictions)
        reproduced = validation["balanced_accuracy"]
        print(f"Seed {seed}: recorded={expected:.6f}, reloaded balanced accuracy={reproduced:.6f}")
        if not np.isclose(expected, reproduced, atol=1e-5, rtol=0):
            raise RuntimeError("Validation score mismatch; stop before test evaluation.")

        evaluations.append(
            {
                "method": "simclr_online_linear_probe",
                "seed": seed,
                "checkpoint": best_path.relative_to(args.checkpoint_root).as_posix()
                if best_path.is_relative_to(args.checkpoint_root)
                else best_path.name,
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "recorded_validation_score": expected,
                "validation": validation,
            }
        )
        models.append((backbone, head))
        del checkpoint, state

    majority = int(np.bincount(targets["train"], minlength=4).argmax())
    majority_result = {
        "method": "training_majority_class",
        "predicted_class": majority,
        "validation": metrics(targets["validation"], np.full(len(targets["validation"]), majority)),
    }

    # Select regularization using validation only; fit on training only.
    best_model, best_c, best_score = None, None, -1.0
    for c_value in (0.01, 0.1, 1.0, 10.0):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=5000, random_state=42),
        )
        classifier.fit(pixels["train"], targets["train"])
        score = balanced_accuracy_score(targets["validation"], classifier.predict(pixels["validation"]))
        if score > best_score:
            best_model, best_c, best_score = classifier, c_value, score

    pixel_result = {
        "method": "standardized_pixels_logistic_regression",
        "selected_C": best_c,
        "validation": metrics(targets["validation"], best_model.predict(pixels["validation"])),
    }

    # All checkpoint consistency checks and model selection precede test scoring.
    if args.include_test:
        for result, (backbone, head) in zip(evaluations, models):
            result["test"] = metrics(targets["test"], predict(backbone, head, images["test"], device))
        majority_result["test"] = metrics(targets["test"], np.full(len(targets["test"]), majority))
        pixel_result["test"] = metrics(targets["test"], best_model.predict(pixels["test"]))

    results = evaluations + [majority_result, pixel_result]
    summary = {}
    for split in ("validation", "test"):
        if split == "test" and not args.include_test:
            continue
        summary[split] = {}
        for name in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.array([row[split][name] for row in evaluations])
            summary[split][name] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "n_seeds": len(values),
            }
    report = {
        "class_order": ["straight", "left", "right", "up"],
        "checkpoint_selection_metric": "validation balanced accuracy",
        "test_evaluated": args.include_test,
        "results": results,
        "simclr_summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for result in results:
        print(
            result["method"],
            result.get("seed", ""),
            "validation:",
            result["validation"],
            "test:",
            result.get("test", "not evaluated"),
        )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
