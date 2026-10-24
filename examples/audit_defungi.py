"""Audit DeFungi group splits, duplicates, and training-only normalization."""

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from stable_datasets.images.defungi import _CLASSES, _scan_archive, _split_groups


def audit(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        records = _scan_archive(archive)
        groups = _split_groups(records)
        digest_groups = defaultdict(set)
        digest_counts = Counter()
        for _, _, source, digest in records:
            digest_groups[digest].add(source)
            digest_counts[digest] += 1
        report = {
            "dataset": "DeFungi",
            "archive_sha256": hashlib.sha256(Path(archive_path).read_bytes()).hexdigest(),
            "images": len(records),
            "class_order": list(_CLASSES),
            "class_counts": {label: sum(row[1] == index for row in records) for index, label in enumerate(_CLASSES)},
            "source_groups": len(groups),
            "duplicate_linked_groups": len({value[0] for value in groups.values()}),
            "byte_duplicate_clusters": sum(count > 1 for count in digest_counts.values()),
            "duplicate_extra_images": sum(count - 1 for count in digest_counts.values()),
            "duplicate_clusters_crossing_source_groups": sum(len(sources) > 1 for sources in digest_groups.values()),
            "split_rule": "Class-wise SHA256 rank of defungi-v1:42:<component>; floor 70/15%, remainder test.",
            "splits": {},
        }
        for split in ("train", "validation", "test"):
            rows = [row for row in records if groups[row[2]][1] == split]
            report["splits"][split] = {
                "images": len(rows),
                "class_counts": {label: sum(row[1] == index for row in rows) for index, label in enumerate(_CLASSES)},
                "source_groups": len({row[2] for row in rows}),
                "split_groups": len({groups[row[2]][0] for row in rows}),
            }
        totals = np.zeros(3)
        squares = np.zeros(3)
        pixels = 0
        for name, _, source, _ in records:
            if groups[source][1] != "train":
                continue
            with Image.open(io.BytesIO(archive.read(name))) as image:
                array = np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
            totals += array.sum(axis=(0, 1))
            squares += (array * array).sum(axis=(0, 1))
            pixels += array.shape[0] * array.shape[1]
        mean = totals / pixels
        report["normalization"] = {
            "mean": mean.tolist(),
            "std": np.sqrt(squares / pixels - mean * mean).tolist(),
            "method": "Training images only; Pillow bilinear resize to 96x96; RGB/255; population moments in float64.",
        }
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.archive), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
