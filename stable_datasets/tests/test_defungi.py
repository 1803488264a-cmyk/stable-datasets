"""Real-archive checks for the released DeFungi data and custom splits."""

import hashlib
import io
import os
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from stable_datasets.images import DeFungi


@pytest.fixture(scope="module")
def real_archive():
    location = os.environ.get("DEFUNGI_ARCHIVE")
    if not location:
        pytest.skip("DEFUNGI_ARCHIVE is required for real-data tests.")
    path = Path(location)
    if not path.is_file():
        pytest.fail(f"DeFungi archive does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def records(real_archive):
    builder = object.__new__(DeFungi)
    DeFungi.__init__(builder)
    return {split: list(builder._generate_examples(real_archive, split)) for split in ("train", "validation", "test")}


def test_released_archive_checksum(real_archive):
    assert hashlib.sha256(real_archive.read_bytes()).hexdigest() == (
        "02020311d1f8207f3aa3bf8b2303a5106463ac0bf35adde4ae042e8b2716ff6c"
    )


def test_exact_split_sizes_and_class_counts(records):
    expected = {
        "train": [2976, 1646, 635, 574, 537],
        "validation": [734, 377, 50, 109, 83],
        "test": [694, 311, 134, 135, 119],
    }
    for split, rows in records.items():
        counts = Counter(row["label"] for _, row in rows)
        assert [counts[label] for label in range(5)] == expected[split]
    keys = [key for rows in records.values() for key, _ in rows]
    assert len(keys) == len(set(keys)) == 9114


def test_source_and_linked_groups_are_disjoint(records):
    for field, expected in (("source_group", [312, 64, 71]), ("split_group", [306, 64, 70])):
        groups = [{row[field] for _, row in records[split]} for split in ("train", "validation", "test")]
        assert [len(values) for values in groups] == expected
        assert groups[0].isdisjoint(groups[1])
        assert groups[0].isdisjoint(groups[2])
        assert groups[1].isdisjoint(groups[2])


def test_duplicate_bytes_do_not_cross_splits(records):
    hashes = [{hashlib.sha256(row["image"]).digest() for _, row in rows} for rows in records.values()]
    assert hashes[0].isdisjoint(hashes[1])
    assert hashes[0].isdisjoint(hashes[2])
    assert hashes[1].isdisjoint(hashes[2])
    assert sum(map(len, hashes)) == 9055


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_all_images_decode_and_metadata_matches(records, split):
    labels = ["H1", "H2", "H3", "H5", "H6"]
    for key, row in records[split]:
        assert key == row["image_id"]
        assert key.startswith(labels[row["label"]] + "/")
        assert row["source_group"].startswith(labels[row["label"]] + "_")
        with Image.open(io.BytesIO(row["image"])) as image:
            image.load()
            assert image.mode == "RGB"
            assert image.size == (500, 500)
