"""Integration tests using a locally downloaded CMU Face Images archive."""

import io
import os
from pathlib import Path

import pytest
from PIL import Image

from stable_datasets.images.cmu_faces import CMUFaces


@pytest.fixture(scope="module")
def cmu_records():
    archive_path = os.environ.get("CMU_FACES_ARCHIVE")
    if not archive_path:
        pytest.skip("Set CMU_FACES_ARCHIVE to run real-data integration tests.")

    path = Path(archive_path)
    if not path.is_file():
        pytest.fail(f"CMU archive does not exist: {path}")

    # Inspect the builder directly without triggering downloads or caching.
    builder = object.__new__(CMUFaces)
    CMUFaces.__init__(builder)

    return {split: list(builder._generate_examples(path, split)) for split in ("train", "validation", "test")}


def test_total_and_unique_keys(cmu_records):
    keys = [key for records in cmu_records.values() for key, _ in records]
    assert {split: len(rows) for split, rows in cmu_records.items()} == {
        "train": 378,
        "validation": 121,
        "test": 125,
    }
    assert len(keys) == 624
    assert len(set(keys)) == 624


def test_subject_disjoint_splits(cmu_records):
    subjects = {split: {record["subject"] for _, record in records} for split, records in cmu_records.items()}

    assert len(subjects["train"]) == 12
    assert len(subjects["validation"]) == 4
    assert len(subjects["test"]) == 4

    assert subjects["train"].isdisjoint(subjects["validation"])
    assert subjects["train"].isdisjoint(subjects["test"])
    assert subjects["validation"].isdisjoint(subjects["test"])

    assert len(set.union(*subjects.values())) == 20


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_images_and_labels(cmu_records, split):
    records = cmu_records[split]
    assert records
    assert {record["label"] for _, record in records} == {0, 1, 2, 3}

    for _, record in records:
        assert 0 <= record["expression"] < 4
        assert 0 <= record["eyes"] < 2

        with Image.open(io.BytesIO(record["image"])) as image:
            image.load()
            assert image.size == (32, 30)
            assert image.mode == "L"
