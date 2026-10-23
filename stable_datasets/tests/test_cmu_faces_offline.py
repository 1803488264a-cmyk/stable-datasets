"""Offline tests for CMU parsing and the public loading interface."""

import io
import tarfile
import zipfile

import pytest
from PIL import Image

import stable_datasets.utils as dataset_utils
from stable_datasets.images import CMUFaces


@pytest.fixture
def miniature_archive(tmp_path):
    """Create a tiny archive with one person from each split."""
    image_buffer = io.BytesIO()
    Image.new("L", (32, 30), color=100).save(image_buffer, format="PPM")
    image_bytes = image_buffer.getvalue()

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
        for subject in ("an2i", "megak", "saavik"):
            name = f"faces_4/{subject}/{subject}_left_angry_sunglasses_4.pgm"
            member = tarfile.TarInfo(name)
            member.size = len(image_bytes)
            archive.addfile(member, io.BytesIO(image_bytes))

        # Source-marked bad files must never become examples.
        member = tarfile.TarInfo("faces_4/an2i/unreadable.pgm.bad")
        member.size = 3
        archive.addfile(member, io.BytesIO(b"bad"))

    path = tmp_path / "cmu_faces.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("faces_4.tar.gz", tar_buffer.getvalue())
    return path


@pytest.mark.parametrize(
    ("split", "subject"),
    [
        ("train", "an2i"),
        ("validation", "megak"),
        ("test", "saavik"),
    ],
)
def test_parsing_and_split_membership(miniature_archive, split, subject):
    builder = object.__new__(CMUFaces)
    CMUFaces.__init__(builder)

    records = list(builder._generate_examples(miniature_archive, split))
    assert len(records) == 1

    key, record = records[0]
    assert key == f"faces_4/{subject}/{subject}_left_angry_sunglasses_4.pgm"
    assert record["subject"] == subject
    assert record["label"] == 1
    assert record["expression"] == 3
    assert record["eyes"] == 1

    with Image.open(io.BytesIO(record["image"])) as image:
        image.load()
        assert image.size == (32, 30)
        assert image.mode == "L"


def test_unknown_split(miniature_archive):
    builder = object.__new__(CMUFaces)
    CMUFaces.__init__(builder)

    with pytest.raises(ValueError, match="Unknown CMUFaces split"):
        list(builder._generate_examples(miniature_archive, "unknown"))


def test_public_loading_and_cache_reuse(miniature_archive, tmp_path, monkeypatch):
    download_calls = []

    def local_download(specs, *args, **kwargs):
        download_calls.append(len(specs))
        return [miniature_archive for _ in specs]

    monkeypatch.setattr(dataset_utils, "bulk_download", local_download)

    options = {
        "download_dir": tmp_path / "downloads",
        "processed_cache_dir": tmp_path / "processed",
    }
    first = CMUFaces(**options)

    assert set(first) == {"train", "validation", "test"}
    assert {split: len(data) for split, data in first.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert download_calls == [1]

    def unexpected_work(*args, **kwargs):
        pytest.fail("Cached loading attempted downloading or generation.")

    monkeypatch.setattr(dataset_utils, "bulk_download", unexpected_work)
    monkeypatch.setattr(CMUFaces, "_generate_examples", unexpected_work)

    second = CMUFaces(**options)

    for split in first:
        assert len(second[split]) == len(first[split])
        before = first[split][0]
        after = second[split][0]
        for field in ("label", "subject", "expression", "eyes"):
            assert after[field] == before[field]
        assert after["image"].size == before["image"].size
        assert after["image"].tobytes() == before["image"].tobytes()


def test_all_metadata_combinations_and_stable_order(tmp_path):
    image_buffer = io.BytesIO()
    Image.new("L", (32, 30), color=127).save(image_buffer, format="PPM")
    payload = image_buffer.getvalue()
    expected = {}
    for pose_id, pose in enumerate(("straight", "left", "right", "up")):
        for expression_id, expression in enumerate(("neutral", "happy", "sad", "angry")):
            for eyes_id, eyes in enumerate(("open", "sunglasses")):
                name = f"faces_4/an2i/an2i_{pose}_{expression}_{eyes}_4.pgm"
                expected[name] = (pose_id, expression_id, eyes_id)
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
        for name in sorted(expected, reverse=True):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    path = tmp_path / "all_combinations.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("faces_4.tar.gz", tar_buffer.getvalue())
    builder = object.__new__(CMUFaces)
    CMUFaces.__init__(builder)
    rows = list(builder._generate_examples(path, "train"))
    assert [key for key, _ in rows] == sorted(expected)
    assert len(rows) == 32
    for key, row in rows:
        assert (row["label"], row["expression"], row["eyes"]) == expected[key]


@pytest.mark.parametrize(
    "filename",
    [
        "unexpected.pgm",
        "an2i_unknown_angry_open_4.pgm",
        "an2i_left_unknown_open_4.pgm",
        "an2i_left_angry_unknown_4.pgm",
        "an2i_left_angry_open_2.pgm",
        "unknown_left_angry_open_4.pgm",
    ],
)
def test_invalid_metadata_is_rejected(tmp_path, filename):
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"faces_4/an2i/{filename}")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("faces_4.tar.gz", tar_buffer.getvalue())
    builder = object.__new__(CMUFaces)
    CMUFaces.__init__(builder)
    with pytest.raises(ValueError):
        list(builder._generate_examples(path, "train"))


def test_public_exports():
    import stable_datasets.images as images

    assert "CMUFaces" in images.__all__
    assert "Country211" in images.__all__
    assert images.CMUFaces is CMUFaces
    assert all(hasattr(images, name) for name in images.__all__)
