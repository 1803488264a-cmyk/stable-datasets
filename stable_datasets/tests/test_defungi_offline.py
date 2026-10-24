"""Offline checks of parsing, leakage prevention and the public cache interface."""

import io
import zipfile

import pytest
from PIL import Image

import stable_datasets.utils as dataset_utils
from stable_datasets.images import DeFungi
from stable_datasets.images.defungi import _scan_archive, _split_groups


def _jpeg(index):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (index * 7 % 256, index * 13 % 256, index * 23 % 256)).save(
        buffer, format="JPEG", quality=100, subsampling=0
    )
    return buffer.getvalue()


def _write_zip(path, files, reverse=False):
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(files, reverse=reverse):
            archive.writestr(name, files[name])
    return path


@pytest.fixture
def archive_files():
    return {
        f"{label}/{label}_{group}a_1.jpg": _jpeg(class_id * 10 + group)
        for class_id, label in enumerate(("H1", "H2", "H3", "H5", "H6"))
        for group in range(1, 7)
    }


@pytest.fixture
def archive_path(tmp_path, archive_files):
    return _write_zip(tmp_path / "defungi.zip", archive_files)


def _builder():
    builder = object.__new__(DeFungi)
    DeFungi.__init__(builder)
    return builder


def test_schema_and_exports():
    import stable_datasets.images as images

    builder = _builder()
    assert "DeFungi" in images.__all__
    assert builder.info.features["label"].names == ["H1", "H2", "H3", "H5", "H6"]
    assert builder.info.supervised_keys == ("image", "label")
    assert builder.SOURCE["license"] == "CC BY 4.0"
    assert all(asset.checksum.startswith("sha256:") for asset in builder.SOURCE["assets"].values())


def test_split_coverage_and_archive_order_independence(tmp_path, archive_files):
    first = _write_zip(tmp_path / "first.zip", archive_files)
    second = _write_zip(tmp_path / "second.zip", archive_files, reverse=True)
    builder = _builder()
    all_keys = []
    for split in ("train", "validation", "test"):
        rows = list(builder._generate_examples(first, split))
        repeated = list(builder._generate_examples(second, split))
        assert rows == repeated
        assert {row["label"] for _, row in rows} == set(range(5))
        assert [key for key, _ in rows] == sorted(key for key, _ in rows)
        for key, row in rows:
            assert set(row) == set(builder.info.features)
            assert key == row["image_id"]
            assert row["source_group"] == row["split_group"]
            with Image.open(io.BytesIO(row["image"])) as image:
                image.load()
                assert image.mode == "RGB"
        all_keys.extend(key for key, _ in rows)
    assert len(all_keys) == len(set(all_keys)) == len(archive_files)


def test_numeric_source_group_keeps_letter_variants_together(tmp_path, archive_files):
    archive_files["H1/H1_001b_9.jpg"] = _jpeg(90)
    path = _write_zip(tmp_path / "variants.zip", archive_files)
    with zipfile.ZipFile(path) as archive:
        records = _scan_archive(archive)
    assert {row[2] for row in records if row[0] in ("H1/H1_1a_1.jpg", "H1/H1_001b_9.jpg")} == {"H1_1"}


def test_duplicate_links_are_transitive_and_order_independent():
    records = [
        ("a", 0, "H1_1", "shared-a"),
        ("b", 0, "H1_2", "shared-a"),
        ("c", 0, "H1_2", "shared-b"),
        ("d", 0, "H1_3", "shared-b"),
        ("e", 0, "H1_4", "unique-e"),
        ("f", 0, "H1_5", "unique-f"),
    ]
    groups = _split_groups(records)
    assert groups == _split_groups(list(reversed(records)))
    assert groups["H1_1"] == groups["H1_2"] == groups["H1_3"]
    assert groups["H1_1"][0] == "H1_1"
    assert {assignment[1] for assignment in groups.values()} == {"train", "validation", "test"}


def test_duplicate_linked_sources_never_cross_splits(tmp_path, archive_files):
    archive_files["H1/H1_2b_2.jpg"] = archive_files["H1/H1_1a_1.jpg"]
    path = _write_zip(tmp_path / "duplicates.zip", archive_files)
    observed = {}
    for split in ("train", "validation", "test"):
        for name, row in _builder()._generate_examples(path, split):
            observed[name] = (split, row["split_group"])
    assert observed["H1/H1_1a_1.jpg"] == observed["H1/H1_2b_2.jpg"]
    assert observed["H1/H1_1a_1.jpg"] == observed["H1/H1_2a_1.jpg"]
    assert len(observed) == len(archive_files)


def test_identical_bytes_with_conflicting_labels_are_rejected():
    with pytest.raises(ValueError, match="conflicting labels"):
        _split_groups([("first", 0, "H1_1", "same"), ("second", 1, "H2_1", "same")])


def test_insufficient_groups_are_rejected():
    with pytest.raises(ValueError, match="at least three"):
        _split_groups([("first", 0, "H1_1", "one"), ("second", 0, "H1_2", "two")])


@pytest.mark.parametrize("split", ["valid", "unknown", ""])
def test_unknown_split_is_rejected(archive_path, split):
    with pytest.raises(ValueError, match="Unknown DeFungi split"):
        list(_builder()._generate_examples(archive_path, split))


@pytest.mark.parametrize("name", ["H1/H2_1a_1.jpg", "H4/H4_1a_1.jpg", "../H1_1a_1.jpg", "H1/unexpected.jpg"])
def test_invalid_archive_names_are_rejected(tmp_path, name):
    path = _write_zip(tmp_path / "invalid.zip", {name: _jpeg(1)})
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="filename"):
        _scan_archive(archive)


def test_empty_archive_is_rejected(tmp_path):
    path = _write_zip(tmp_path / "empty.zip", {"README.txt": b"No images"})
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="no supported images"):
        _scan_archive(archive)


def test_repeated_archive_member_is_rejected(tmp_path):
    path = tmp_path / "repeated.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("H1/H1_1a_1.jpg", _jpeg(1))
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("H1/H1_1a_1.jpg", _jpeg(2))
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="Duplicate DeFungi archive member"):
        _scan_archive(archive)


def test_public_loading_shared_download_and_cache_reuse(archive_path, tmp_path, monkeypatch):
    calls = []

    def download(specs, **kwargs):
        calls.append(len(specs))
        return [archive_path for _ in specs]

    monkeypatch.setattr(dataset_utils, "bulk_download", download)
    options = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    dataset = DeFungi(**options)
    assert {split: len(data) for split, data in dataset.items()} == {"train": 20, "validation": 5, "test": 5}
    assert calls == [1]

    def unexpected(*args, **kwargs):
        pytest.fail("A cached dataset attempted a download or rebuild.")

    monkeypatch.setattr(dataset_utils, "bulk_download", unexpected)
    monkeypatch.setattr(DeFungi, "_generate_examples", unexpected)
    cached = DeFungi(**options)
    train = DeFungi(split="train", **options)
    assert len(train) == 20
    for split in dataset:
        for first, second in zip(dataset[split], cached[split]):
            assert first["image"].tobytes() == second["image"].tobytes()
            for field in ("label", "image_id", "source_group", "split_group"):
                assert first[field] == second[field]
