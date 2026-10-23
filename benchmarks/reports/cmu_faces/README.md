# CMU Face Images: dataset integration and evaluation

This contribution adds a reproducible loader for the quarter-resolution CMU Face Images dataset and evaluates head-pose classification on held-out people. The loader preserves grayscale pixels and annotations, provides deterministic subject-disjoint splits, and uses the existing download and Arrow-cache interfaces.

## Source and scope

- Source: [UCI CMU Face Images](https://archive.ics.uci.edu/dataset/124/cmu+face+images).
- DOI: [10.24432/C5JC79](https://doi.org/10.24432/C5JC79).
- Citation: Mitchell, T. (1997). *CMU Face Images*. UCI Machine Learning Repository. UCI lists the donation date separately as June 23, 1999.
- License: CC BY 4.0, as listed by UCI. Dataset copyright/licensing is distinct from the repository code license.
- Raw ZIP SHA-256: `01a8b6cc58ac2f825f72abd6cf2b393df067c6529d96f7a90b22dede71b142cc`.
- Read `faces_4.tar.gz` inside the ZIP without extracting archive paths onto disk.
- Include 624 valid `.pgm` images, width 32 and height 30. Exclude entries marked `.bad` and other resolutions. The source describes 640 nominal images; this loader does not fabricate the 16 unavailable/corrupt images.
- This integration supports the quarter-resolution variant only. It does not introduce a new raw dataset or claim a new learning algorithm.

## Labels and custom split

The supervised task is four-class head pose. Label ids are fixed: `0=straight`, `1=left`, `2=right`, `3=up`. Metadata includes `subject` as a string, `expression` (`neutral`, `happy`, `sad`, `angry`), and `eyes` (`open`, `sunglasses`), with ids following the listed order.

This is an explicitly defined custom split, not an official CMU benchmark protocol. Subject groups are fixed before training and remain identical across seeds. No identity appears in more than one split, and resized copies are not mixed across splits.

| Split | Images | People | Subject ids |
|---|---:|---:|---|
| Train | 378 | 12 | an2i, at33, boland, bpm, ch4f, cheyer, choon, danieln, glickman, karyadi, kawamura, kk49 |
| Validation | 121 | 4 | megak, mitchell, night, phoebe |
| Test | 125 | 4 | saavik, steffi, sz24, tammo |

## Load the dataset

```python
from stable_datasets.images import CMUFaces

def main():
    dataset = CMUFaces()  # Download/build once, then reuse the cache.
    assert {split: len(data) for split, data in dataset.items()} == {
        "train": 378, "validation": 121, "test": 125,
    }
    sample = dataset["train"][0]
    print(sample["image"].size, sample["subject"], sample["label"])


if __name__ == "__main__":
    main()
```

The main guard supports the library downloader's multiprocessing on Windows when this example is saved as a Python script. `CMUFaces(split="train")` returns one split. Optional `download_dir` and `processed_cache_dir` relocate raw and processed files. Shared split assets are downloaded once. Images remain grayscale PIL images when decoded; conversion to RGB occurs in benchmark transforms only.

## Evaluation protocol

- Existing method: SimCLR, using `stable_pretraining.forward.simclr` and its NT-Xent loss, temperature 0.5.
- Backbone: randomly initialized `vit_tiny_patch16_224`, instantiated for 64×64 inputs. No pretrained weights. Patch size 16; embedding dimension 192.
- Projection head: the repository's three-layer MLP, hidden dimension 256 and output dimension 128.
- Training: 50 epochs, batch size 32, drop the final incomplete training batch, one optimizer step per batch, float32, zero loader workers, seeds 42/43/44.
- Backbone/projector optimizer configuration: AdamW, configured learning rate 0.0003, weight decay 0.05, betas (0.9, 0.999). The existing benchmark builds a step-based LinearWarmupCosineAnnealing schedule over 550 training steps, with a five-step warmup.
- Two independently augmented training views use random resized crops with area scale (0.8, 1.0), the existing aspect-ratio range (3/4, 4/3), and Gaussian blur with probability 0.5, sigma (0.1, 2.0), kernel size 23. Horizontal flips are disabled because left/right pose must not be treated as an invariant. The existing grayscale branch omits color jitter and random grayscale.
- Images are converted to RGB for the backbone. Normalize each repeated channel using native training-image mean 0.27856666 and standard deviation 0.24860810, computed over all training pixels in [0,1]. These are training-derived normalization constants, not statistics fitted to validation/test data.
- Validation/test transforms only convert to RGB, resize to 64×64, and normalize; there is no random evaluation augmentation.
- An online linear classifier is trained on detached backbone features using the benchmark's `OnlineProbe` defaults. Labels supervise this classifier, not the SimCLR encoder loss. This online probe is not a separately converged post-training linear-evaluation protocol.
- Select each seed's checkpoint by maximum validation macro-averaged class recall (the configured TorchMetrics `MulticlassAccuracy` default). Reloaded balanced accuracy reproduced all three recorded scores within 1e-5 before test scoring.
- Evaluate all three selected checkpoints on the complete test split. No seed is discarded, and test results are not used to select hyperparameters or retrain models.

Baselines use the same people and split assignments. The majority baseline predicts the most frequent training label. The pixel baseline standardizes 960 native grayscale pixel features using training-only means/variances and fits multinomial logistic regression on training labels. Choose C from {0.01, 0.1, 1, 10} by validation balanced accuracy (first value wins a tie). Selected C: **0.01**. Neither baseline uses external pretraining data. These are task-level comparisons, not a controlled ablation isolating representation quality.

## Results

All values below are percentages. Balanced accuracy is mean per-class recall; macro-F1 is the unweighted mean class F1. Raw confusion matrices in `evaluation_final.json` have true classes in rows and predicted classes in columns, ordered straight/left/right/up. Standard deviations use `ddof=1` across three training seeds and are not confidence intervals.

| Method | Validation balanced accuracy | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| SimCLR + online linear probe (seed 42) | 68.30 | 48.80 | 49.28 | 44.56 |
| SimCLR + online linear probe (seed 43) | 61.71 | 38.40 | 38.57 | 32.21 |
| SimCLR + online linear probe (seed 44) | 59.78 | 40.80 | 40.84 | 34.23 |
| Training majority class | 25.00 | 24.80 | 25.00 | 9.94 |
| Standardized pixels + logistic regression | 75.67 | 76.00 | 76.10 | 75.83 |
| SimCLR mean ± sample SD (3 seeds) | 63.27 ± 4.47 | 42.67 ± 5.45 | 42.90 ± 5.64 | 37.00 ± 6.62 |

Best checkpoint epochs (zero-based): seed 42 → 42, seed 43 → 49, seed 44 → 47. The three training logs reached epoch 49. The saved `last.ckpt` files contained epochs 42/49/47, so their epoch fields do not describe the completed training length. Evaluation explicitly follows the recorded `best_model_path`.

The pixel baseline outperforms these SimCLR runs on this custom split. SimCLR also varies across seeds and performs worse on held-out test people than validation people. This experiment establishes loader/benchmark functionality and provides an honest baseline; it does not establish that SSL is generally inferior. Only 20 people and four test identities are available. Seed variation does not measure uncertainty across alternative subject splits, and broader generalization cannot be inferred from these results.

## Reproduce

Run commands from the repository root. Match Python/package versions in `runtime_versions.json`; the recorded GPU was an NVIDIA RTX 4060 Laptop GPU. The exact stable-pretraining source revision was `ab836bf699a2be712dfcf980a5eb70d36391e876` (package `0.1.9.dev10+gab836bf69`). Install that revision rather than an unpinned main branch when reproducing these runs. CUDA wheels must match the environment; the recorded torch/torchvision builds were 2.13.0+cu126 / 0.28.0+cu126.

The dataset loader requires this contribution's source checkout; an older PyPI release does not contain `CMUFaces`. From that checkout, install the dataset library in an activated Python environment:

```text
python -m pip install -e .
```

For the benchmark scripts, install the pinned SSL implementation and the remaining benchmark/test dependencies, then check dependency consistency:

```text
python -m pip install "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining.git@ab836bf699a2be712dfcf980a5eb70d36391e876" "hydra-joblib-launcher>=1.2" psutil pytest
python -m pip check
```

The pinned SSL package declares its torch, torchvision, Lightning, timm, W&B, Hydra and datasets dependencies. These commands resolve dependencies for the current platform; `runtime_versions.json` records the versions used for the reported experiment rather than a complete environment lockfile. A supported CUDA installation is needed for GPU training. CPU execution is also supported but takes longer; identical results across hardware/software versions are not guaranteed.

The following commands use paths relative to the repository root and work in Windows Command Prompt and POSIX shells. First run a smoke test (three training and three validation batches):

```text
python -m benchmarks.run_cmu_faces --seed 42 --data-root ./cmu_data --output-root ./cmu_runs --smoke-test
```

Run the fixed full protocol for all three seeds:

```text
python -m benchmarks.run_cmu_faces --seed 42 --data-root ./cmu_data --output-root ./cmu_runs
python -m benchmarks.run_cmu_faces --seed 43 --data-root ./cmu_data --output-root ./cmu_runs
python -m benchmarks.run_cmu_faces --seed 44 --data-root ./cmu_data --output-root ./cmu_runs
```

Use a fresh `cmu_runs` directory for a new independent reproduction. The original runner may resume if a matching checkpoint already exists. Smoke and formal checkpoints are kept separate. The launcher uses the same training overrides as the recorded experiments. On Windows it replaces only a POSIX-specific signal diagnostic with portable logging, retaining the Manager's training/checkpoint behavior. The SimCLR builder uses the verified `forward.simclr` API; KNN receives an explicit number of classes.

Validate checkpoint reloads and baseline selection before test scoring:

```text
python -m benchmarks.evaluate_cmu_faces --data-root ./cmu_data --checkpoint-root ./cmu_runs/formal_checkpoints --output ./cmu_runs/reloaded_validation.json
```

After the protocol is fixed, score the test split with:

```text
python -m benchmarks.evaluate_cmu_faces --data-root ./cmu_data --checkpoint-root ./cmu_runs/formal_checkpoints --output ./cmu_runs/evaluation_final.json --include-test
```

The evaluator accepts relocated checkpoint directories, uses strict parameter loading, and stops on a validation-score mismatch before scoring test predictions. Downloaded images and binary checkpoints are not part of the repository submission. Consequently the committed JSON provides the recorded results, while rerunning inference requires retained or reproduced checkpoints.

## Tests

Offline tests construct miniature nested archives and cover all 32 pose/expression/eye combinations, fixed ordering, subject routing, bad-file exclusion, invalid metadata, public exports, shared downloads and cache reuse. Real-data tests additionally check exact split sizes, all 624 image decodes, label coverage and disjoint subject groups.

Offline and optional benchmark tests (from the repository root):

```text
python -m pytest stable_datasets/tests/test_cmu_faces.py stable_datasets/tests/test_cmu_faces_offline.py stable_datasets/tests/test_schema.py -q
python -m pytest stable_datasets/tests/test_cmu_benchmark.py -q
python -m pytest stable_datasets/tests/test_benchmarks.py::TestDatasetConfig stable_datasets/tests/test_benchmarks.py::TestTransforms -q
```

Without `CMU_FACES_ARCHIVE`, the real-data tests skip; the offline archive tests still run. To enable real-data tests after a smoke test has downloaded the ZIP, Windows Command Prompt users can run:

```text
for %F in (cmu_data\downloads\*cmu*face*images*.zip) do set "CMU_FACES_ARCHIVE=%~fF"
python -m pytest stable_datasets/tests/test_cmu_faces.py -q
```

In a POSIX shell:

```sh
CMU_FACES_ARCHIVE="$(find cmu_data/downloads -maxdepth 1 -type f -name '*cmu*face*images*.zip' -print -quit)" python -m pytest stable_datasets/tests/test_cmu_faces.py -q
```

The optional benchmark tests need torch and stable-pretraining and check actual image transforms, the CMU crop/flip policy including local-crop requests, and unchanged CIFAR defaults. A separate audit recomputed accuracy, balanced accuracy and macro-F1 from every committed confusion matrix and found agreement within 1e-12.
