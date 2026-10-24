# DeFungi: grouped-split SimCLR benchmark

## Data and split

The UCI DeFungi release contains 9,114 RGB microscopy patches (500x500) in classes H1, H2, H3, H5 and H6. Source, citation, CC BY 4.0 license, checksum, class counts and loader usage are documented in [the dataset page](../../../docs/source/datasets/defungi.rst).

This custom split uses filename-derived source groups joined transitively by exact JPEG-byte duplicate links. All 59 duplicate pairs remain within a partition; all released images are retained. There are 440 resulting groups, with 306/64/70 groups and 6,368/1,353/1,393 images in train/validation/test. The split is fixed independently of training seeds. These groups are not verified patient or slide identities; unknown related images can remain across partitions. Results are not directly comparable with papers using other splits.

## Protocol

Two independent runs use seeds 42 and 43, each with a 50-epoch training budget. The encoder is a randomly initialized `vit_tiny_patch16_224` adapted by the benchmark to 96x96 inputs. SimCLR uses two augmented views, batch size 32, projector hidden/output dimensions 256/128, temperature 0.5, full float32 precision and zero data-loader workers. Optimizer and scheduler settings are inherited unchanged from `benchmarks/conf/model/simclr.yaml`. No external pretrained weights are used.

The DeFungi configuration uses random resized crops with minimum scale 0.5 and the shared SimCLR color/flip augmentation. Normalization is computed on training images only: mean [0.53540354, 0.53722106, 0.50681663], standard deviation [0.07524247, 0.06417255, 0.08008631]. Evaluation uses deterministic RGB conversion, resize and normalization.

An online linear classifier learns from detached encoder features and training labels. Labels do not supervise the SimCLR encoder loss. This is an online probe, not a separately converged post-training linear evaluation. Each checkpoint is selected by validation balanced accuracy (macro-averaged recall). Reloading reproduced both recorded validation scores within 1e-5 before test predictions were scored. Both seeds are reported; no test-based seed or hyperparameter selection is performed.

Terminal logs reached epoch 49 for both runs. The retained `last.ckpt` files record epochs 35 and 29 and refer to the validation-selected checkpoints. The evaluator follows the recorded best-model path. The committed results describe these selected checkpoints; the checkpoint epoch fields do not establish the completed training length.

The majority baseline predicts H1, the most frequent training class. The pixel baseline resizes images to 16x16 RGB using bilinear interpolation, standardizes its 768 features using training-only statistics, and fits logistic regression on training labels. C is selected from [0.01, 0.1, 1, 10] using validation balanced accuracy; selected C is 0.01. These are simple task-level baselines, not a controlled representation ablation.

## Results

All values are percentages. Confusion matrices in `evaluation_final.json` use true labels in rows and predicted labels in columns, ordered H1/H2/H3/H5/H6. Sample SD uses ddof=1 across two seeds; it is not a confidence interval.

| Method | Validation balanced accuracy | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| SimCLR + online probe (seed 42) | 63.51 | 63.68 | 62.91 | 52.29 |
| SimCLR + online probe (seed 43) | 64.84 | 64.82 | 68.52 | 64.16 |
| Training majority class | 20.00 | 49.82 | 20.00 | 13.30 |
| 16x16 RGB pixels + logistic regression | 35.70 | 55.64 | 35.49 | 35.50 |
| SimCLR mean ± sample SD (2 seeds) | 64.18 ± 0.94 | 64.25 ± 0.81 | 65.72 ± 3.96 | 58.22 ± 8.40 |

Both SimCLR runs exceed the two simple baselines on the reported test metrics. However, seed 42 never predicts H2 on either validation or test images (zero H2 recall), while seed 43 has test H2 recall 160/311. Thus overall accuracy obscures a substantial class-specific failure and seed sensitivity. Two runs provide limited evidence about training variability and none about alternative split uncertainty. The benchmark demonstrates loader integration and a reproducible experimental protocol, not diagnostic suitability or state-of-the-art performance.

## Reproduction

Run from this contribution's repository checkout. Install the dataset package and benchmark dependencies in an activated environment:

```text
python -m pip install -e .
python -m pip install "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining.git@ab836bf699a2be712dfcf980a5eb70d36391e876" "hydra-joblib-launcher>=1.2" psutil pytest scikit-learn threadpoolctl
python -m pip check
```

The runs used the existing Windows CMU benchmark environment with an NVIDIA RTX 4060 Laptop GPU; the observed PyTorch build was 2.13.0+cu126. The source revision above is the CMU reproduction dependency pin; a complete DeFungi-specific environment export has not been captured in this report. Dependency resolution and hardware can affect exact reproducibility.

Use a fresh output directory for an independent reproduction. Existing paths may trigger the upstream runner's auto-resume behavior, and a retained `last.ckpt` may predate the completed training run.

```text
python -m benchmarks.run_defungi --seed 42 --data-root ./defungi_data --output-root ./defungi_runs --smoke-test
python -m benchmarks.run_defungi --seed 42 --data-root ./defungi_data --output-root ./defungi_runs
python -m benchmarks.run_defungi --seed 43 --data-root ./defungi_data --output-root ./defungi_runs
python -m benchmarks.evaluate_defungi --data-root ./defungi_data --checkpoint-root ./defungi_runs/formal_checkpoints --output ./defungi_runs/evaluation_final.json --include-test
```

The launcher preserves the training overrides used in the recorded command-line runs and replaces only the Windows-incompatible signal diagnostic. Smoke and formal checkpoint directories are separate. The evaluator strictly loads parameters and stops if validation scores disagree. Weights and raw images are not committed; rerunning inference requires retained or reproduced checkpoints.

## Validation

The original DeFungi loader/configuration suite passed 28 tests in the training environment. It covers real-image decoding, grouping/duplicate separation, offline errors, caching and benchmark transforms. All eight result confusion matrices and both seeds' summary statistics were independently recomputed from the supplied JSON and agree within 1e-12.

```text
python -m pytest stable_datasets/tests/test_defungi.py stable_datasets/tests/test_defungi_offline.py stable_datasets/tests/test_defungi_benchmark.py -q
```

Set `DEFUNGI_ARCHIVE` to the downloaded original ZIP to enable real-data tests; otherwise those tests skip. Training dependencies are needed for benchmark tests.
