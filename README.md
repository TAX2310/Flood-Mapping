# Flood-Mapping — Instruction Manual

This repository trains and evaluates deep-learning flood segmentation models on the
**STURM-Flood-24** dataset, using three input modalities:

- **Sentinel-1 SAR** (2-channel radar)
- **Sentinel-2 Optical** (9-channel multispectral)
- **Late Fusion** of SAR + Optical (dual-encoder U-Net)

All three pipelines (download → train → test → run inference → plot results) are
self-contained, one-notebook-per-modality:

| Notebook | Modality | Config class |
|---|---|---|
| [01_SAR.ipynb](01_SAR.ipynb) | Sentinel-1 SAR | `S1_CFG` |
| [02_Optical.ipynb](02_Optical.ipynb) | Sentinel-2 Optical | `S2_CFG` |
| [03_Fusion.ipynb](03_Fusion.ipynb) | SAR + Optical fusion | `Fusion_CFG` |

Each notebook is independent — you can run any one of them without the others, though
`03_Fusion.ipynb` is most meaningful to compare once you have S1 and S2 results.

---

## 1. Repository layout

```text
Flood-Mapping/
├── 01_SAR.ipynb            # Sentinel-1 pipeline: setup → train → test → infer → plot
├── 02_Optical.ipynb        # Sentinel-2 pipeline (same stages)
├── 03_Fusion.ipynb         # Fusion pipeline (same stages, + cross-modality comparison plots)
├── requirements.txt
├── src/
│   ├── config.py           # CFG / S1_CFG / S2_CFG / Fusion_CFG dataclasses
│   ├── data/
│   │   ├── sturm_fusion.py # download_and_extract(): fetch + unzip the dataset
│   │   ├── split.py        # builds sample index from metadata.csv, train/val/test split
│   │   ├── dataset.py      # SturmS1Dataset / SturmS2Dataset / SturmFusionDataset
│   │   ├── dataloader.py   # make_s1_dataloaders / make_s2_dataloaders / make_fusion_dataloaders
│   │   └── preprocessing.py# Preprocessing functions
│   ├── models/
│   │   ├── models.py       # Load models (U-Net / U-Net++ via segmentation-models-pytorch and custom models)
│   │   └── LateFusionUNetResNet34.py  # dual ResNet34-encoder fusion U-Net
│   ├── losses/losses.py    # Loss functions 
│   ├── train/
│   │   ├── training.py     # Training functions
│   │   └── train_from_file.py  # CLI entry point run as a subprocess
│   ├── test/testing.py     # Training functions
│   ├── inference/inference.py  # Inference functions — run a trained model on chosen/all test tiles
│   └── util/
│       ├── io.py           # save/load config & checkpoints, GeoTIFF I/O, results CSV
│       ├── metrics.py      # metrics from logits
│       ├── plotting.py     # all plotting functions used in the notebooks
│       └── seed.py         # Set random seed for reproducability 
```

---

## 2. Setup

Every notebook starts with the same **Setup** cell (a Colab form):

```python
root_path = "/content/drive/MyDrive/MSc/Flood-Mapping"
Dataset_url = "https://huggingface.co/datasets/tax2310/STURM-fusion-24/resolve/main/Dataset.zip"
mount_drive = False
clone_repo = False
```

| Variable | Effect |
|---|---|
| `mount_drive` | `True` mounts Google Drive at `/content/drive` and uses `root_path` as the project root (persistent storage). `False` works inside the ephemeral Colab/local runtime under `Flood-Mapping/`. |
| `clone_repo` | `True` clones `https://github.com/TAX2310/Flood-Mapping.git` into `root_path` if it doesn't already exist there. `False` assumes the repo is already present. |
| `root_path` | Where the project (code + `Dataset/` + `experiments/`) lives. Only used if `mount_drive=True`. |
| `Dataset_url` | URL of the dataset zip; passed into the config as `cfg.DATASET_URL`. |


Combinations:

At least one of `mount_drive` or `clone_repo` must be `True`, or the cell raises a `ValueError`.

- **mount + clone** → first-time Colab setup with persistent Drive storage.
- **mount only** → resume a previous Colab session, repo already on Drive.
- **clone only** → fresh temporary Colab/VM run, nothing persists after the session.
- **neither** → local development against an already-cloned repo.

After this cell, `cfg` is one of `S1_CFG()`, `S2_CFG()`, or `Fusion_CFG()` (from
[src/config.py](src/config.py)), with `cfg.ROOT`, `cfg.DATASET_URL`, and `cfg.DEVICE`
("cuda" if available, else "cpu") set.

### Install dependencies

```python
requirements = cfg.ROOT / "requirements.txt"
!pip install -r {requirements}
```

Key dependency: `segmentation-models-pytorch` (U-Net / U-Net++ encoders + decoders, Dice loss).

### Download and extract the dataset

```python
import src.data.sturm_fusion as SturmFusion
data_root = SturmFusion.download_and_extract(cfg)
```

`download_and_extract` ([src/data/sturm_fusion.py](src/data/sturm_fusion.py)):

1. Skips downloading if `cfg.S1_PATH` and `cfg.MASK_PATH` already exist, or if the zip is already on disk.
2. Otherwise `wget`s `cfg.ZIP_URL` to `cfg.ZIP_PATH` and extracts it under `cfg.ROOT`.
3. Deletes the zip after extraction.

Expected resulting structure under `cfg.DATA_PATH` (`Dataset/`):

```text
Dataset/
├── S1/            # Sentinel-1 GeoTIFFs, 2 bands, 128×128
├── S2/            # Sentinel-2 GeoTIFFs, 9 bands, 128×128
├── floodmaps/     # ground-truth mask GeoTIFFs (same tile_id filenames)
└── metadata/
    └── metadata.csv   # one row per tile_id, with ems_code (event), aoi_code, etc.
```

---

## 3. Configuration (`src/config.py`)

`CFG` is the shared base dataclass; `S1_CFG`, `S2_CFG`, `Fusion_CFG` extend it per modality.

Notable fields you may want to change before training:

| Field | Meaning |
|---|---|
| `SPLIT_METHOD` | `"random"` (default, ratio-based) or `"by_event"` (held-out EMSR events listed in `TRAIN_EVENTS`/`VAL_EVENTS`/`TEST_EVENTS`) |
| `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | Used only when `SPLIT_METHOD="random"`; must sum to 1.0 |
| `BINARY_MASK` | Collapses multi-class flood masks to binary water/non-water using `WATER_CLASSES`/`IGNORE_CLASSES` |
| `EPOCHS` / `PATIENCE` | Max epochs and early-stopping patience (on IoU, then F1 as tiebreak) |
| `THRESHOLD` | Sigmoid probability threshold for converting logits → binary prediction |
| `USE_ROTATIONS` | If `True`, training set is 4x augmented with 90°/180°/270° rotations |
| `MODEL` | Model name passed to `get_model()`, e.g. `"unet_resnet34_sar"`, `"unet_optical"`, `"unet_resnet34_fusion"` |
| `LR`, `BATCH_SIZE`, `WEIGHT_DECAY`, `DROPOUT_RATE` | Set per hyperparameter-sweep iteration (see §4) |

Each config also exposes derived `Path` properties (`DATA_PATH`, `S1_PATH`, `S2_PATH`,
`MASK_PATH`, `METADATA_CSV`, `EXP_DIR`, `EXPORT_DIR`, `S1_MODEL`/`S2_MODEL`/`FUSION_MODEL`
for the best-run experiment directory, and `*_TEST_RESULTS_CSV` paths) — use
these instead of hardcoding paths.

---

## 4. Training

```python
import src.train.training as training
import src.test.testing as testing
import src.util.io as io
import src.util.plotting as plot

learning_rates = [1e-3, 1e-4]
batch_sizes = [32, 64]
weight_decays = [0.0, 1e-5]
dropout_rates = [0.0, 0.2]
num_workers = 8

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for weight_decay in weight_decays:
            for dropout_rate in dropout_rates:
                cfg.LR = learning_rate
                cfg.BATCH_SIZE = batch_size
                cfg.WEIGHT_DECAY = weight_decay
                cfg.DROPOUT_RATE = dropout_rate
                training.train_from_file(cfg, num_workers=num_workers)
```

This performs a **grid search** over the four hyperparameter lists (16 combinations by
default). For each combination:

1. **`train_from_file(cfg, num_workers)`** ([src/train/training.py](src/train/training.py))
   pickles `cfg` to `tmp_config.pkl` and launches
   [src/train/train_from_file.py](src/train/train_from_file.py) as a **subprocess**
   (avoids Jupyter/Colab memory buildup across many training runs and allows for use of multiple workers), streaming its stdout
   back into the notebook.
2. The subprocess loads the config and calls **`training.train_model(cfg)`**, which:
   - Computes an experiment directory via `io.experiment_dir(cfg)`:
     `experiments/<DATASET>/<DATA_TYPE>/lr_<LR>/bs_<BATCH_SIZE>/wd_<WEIGHT_DECAY>/dr_<DROPOUT_RATE>/`
   - **Skips the run entirely** if `summary.json` already exists in that directory
     (safe to re-run the grid search — already-completed runs are not repeated).
   - Saves `config.json` / `config.pkl`, builds the model (`models.get_model(cfg)`) and
     dataloaders for the configured `DATA_TYPE`.
   - Optimizer: `AdamW(lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)`.
     Scheduler: `ReduceLROnPlateau(mode="min", factor=0.5, patience=3)` on val loss.
   - Loss: `losses.bce_dice` (0.5 · BCE-with-logits + 0.5 · Dice).
   - **Resumes from `checkpoint.pth`** if one exists in the experiment directory (so an
     interrupted run continues from the last completed epoch).
   - Each epoch: trains, validates, computes `accuracy/precision/recall/f1/iou`
     (`util/metrics.py`), and saves `best_model.pth` whenever IoU improves (F1 as
     tiebreaker, see `is_best()`).
   - **Early stopping**: stops after `cfg.PATIENCE` epochs without IoU improvement.
   - Writes `last_model.pth`, `metrics.csv` (full history), and `checkpoint.pth` after
     every epoch, and a final `summary.json` once done. The checkpoint file is deleted
     on successful completion (its presence signals "still running/interrupted").

**Outputs per run**, under `experiments/STURM-fusion-24/<DATA_TYPE>/lr_.../bs_.../wd_.../dr_.../`:

| File | Contents |
|---|---|
| `config.json`, `config.pkl` | The exact `cfg` used (pickle is what test/inference reload) |
| `best_model.pth` | State dict of the best-IoU epoch |
| `last_model.pth` | State dict of the most recently completed epoch |
| `metrics.csv` | Per-epoch train/val loss + val metrics |
| `summary.json` | Best epoch, best val loss, best val metrics, descriptive `title` |
| `checkpoint.pth` | Resume state (deleted when training finishes normally) |

### Comparing hyperparameter runs

```python
plot.plot_hp_comparison_bar(cfg)   # bar chart of F1/IoU across every completed run for this DATA_TYPE
plot.view_training_metrics(cfg)    # dropdown widget -> pick a run -> plots train/val loss, P/R, IoU/F1
```

`plot_hp_comparison_bar` reads every `summary.json` under
`experiments/<DATASET>/<DATA_TYPE>/`, so it reflects whatever grid search has completed
so far — partial sweeps are fine.

---

## 5. Testing

```python
testing.select_model_to_test(cfg)
```

This shows an `ipywidgets` dropdown of every leaf experiment directory under
`experiments/<DATASET>/<DATA_TYPE>/` (i.e., every hyperparameter combination trained so
far). Selecting one and clicking **"Test Model"** runs
**`testing.test_model(cfg, model_dir)`** ([src/test/testing.py](src/test/testing.py)):

- Reloads that run's own `config.pkl` (so test-time settings match training exactly).
- Loads `best_model.pth` into a freshly constructed model.
- Builds the **test split** dataloader (same `SPLIT_METHOD`/seed as training, so the
  held-out test set is reproduced deterministically).
- Computes test loss + metrics and **appends** them into that run's `summary.json` under
  a `"test_metrics"` key (via `io.update_summary`).

Run this once per model you intend to report on.

---

## 6. Inference & exporting predictions

```python
import src.inference.inference as inference

samples = ["EMSR470_AOI01_46_07_2_1.tif",
           "EMSR441_AOI05_2_3_2_2.tif",
           "EMSR570_AOI02_07_03_2_1.tif"]

results = inference.inference(cfg, cfg.S1_MODEL, samples)      # a handful of tiles, for visual inspection
all_results = inference.inference(cfg, cfg.S1_MODEL)            # every tile in the test split
io.create_inference_results_csv(cfg, all_results, cfg.METADATA_CSV, cfg.S1_TEST_RESULTS_CSV)
```

(Replace `cfg.S1_MODEL` with `cfg.S2_MODEL` / `cfg.FUSION_MODEL` in the respective
notebooks — these point at the canonical best hyperparameter combination defined in
`src/config.py`. Point `model_dir` at any other experiment directory to inspect a
different run.)

- **`inference.inference(cfg, model_dir, samples=None, export=False)`**
  ([src/inference/inference.py](src/inference/inference.py)):
  - `samples=None` → runs over the **entire test split** for that `DATA_TYPE`.
  - `samples=[...]` → filters to only the listed `sample_id`s (tile filenames), useful
    for quick qualitative checks on specific events/tiles.
  - For every sample returns image(s), ground-truth mask, predicted probability map,
    thresholded prediction, file paths, and per-sample metrics.
  - `export=True` additionally writes probability/prediction GeoTIFFs to
    `cfg.EXPORT_DIR/probability/` and `cfg.EXPORT_DIR/prediction/` (via
    `io.export_prediction_tifs`), reusing the georeferencing of the input tile.
- **`io.create_inference_results_csv(...)`** joins per-sample metrics with the dataset
  metadata (`metadata.csv`) and per-mask flood-pixel statistics, producing one row per
  test tile — this CSV is what all the distribution/scatter plots consume.

---

## 7. Visualizing results

Qualitative, single-sample plots (used on the small `samples` list):

```python
plot.plot_s1_results(results)      # 01_SAR.ipynb
plot.plot_s2_results(results)      # 02_Optical.ipynb
plot.plot_fusion_results(results)  # 03_Fusion.ipynb
```

Each prints sample tensor shapes, then for every sample shows: the input image(s)
(false-colour for SAR/S2), ground-truth mask, predicted mask, an FP/FN overlay
(black=no flood, blue=false positive, red=false negative, white=true flood), and the
raw probability map.

Aggregate, whole-test-set plots (used on the `*_TEST_RESULTS_CSV` from §6):

```python
plot.plot_metric_distribution_from_csv(cfg.S1_TEST_RESULTS_CSV)        # histogram of IoU (or any metric) across all test tiles
plot.plot_iou_vs_flood_scatter([cfg.S1_TEST_RESULTS_CSV])              # IoU vs. % flood coverage per tile
plot.plot_average_iou_per_event(cfg.S1_TEST_RESULTS_CSV)               # bar chart of mean IoU per EMSR event
```

`03_Fusion.ipynb` additionally compares all three modalities once their result CSVs exist:

```python
plot.plot_iou_vs_flood_scatter([cfg.S2_TEST_RESULTS_CSV,
                                 cfg.S1_TEST_RESULTS_CSV,
                                 cfg.FUSION_TEST_RESULTS_CSV], figsize=(10, 10))

plot.plot_fusion_improvement_distribution([cfg.S2_TEST_RESULTS_CSV,
                                            cfg.S1_TEST_RESULTS_CSV,
                                            cfg.FUSION_TEST_RESULTS_CSV])
```

`plot_fusion_improvement_distribution` merges all three CSVs by tile, computes
`fusion_iou - max(s1_iou, s2_iou)` per tile, and histograms that delta — a positive mean
shows fusion outperforming the best single modality on each tile.

> Run `01_SAR.ipynb` and `02_Optical.ipynb` (through §6, to produce their
> `*_TEST_RESULTS_CSV` files) **before** running these cross-modality comparison cells
> in `03_Fusion.ipynb`.

---

## 8. End-to-end checklist

1. Run the **Setup** cell (choose `mount_drive`/`clone_repo` for your environment).
2. Run **download + extract** and **`pip install -r requirements.txt`**.
3. Define your hyperparameter grid and run the **training loop** (§4). Re-running is
   safe — completed combinations are skipped automatically.
4. Use `plot.plot_hp_comparison_bar` / `plot.view_training_metrics` to pick the best run.
5. `testing.select_model_to_test(cfg)` → select that run → test it.
6. Run **inference** (§6) on a few samples for a sanity check, then on the full test
   split to build `*_TEST_RESULTS_CSV`.
7. Visualize qualitative and aggregate results (§7).
8. For fusion analysis, repeat 1–6 for S1 and S2 first, then run `03_Fusion.ipynb`
   through to its cross-modality comparison plots.

---

## 9. Key implementation notes

- **Reproducibility**: `seed.set_seed(cfg.RANDOM_SEED)` is called at the start of every
  `make_*_dataloaders` call, so the random train/val/test split (`SPLIT_METHOD="random"`)
  is reproducible given the same seed and sample list.
- **Mask binarization**: ground-truth masks are multi-class GeoTIFFs; `BINARY_MASK=True`
  (default) remaps `WATER_CLASSES=(1,2,3,4,5)` → `1` and `IGNORE_CLASSES=(99,)` → `255`,
  everything else → `0` (`preprocessing.remap_mask_to_binary`).
- **Fusion model** ([src/models/LateFusionUNetResNet34.py](src/models/LateFusionUNetResNet34.py)):
  two independent ResNet34 encoders (one per modality) feed 1×1-conv fusion blocks at
  every decoder skip level, into a single shared U-Net decoder + segmentation head.
- **`train_from_file` runs training out-of-process** deliberately — running 16+
  sequential trainings inside one notebook kernel can leak GPU/CPU memory over time;
  each subprocess starts clean and exits when done.
- **Why subprocess + pickle config**: the config dataclass instance is pickled rather
  than passed as CLI args because it carries `Path` objects and tuples that don't survive
  `argparse` cleanly.
