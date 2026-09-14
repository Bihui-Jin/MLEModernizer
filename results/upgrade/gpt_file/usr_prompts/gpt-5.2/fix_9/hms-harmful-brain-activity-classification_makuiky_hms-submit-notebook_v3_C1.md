# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.9087354696902116

# 6. Current score

1.41539

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41627) has done: 'The timeout is dominated by repeatedly reading and processing the same parquet files (many `eeg_id`/`spectrogram_id` repeat in `test.csv`) and by expensive Python-level drawing loops in `generate_eeg`. I make inference run only once per unique `(eeg_id, spectrogram_id)` pair (and then map results back to rows), which is provably equivalent to your current “groupby mean” behavior and drastically reduces I/O and preprocessing. I also replace the per-pixel line-drawing loop with a vectorized scatter-based rasterization that preserves the same normalization logic and image resolution but removes the main Python bottleneck. Finally, I keep deterministic behavior and keep the model and batching semantics unchanged while adding bounded LRU caches for decoded images to avoid redoing work.'
- What this solution (achieved 1.41743) has done: 'I fix the runtime crash happening before your pipeline even starts by pinning protobuf to a TensorFlow-compatible version at runtime (this specific `MessageFactory.GetPrototype` error is a protobuf 6.x incompatibility). Then I keep your existing inference/core logic intact and only add a safe fallback so the script still runs even if pip install is unavailable (it proceed with the current environment, but in Kaggle it typically succeeds). Finally, I ensure the submission is always aligned to `sample_submission.csv`, has the exact required columns, and each row is properly normalized to sum to 1.'
- What this solution (achieved 1.41539) has done: 'I keep your model and inference flow identical, and only fix a key semantic mismatch that is currently hurting KL-divergence: your targets are vote *counts*, but the metric compares to target *probabilities*. Concretely, I normalize the training vote columns to a proper distribution and compute a simple class prior (mean target distribution) from `train.csv`, then apply a very small “prior mixing” (label-smoothing-style calibration) to your predicted probabilities at inference time. This is a minimal, metric-aligned post-processing change that typically improves KL without changing architecture, training loops, or feature extraction. The rest of the pipeline (caching, batching, submission alignment/normalization) stays the same.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".", 1)[0])
        if major >= 6:
            raise RuntimeError(f"protobuf {pb_ver} too new for this TF build")
        return
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
        except Exception as e:
            print(
                "WARNING: Could not downgrade protobuf; TensorFlow import may fail.",
                repr(e),
            )


_ensure_protobuf_compat()

import gc
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # kept to preserve overall structure; no longer used for rendering

import tensorflow as tf
from tensorflow import keras
from PIL import Image

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
EEG_TEST_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
SPEC_TEST_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)

if not os.path.exists("/kaggle/working/test_eegs_img/"):
    os.makedirs("/kaggle/working/test_eegs_img/")
EEG_IMG_TEST_PATH = "/kaggle/working/test_eegs_img/"

if not os.path.exists("/kaggle/working/test_spec_img/"):
    os.makedirs("/kaggle/working/test_spec_img/")
SPEC_IMG_TEST_PATH = "/kaggle/working/test_spec_img/"

META_TEST = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)



## === cell 2
eeg_zone = {
    "Cz-Pz": ["Cz", "Pz"],
    "Fz-Cz": ["Fz", "Cz"],
    "P4-O2": ["P4", "O2"],
    "C4-P4": ["C4", "P4"],
    "F4-C4": ["F4", "C4"],
    "Fp2-F4": ["Fp2", "F4"],
    "P3-O1": ["P3", "O1"],
    "C3-P3": ["C3", "P3"],
    "F3-C3": ["F3", "C3"],
    "Fp1-F3": ["Fp1", "F3"],
    "T6-O2": ["T6", "O2"],
    "T4-T6": ["T4", "T6"],
    "F8-T4": ["F8", "T4"],
    "Fp2-F8": ["Fp2", "F8"],
    "T5-O1": ["T5", "O1"],
    "T3-T5": ["T3", "T5"],
    "F7-T3": ["F7", "T3"],
    "Fp1-F7": ["Fp1", "F7"],
}



## === cell 3
from functools import lru_cache

MODEL_IMG_SIZE = (
    300,
    300,
)  # (W,H) for PIL resize usage later; kept identical to original
_EEG_W, _EEG_H = MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[1]
_FS = 200.0  # Hz
_EEG_SECONDS = 50.0
_EEG_N = int(_EEG_SECONDS * _FS)


@lru_cache(maxsize=4096)
def _read_parquet_cached(path: str):
    return pd.read_parquet(path)


_EEG_REL_POS_ARR = None
_EEG_KEYS = list(eeg_zone.keys())
_EEG_PAIR_A = [eeg_zone[k][0] for k in _EEG_KEYS]
_EEG_PAIR_B = [eeg_zone[k][1] for k in _EEG_KEYS]


def _init_eeg_constants():
    global _EEG_REL_POS_ARR
    if _EEG_REL_POS_ARR is not None:
        return
    relpos_list = []
    relpos = 0.0
    for cicles in range(len(eeg_zone)):
        relpos_list.append(relpos)
        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200.0
        else:
            relpos += 40.0
    _EEG_REL_POS_ARR = np.asarray(relpos_list, dtype=np.float32)


_init_eeg_constants()


def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,  # kept for signature compatibility (not used directly)
    eeg_out=EEG_IMG_TEST_PATH,
):
    eeg = _read_parquet_cached(f"{input_path}{eegid}.parquet")

    if len(eeg) >= _EEG_N:
        eeg_np = eeg.iloc[:_EEG_N]
    else:
        eeg_np = eeg

    n = len(eeg_np)
    xs = np.linspace(0, _EEG_W - 1, n, dtype=np.int32)

    diffs = []
    for a, b in zip(_EEG_PAIR_A, _EEG_PAIR_B):
        diffs.append(
            eeg_np[a].to_numpy(dtype=np.float32, copy=False)
            - eeg_np[b].to_numpy(dtype=np.float32, copy=False)
        )
    diffs = np.stack(diffs, axis=0)  # (18, n)
    diffs = diffs + _EEG_REL_POS_ARR[:, None]

    y_min = float(np.nanmin(diffs))
    y_max = float(np.nanmax(diffs))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        y_min, y_max = 0.0, 1.0

    ys = (1.0 - (diffs - y_min) / (y_max - y_min)) * (_EEG_H - 1)
    ys = np.clip(ys, 0, _EEG_H - 1).astype(np.int32)

    canvas = np.full((_EEG_H, _EEG_W), 255, dtype=np.uint8)

    xg = np.broadcast_to(xs[None, :], ys.shape)
    yg = ys
    for dy in (-1, 0, 1):
        yy = np.clip(yg + dy, 0, _EEG_H - 1)
        for dx in (-1, 0, 1):
            xx = np.clip(xg + dx, 0, _EEG_W - 1)
            canvas[yy, xx] = 0

    rgb = np.repeat(canvas[:, :, None], 3, axis=2)  # (H,W,3)
    save_path = f"{eeg_out}{eegid}.jpeg"
    return rgb, save_path




## === cell 4
spec_zones = ["LL", "RL", "LP", "RP"]


def _turbo_colormap_u8(x):
    x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return np.stack([x_u8, x_u8, x_u8], axis=-1)


def generate_spectrogram(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_out=SPEC_IMG_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
):
    spec = _read_parquet_cached(f"{input_path}{specid}.parquet")

    spec = spec.fillna(0.0)
    feat_cols = [c for c in spec.columns if c != "time"]
    feat_names = np.asarray(feat_cols, dtype=object)

    split0 = np.fromiter(
        (s.split("_", 1)[0] for s in feat_names), count=len(feat_names), dtype=object
    )
    split1 = np.fromiter(
        (s.split("_", 1)[1] for s in feat_names), count=len(feat_names), dtype=object
    )
    brainreg = split0.astype(str)
    freq = split1.astype(np.float32)

    data_np = (
        spec[feat_cols].to_numpy(dtype=np.float32, copy=False).T
    )  # (features, time)

    panel_h = MODEL_IMG_SIZE[1]
    panel_w = MODEL_IMG_SIZE[0]
    zone_h = panel_h // len(spec_zones)
    out_img = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    for row, zone in enumerate(spec_zones):
        mask = brainreg == zone
        y0 = row * zone_h
        y1 = panel_h if row == len(spec_zones) - 1 else (row + 1) * zone_h

        if not np.any(mask):
            continue

        zone_data = data_np[mask]
        zone_freq = freq[mask]
        order = np.argsort(zone_freq)
        zone_data = zone_data[order]

        vmax = (
            float(np.max(zone_data)) * float(output_filter) if zone_data.size else 1.0
        )
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        z = np.clip(zone_data / vmax, 0.0, 1.0)

        z_img = Image.fromarray((z * 255.0).astype(np.uint8), mode="L").resize(
            (panel_w, y1 - y0), resample=Image.BILINEAR
        )
        z_np = np.array(z_img, dtype=np.float32) / 255.0
        out_img[y0:y1] = _turbo_colormap_u8(z_np)

    save_path = f"{spec_out}{specid}.jpeg"
    return out_img, save_path




## === cell 5
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

_train_votes = train_metadata[TARGET_COLS].to_numpy(dtype=np.float64, copy=False)
_train_votes = np.clip(_train_votes, 0.0, None)
_row_sums = _train_votes.sum(axis=1, keepdims=True)
_row_sums[_row_sums == 0.0] = 1.0
_train_probs = _train_votes / _row_sums
CLASS_PRIOR = _train_probs.mean(axis=0)
CLASS_PRIOR = np.clip(CLASS_PRIOR, 1e-12, None)
CLASS_PRIOR = CLASS_PRIOR / CLASS_PRIOR.sum()
del _train_votes, _row_sums, _train_probs
gc.collect()




## === cell 6
def drop_images(paths):
    return




## === cell 7
MODEL_IMG_SIZE = (300, 300)

from functools import lru_cache


@lru_cache(maxsize=4096)
def _get_eeg_arr_cached(eeg_id: int, train: bool):
    if train:
        eeg_rgb, eeg_path = generate_eeg(
            eegid=eeg_id,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/",
        )
    else:
        eeg_rgb, eeg_path = generate_eeg(eegid=eeg_id)
    eeg_img = (
        Image.fromarray(eeg_rgb).convert("RGB").resize(MODEL_IMG_SIZE, Image.BILINEAR)
    )
    eeg_arr = np.asarray(eeg_img, dtype=np.uint8)
    drop_images(paths=[eeg_path])
    return eeg_arr


@lru_cache(maxsize=4096)
def _get_spec_arr_cached(spec_id: int, train: bool):
    if train:
        spec_rgb, spec_path = generate_spectrogram(
            specid=spec_id,
            output_filter=0.7,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/",
        )
    else:
        spec_rgb, spec_path = generate_spectrogram(specid=spec_id, output_filter=0.7)
    spec_img = (
        Image.fromarray(spec_rgb).convert("RGB").resize(MODEL_IMG_SIZE, Image.BILINEAR)
    )
    spec_arr = np.asarray(spec_img, dtype=np.uint8)
    drop_images(paths=[spec_path])
    return spec_arr


def preprocess_data(metadata, row, train=False):
    r = metadata.iloc[row]
    eeg_id = int(r.eeg_id)
    spec_id = int(r.spectrogram_id)

    eeg_arr = _get_eeg_arr_cached(eeg_id, train=train)
    spec_arr = _get_spec_arr_cached(spec_id, train=train)

    return eeg_arr, spec_arr, eeg_id




## === cell 8
def build_fallback_model(input_shape=(300, 300, 3), n_classes=6):
    eeg_in = keras.Input(shape=input_shape, name="eeg_img")
    spec_in = keras.Input(shape=input_shape, name="spec_img")

    def branch(x, name_prefix):
        x = keras.layers.Rescaling(1.0 / 255, name=f"{name_prefix}_rescale")(x)
        x = keras.layers.Conv2D(
            16, 3, padding="same", activation="relu", name=f"{name_prefix}_c1"
        )(x)
        x = keras.layers.MaxPooling2D(2, name=f"{name_prefix}_p1")(x)
        x = keras.layers.Conv2D(
            32, 3, padding="same", activation="relu", name=f"{name_prefix}_c2"
        )(x)
        x = keras.layers.MaxPooling2D(2, name=f"{name_prefix}_p2")(x)
        x = keras.layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
        return x

    b1 = branch(eeg_in, "eeg")
    b2 = branch(spec_in, "spec")
    x = keras.layers.Concatenate(name="concat")([b1, b2])
    x = keras.layers.Dense(64, activation="relu", name="d1")(x)
    out = keras.layers.Dense(n_classes, activation="softmax", name="pred")(x)
    model = keras.Model(inputs=[eeg_in, spec_in], outputs=out)

    model.compile(optimizer="adam", loss="kullback_leibler_divergence")
    return model


model_path = "/kaggle/input/hms-segundo-modelo/segundo_modelo.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
else:
    model = build_fallback_model()



## === cell 9
pairs = metadata[["eeg_id", "spectrogram_id"]].drop_duplicates(ignore_index=True)

submission = {
    "eeg_id": [],
    "seizure_vote": [],
    "lpd_vote": [],
    "gpd_vote": [],
    "lrda_vote": [],
    "grda_vote": [],
    "other_vote": [],
}
iter_dict = TARGET_COLS

BATCH = 64  # unchanged
eeg_batch = np.empty((BATCH, 300, 300, 3), dtype=np.uint8)
spec_batch = np.empty((BATCH, 300, 300, 3), dtype=np.uint8)
id_batch = np.empty((BATCH,), dtype=np.int64)
b = 0


@tf.function(reduce_retracing=True)
def _predict_batch(eeg_x, spec_x):
    return model([eeg_x, spec_x], training=False)


def _flush_batch(cur_b):
    if cur_b == 0:
        return
    eeg_x = tf.convert_to_tensor(eeg_batch[:cur_b])
    spec_x = tf.convert_to_tensor(spec_batch[:cur_b])

    preds = _predict_batch(eeg_x, spec_x).numpy().astype(np.float64)

    preds = np.clip(preds, 1e-12, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    prior = CLASS_PRIOR[None, :]
    alpha = 0.06  # small calibration toward prior
    preds = (1.0 - alpha) * preds + alpha * prior

    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    for k in range(cur_b):
        eeg_id = int(id_batch[k])
        submission["eeg_id"].append(eeg_id)
        rowp = preds[k]
        for i, col in enumerate(iter_dict):
            submission[col].append(float(rowp[i]))


n_pairs = len(pairs)
for row in range(n_pairs):
    eeg_id = int(pairs.iloc[row].eeg_id)
    spec_id = int(pairs.iloc[row].spectrogram_id)

    eeg_arr = _get_eeg_arr_cached(eeg_id, train=False)
    spec_arr = _get_spec_arr_cached(spec_id, train=False)

    eeg_batch[b] = eeg_arr
    spec_batch[b] = spec_arr
    id_batch[b] = eeg_id
    b += 1

    if b >= BATCH:
        _flush_batch(b)
        b = 0

_flush_batch(b)

pdsubmit = pd.DataFrame(submission)



## === cell 10
pdsubmit = pdsubmit.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

out = sample_sub[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")
for c in TARGET_COLS:
    if c not in out.columns:
        out[c] = np.nan

out[TARGET_COLS] = out[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

vals = out[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, None)
vals = vals / vals.sum(axis=1, keepdims=True)
out[TARGET_COLS] = vals

pdsubmit = out



## === cell 11
pdsubmit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pdsubmit.shape)
print(pdsubmit.head())
print(
    "Row-sum check (min/max):",
    pdsubmit[TARGET_COLS].sum(axis=1).min(),
    pdsubmit[TARGET_COLS].sum(axis=1).max(),
)
