# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend in Kaggle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image
import gc
import warnings
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
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



## === cell 2
EEG_TRAIN_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
SPEC_TRAIN_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
META_TRAIN = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"



## === cell 3
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




## === cell 4
def _rasterize_polyline_min_at(canvas, xs, ys, color_val=0.0, thickness=1):
    h, w = canvas.shape
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    if xs.size < 2:
        return

    x0 = xs[:-1]
    y0 = ys[:-1]
    x1 = xs[1:]
    y1 = ys[1:]

    steps = np.maximum(np.abs(x1 - x0), np.abs(y1 - y0)).astype(np.int32) + 1
    total = int(steps.sum())
    if total <= 0:
        return

    seg_offsets = np.cumsum(np.concatenate(([0], steps[:-1]))).astype(np.int64)
    pos = np.arange(total, dtype=np.int64)
    seg_id = np.searchsorted(seg_offsets, pos, side="right") - 1
    seg_id = np.clip(seg_id, 0, steps.size - 1)

    local = pos - seg_offsets[seg_id]
    denom = (steps[seg_id] - 1).astype(np.float32)
    denom = np.where(denom <= 0, 1.0, denom)
    t = local.astype(np.float32) / denom

    xline = np.rint(
        x0[seg_id].astype(np.float32) + (x1[seg_id] - x0[seg_id]).astype(np.float32) * t
    ).astype(np.int32)
    yline = np.rint(
        y0[seg_id].astype(np.float32) + (y1[seg_id] - y0[seg_id]).astype(np.float32) * t
    ).astype(np.int32)

    xline = np.clip(xline, 0, w - 1)
    yline = np.clip(yline, 0, h - 1)

    if thickness <= 1:
        np.minimum.at(canvas, (yline, xline), color_val)
    else:
        for tt in range(-thickness // 2, thickness // 2 + 1):
            yt = np.clip(yline + tt, 0, h - 1)
            np.minimum.at(canvas, (yt, xline), color_val)


def _precompute_eeg_offsets(eeg_zone_dict):
    cicles = 0
    relpos = 0.0
    offsets = []
    for _ in eeg_zone_dict.items():
        offsets.append(relpos)
        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200
        else:
            relpos += 40
        cicles += 1
    return np.asarray(offsets, dtype=np.float32)


_EEG_OFFSETS = _precompute_eeg_offsets(eeg_zone)
_EEG_COLS_NEEDED = sorted({c for ab in eeg_zone.values() for c in ab})

_EEG_CANVAS_HW = (300, 300)
_EEG_XS_300 = None


def _read_eeg_np(path):
    table = pq.read_table(path, columns=_EEG_COLS_NEEDED)  # pyarrow.Table
    out = {}
    for c in _EEG_COLS_NEEDED:
        col = table[c]
        if hasattr(col, "combine_chunks"):
            col = col.combine_chunks()
        out[c] = col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return out


def generate_eeg_tensor(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,  # kept for signature compatibility
    out_hw=(300, 300),
):
    global _EEG_XS_300
    eeg = _read_eeg_np(f"{input_path}{eegid}.parquet")

    h, w = out_hw
    canvas = np.ones((h, w), dtype=np.float32)

    n = len(next(iter(eeg.values())))
    if out_hw == _EEG_CANVAS_HW and _EEG_XS_300 is not None and _EEG_XS_300.size == n:
        xs = _EEG_XS_300
    else:
        xs = np.linspace(0, w - 1, n, dtype=np.int32)
        if out_hw == _EEG_CANVAS_HW:
            _EEG_XS_300 = xs

    offsets = _EEG_OFFSETS

    diffs = []
    for _, (a, b) in eeg_zone.items():
        diffs.append(eeg[a] - eeg[b])
    diffs = np.stack(diffs, axis=0)  # (n_ch, n)

    diffs = diffs + offsets[:, None]

    y_min = float(np.nanmin(diffs))
    y_max = float(np.nanmax(diffs))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        y_min, y_max = -1.0, 1.0

    ys_all = (h - 1) - ((diffs - y_min) / (y_max - y_min) * (h - 1))
    ys_all = np.clip(ys_all, 0, h - 1).astype(np.int32)

    thickness = 1
    for ch in range(ys_all.shape[0]):
        _rasterize_polyline_min_at(
            canvas, xs, ys_all[ch], color_val=0.0, thickness=thickness
        )

    arr = np.repeat(canvas[:, :, None], 3, axis=2)
    return arr




## === cell 5
spec_zones = ["LL", "RL", "LP", "RP"]

_SPEC_COL_CACHE = {
    "cols_tuple": None,
    "zone_sorted_idx": None,  # dict zone -> indices into feature rows (excluding time)
}


def _get_spec_col_meta(data_cols):
    key = tuple(data_cols)
    if _SPEC_COL_CACHE["cols_tuple"] == key:
        return _SPEC_COL_CACHE["zone_sorted_idx"]

    cols = np.asarray(data_cols, dtype=object)
    parts = np.char.split(cols.astype(str), "_")
    zone = np.array([p[0] if len(p) > 0 else "" for p in parts], dtype=object)
    freq = np.array(
        [float(p[1]) if len(p) > 1 else np.nan for p in parts], dtype=np.float32
    )

    zone_sorted_idx = {}
    for z in spec_zones:
        mask = zone == z
        if not np.any(mask):
            zone_sorted_idx[z] = np.empty((0,), dtype=np.int64)
            continue
        idx = np.nonzero(mask)[0]
        order = np.argsort(freq[idx])
        zone_sorted_idx[z] = idx[order].astype(np.int64)

    _SPEC_COL_CACHE["cols_tuple"] = key
    _SPEC_COL_CACHE["zone_sorted_idx"] = zone_sorted_idx
    return zone_sorted_idx


def _read_spec_np(path):
    table = pq.read_table(path)
    names = table.schema.names
    time_col = table["time"]
    if hasattr(time_col, "combine_chunks"):
        time_col = time_col.combine_chunks()
    time = time_col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    data_cols = [c for c in names if c != "time"]
    cols_arr = []
    for c in data_cols:
        col = table[c]
        if hasattr(col, "combine_chunks"):
            col = col.combine_chunks()
        cols_arr.append(
            col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        )
    M = np.stack(cols_arr, axis=1).T  # (n_features, n_time)
    if np.isnan(M).any():
        np.nan_to_num(M, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return time, data_cols, M


def generate_spectrogram_tensor(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
    out_hw=(300, 300),
):
    time, data_cols, M = _read_spec_np(f"{input_path}{specid}.parquet")
    zone_sorted_idx = _get_spec_col_meta(data_cols)

    panels = []
    for z in spec_zones:
        idx = zone_sorted_idx.get(z, None)
        if idx is None or idx.size == 0:
            panels.append(np.zeros((1, len(time)), dtype=np.float32))
            continue
        m = M[idx]
        vmax = float(np.max(m) * output_filter) if m.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        m = np.clip(m / vmax, 0.0, 1.0)
        panels.append(m)

    total_rows = sum(p.shape[0] for p in panels)
    if total_rows <= 0:
        img = np.zeros(out_hw, dtype=np.float32)
    else:
        stacked = np.concatenate(panels, axis=0)  # (freq_bins_total, n_time)
        t = tf.convert_to_tensor(stacked[:, :, None], dtype=tf.float32)
        t = tf.image.resize(t, out_hw, method="bilinear", antialias=False)
        img = tf.squeeze(t, axis=-1).numpy().astype(np.float32, copy=False)

    arr = np.repeat(img[:, :, None], 3, axis=2)
    return arr




## === cell 6
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(META_TRAIN)

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 7
def _image_array_to_tensor(arr):
    t = tf.convert_to_tensor(arr, dtype=tf.float32)
    t = tf.expand_dims(t, axis=0)
    return t


def preprocess_data(metadata_df, row, train=False):
    r = metadata_df.iloc[row]
    if train:
        eeg_arr = generate_eeg_tensor(
            eegid=int(r.eeg_id),
            input_path=EEG_TRAIN_PATH,
        )
        spec_arr = generate_spectrogram_tensor(
            specid=int(r.spectrogram_id),
            input_path=SPEC_TRAIN_PATH,
            output_filter=0.7,
        )
    else:
        eeg_arr = generate_eeg_tensor(eegid=int(r.eeg_id))
        spec_arr = generate_spectrogram_tensor(
            specid=int(r.spectrogram_id), output_filter=0.7
        )

    eeg_img_tr = _image_array_to_tensor(eeg_arr)
    spec_img_tr = _image_array_to_tensor(spec_arr)

    eeg_id = int(r.eeg_id)
    return eeg_img_tr, spec_img_tr, eeg_id




## === cell 8
def build_fallback_model(input_shape=(300, 300, 3), n_classes=6):
    eeg_in = keras.Input(shape=input_shape, name="eeg_img")
    spec_in = keras.Input(shape=input_shape, name="spec_img")

    def branch(x, name_prefix):
        x = layers.Resizing(300, 300, name=f"{name_prefix}_resize")(x)
        x = layers.Conv2D(
            16, 3, padding="same", activation="relu", name=f"{name_prefix}_c1"
        )(x)
        x = layers.MaxPooling2D(2, name=f"{name_prefix}_p1")(x)
        x = layers.Conv2D(
            32, 3, padding="same", activation="relu", name=f"{name_prefix}_c2"
        )(x)
        x = layers.MaxPooling2D(2, name=f"{name_prefix}_p2")(x)
        x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
        return x

    b1 = branch(eeg_in, "eeg")
    b2 = branch(spec_in, "spec")
    x = layers.Concatenate(name="concat")([b1, b2])
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs=[eeg_in, spec_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss="kullback_leibler_divergence"
    )
    return model


model = build_fallback_model()


@tf.function(reduce_retracing=True)
def _predict_fn(eeg_img, spec_img):
    return model([eeg_img, spec_img], training=False)


@tf.function(reduce_retracing=True)
def _predict_batch_fn(eeg_imgs, spec_imgs):
    return model([eeg_imgs, spec_imgs], training=False)




## === cell 9
def votes_to_prob(row):
    v = row[TARGET_COLS].astype(np.float32).values
    s = float(np.sum(v))
    if s <= 0:
        return np.ones(len(TARGET_COLS), dtype=np.float32) / len(TARGET_COLS)
    return v / s


train_subset = train_metadata.sample(
    n=min(256, len(train_metadata)), random_state=SEED
).reset_index(drop=True)

X_eeg, X_spec, Y = [], [], []
for i in range(len(train_subset)):
    try:
        eeg_id = int(train_subset.at[i, "eeg_id"])
        spec_id = int(train_subset.at[i, "spectrogram_id"])
        eeg_arr = generate_eeg_tensor(eegid=eeg_id, input_path=EEG_TRAIN_PATH)
        spec_arr = generate_spectrogram_tensor(
            specid=spec_id, input_path=SPEC_TRAIN_PATH, output_filter=0.7
        )
        X_eeg.append(eeg_arr)
        X_spec.append(spec_arr)
        Y.append(votes_to_prob(train_subset.loc[i]))
    except Exception:
        continue

X_eeg = np.asarray(X_eeg, dtype=np.float32)
X_spec = np.asarray(X_spec, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

if len(Y) >= 8:
    model.fit([X_eeg, X_spec], Y, epochs=2, batch_size=16, verbose=0)

del X_eeg, X_spec, Y, train_subset
gc.collect()




## === cell 10
def _make_features_pair(eeg_id, spec_id):
    eeg_arr = generate_eeg_tensor(eegid=int(eeg_id))
    spec_arr = generate_spectrogram_tensor(specid=int(spec_id), output_filter=0.7)
    return eeg_arr, spec_arr


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

eeg_ids = metadata["eeg_id"].to_numpy(dtype=np.int64, copy=False)
spec_ids = metadata["spectrogram_id"].to_numpy(dtype=np.int64, copy=False)

BATCH = 32
n = len(metadata)

MAX_WORKERS = 4
PREFETCH_BATCHES = (
    4  # bounded queue to limit memory while ensuring steady-state throughput
)


def _producer(q, eeg_ids_arr, spec_ids_arr, batch, total_n, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        i = 0
        while i < total_n:
            j = min(i + batch, total_n)
            eeg_id_batch = eeg_ids_arr[i:j]
            spec_id_batch = spec_ids_arr[i:j]

            eeg_np = np.empty((j - i, 300, 300, 3), dtype=np.float32)
            spec_np = np.empty((j - i, 300, 300, 3), dtype=np.float32)
            ok_mask = np.ones((j - i,), dtype=bool)

            for k, p in enumerate(
                ex.map(_make_features_pair, eeg_id_batch, spec_id_batch, chunksize=2)
            ):
                try:
                    eeg_arr, spec_arr = p
                    eeg_np[k] = eeg_arr
                    spec_np[k] = spec_arr
                except Exception:
                    ok_mask[k] = False
                    eeg_np[k] = 1.0
                    spec_np[k] = 0.0

            q.put((i, j, eeg_id_batch.copy(), eeg_np, spec_np, ok_mask), block=True)
            i = j
    q.put(None, block=True)


q = Queue(maxsize=PREFETCH_BATCHES)
prod = tf.compat.v1.train.Coordinator()  # just a lightweight holder; no TF graph usage

import threading

t = threading.Thread(
    target=_producer, args=(q, eeg_ids, spec_ids, BATCH, n, MAX_WORKERS), daemon=True
)
t.start()

while True:
    item = q.get(block=True)
    if item is None:
        break
    i, j, eeg_id_batch, eeg_np, spec_np, ok_mask = item

    preds = _predict_batch_fn(eeg_np, spec_np).numpy().astype(np.float64)

    eeg_id_list = eeg_id_batch.astype(np.int64).tolist()
    submission["eeg_id"].extend(eeg_id_list)

    pred_batch = preds
    if not ok_mask.all():
        uniform = np.ones((len(TARGET_COLS),), dtype=np.float64) / len(TARGET_COLS)
        pred_batch = pred_batch.copy()
        pred_batch[~ok_mask] = uniform

    pred_batch = np.clip(pred_batch, 1e-12, 1.0)
    pred_batch = pred_batch / pred_batch.sum(axis=1, keepdims=True)

    for jj, col in enumerate(iter_dict):
        submission[col].extend(pred_batch[:, jj].astype(float).tolist())

pdsubmit = pd.DataFrame(submission)

sample = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
pdsubmit = sample[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")
for c in TARGET_COLS:
    if c not in pdsubmit.columns:
        pdsubmit[c] = 1.0 / len(TARGET_COLS)
pdsubmit[TARGET_COLS] = pdsubmit[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

vals = pdsubmit[TARGET_COLS].values.astype(np.float64)
vals = np.clip(vals, 1e-12, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
pdsubmit[TARGET_COLS] = vals



## === cell 11
pdsubmit.to_csv("submission.csv", index=False)
print(pdsubmit.head())
print("Saved submission.csv with shape:", pdsubmit.shape)
assert os.path.exists("submission.csv")
assert pdsubmit.shape[0] == 9850
assert list(pdsubmit.columns) == ["eeg_id"] + TARGET_COLS
row_sums = pdsubmit[TARGET_COLS].sum(axis=1).values
assert np.all(np.isfinite(row_sums))
assert np.max(np.abs(row_sums - 1.0)) < 1e-6
