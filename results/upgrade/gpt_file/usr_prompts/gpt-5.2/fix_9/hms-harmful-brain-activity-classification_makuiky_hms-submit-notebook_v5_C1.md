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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image  # kept (original dependency), though no longer used for main path
import gc

import pyarrow.parquet as pq
import pyarrow as pa

import pyarrow.dataset as ds

from concurrent.futures import ThreadPoolExecutor



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
plt.switch_backend("Agg")

tf.random.set_seed(0)
np.random.seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



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
def _resize_nearest(img_uint8, out_h=300, out_w=300):
    h, w = img_uint8.shape[:2]
    if h == out_h and w == out_w:
        return img_uint8
    y_idx = (np.linspace(0, h - 1, out_h)).astype(np.int32, copy=False)
    x_idx = (np.linspace(0, w - 1, out_w)).astype(np.int32, copy=False)
    return img_uint8[y_idx][:, x_idx]


def _draw_polyline_fast(img, xs, ys, color=0):
    h, w = img.shape
    xs = xs.astype(np.int32, copy=False)
    ys = ys.astype(np.int32, copy=False)
    n = xs.size
    if n < 2:
        return

    x0 = xs[:-1].astype(np.int32, copy=False)
    y0 = ys[:-1].astype(np.int32, copy=False)
    x1 = xs[1:].astype(np.int32, copy=False)
    y1 = ys[1:].astype(np.int32, copy=False)

    dx = x1 - x0
    dy = y1 - y0
    adx = np.abs(dx)
    ady = np.abs(dy)
    steps = np.maximum(adx, ady).astype(np.int32, copy=False)

    deg = steps == 0
    if np.any(deg):
        xd = x0[deg]
        yd = y0[deg]
        m = (xd >= 0) & (xd < w) & (yd >= 0) & (yd < h)
        img[yd[m], xd[m]] = color

    nd = ~deg
    if not np.any(nd):
        return

    x0 = x0[nd]
    y0 = y0[nd]
    dx = dx[nd]
    dy = dy[nd]
    steps = steps[nd]

    lens = steps + 1
    total = int(lens.sum())
    seg_idx = np.repeat(np.arange(lens.size, dtype=np.int32), lens)
    local = np.arange(total, dtype=np.int32) - np.repeat(
        np.cumsum(lens, dtype=np.int64) - lens, lens
    )
    t = local.astype(np.float32) / steps[seg_idx].astype(np.float32)

    xi = (x0[seg_idx].astype(np.float32) + dx[seg_idx].astype(np.float32) * t).astype(
        np.int32
    )
    yi = (y0[seg_idx].astype(np.float32) + dy[seg_idx].astype(np.float32) * t).astype(
        np.int32
    )

    m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    img[yi[m], xi[m]] = color


_needed_eeg_cols = []
for a, b in eeg_zone.values():
    _needed_eeg_cols.append(a)
    _needed_eeg_cols.append(b)
_seen = set()
_needed_eeg_cols = [c for c in _needed_eeg_cols if not (c in _seen or _seen.add(c))]

_ZONE_PAIRS = [(a, b) for (a, b) in eeg_zone.values()]
_COLS = _needed_eeg_cols
_COL_TO_IDX = {c: i for i, c in enumerate(_COLS)}
_PAIR_IDX = np.array(
    [(_COL_TO_IDX[a], _COL_TO_IDX[b]) for a, b in _ZONE_PAIRS], dtype=np.int32
)

_XPIX_CACHE = {}  # key: n -> xpix int32 (size n)

_rel = []
relpos = 0.0
for i in range(len(_ZONE_PAIRS)):
    _rel.append(relpos)
    if i in (1, 5, 9, 13):
        relpos += 200.0
    else:
        relpos += 40.0
_REL_OFFSETS = np.asarray(_rel, dtype=np.float32)[:, None]  # (n_traces,1)


def _arrow_table_to_2d_float32(table: pa.Table) -> np.ndarray:
    if table.num_columns == 0:
        return np.zeros((0, 0), dtype=np.float32)
    df = table.to_pandas(split_blocks=False, self_destruct=True)
    X = df.to_numpy(dtype=np.float32, copy=False)
    return X


def _read_parquet_table_fast(path: str, columns=None) -> pa.Table:
    dset = ds.dataset(path, format="parquet")
    return dset.to_table(columns=columns)


def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,  # kept for signature compatibility; rasterizer uses 1px lines
    eeg_out=EEG_IMG_TEST_PATH,
):
    path = f"{input_path}{eegid}.parquet"
    table = _read_parquet_table_fast(path, columns=_needed_eeg_cols)
    eeg_np = _arrow_table_to_2d_float32(table)
    n = eeg_np.shape[0]
    if n <= 1:
        return np.zeros((300, 300, 3), dtype=np.uint8)

    xpix = _XPIX_CACHE.get(n)
    if xpix is None:
        xs = np.arange(n, dtype=np.float32) / 200.0
        xpix = np.clip((xs / 50.0) * (300 - 1), 0, 300 - 1).astype(np.int32)
        _XPIX_CACHE[n] = xpix

    a_idx = _PAIR_IDX[:, 0]
    b_idx = _PAIR_IDX[:, 1]
    traces = eeg_np[:, a_idx].T - eeg_np[:, b_idx].T
    traces = traces + _REL_OFFSETS

    canvas = np.full((300, 300), 255, dtype=np.uint8)

    ymin = float(np.nanmin(traces))
    ymax = float(np.nanmax(traces))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return np.zeros((300, 300, 3), dtype=np.uint8)

    denom = ymax - ymin
    for k in range(traces.shape[0]):
        y = traces[k]
        ypix = ((ymax - y) / denom * (300 - 1)).astype(np.int32, copy=False)
        np.clip(ypix, 0, 300 - 1, out=ypix)
        _draw_polyline_fast(canvas, xpix, ypix, color=0)

    rgb = np.repeat(canvas[:, :, None], 3, axis=2)
    return rgb




## === cell 5
spec_zones = ["LL", "RL", "LP", "RP"]

_turbo = plt.get_cmap("turbo", 256)
_TURBO_LUT = (_turbo(np.arange(256))[:, :3] * 255.0).astype(np.uint8)  # (256,3)

_SPEC_COLMETA_CACHE = {}


def generate_spectrogram(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_out=SPEC_IMG_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
):
    path = f"{input_path}{specid}.parquet"
    table = _read_parquet_table_fast(path, columns=None)  # columns vary; read all once

    names = table.schema.names
    cols = tuple(names)
    meta = _SPEC_COLMETA_CACHE.get(cols)
    if meta is None:
        feat_cols = [c for c in names if c != "time"]
        if not feat_cols:
            return np.zeros((300, 300, 3), dtype=np.uint8)

        col_arr = np.asarray(feat_cols, dtype=object)
        zones = np.array([s.split("_", 1)[0] for s in col_arr], dtype=object)
        freqs = np.array([float(s.split("_", 1)[1]) for s in col_arr], dtype=np.float32)

        name_to_idx = {n: i for i, n in enumerate(names)}
        feat_idx = np.array([name_to_idx[c] for c in feat_cols], dtype=np.int32)

        zone_meta = {}
        for z in spec_zones:
            mask = zones == z
            if np.any(mask):
                zfreq = freqs[mask]
                order = np.argsort(zfreq)
                zone_meta[z] = (mask, order)
            else:
                zone_meta[z] = (mask, None)

        meta = (feat_idx, zone_meta)
        _SPEC_COLMETA_CACHE[cols] = meta

    feat_idx, zone_meta = meta

    feat_cols = [names[int(i)] for i in feat_idx]
    df = table.select(feat_cols).to_pandas(split_blocks=False, self_destruct=True)
    X = df.to_numpy(dtype=np.float32, copy=False)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    out_h_each = 75  # 4 * 75 = 300
    out_w = 300
    rgb_panels = []

    for z in spec_zones:
        mask, order = zone_meta[z]
        if not np.any(mask):
            rgb_panels.append(np.zeros((out_h_each, out_w, 3), dtype=np.uint8))
            continue

        zX = X[:, mask]
        if order is not None:
            zX = zX[:, order]

        img = zX.T  # (f, t)
        vmax = float(np.max(img)) if img.size else 0.0
        vmax = (vmax * output_filter) if vmax > 0 else 1.0

        norm = np.clip(img / vmax, 0.0, 1.0)
        idx = (norm * 255.0 + 0.5).astype(np.uint8, copy=False)
        rgb = _TURBO_LUT[idx]  # (f,t,3)
        rgb = _resize_nearest(rgb, out_h=out_h_each, out_w=out_w)
        rgb_panels.append(rgb)

    rgb = np.concatenate(rgb_panels, axis=0)  # (300,300,3)
    return rgb




## === cell 6
metadata = pd.read_csv(META_TEST)




## === cell 7
def drop_images(paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass




## === cell 8
eeg_ids_arr = metadata["eeg_id"].to_numpy()
spec_ids_arr = metadata["spectrogram_id"].to_numpy()


def preprocess_data_fast(eeg_id, spec_id, train=False):
    if train:
        eeg_rgb = generate_eeg(
            eegid=eeg_id,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/",
        )
        spec_rgb = generate_spectrogram(
            specid=spec_id,
            output_filter=0.7,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/",
        )
    else:
        eeg_rgb = generate_eeg(eegid=eeg_id)
        spec_rgb = generate_spectrogram(specid=spec_id, output_filter=0.7)

    eeg_arr = eeg_rgb.astype(np.float32) / 255.0
    spec_arr = spec_rgb.astype(np.float32) / 255.0
    return eeg_arr, spec_arr, int(eeg_id)




## === cell 9
def build_fallback_model():
    eeg_in = keras.Input(shape=(None, None, 3), name="eeg_img")
    spec_in = keras.Input(shape=(None, None, 3), name="spec_img")

    def tower(x):
        x = keras.layers.Resizing(224, 224)(x)
        x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        return x

    eeg_feat = tower(eeg_in)
    spec_feat = tower(spec_in)
    x = keras.layers.Concatenate()([eeg_feat, spec_feat])
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(6, activation="softmax", name="votes")(x)
    model = keras.Model(inputs=[eeg_in, spec_in], outputs=out)
    return model


model = build_fallback_model()


@tf.function(reduce_retracing=True)
def _predict_batch_fn(eeg_img_batch, spec_img_batch):
    return model([eeg_img_batch, spec_img_batch], training=False)




## === cell 10
iter_dict = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

n_rows = int(eeg_ids_arr.shape[0])
out_ids = np.empty((n_rows,), dtype=np.int64)
out_pred = np.empty((n_rows, 6), dtype=np.float64)

BATCH = 32  # keep same as prior (does not change predictions)


def _process_one(i: int):
    eeg_id = int(eeg_ids_arr[i])
    spec_id = int(spec_ids_arr[i])
    eeg_arr, spec_arr, eeg_id_int = preprocess_data_fast(eeg_id, spec_id, train=False)
    return i, eeg_id_int, eeg_arr, spec_arr


max_workers = min(8, (os.cpu_count() or 4))

eeg_b = np.empty((BATCH, 300, 300, 3), dtype=np.float32)
spec_b = np.empty((BATCH, 300, 300, 3), dtype=np.float32)
id_b = np.empty((BATCH,), dtype=np.int64)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    write_pos = 0
    for chunk_start in range(0, n_rows, BATCH):
        chunk_end = min(n_rows, chunk_start + BATCH)
        idxs = range(chunk_start, chunk_end)
        chunk = list(ex.map(_process_one, idxs))  # ordered results

        bs = len(chunk)
        for j, (_, eeg_id_int, eeg_arr, spec_arr) in enumerate(chunk):
            id_b[j] = eeg_id_int
            eeg_b[j] = eeg_arr
            spec_b[j] = spec_arr

        pred_b = _predict_batch_fn(eeg_b[:bs], spec_b[:bs]).numpy().astype(np.float64)

        pred_b = np.nan_to_num(
            pred_b,
            nan=1.0 / 6.0,
            posinf=1.0 / 6.0,
            neginf=1.0 / 6.0,
        )
        s = pred_b.sum(axis=1, keepdims=True)
        bad = (~np.isfinite(s)) | (s <= 0)
        if np.any(bad):
            pred_b[bad[:, 0]] = 1.0 / 6.0
            s = pred_b.sum(axis=1, keepdims=True)
        pred_b = pred_b / s

        out_ids[write_pos : write_pos + bs] = id_b[:bs]
        out_pred[write_pos : write_pos + bs] = pred_b
        write_pos += bs

        if write_pos % 1024 == 0:
            gc.collect()

pdsubmit = pd.DataFrame(
    np.column_stack([out_ids, out_pred]),
    columns=["eeg_id"] + iter_dict,
)
pdsubmit["eeg_id"] = pdsubmit["eeg_id"].astype(np.int64)

sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
pdsubmit = sample_sub[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")

for c in iter_dict:
    if c not in pdsubmit.columns:
        pdsubmit[c] = 1.0 / 6.0
pdsubmit[iter_dict] = pdsubmit[iter_dict].fillna(1.0 / 6.0)

vals = pdsubmit[iter_dict].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, None)
vals = vals / vals.sum(axis=1, keepdims=True)
pdsubmit[iter_dict] = vals

pdsubmit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pdsubmit.shape)
print(pdsubmit.head())
