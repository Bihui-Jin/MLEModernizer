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

import gc
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # kept, but we avoid any rendering for speed
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    import pyarrow  # noqa: F401

    _PARQUET_ENGINE = "pyarrow"
except Exception:
    _PARQUET_ENGINE = None



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

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



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
from functools import lru_cache


@lru_cache(maxsize=8192)
def _read_parquet_cached(path: str):
    if _PARQUET_ENGINE is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, engine=_PARQUET_ENGINE)


def _scale_to_uint8(x, vmin, vmax):
    x = np.asarray(x, dtype=np.float32)
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _rasterize_polyline(canvas, xs, ys, value=0):
    """
    Draw polyline into a 2D uint8 canvas using simple segment stepping.
    Kept equivalent to prior "visual trace" intent but avoids matplotlib overhead.
    """
    h, w = canvas.shape
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    n = len(xs)
    for i in range(n - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy))) + 1
        if steps <= 0:
            continue
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        xi = (x0 + dx * t).astype(np.int32)
        yi = (y0 + dy * t).astype(np.int32)
        m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        canvas[yi[m], xi[m]] = value


def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,  # kept for signature compatibility; rasterization ignores it
    eeg_out=EEG_IMG_TEST_PATH,  # unused but kept
):
    eeg = _read_parquet_cached(f"{input_path}{int(eegid)}.parquet")

    H = W = 300
    canvas = np.full((H, W), 255, dtype=np.uint8)  # white background

    x = eeg.index.to_numpy(dtype=np.float32) / 200.0
    xs = (x / 50.0 * (W - 1)).astype(np.int32)

    relpos = 0.0
    cicles = 0
    for _, values in eeg_zone.items():
        y = (
            eeg[values[0]].to_numpy(dtype=np.float32)
            - eeg[values[1]].to_numpy(dtype=np.float32)
        ) + relpos

        y_min = relpos - 250.0
        y_max = relpos + 250.0
        ys = ((1.0 - (y - y_min) / (y_max - y_min)) * (H - 1)).astype(np.int32)

        _rasterize_polyline(canvas, xs, ys, value=0)  # black trace

        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200.0
        else:
            relpos += 40.0
        cicles += 1

    rgb = np.repeat(canvas[..., None], 3, axis=2)
    return rgb


_TURBO_LUT = (
    plt.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3] * 255.0 + 0.5
).astype(np.uint8)

spec_zones = ["LL", "RL", "LP", "RP"]


@lru_cache(maxsize=4096)
def _spec_col_index(spec_path: str):
    spec = _read_parquet_cached(spec_path)
    cols = spec.columns.tolist()

    feat_cols = cols[1:]
    zones = np.empty(len(feat_cols), dtype=object)
    freqs = np.empty(len(feat_cols), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        z, f = c.split("_", 1)
        zones[i] = z
        freqs[i] = np.float32(f)

    idx_by_zone = {}
    order_by_zone = {}
    for z in spec_zones:
        mask = zones == z
        idx = np.nonzero(mask)[0]  # indices into feat_cols
        idx_by_zone[z] = idx
        if idx.size:
            order_by_zone[z] = np.argsort(freqs[idx])
        else:
            order_by_zone[z] = np.array([], dtype=np.int64)

    return idx_by_zone, order_by_zone


def generate_spectrogram(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_out=SPEC_IMG_TEST_PATH,  # unused but kept
    spec_zones=spec_zones,
    output_filter=0.5,
):
    spec_path = f"{input_path}{int(specid)}.parquet"

    spec = _read_parquet_cached(spec_path)

    times = spec["time"].to_numpy(dtype=np.float32)
    X = spec.iloc[:, 1:].to_numpy(dtype=np.float32, na_value=0.0)  # (T, F)
    X = X.T  # (F, T) to match previous df.set_index("time").T layout

    idx_by_zone, order_by_zone = _spec_col_index(spec_path)

    H = 300
    W = 300
    panel_h = H // len(spec_zones)
    out = np.zeros((panel_h * len(spec_zones), W, 3), dtype=np.uint8)

    for zi, zone in enumerate(spec_zones):
        idx = idx_by_zone[zone]
        if idx.size == 0:
            out[zi * panel_h : (zi + 1) * panel_h] = 0
            continue

        zmat = X[idx]
        order = order_by_zone[zone]
        if order.size:
            zmat = zmat[order]

        vmax = float(zmat.max()) * float(output_filter) if zmat.size else 1.0
        im = _scale_to_uint8(zmat, 0.0, vmax)  # 0..255

        src_h, src_w = im.shape
        if src_h <= 1 or src_w <= 1:
            im_rs = np.zeros((panel_h, W), dtype=np.uint8)
        else:
            y_idx = (np.linspace(0, src_h - 1, panel_h)).astype(np.int32)
            x_idx = (np.linspace(0, src_w - 1, W)).astype(np.int32)
            im_rs = im[y_idx][:, x_idx]

        rgb = _TURBO_LUT[im_rs]
        out[zi * panel_h : (zi + 1) * panel_h] = rgb

    if out.shape[0] < H:
        pad = np.zeros((H - out.shape[0], W, 3), dtype=np.uint8)
        out = np.concatenate([out, pad], axis=0)
    elif out.shape[0] > H:
        out = out[:H]

    return out




## === cell 5
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(META_TRAIN)

train_grouped = train_metadata.groupby(["eeg_id", "spectrogram_id"], as_index=False)[
    TARGET_COLS
].mean()
votes = train_grouped[TARGET_COLS].to_numpy(dtype=np.float32)
vote_sums = votes.sum(axis=1, keepdims=True)
vote_sums = np.where(vote_sums <= 0, 1.0, vote_sums)
train_grouped[TARGET_COLS] = votes / vote_sums

train_grouped.shape, metadata.shape




## === cell 6
def drop_images(paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@lru_cache(maxsize=4096)
def _preprocess_eeg_tensor(eeg_id: int, train: bool):
    eeg_arr = generate_eeg(
        eegid=int(eeg_id),
        input_path=EEG_TRAIN_PATH if train else EEG_TEST_PATH,
        eeg_out=EEG_IMG_TEST_PATH,  # unused but kept
    )
    eeg_img_tr = tf.image.resize(tf.convert_to_tensor(eeg_arr), [224, 224])
    eeg_img_tr = tf.cast(eeg_img_tr, tf.float32) / 255.0
    return eeg_img_tr


@lru_cache(maxsize=4096)
def _preprocess_spec_tensor(spectrogram_id: int, train: bool, output_filter: float):
    spec_arr = generate_spectrogram(
        specid=int(spectrogram_id),
        input_path=SPEC_TRAIN_PATH if train else SPEC_TEST_PATH,
        spec_out=SPEC_IMG_TEST_PATH,  # unused but kept
        output_filter=float(output_filter),
    )
    spec_img_tr = tf.image.resize(tf.convert_to_tensor(spec_arr), [240, 240])
    spec_img_tr = tf.cast(spec_img_tr, tf.float32) / 255.0
    return spec_img_tr


def preprocess_data_from_ids(eeg_id: int, spectrogram_id: int, train: bool = False):
    eeg_img_tr = _preprocess_eeg_tensor(int(eeg_id), bool(train))
    spec_img_tr = _preprocess_spec_tensor(int(spectrogram_id), bool(train), 0.7)
    return eeg_img_tr, spec_img_tr, int(eeg_id)


def preprocess_data(metadata_df, row, train=False):
    rec = metadata_df.iloc[row]
    return preprocess_data_from_ids(
        int(rec.eeg_id), int(rec.spectrogram_id), train=train
    )




## === cell 7
def build_model():
    eeg_in = keras.Input(shape=(224, 224, 3), name="eeg_img")
    spec_in = keras.Input(shape=(240, 240, 3), name="spec_img")

    x1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(eeg_in)
    x1 = keras.layers.MaxPooling2D()(x1)
    x1 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x1)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)

    x2 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(spec_in)
    x2 = keras.layers.MaxPooling2D()(x2)
    x2 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x2)
    x2 = keras.layers.GlobalAveragePooling2D()(x2)

    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(6, activation="softmax", name="probs")(x)

    model = keras.Model(inputs=[eeg_in, spec_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="kullback_leibler_divergence",
        metrics=[],
    )
    return model


model = build_model()
model.summary()


@tf.function(reduce_retracing=True)
def _predict_fn(eeg_batch, spec_batch):
    return model([eeg_batch, spec_batch], training=False)




## === cell 8
MAX_TRAIN_SAMPLES = 256  # unchanged
train_subset = train_grouped.sample(
    n=min(MAX_TRAIN_SAMPLES, len(train_grouped)), random_state=SEED
).reset_index(drop=True)

n_tr = len(train_subset)
X_eeg = np.empty((n_tr, 224, 224, 3), dtype=np.float32)
X_spec = np.empty((n_tr, 240, 240, 3), dtype=np.float32)
Y = train_subset[TARGET_COLS].to_numpy(dtype=np.float32)

for i in range(n_tr):
    rec = train_subset.iloc[i]
    eeg_img, spec_img, _ = preprocess_data_from_ids(
        int(rec.eeg_id), int(rec.spectrogram_id), train=True
    )
    X_eeg[i] = np.asarray(eeg_img)
    X_spec[i] = np.asarray(spec_img)
    if (i + 1) % 64 == 0:
        gc.collect()

Y = Y / np.clip(Y.sum(axis=1, keepdims=True), 1e-8, None)

model.fit([X_eeg, X_spec], Y, batch_size=16, epochs=2, verbose=1)

del X_eeg, X_spec, Y
gc.collect()



## === cell 9
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

BATCH = 64
n_te = len(metadata)

eeg_ids_all = metadata["eeg_id"].to_numpy(dtype=np.int64)
spec_ids_all = metadata["spectrogram_id"].to_numpy(dtype=np.int64)

for start in range(0, n_te, BATCH):
    end = min(start + BATCH, n_te)
    bs = end - start

    eeg_batch = np.empty((bs, 224, 224, 3), dtype=np.float32)
    spec_batch = np.empty((bs, 240, 240, 3), dtype=np.float32)
    eeg_ids = eeg_ids_all[start:end].copy()

    for j in range(bs):
        eeg_id = int(eeg_ids_all[start + j])
        spec_id = int(spec_ids_all[start + j])
        eeg_img, spec_img, _ = preprocess_data_from_ids(eeg_id, spec_id, train=False)
        eeg_batch[j] = np.asarray(eeg_img)
        spec_batch[j] = np.asarray(spec_img)

    preds = (
        _predict_fn(tf.convert_to_tensor(eeg_batch), tf.convert_to_tensor(spec_batch))
        .numpy()
        .astype(np.float64)
    )
    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    submission["eeg_id"].extend(eeg_ids.tolist())
    for k, col in enumerate(iter_dict):
        submission[col].extend(preds[:, k].tolist())

    if end % 512 == 0:
        gc.collect()

pdsubmit = pd.DataFrame(submission)

sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
pdsubmit = sample_sub[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")

missing_mask = pdsubmit[iter_dict].isna().any(axis=1)
if missing_mask.any():
    pdsubmit.loc[missing_mask, iter_dict] = 1.0 / len(iter_dict)

probs = pdsubmit[iter_dict].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
pdsubmit[iter_dict] = probs

pdsubmit.head()



## === cell 10
pdsubmit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pdsubmit.shape)
print(
    "Row sum check (min,max):",
    pdsubmit[TARGET_COLS].sum(axis=1).min(),
    pdsubmit[TARGET_COLS].sum(axis=1).max(),
)
print(pdsubmit.columns.tolist())
