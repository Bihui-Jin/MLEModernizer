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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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


@lru_cache(maxsize=256)
def _read_parquet_cached(path: str):
    return pd.read_parquet(path)


def _draw_polyline(img, xs, ys, thickness=1):
    h, w = img.shape
    for i in range(len(xs) - 1):
        x0, y0 = int(xs[i]), int(ys[i])
        x1, y1 = int(xs[i + 1]), int(ys[i + 1])
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            if 0 <= x < w and 0 <= y < h:
                y0t = max(0, y - thickness)
                y1t = min(h, y + thickness + 1)
                x0t = max(0, x - thickness)
                x1t = min(w, x + thickness + 1)
                img[y0t:y1t, x0t:x1t] = 0  # black
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy


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

    relpos_list = []
    relpos = 0.0
    for cicles in range(len(eeg_zone)):
        relpos_list.append(relpos)
        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200.0
        else:
            relpos += 40.0
    relpos_arr = np.asarray(relpos_list, dtype=np.float32)

    diffs = []
    for key, (a, b) in eeg_zone.items():
        diffs.append(
            (
                eeg_np[a].to_numpy(dtype=np.float32, copy=False)
                - eeg_np[b].to_numpy(dtype=np.float32, copy=False)
            )
        )
    diffs = np.stack(diffs, axis=0)  # (18, n)
    diffs = diffs + relpos_arr[:, None]

    y_min = float(np.nanmin(diffs))
    y_max = float(np.nanmax(diffs))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        y_min, y_max = 0.0, 1.0

    ys = (1.0 - (diffs - y_min) / (y_max - y_min)) * (_EEG_H - 1)
    ys = np.clip(ys, 0, _EEG_H - 1).astype(np.int32)

    canvas = np.full((_EEG_H, _EEG_W), 255, dtype=np.uint8)
    for i in range(ys.shape[0]):
        _draw_polyline(canvas, xs, ys[i], thickness=1)

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
    spec = spec.fillna(0)
    spec = spec.set_index("time").T

    parts = spec.index.to_series().str.split("_", n=1, expand=True)
    brainreg = parts[0].astype(str).to_numpy()
    freq = parts[1].astype(float).to_numpy()

    data_np = spec.to_numpy(dtype=np.float32, copy=False)  # (features, time)
    times = spec.columns.to_numpy(dtype=np.float32, copy=False)

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




## === cell 6
def drop_images(paths):
    return




## === cell 7
MODEL_IMG_SIZE = (300, 300)


def preprocess_data(metadata, row, train=False):
    r = metadata.iloc[row]
    if train:
        eeg_rgb, eeg_path = generate_eeg(
            eegid=r.eeg_id,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/",
        )
        spec_rgb, spec_path = generate_spectrogram(
            specid=r.spectrogram_id,
            output_filter=0.7,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/",
        )
    else:
        eeg_rgb, eeg_path = generate_eeg(eegid=r.eeg_id)
        spec_rgb, spec_path = generate_spectrogram(
            specid=r.spectrogram_id, output_filter=0.7
        )

    eeg_img = (
        Image.fromarray(eeg_rgb).convert("RGB").resize(MODEL_IMG_SIZE, Image.BILINEAR)
    )
    spec_img = (
        Image.fromarray(spec_rgb).convert("RGB").resize(MODEL_IMG_SIZE, Image.BILINEAR)
    )

    eeg_arr = np.asarray(eeg_img, dtype=np.uint8)
    spec_arr = np.asarray(spec_img, dtype=np.uint8)

    eeg_img_tr = tf.convert_to_tensor(eeg_arr)
    spec_img_tr = tf.convert_to_tensor(spec_arr)

    eeg_img_tr = tf.expand_dims(eeg_img_tr, axis=0)
    spec_img_tr = tf.expand_dims(spec_img_tr, axis=0)

    drop_images(paths=[eeg_path, spec_path])
    eeg_id = int(r.eeg_id)
    return eeg_img_tr, spec_img_tr, eeg_id




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

BATCH = 32  # Speed: larger batch reduces TF call overhead; memory remains manageable for 300x300x3x2 inputs.

eeg_batch = []
spec_batch = []
id_batch = []


@tf.function(reduce_retracing=True)
def _predict_batch(eeg_x, spec_x):
    return model([eeg_x, spec_x], training=False)


def _flush_batch():
    if not eeg_batch:
        return
    eeg_x = tf.concat(eeg_batch, axis=0)
    spec_x = tf.concat(spec_batch, axis=0)

    preds = _predict_batch(eeg_x, spec_x).numpy().astype(np.float64)

    preds = np.clip(preds, 1e-12, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    for k, eeg_id in enumerate(id_batch):
        submission["eeg_id"].append(int(eeg_id))
        for i, col in enumerate(iter_dict):
            submission[col].append(float(preds[k, i]))

    eeg_batch.clear()
    spec_batch.clear()
    id_batch.clear()


n_rows = len(metadata)
for row in range(n_rows):
    eeg_img, spec_img, eeg_id = preprocess_data(metadata, row)
    eeg_batch.append(eeg_img)
    spec_batch.append(spec_img)
    id_batch.append(eeg_id)

    if len(eeg_batch) >= BATCH:
        _flush_batch()

_flush_batch()

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
