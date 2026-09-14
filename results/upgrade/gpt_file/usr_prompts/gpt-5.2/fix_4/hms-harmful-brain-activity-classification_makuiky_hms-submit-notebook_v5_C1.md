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
    y_idx = (np.linspace(0, h - 1, out_h)).astype(np.int32)
    x_idx = (np.linspace(0, w - 1, out_w)).astype(np.int32)
    return img_uint8[y_idx][:, x_idx]


def _draw_polyline(img, xs, ys, color=0):
    h, w = img.shape
    xs = xs.astype(np.int32, copy=False)
    ys = ys.astype(np.int32, copy=False)
    n = xs.size
    if n < 2:
        return
    for i in range(n - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steps = dx if dx > dy else dy
        if steps == 0:
            if 0 <= x0 < w and 0 <= y0 < h:
                img[y0, x0] = color
            continue
        xi = (x0 + (x1 - x0) * np.arange(steps + 1) / steps).astype(np.int32)
        yi = (y0 + (y1 - y0) * np.arange(steps + 1) / steps).astype(np.int32)
        m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        img[yi[m], xi[m]] = color


def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,  # kept for signature compatibility; rasterizer uses 1px lines
    eeg_out=EEG_IMG_TEST_PATH,
):
    needed_cols = []
    for a, b in eeg_zone.values():
        needed_cols.append(a)
        needed_cols.append(b)
    seen = set()
    needed_cols = [c for c in needed_cols if not (c in seen or seen.add(c))]

    eeg = pd.read_parquet(f"{input_path}{eegid}.parquet", columns=needed_cols)
    n = len(eeg.index)
    if n <= 1:
        return np.zeros((300, 300, 3), dtype=np.uint8)

    traces = []
    relpos = 0.0
    cicles = 0
    for key, (c1, c2) in eeg_zone.items():
        traces.append(
            (eeg[c1].to_numpy(np.float32) - eeg[c2].to_numpy(np.float32)) + relpos
        )
        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200
        else:
            relpos += 40
        cicles += 1
    traces = np.stack(traces, axis=0)  # (n_traces, n)

    out_h = 300
    out_w = 300
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)

    xs = np.arange(n, dtype=np.float32) / 200.0
    xpix = np.clip((xs / 50.0) * (out_w - 1), 0, out_w - 1).astype(np.int32)

    ymin = float(np.nanmin(traces))
    ymax = float(np.nanmax(traces))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return np.zeros((300, 300, 3), dtype=np.uint8)

    for k in range(traces.shape[0]):
        y = traces[k]
        ypix = ((ymax - y) / (ymax - ymin) * (out_h - 1)).astype(np.int32)
        ypix = np.clip(ypix, 0, out_h - 1)
        _draw_polyline(canvas, xpix, ypix, color=0)

    rgb = np.repeat(canvas[:, :, None], 3, axis=2)
    return rgb




## === cell 5
spec_zones = ["LL", "RL", "LP", "RP"]


def generate_spectrogram(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_out=SPEC_IMG_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
):
    spec = pd.read_parquet(f"{input_path}{specid}.parquet")
    spec = spec.fillna(0)
    times = spec["time"].to_numpy()
    feat_cols = [c for c in spec.columns if c != "time"]
    if not feat_cols:
        return np.zeros((300, 300, 3), dtype=np.uint8)

    col_arr = np.asarray(feat_cols, dtype=object)
    zones = np.array([s.split("_", 1)[0] for s in col_arr], dtype=object)
    freqs = np.array([float(s.split("_", 1)[1]) for s in col_arr], dtype=np.float32)

    X = spec[feat_cols].to_numpy(np.float32)  # (t, features)
    out_h_each = 75  # 4 * 75 = 300, matching 3x3@100dpi output size
    out_w = 300
    panels = []

    for z in spec_zones:
        mask = zones == z
        if not np.any(mask):
            panel = np.zeros((out_h_each, out_w), dtype=np.uint8)
            panels.append(panel)
            continue
        zX = X[:, mask]  # (t, fz)
        zfreq = freqs[mask]
        order = np.argsort(zfreq)
        zX = zX[:, order]
        zfreq = zfreq[order]

        img = zX.T  # (f, t)
        vmax = float(np.max(img)) if img.size else 0.0
        vmax = (vmax * output_filter) if vmax > 0 else 1.0

        norm = np.clip(img / vmax, 0.0, 1.0)
        panels.append(norm)

    turbo = plt.get_cmap("turbo", 256)
    lut = (turbo(np.arange(256))[:, :3] * 255.0).astype(np.uint8)  # (256,3)

    rgb_panels = []
    for norm in panels:
        idx = (norm * 255.0 + 0.5).astype(np.uint8)
        rgb = lut[idx]  # (f,t,3)
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

    eeg_img_tr = tf.convert_to_tensor(eeg_arr, dtype=tf.float32)
    spec_img_tr = tf.convert_to_tensor(spec_arr, dtype=tf.float32)

    eeg_img_tr = tf.expand_dims(eeg_img_tr, axis=0)
    spec_img_tr = tf.expand_dims(spec_img_tr, axis=0)

    return eeg_img_tr, spec_img_tr, int(eeg_id)




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
def _predict_fn(eeg_img, spec_img):
    return model([eeg_img, spec_img], training=False)




## === cell 10
submission = {
    "eeg_id": [],
    "seizure_vote": [],
    "lpd_vote": [],
    "gpd_vote": [],
    "lrda_vote": [],
    "grda_vote": [],
    "other_vote": [],
}
iter_dict = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

n_rows = eeg_ids_arr.shape[0]
for row in range(n_rows):
    eeg_id = eeg_ids_arr[row]
    spec_id = spec_ids_arr[row]
    eeg_img, spec_img, eeg_id_int = preprocess_data_fast(eeg_id, spec_id)
    pred = _predict_fn(eeg_img, spec_img).numpy().squeeze()
    pred = np.asarray(pred, dtype=np.float64)
    pred = np.nan_to_num(pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    s = pred.sum()
    if not np.isfinite(s) or s <= 0:
        pred = np.ones(6, dtype=np.float64) / 6.0
    else:
        pred = pred / s

    submission["eeg_id"].append(eeg_id_int)
    for i, illness in enumerate(iter_dict):
        submission[illness].append(float(pred[i]))

    del eeg_img, spec_img
    if row % 500 == 0:
        gc.collect()

pdsubmit = pd.DataFrame(submission)



## === cell 11
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
