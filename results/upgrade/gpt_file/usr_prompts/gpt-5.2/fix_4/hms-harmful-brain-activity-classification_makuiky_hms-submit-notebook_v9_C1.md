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
import io

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
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
def _fig_to_rgb_array(fig, out_hw=(300, 300)):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize(out_hw, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    buf.close()
    return arr


def generate_eeg_tensor(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,
    out_hw=(300, 300),
):
    eeg = pd.read_parquet(f"{input_path}{eegid}.parquet")

    ysticks = []
    labels = []
    cicles = 0
    relpos = 0
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True)

    for key, values in eeg_zone.items():
        ax.plot(
            eeg.index / 200,
            eeg[values[0]] - eeg[values[1]] + relpos,
            color="black",
            linewidth=linewidth,
        )

        ysticks.append(relpos)
        labels.append(key)
        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200
        else:
            relpos += 40

        cicles += 1

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 50)

    arr = _fig_to_rgb_array(fig, out_hw=out_hw)
    return arr




## === cell 5
spec_zones = ["LL", "RL", "LP", "RP"]

_SPEC_COL_CACHE = {"cols": None, "zone": None, "freq": None}


def _get_spec_col_meta(spec_columns):
    cols = np.asarray(spec_columns, dtype=object)
    if _SPEC_COL_CACHE["cols"] is not None and np.array_equal(
        _SPEC_COL_CACHE["cols"], cols
    ):
        return _SPEC_COL_CACHE["zone"], _SPEC_COL_CACHE["freq"]

    parts = np.char.split(cols.astype(str), "_")
    zone = np.array([p[0] if len(p) > 0 else "" for p in parts], dtype=object)
    freq = np.array(
        [float(p[1]) if len(p) > 1 else np.nan for p in parts], dtype=np.float32
    )

    _SPEC_COL_CACHE["cols"] = cols
    _SPEC_COL_CACHE["zone"] = zone
    _SPEC_COL_CACHE["freq"] = freq
    return zone, freq


def generate_spectrogram_tensor(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
    out_hw=(300, 300),
):
    spec = pd.read_parquet(f"{input_path}{specid}.parquet")
    spec = spec.fillna(0)

    spec = spec.set_index("time")
    spec = spec.T

    zone, freq = _get_spec_col_meta(spec.index)
    spec2 = spec.copy()
    spec2["freq"] = freq
    spec2["brainreg"] = zone
    spec2 = spec2.set_index("freq", drop=True)

    subspec = {}
    for z in spec_zones:
        d = spec2[spec2["brainreg"] == z].drop("brainreg", axis=1)
        subspec[f"{z}_sub"] = d

    fig, ax = plt.subplots(nrows=len(spec_zones), figsize=(3, 3), sharex=True)
    if len(spec_zones) == 1:
        ax = [ax]
    for row in range(len(spec_zones)):
        data = subspec[f"{spec_zones[row]}_sub"]
        vmax = (data.to_numpy().max() * output_filter) if data.size else 1.0
        ax[row].imshow(
            data,
            cmap="turbo",
            aspect="auto",
            origin="lower",
            extent=[
                data.columns.min(),
                data.columns.max(),
                data.index.min(),
                data.index.max(),
            ],
            vmin=0,
            vmax=vmax,
        )
        ax[row].set_xticks([])
        ax[row].set_yticks([])

    plt.subplots_adjust(hspace=0.01)

    arr = _fig_to_rgb_array(fig, out_hw=out_hw)
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
    if train:
        eeg_arr = generate_eeg_tensor(
            eegid=int(metadata_df.loc[row].eeg_id),
            input_path=EEG_TRAIN_PATH,
        )
        spec_arr = generate_spectrogram_tensor(
            specid=int(metadata_df.loc[row].spectrogram_id),
            input_path=SPEC_TRAIN_PATH,
            output_filter=0.7,
        )
    else:
        eeg_arr = generate_eeg_tensor(eegid=int(metadata_df.loc[row].eeg_id))
        spec_arr = generate_spectrogram_tensor(
            specid=int(metadata_df.loc[row].spectrogram_id), output_filter=0.7
        )

    eeg_img_tr = _image_array_to_tensor(eeg_arr)
    spec_img_tr = _image_array_to_tensor(spec_arr)

    eeg_id = int(metadata_df.loc[row].eeg_id)
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
        eeg_t, spec_t, _ = preprocess_data(train_subset, i, train=True)
        X_eeg.append(eeg_t[0].numpy())
        X_spec.append(spec_t[0].numpy())
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

eeg_ids = metadata["eeg_id"].to_numpy()
spec_ids = metadata["spectrogram_id"].to_numpy()

for idx in range(len(metadata)):
    eeg_id = int(eeg_ids[idx])
    try:
        eeg_arr = generate_eeg_tensor(eegid=eeg_id)
        spec_arr = generate_spectrogram_tensor(
            specid=int(spec_ids[idx]), output_filter=0.7
        )
        eeg_img = _image_array_to_tensor(eeg_arr)
        spec_img = _image_array_to_tensor(spec_arr)

        pred = _predict_fn(eeg_img, spec_img)[0].numpy().astype(np.float64)
    except Exception:
        pred = np.ones(len(TARGET_COLS), dtype=np.float64) / len(TARGET_COLS)

    pred = np.clip(pred, 1e-12, 1.0)
    pred = pred / pred.sum()

    submission["eeg_id"].append(eeg_id)
    for j, col in enumerate(iter_dict):
        submission[col].append(float(pred[j]))

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
