# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

1.160911075688366

# 6. Current score

1.40713

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40713) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure‑Python protobuf implementation before TensorFlow imports, which avoids the `MessageFactory.GetPrototype` attribute error in this environment. Then I fix the submission length mismatch by ensuring we predict exactly one row per `eeg_id` in `sample_submission.csv` and that we always merge back onto `sample_submission` as the authoritative index. Finally, I keep the model and preprocessing logic unchanged, but make inference deterministic and robust to any unexpected duplicate/missing IDs so a valid `submission.csv` is always produced.'
- What this solution (achieved 1.40713) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import* and by clearing any preloaded `google.protobuf` modules that might already be in memory. Then I fix the “submission length mismatch” by **not deduplicating** `sample_submission.csv` (the official evaluator expects exactly that row count) and instead using `sample_submission` as the authoritative list/order of `eeg_id`s. Finally, I keep your model and preprocessing intact, but ensure we generate exactly one prediction per `eeg_id` in `sample_submission`, normalize probabilities, and always write a valid `submission.csv` with the exact required columns and row count.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
from tensorflow import keras

os.environ["PYTHONHASHSEED"] = "0"
tf.keras.utils.set_random_seed(42)

print("Python:", os.sys.version)
print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _pick_comp_root():
    candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification/",
        "/kaggle/data/hms-harmful-brain-activity-classification/",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    candidates2 = [
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
    ]
    for p in candidates2:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not locate competition dataset root under /kaggle/input or /kaggle/data"
    )


COMP_ROOT = _pick_comp_root()
print("Using COMP_ROOT:", COMP_ROOT)

EEG_TEST_PATH = os.path.join(COMP_ROOT, "test_eegs/")
SPEC_TEST_PATH = os.path.join(COMP_ROOT, "test_spectrograms/")

if not os.path.exists("/kaggle/working/test_eegs_img/"):
    os.makedirs("/kaggle/working/test_eegs_img/")
EEG_IMG_TEST_PATH = "/kaggle/working/test_eegs_img/"

if not os.path.exists("/kaggle/working/test_spec_img/"):
    os.makedirs("/kaggle/working/test_spec_img/")
SPEC_IMG_TEST_PATH = "/kaggle/working/test_spec_img/"

META_TEST = os.path.join(COMP_ROOT, "test.csv")



## === cell 2
SAMPLE_SUB_PATH = os.path.join(COMP_ROOT, "sample_submission.csv")
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]
print("Target columns:", TARGET_COLS)
print("sample_submission rows:", len(sample_sub))
print("sample_submission unique eeg_id:", sample_sub["eeg_id"].nunique())



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
import io


def _fig_to_rgb_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.asarray(img)
    buf.close()
    return arr


def generate_eeg_array(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,
):
    eeg = pd.read_parquet(f"{input_path}{eegid}.parquet")

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

        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200
        else:
            relpos += 40
        cicles += 1

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 50)

    return _fig_to_rgb_array(fig)




## === cell 5
spec_zones = ["LL", "RL", "LP", "RP"]


def generate_spectrogram_array(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
):
    spec = pd.read_parquet(f"{input_path}{specid}.parquet")
    spec = spec.fillna(0)
    spec = spec.set_index("time")
    spec = spec.T

    parts = spec.index.to_series().str.split("_", expand=True)
    brainreg = parts[0].astype(str).values
    freq = parts[1].astype(float).values

    spec = spec.copy()
    spec["brainreg"] = brainreg
    spec["freq"] = freq
    spec.set_index("freq", inplace=True)

    subspec = {}
    for zone in spec_zones:
        z = spec[spec.brainreg == zone].drop("brainreg", axis=1)
        subspec[f"{zone}_sub"] = z

    fig, ax = plt.subplots(nrows=len(spec_zones), figsize=(3, 3), sharex=True)
    for row in range(len(spec_zones)):
        data = subspec[f"{spec_zones[row]}_sub"]
        dmax = float(data.to_numpy().max()) if data.size else 0.0
        vmax = (dmax * output_filter) if dmax > 0 else 1.0
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
    return _fig_to_rgb_array(fig)




## === cell 6
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(os.path.join(COMP_ROOT, "train.csv"))
print("test rows:", len(metadata), "train rows:", len(train_metadata))
print("test columns:", list(metadata.columns))




## === cell 7
def drop_images(paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass




## === cell 8
IMG_SIZE = (224, 224)


def _rgb_array_to_tensor(arr_rgb, img_size=IMG_SIZE):
    img = Image.fromarray(arr_rgb).convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = tf.convert_to_tensor(arr, dtype=tf.float32)
    ten = tf.expand_dims(ten, axis=0)  # [1,H,W,3]
    return ten


_EEG_TENSOR_CACHE = {}
_SPEC_TENSOR_CACHE = {}


def preprocess_data(metadata, row, train=False):
    if train:
        eeg_id = int(metadata.loc[row].eeg_id)
        spec_id = int(metadata.loc[row].spectrogram_id)
        eeg_path = os.path.join(COMP_ROOT, "train_eegs/")
        spec_path = os.path.join(COMP_ROOT, "train_spectrograms/")
        output_filter = 0.7
    else:
        eeg_id = int(metadata.loc[row].eeg_id)
        spec_id = int(metadata.loc[row].spectrogram_id)
        eeg_path = EEG_TEST_PATH
        spec_path = SPEC_TEST_PATH
        output_filter = 0.7

    if eeg_id in _EEG_TENSOR_CACHE:
        eeg_img_tr = _EEG_TENSOR_CACHE[eeg_id]
    else:
        eeg_arr = generate_eeg_array(eegid=eeg_id, input_path=eeg_path)
        eeg_img_tr = _rgb_array_to_tensor(eeg_arr)
        _EEG_TENSOR_CACHE[eeg_id] = eeg_img_tr

    if spec_id in _SPEC_TENSOR_CACHE:
        spec_img_tr = _SPEC_TENSOR_CACHE[spec_id]
    else:
        spec_arr = generate_spectrogram_array(
            specid=spec_id, input_path=spec_path, output_filter=output_filter
        )
        spec_img_tr = _rgb_array_to_tensor(spec_arr)
        _SPEC_TENSOR_CACHE[spec_id] = spec_img_tr

    return eeg_img_tr, spec_img_tr, eeg_id




## === cell 9
def build_two_input_model(img_size=IMG_SIZE, n_classes=6):
    eeg_in = keras.Input(shape=(img_size[0], img_size[1], 3), name="eeg_img")
    spec_in = keras.Input(shape=(img_size[0], img_size[1], 3), name="spec_img")

    def tower(x, name):
        x = keras.layers.Conv2D(
            16, 3, padding="same", activation="relu", name=f"{name}_c1"
        )(x)
        x = keras.layers.MaxPool2D(2, name=f"{name}_p1")(x)
        x = keras.layers.Conv2D(
            32, 3, padding="same", activation="relu", name=f"{name}_c2"
        )(x)
        x = keras.layers.MaxPool2D(2, name=f"{name}_p2")(x)
        x = keras.layers.Conv2D(
            64, 3, padding="same", activation="relu", name=f"{name}_c3"
        )(x)
        x = keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
        return x

    eeg_feat = tower(eeg_in, "eeg")
    spec_feat = tower(spec_in, "spec")

    x = keras.layers.Concatenate(name="concat")([eeg_feat, spec_feat])
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    out = keras.layers.Dense(n_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs=[eeg_in, spec_in], outputs=out)
    return model


model = build_two_input_model(img_size=IMG_SIZE, n_classes=len(TARGET_COLS))
model.compile(optimizer="adam", loss="kullback_leibler_divergence")
model.summary()




## === cell 10
def normalize_probs(p, eps=1e-7):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    s = p.sum(axis=-1, keepdims=True)
    p = p / s
    return p.astype(np.float32)




## === cell 11
BATCH_SIZE = 64  # safe CPU batch size to reduce Python/TF overhead

test_meta = pd.read_csv(META_TEST)[["eeg_id", "spectrogram_id"]].drop_duplicates(
    "eeg_id"
)

infer_unique = (
    sample_sub[["eeg_id"]]
    .drop_duplicates("eeg_id")
    .merge(test_meta, on="eeg_id", how="left")
)

if infer_unique["spectrogram_id"].isna().any():
    missing_ids = infer_unique.loc[
        infer_unique["spectrogram_id"].isna(), "eeg_id"
    ].tolist()
    raise ValueError(
        f"Missing spectrogram_id after merge for eeg_id(s): {missing_ids[:10]} (showing up to 10)"
    )

pred_map = {}  # eeg_id -> prob vector

eeg_batch = []
spec_batch = []
id_batch = []


def _flush_batch():
    if not id_batch:
        return
    eeg_tensor = tf.concat(eeg_batch, axis=0)
    spec_tensor = tf.concat(spec_batch, axis=0)
    preds = model.predict([eeg_tensor, spec_tensor], verbose=0)
    preds = normalize_probs(preds)
    for j, eeg_id in enumerate(id_batch):
        pred_map[int(eeg_id)] = preds[j]
    eeg_batch.clear()
    spec_batch.clear()
    id_batch.clear()


for idx in range(len(infer_unique)):
    eeg_img, spec_img, eeg_id = preprocess_data(infer_unique, idx, train=False)
    eeg_batch.append(eeg_img)
    spec_batch.append(spec_img)
    id_batch.append(eeg_id)

    if len(id_batch) >= BATCH_SIZE:
        _flush_batch()

    if idx % 500 == 0:
        gc.collect()

_flush_batch()

probs = np.zeros((len(sample_sub), len(TARGET_COLS)), dtype=np.float32)
default = np.full((len(TARGET_COLS),), 1.0 / len(TARGET_COLS), dtype=np.float32)

for i, eid in enumerate(sample_sub["eeg_id"].astype(int).values):
    probs[i] = pred_map.get(int(eid), default)

probs = normalize_probs(probs)

pdsubmit = pd.DataFrame({"eeg_id": sample_sub["eeg_id"].astype(int).values})
for j, c in enumerate(TARGET_COLS):
    pdsubmit[c] = probs[:, j]

pdsubmit = pdsubmit[["eeg_id"] + TARGET_COLS]
print(pdsubmit.head())
print("Rows:", len(pdsubmit), "expected:", len(sample_sub))
print(
    "Row sum stats:",
    float(pdsubmit[TARGET_COLS].sum(axis=1).min()),
    float(pdsubmit[TARGET_COLS].sum(axis=1).max()),
)



## === cell 12
out_path = "submission.csv"
pdsubmit.to_csv(out_path, index=False)
print("Wrote submission.csv:", os.path.getsize(out_path), "bytes")
chk = pd.read_csv(out_path)
print(chk.head())
print("Submission rows:", len(chk), "expected:", len(sample_sub))
assert len(chk) == len(
    sample_sub
), "Submission and sample_submission must have the same length"
assert list(chk.columns) == ["eeg_id"] + TARGET_COLS, "Submission columns mismatch"
row_sums = chk[TARGET_COLS].sum(axis=1).values
assert np.all(np.isfinite(row_sums)) and np.allclose(
    row_sums, 1.0, atol=1e-4
), "Row probabilities must sum to 1"
print("Submission sanity checks passed.")
