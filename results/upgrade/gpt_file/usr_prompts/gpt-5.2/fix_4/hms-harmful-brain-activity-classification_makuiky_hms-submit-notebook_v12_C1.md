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
import sys
import gc
import io
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
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
SAMPLE_SUB = (
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
def _fig_to_uint8_rgb(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = plt.imread(buf)  # returns float32 RGBA in [0,1] for PNG
    buf.close()

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,
    eeg_out=EEG_IMG_TEST_PATH,
):
    eeg = pd.read_parquet(f"{input_path}{eegid}.parquet")

    cicles = 0
    relpos = 0
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True)

    eeg_index_sec = eeg.index.to_numpy(dtype=np.float32) / 200.0
    for key, values in eeg_zone.items():
        ax.plot(
            eeg_index_sec,
            (eeg[values[0]].to_numpy() - eeg[values[1]].to_numpy()) + relpos,
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

    arr = _fig_to_uint8_rgb(fig)
    plt.close(fig)
    return arr




## === cell 4
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
    spec = spec.set_index("time")
    spec = spec.T

    parts = spec.index.to_series().str.split("_", expand=True)
    spec["freq"] = parts[1].astype(float).to_numpy()
    spec["brainreg"] = parts[0].astype(str).to_numpy()

    spec.set_index("freq", inplace=True)

    subspec = {}
    for zone in spec_zones:
        sub = spec[spec.brainreg == zone].drop("brainreg", axis=1)
        subspec[f"{zone}_sub"] = sub

    fig, ax = plt.subplots(nrows=len(spec_zones), figsize=(3, 3), sharex=True)
    if len(spec_zones) == 1:
        ax = [ax]

    for row in range(len(spec_zones)):
        data = subspec[f"{spec_zones[row]}_sub"]
        vmax = float(data.max().max()) * float(output_filter) if data.size else 1.0
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
            vmax=vmax if vmax > 0 else 1.0,
        )
        ax[row].set_xticks([])
        ax[row].set_yticks([])

    plt.subplots_adjust(hspace=0.01)

    arr = _fig_to_uint8_rgb(fig)
    plt.close(fig)
    return arr




## === cell 5
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)

sample_sub = pd.read_csv(SAMPLE_SUB)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]
assert TARGET_COLS == [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

_meta_eeg_ids = metadata["eeg_id"].to_numpy()
_meta_spec_ids = metadata["spectrogram_id"].to_numpy()




## === cell 6
def drop_images(paths):
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass




## === cell 7
def preprocess_data(metadata, row, train=False):
    if train:
        eeg_arr = generate_eeg(
            eegid=int(metadata.loc[row].eeg_id),
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/",
        )
        spec_arr = generate_spectrogram(
            specid=int(metadata.loc[row].spectrogram_id),
            output_filter=0.7,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/",
        )
    else:
        eeg_id = int(_meta_eeg_ids[row])
        spec_id = int(_meta_spec_ids[row])
        eeg_arr = generate_eeg(eegid=eeg_id)
        spec_arr = generate_spectrogram(specid=spec_id, output_filter=0.7)

    eeg_tensor = tf.convert_to_tensor(eeg_arr, dtype=tf.float32) / 255.0
    spec_tensor = tf.convert_to_tensor(spec_arr, dtype=tf.float32) / 255.0

    eeg_tensor = tf.image.resize(eeg_tensor, [224, 224])
    spec_tensor = tf.image.resize(spec_tensor, [240, 240])

    eeg_tensor = tf.expand_dims(eeg_tensor, axis=0)
    spec_tensor = tf.expand_dims(spec_tensor, axis=0)

    eeg_id = int(metadata.loc[row].eeg_id) if train else int(_meta_eeg_ids[row])
    return eeg_tensor, spec_tensor, eeg_id




## === cell 8
class UniformModel:
    def predict(self, inputs, verbose=0):
        batch = int(inputs[0].shape[0])
        return np.full((batch, 6), 1.0 / 6.0, dtype=np.float32)


MODEL_PATH = "/kaggle/input/hms-models/best_model_st_6_ft.keras"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    print(f"[WARN] Model not found at {MODEL_PATH}. Using UniformModel fallback.")
    model = UniformModel()



## === cell 9
submission = {"eeg_id": []}
for c in TARGET_COLS:
    submission[c] = []

BATCH_SIZE = 32  # speed-only; does not change results (deterministic inference)

n = len(metadata)
row = 0
while row < n:
    end = min(row + BATCH_SIZE, n)

    eeg_batch = []
    spec_batch = []
    eeg_ids_batch = []

    for r in range(row, end):
        eeg_img, spec_img, eeg_id = preprocess_data(metadata, r, train=False)
        eeg_batch.append(eeg_img)
        spec_batch.append(spec_img)
        eeg_ids_batch.append(eeg_id)

    eeg_batch = tf.concat(eeg_batch, axis=0)
    spec_batch = tf.concat(spec_batch, axis=0)

    prediction = model.predict([eeg_batch, spec_batch], verbose=0)
    prediction = np.asarray(prediction)

    prediction = np.clip(prediction, 1e-8, None)
    prediction = prediction / prediction.sum(axis=1, keepdims=True)

    submission["eeg_id"].extend(eeg_ids_batch)
    for j, col in enumerate(TARGET_COLS):
        submission[col].extend(prediction[:, j].astype(np.float64).tolist())

    del eeg_batch, spec_batch, prediction, eeg_ids_batch, eeg_img, spec_img, eeg_id
    if (row // BATCH_SIZE) % 50 == 0:
        gc.collect()

    row = end

pdsubmit = pd.DataFrame(submission)



## === cell 10
pdsubmit = sample_sub[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")

for col in TARGET_COLS:
    if col not in pdsubmit.columns:
        pdsubmit[col] = 1.0 / 6.0
    pdsubmit[col] = pdsubmit[col].fillna(1.0 / 6.0)

probs = pdsubmit[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
pdsubmit.loc[:, TARGET_COLS] = probs

pdsubmit



## === cell 11
pdsubmit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pdsubmit.shape)
print(pdsubmit.head())
