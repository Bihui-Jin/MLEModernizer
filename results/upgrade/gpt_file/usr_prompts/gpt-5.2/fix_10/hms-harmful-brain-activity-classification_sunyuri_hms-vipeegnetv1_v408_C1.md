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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.2849029597490072

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I remove the protobuf/TF determinism configuration that is triggering the `MessageFactory.GetPrototype` crash on Kaggle/Python 3.13, while keeping the model/training logic unchanged. Then I fix the Kaggle inference path so it no longer depends on an external `/kaggle/input/models*` dataset being present; if no pretrained weights are found, it fall back to generating a valid, properly-normalized submission from the train label prior (so a `.csv` is always produced). Finally, I keep the original “load weights and ensemble” inference when weights are available, and I harden probability normalization to guarantee each row sums to 1 and matches the required column order.'
- What this solution (achieved 1.39779) has done: 'I fix the protobuf/TensorFlow crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF, which avoids the `MessageFactory.GetPrototype` error on Kaggle Python 3.13 while keeping your model/training/inference logic intact. I also make model discovery consistent with your actual fold naming (`fold{i}_stage2.weights.h5` for i in 0..SPLITS-1) so any attached weights are actually loaded; this is the smallest change likely to move your score down toward the target because your current 1.39779 looks like it’s mostly falling back to the train-prior submission. Finally, I keep (and slightly harden) the submission probability normalization to guarantee valid rows that sum to 1 with the required column order.'
- What this solution (achieved 1.39779) has done: 'I fix two runtime blockers so the notebook runs end-to-end on Kaggle/Python 3.13: (1) avoid the protobuf `MessageFactory.GetPrototype` crash by switching Keras to the NumPy backend (no TensorFlow import needed for inference fallback), and (2) fix the Keras error from trying to assign to a read-only `name` property by not renaming the EfficientNet model. To ensure you always get a valid `submission.csv` even when no pretrained weights are attached, I keep your existing “train-prior” fallback but make it unconditional for Kaggle by disabling training and skipping any TensorFlow-dependent code paths. Finally, I harden the submission to guarantee finite probabilities, correct columns, and rows summing to 1 (required for a valid submission and improves score vs invalid/NaN outputs).'
- What this solution (achieved 1.39779) has done: 'Your current 1.39779 is consistent with always falling back to the global train-prior, which is far from the target 0.2849 (lower is better), so we should minimally enable real inference with your existing model definition and ensembling logic. I keep your core model architecture unchanged, but switch the Kaggle path back to a safe TensorFlow-based inference mode by forcing pure-Python protobuf before importing TF (to avoid the Python 3.13 crash) and by actually loading fold weights when present. I also implement the missing test EEG feature extraction in the same way as your training-time EEG preprocessing (channel differencing + bandpass + clipping), but only for test and only when weights exist, so it stays within runtime. Finally, I keep and harden the probability normalization to guarantee valid rows summing to 1 for KL-divergence evaluation.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) strongly suggests you’re still submitting the “train-prior fallback” rather than real model inference, so the smallest change that should move you much closer to the target is to reliably find and load your fold weights. I keep your model and preprocessing identical, but (1) broaden and harden model-weight discovery (including recursively searching common Kaggle input locations), and (2) ensure the test batch passed to the model matches the model’s expected input dict name (`"eeg"`) to avoid silent mis-feeding when Keras expects named inputs. I also keep your existing probability normalization (needed for a valid KL submission) and keep the train-prior fallback only if no weights can be loaded. These changes are directly targeted at turning on the intended ensemble inference without altering architecture/training semantics.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far above the target (0.2849), and it strongly suggests you’re still mostly submitting the global train-prior fallback rather than real fold-ensemble inference. The smallest change likely to move the score down toward the target is to (1) actually find/load fold weights when they exist and (2) ensure inference uses the same input preprocessing shape the model expects (including the EEG_MULTIPLY downsampling that your model’s Input shape implies). I keep the model architecture and prediction loop intact, but fix the test EEG preprocessing to correctly resample to `RSFREQ/EEG_MULTIPLY` so the time dimension matches the model’s expected length, avoiding silent mismatch/poor predictions. I also slightly harden the weight discovery to prefer the explicitly configured `LOAD_MODELS_FROM` first (so we don’t accidentally pick up wrong .h5 files), while keeping the same fallback behavior and the same probability normalization for a valid KL submission.'
- What this solution (achieved 1.39779) has done: 'We need to move your score down (lower-is-better) from 1.39779 toward 0.28490, and the biggest likely blocker is that you’re still not actually using the fold ensemble at inference time (so you submit the global prior). I make two minimal, directly score-relevant fixes: (1) ensure model weights actually load by calling `model(..., training=False)` once before `load_weights()` (builds variables for subclassed graphs and avoids silent partial loads), and (2) make weight discovery prefer only fold-matching filenames and avoid accidentally picking unrelated `.h5` files that can degrade predictions. I also make test EEG preprocessing match your training-time exactly by removing the extra `EEG_MULTIPLY` resample branch (your model already encodes EEG_MULTIPLY in its input shape), which prevents subtle time-length mismatches that can hurt KL. Submission normalization and fallback behavior remain unchanged so it always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024
@author: yuri
email: syuri@tju.edu.cn

Patched for Kaggle (Python 3.13) to improve score toward target:
- Enable TF inference safely by forcing pure-Python protobuf BEFORE importing TensorFlow.
- Keep core model architecture unchanged.
- Make fold weight discovery robust (including recursive search) so we actually run real inference instead of train-prior fallback (main driver of score ~1.39).
- Feed named input dict {"eeg": batch} to match Keras named Input and avoid any misalignment.
- Preserve the train-prior fallback if no weights are found, so a valid submission is always produced.
- Enforce finite probs, clip, and renormalize to sum=1 (submission validity for KL metric).
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img ***
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 40
SPE_WIDE = 1000

STFT_LENGTH = 50
STFT_TIME = 0.1
STFT_HIGH = 50
STFT_WIDE = 200

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5

READ_EEG_FILES = False
READ_SPE_FILES = False

spectrograms = {}
eegs = {}
stfts = {}
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",  # LL
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",  # LP
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",  # RP
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",  # RL
]

TEST_BATCHSIZE = 128

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import zoom
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

TARGETS_RAW = [c + "_raw" for c in TARGETS]

if NEEDTRAIN:
    if READ_EEG_FILES:
        train = df.drop_duplicates(
            [
                "eeg_id",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]
        ).reset_index(drop=True)
        train["sign_id"] = train.index.values
        df["sign_id"] = df.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data
        train.to_csv("train.csv", index=False)
    else:
        if os.path.exists("train.csv"):
            train = pd.read_csv("train.csv")
        else:
            train = df.drop_duplicates(
                [
                    "eeg_id",
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
            ).reset_index(drop=True)
            train["sign_id"] = train.index.values
            df["sign_id"] = df.index.values
            y_data = train[TARGETS].values
            train[TARGETS_RAW] = y_data
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## === cell 1
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        if ("stft" in DATATYPE) or ("img" in DATATYPE):
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )
        time_start_time = time.time()

        for i, eeg_id in enumerate(train.eeg_id.unique()):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet"))
            )

            eeg = []
            for channel in BRAIN:
                eeg_temp = (
                    eeg_default.loc[:, channel.split("-")[0]]
                    - eeg_default.loc[:, channel.split("-")[1]]
                ).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                train_plot = train[train.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(train_plot)):
                    row = train_plot.iloc[j]
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )

                    eeg_plot = eeg2[
                        :,
                        round(row.eeg_label_offset_seconds * RSFREQ) : round(
                            (row.eeg_label_offset_seconds + EEG_LENGTH) * RSFREQ
                        ),
                    ]
                    eeg_plot = eeg_plot[
                        :,
                        round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                            (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                        ),
                    ]

                    img_save = np.zeros(
                        (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                    )
                    for ii in range(eeg_plot.shape[0]):
                        fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                        fig.patch.set_facecolor("black")
                        plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)

                        plt.xlim(-5, eeg_plot.shape[1] + 5)
                        plt.ylim(0, 200)
                        plt.axis("off")

                        byte_stream = io.BytesIO()
                        plt.savefig(
                            byte_stream, format="png", bbox_inches="tight", dpi=100
                        )
                        byte_stream.seek(0)
                        img = Image.open(byte_stream)
                        img = np.array(img)[:, :, :1]
                        img = img / 255
                        img = np.array(img, dtype=np.float32)
                        byte_stream.truncate()
                        plt.close("all")

                        if img.shape != (36, IMG_WIDE, 1):
                            img = np.concatenate((img, img, img), 2)
                            import tensorflow as tf  # only if training locally

                            img = np.array(
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]

                        img_save[ii, :, :] = img

                    imgs[train_plot.sign_id[j]] = img_save

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            if filter_range is not None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.array(eeg, dtype=np.float32)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()

if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    time_start_time = time.time()
    if READ_SPE_FILES:
        for i, f in enumerate(files):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(files)
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        np.save("./input/preprocess/spectrograms.npy", spectrograms, allow_pickle=True)
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()




## === cell 2
def build_model():
    import tensorflow as tf  # only invoked if training or TF inference is used

    inp = []
    y = 0

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            name="eeg",
        )
        x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        strides = 10
        eeg_embed = tf.keras.layers.Conv1D(
            filters=strides * 3,
            kernel_size=strides,
            strides=strides,
            padding="same",
            use_bias=False,
            activation=None,
        )

        x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)

        x_eeg = tf.keras.layers.Concatenate(axis=-1)(
            [
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 0 * strides : 1 * strides]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 1 * strides : 2 * strides]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 2 * strides : 3 * strides]
                ),
            ]
        )
        x_eeg = tf.keras.layers.Permute([4, 2, 1, 3])(x_eeg)
        x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
        x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

        base_model_eeg = tf.keras.applications.EfficientNetV2B3(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)
        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

    y = y_eeg * 1
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 3
def train_fold(
    i,
    stage,
    train_index,
    valid_index,
    df_train_stage1,
    df_valid_stage1,
    df_train_stage2,
    df_valid_stage2,
    build_model,
    BATCHSIZE,
    EPOCHS,
    LEARN_RATE,
    TARGETS,
    TARGETS_RAW,
):
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from sklearn.metrics import confusion_matrix

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super().__init__()
            self.total_step = total_step
            self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
            self.lr_max = lr_max
            self.lr_min = lr_min
            self.begin = 1

        def __call__(self, step):
            if step == self.total_step:
                self.begin = 0
                self.lr_max = self.lr_max * 0.5
                self.lr_min = self.lr_min * 0.1

            step = step % self.total_step
            step = step + 1

            if (self.begin == 1) and (step < self.warm_step):
                lr = self.lr_max / self.warm_step * step
            else:
                if self.begin == 1:
                    if self.total_step == 1:
                        lr = self.lr_max
                    else:
                        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                            1.0
                            + tf.cos(
                                (step - self.warm_step)
                                / (self.total_step - self.warm_step)
                                * np.pi
                            )
                        )
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0 + tf.cos(step / 10 * np.pi)
                    )
            return np.float32(lr)

    print("#" * 25)
    print(f"### Fold {i + 1}")

    model = build_model()
    loss = tf.keras.losses.KLDivergence()

    raise RuntimeError(
        "Training is disabled in this patched Kaggle script (NEEDTRAIN=False). "
        "Run locally with the original training pipeline if needed."
    )




## === cell 4
def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-7, 1.0)
    s = np.sum(p, axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    p = p / s
    return p.astype(np.float32)


def _candidate_weight_paths(base_dir: str, fold_i: int):
    names = [
        f"fold{fold_i}_stage2.weights.h5",
        f"fold_{fold_i}_stage2.weights.h5",
        f"fold{fold_i}.weights.h5",
        f"fold_{fold_i}.weights.h5",
        f"fold{fold_i}_stage2.h5",
        f"fold_{fold_i}_stage2.h5",
        f"fold{fold_i}.h5",
        f"fold_{fold_i}.h5",
    ]
    return [os.path.join(base_dir, n) for n in names]


def _discover_models_dirs():
    dirs = []
    if os.path.isdir(LOAD_MODELS_FROM):
        dirs.append(LOAD_MODELS_FROM)

    for p in ("./models", "./input/models", "./input/model", "./input/weights"):
        if os.path.isdir(p):
            dirs.append(p)

    if PLATFORM == "kaggle":
        for base in ("/kaggle/input",):
            if os.path.isdir(base):
                try:
                    for d in os.listdir(base):
                        if d.startswith("models") or d in (
                            "models",
                            "model",
                            "weights",
                            "checkpoints",
                        ):
                            p = os.path.join(base, d)
                            if os.path.isdir(p):
                                dirs.append(p)
                except Exception:
                    pass

    def _bounded_walk(root, max_dirs=2000):
        cnt = 0
        for dirpath, dirnames, filenames in os.walk(root):
            cnt += 1
            if cnt > max_dirs:
                break
            if any(("fold" in fn and fn.endswith(".h5")) for fn in filenames):
                yield dirpath

    if PLATFORM == "kaggle" and os.path.isdir("/kaggle/input"):
        for p in _bounded_walk("/kaggle/input", max_dirs=2000):
            dirs.append(p)

    out, seen = [], set()
    for d in dirs:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def _load_test_eeg_array(eeg_id: int) -> np.ndarray:
    path = os.path.join(LOAD_DATA_FROM, "test_eegs", f"{int(eeg_id)}.parquet")
    eeg_default = pd.read_parquet(path)

    eeg = []
    for channel in BRAIN:
        a_ch, b_ch = channel.split("-")
        eeg_temp = (eeg_default.loc[:, a_ch] - eeg_default.loc[:, b_ch]).values
        eeg_temp = np.nan_to_num(eeg_temp, nan=0.0, posinf=0.0, neginf=0.0)
        eeg.append(np.reshape(eeg_temp, (1, -1)))
    eeg = np.concatenate(eeg, axis=0)

    if SFREQ != RSFREQ:
        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

    eeg = np.clip(eeg, a_min=-1024, a_max=1024)
    if filter_range is not None:
        eeg = signal.filtfilt(b, a, eeg, axis=1)
    eeg = np.array(eeg, dtype=np.float32)

    expected_len = round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY)
    if eeg.shape[1] > expected_len:
        eeg = eeg[:, :expected_len]
    elif eeg.shape[1] < expected_len:
        pad = expected_len - eeg.shape[1]
        eeg = np.pad(eeg, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    if EEG_CHANNEL_USED != eeg.shape[0]:
        eeg = eeg[:EEG_CHANNEL_USED, :]

    return eeg


if __name__ == "__main__":
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    model_dirs = _discover_models_dirs()
    print(
        "Model search dirs (deduped):",
        model_dirs[:25],
        "..." if len(model_dirs) > 25 else "",
    )
    print("Total candidate dirs:", len(model_dirs))

    fold_weight_paths = []
    for model_i in range(SPLITS):
        chosen = None
        for d in model_dirs:
            for wpath in _candidate_weight_paths(d, model_i):
                if os.path.exists(wpath):
                    chosen = wpath
                    break
            if chosen is not None:
                break
        fold_weight_paths.append(chosen)

    print("Chosen fold weights:", fold_weight_paths)
    found_any_weight = any(p is not None for p in fold_weight_paths)

    if not found_any_weight:
        print("No fold weights found; using train prior probabilities.")
        prior = df[TARGETS].values.astype(np.float64)
        prior = prior / np.clip(prior.sum(axis=1, keepdims=True), 1e-12, None)
        prior = prior.mean(axis=0)
        prior = np.clip(prior, 1e-7, 1.0)
        prior = prior / prior.sum()
        preds_all = np.tile(prior.reshape(1, -1), (len(test), 1))
        preds_all = _normalize_probs(preds_all)
    else:
        import tensorflow as tf

        tf.random.set_seed(SEED)

        preds_sum = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
        used_folds = 0

        dummy = np.zeros(
            (
                1,
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            dtype=np.float32,
        )

        for fold_i, wpath in enumerate(fold_weight_paths):
            if wpath is None:
                continue
            print(f"Loading fold {fold_i} weights: {wpath}")
            model = build_model()

            _ = model({"eeg": dummy}, training=False)
            model.load_weights(wpath)

            for start in range(0, len(test), TEST_BATCHSIZE):
                end = min(len(test), start + TEST_BATCHSIZE)
                eeg_ids = test.eeg_id.values[start:end]
                batch = np.stack([_load_test_eeg_array(eid) for eid in eeg_ids], axis=0)

                p = model.predict({"eeg": batch}, verbose=0)
                preds_sum[start:end] += np.asarray(p, dtype=np.float64)

            used_folds += 1
            del model
            gc.collect()

        if used_folds == 0:
            print("Weights were detected but none could be loaded; using train prior.")
            prior = df[TARGETS].values.astype(np.float64)
            prior = prior / np.clip(prior.sum(axis=1, keepdims=True), 1e-12, None)
            prior = prior.mean(axis=0)
            prior = np.clip(prior, 1e-7, 1.0)
            prior = prior / prior.sum()
            preds_all = np.tile(prior.reshape(1, -1), (len(test), 1))
        else:
            preds_all = preds_sum / used_folds

        preds_all = _normalize_probs(preds_all)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = preds_all
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", sub.shape)
    print(sub.head())
