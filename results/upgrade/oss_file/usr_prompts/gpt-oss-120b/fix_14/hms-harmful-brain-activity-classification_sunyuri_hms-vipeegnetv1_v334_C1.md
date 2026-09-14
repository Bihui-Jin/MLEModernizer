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

0.2892674952137032

# 6. Current score

0.76636

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf import error by guarding the aliasing logic and make the data‑loading path robust so the script can always locate `test.csv`. These changes prevent the early crash and ensure a valid `submission.csv` is written without altering the core model or training logic.'
- What this solution (achieved 1.39779) has done: 'Implemented a small but critical fix: corrected the faulty import statement (`import pandas as pd, np`) to properly import NumPy as `np`. Added a brief comment explaining the change. No other logic was altered, preserving the original model‑free baseline while ensuring the script runs end‑to‑end and produces a compliant `submission.csv`.'
- What this solution (achieved 1.67825) has done: 'I add a lightweight patient‑level prior: for each patient in the training set I compute the average normalized vote distribution and use it for test rows belonging to the same patient. If a patient is unseen, the global class prior is used. This small change keeps the original “no‑training” logic but provides more informative predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.76636) has done: 'I keep the original data‑loading and patient‑prior logic unchanged but add a lightweight smoothing step: each patient’s vote distribution be blended with the overall class prior (10 % weight). This reduces overly confident predictions, which typically lowers the KL‑divergence score while preserving the core approach and keeping all rows summing to one.'
- What this solution (achieved 0.81911) has done: 'I increase the smoothing weight so predictions rely more on the overall class prior, which reduces over‑confident per‑patient estimates and should lower the KL‑divergence (the metric is lower‑is‑better). This is a minimal change that keeps the coreI keep the overall data loading, patient‑prior computation, and blending logic unchanged, but add a light temperature‑scaling step after the smoothing blend. Making the predictions a bit softer (using a temperature > 1) reduces over‑confidence and typically lowers the KL‑divergence, moving the score closer to the target without altering the core model‑free approach.'
- What this solution (achieved 1.39779) has done: 'I keep the existing data‑loading and patient‑prior logic but replace the per‑patient blend with a full smoothing toward the overall class prior (α = 1.0) and remove the temperature scaling (T = 1.0). This makes every prediction equal to the global class prior, which is less over‑confident than the previous patient‑specific estimates and moves the KL‑divergence closer to the target score while preserving the original workflow.'
- What this solution (achieved 0.80514) has done: 'I blend the per‑patient vote distribution with the overall class prior instead of using the global prior alone. This adds a modest amount of patient‑specific information while keeping predictions well‑calibrated, which should lower the KL‑divergence and move the score closer to the target. The change is confined to the prediction step and retains the original workflow.'
- What this solution (achieved 1.2512) has done: 'I reduce the reliance on patient‑specific priors (set ALPHA to 0.2) and apply a mild temperature scaling (T = 2) after blending, which softens over‑confident predictions and should lower the KL‑divergence toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.76636) has done: 'I increased the weight given to the patient‑specific vote distribution (ALPHA = 0.9) and removed the temperature‑scaling (T = 1.0) so the blended predictions rely more on the patient prior while still being normalised to a proper probability distribution. This modest adjustment keeps the original workflow unchanged but should produce softer, better‑calibrated probabilities and therefore lower the KL‑divergence toward the target score. The only code changes are the two constant values and a brief comment, and the script now starts with cell 1 as required.'
- What this solution (achieved 1.13401) has done: 'I lower the reliance on patient‑specific priors and add a modest temperature scaling so the predictions are softer, which tends to reduce over‑confidence and lower the KL‑divergence toward the target. The only changes are the constants `ALPHA` and `T` and the cell header renumbering.'
- What this solution (achieved 0.76636) has done: 'I keep the original workflow unchanged and only adjust the smoothing constants that control how much we rely on patient‑specific vote distributions versus the global class prior. Raising `ALPHA` to 0.9 gives far more weight to the patient prior (which was shown to improve KL‑divergence), and setting the temperature `T` to 1.0 removes the extra uniform‑making scaling. These minimal changes are expected to lower the score toward the target while preserving all existing logic and ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        fallback = getattr(_mf.MessageFactory, "GetMessageClass", lambda x: x)
        _mf.MessageFactory.GetPrototype = fallback
except Exception:
    pass

NEEDTRAIN = False  # skip training to guarantee fast, reproducible execution
LOAD_MODELS_FROM = "modelsxxxxxxx"  # path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
else:
    PLATFORM = "local"

if PLATFORM == "kaggle":
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name.startswith("models"):
            LOAD_MODELS_FROM = dir_name
            break

DEFAULT_KAGGLE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
DEFAULT_REL_PATH = "./input/hms-harmful-brain-activity-classification"

if os.path.isdir(DEFAULT_KAGGLE_PATH) and os.path.exists(
    os.path.join(DEFAULT_KAGGLE_PATH, "test.csv")
):
    LOAD_DATA_FROM = DEFAULT_KAGGLE_PATH
else:
    LOAD_DATA_FROM = DEFAULT_REL_PATH

print("Platform:", PLATFORM)
print("Load data from:", LOAD_DATA_FROM)

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_HIGH = 64  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / 0.5)  # the width of the STFT (eeg spectrogram)  50 * 5

filter_range = [0.5, 45]  # eeg filtering range
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg
READ_SPE_FILES = False  # preprocess spectrogram

import io
from PIL import Image
import pandas as pd
import numpy as np  # Fixed import: correctly bring in NumPy
from sklearn.metrics import confusion_matrix

import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model
from tensorflow.python.framework.ops import reset_default_graph

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

MIX = True
if MIX:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using full precision")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six vote columns
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

vote_vals = df[TARGETS].values.astype(np.float32)
row_sums = vote_vals.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
norm_votes = vote_vals / row_sums
class_prior = norm_votes.mean(axis=0)  # shape (6,)

patient_prior_df = df.groupby("patient_id")[list(TARGETS)].sum()
patient_row_sums = patient_prior_df.sum(axis=1).replace(0, 1)  # avoid division by zero
patient_prior_df = patient_prior_df.div(patient_row_sums, axis=0)

if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    test = test.merge(
        patient_prior_df,
        left_on="patient_id",
        right_index=True,
        how="left",
        suffixes=("", "_patient"),
    )

    for i, col in enumerate(TARGETS):
        if col in test.columns:
            test[col] = test[col].fillna(class_prior[i])
        else:
            test[col] = class_prior[i]

    ALPHA = 0.9  # weight of patient prior (0 ≤ ALPHA ≤ 1); higher = more patient info
    patient_vals = test[TARGETS].values.astype(np.float32)  # shape (n_test, 6)
    blended_vals = ALPHA * patient_vals + (1.0 - ALPHA) * class_prior  # broadcast

    T = 1.0  # temperature; T=1 leaves distribution unchanged
    scaled_vals = np.power(blended_vals, 1.0 / T)

    row_sums = scaled_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pred_vals = scaled_vals / row_sums

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = pred_vals
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print("Submission preview:")
    print(sub.head())
