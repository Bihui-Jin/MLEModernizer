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

0.3227873128869408

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation early (this is the root cause of the `MessageFactory.GetPrototype` error in many Kaggle TF environments). Then I make model-weight discovery robust on Kaggle by scanning `/kaggle/input/` for a `models*` dataset folder and falling back to a safe uniform-probability submission if no weights are available (so you always get a valid `submission.csv`). Finally, I keep your architecture/training/inference intact, only adding deterministic, score-neutral safety checks to ensure probabilities sum to 1 and submission columns match `sample_submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible `protobuf` version at runtime (via pip in the Kaggle notebook kernel) before importing TensorFlow, while keeping your training/inference/model code intact. Then I make the training-data CSV fallback robust so Kaggle inference never tries to read a missing local `train.csv` artifact (this can silently break subsequent logic). Finally, I remove the intentional `SystemExit(0)` after writing a uniform submission so the notebook completes cleanly and always leaves a valid `submission.csv` behind (score-neutral, but fixes the runtime failure).'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.3228), so the biggest safe gain is to stop falling back to a uniform submission and instead reliably load a real pretrained weights set. I keep your model/training/inference logic unchanged, but make weight discovery robust by scanning recursively under `/kaggle/input/` for `fold*_stage2.weights.h5` files (common Kaggle dataset nesting), and load them directly. I also ensure the correct EEG bandpass filtering is applied consistently in test-time preprocessing (currently the `b,a` filter coefficients are not guaranteed to exist in inference-only runs), and keep the existing probability normalization/safety checks. These are minimal, execution-safe changes that should substantially reduce KL versus the uniform fallback and move the score toward your target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.3228), so the most direct minimal improvement is to fix a silent label-mismatch bug in `DataGenerator` where the row lookup ignores `other_vote`, which makes training/inference inconsistent and hurts calibration. I also make the inference-time weight discovery prefer the intended `LOAD_MODELS_FROM` folder first (so you reliably use the correct trained folds instead of accidentally loading unrelated weights from other datasets under `/kaggle/input`). Finally, I keep your architecture and training loop unchanged, but enforce stable, correct per-row probability normalization at the end (already mostly present) without changing evaluation semantics.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.3228), and the most likely cause is that inference is still often running with missing/incorrect weights or misaligned preprocessing. I (1) make weight discovery prefer the exact `LOAD_MODELS_FROM` subtree and select the best contiguous fold set (to avoid accidentally loading unrelated/partial weights), and (2) make the inference-time EEG preprocessing match the training-time preprocessing (notably: apply the same bandpass filtering order and remove the test-only mirror-padding that changes the signal distribution). These are minimal changes that keep the same model, loss, and prediction semantics but should substantially reduce KL by making the inputs/weights consistent. I also add a strict post-check that enforces finite probabilities summing to 1 to prevent any submission failures.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.3228), and the most likely cause is that inference is not using the intended pretrained weights because the Kaggle-built model graph does not match the training graph (you switch Conv1D initializer/constraint depending on PLATFORM). I make the inference-time model architecture identical to the training-time one by removing the PLATFORM-conditional Conv1D definition so weights load correctly and predictions become meaningful. I also harden weight discovery to prefer the `LOAD_MODELS_FROM` subtree and, if no stage2 weights exist, fall back to stage1 weights (still same model/loss; just better than uniform). Finally, I keep your post-processing normalization but add a strict assertion that row sums are ~1 after normalization to avoid any submission failure.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.3228), and the most likely reason is that inference is often not actually using the trained ensemble because weight discovery/loading is brittle (e.g., stage2 missing, partial folds, or incompatible weights). I make the weight discovery prefer the exact `LOAD_MODELS_FROM` subtree, require a complete fold set when possible, and if stage2 is incomplete fall back to the corresponding stage1 weights (same architecture and inference semantics, just ensuring you actually use real weights). I also add a tiny, score-positive calibration step (very light uniform mixing / label-smoothing at inference) to reduce overconfident probabilities, which typically reduces KL without changing the model. Finally, I keep your probability normalization, but make it strictly safe with float64 normalization once right before writing `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.3228), and the most likely “minimal-change” win is that inference is still often not actually loading the intended fold weights (or is mixing incomplete/mismatched folds), causing near-random predictions. I make weight discovery strictly prefer a complete, consistent fold set from `LOAD_MODELS_FROM` (same stage, same fold indices), and only then fall back to broader `/kaggle/input` scanning—this keeps your model and inference logic identical but makes it far more likely real weights are used. I also force `stage` passed into `DataGenerator` during test-time to match the chosen weights’ stage (currently it is hard-coded to 2), which can change augmentation behavior and slightly shift predictions. Finally, I keep your existing probability normalization but make the uniform-mixing calibration conditional and lighter (only applied when predictions are overconfident / numerically unstable), so we improve KL toward the target without changing core semantics.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.3228), so the smallest score-relevant fixes are to (1) ensure you are actually loading the intended fold weights from the correct dataset path on Kaggle, and (2) make sure the test-time EEG preprocessing matches training-time exactly (notably: channel selection and scaling should be identical, and avoid mixed-float16 inference producing overly peaky/unstable probabilities). I keep your model, loss, and overall training/inference flow unchanged, but make weight discovery prefer `/kaggle/input/<models...>/models/` (common nesting) before any global scan, and I force full-float32 for inference-time prediction only (no architecture change) to stabilize KL. Finally, I keep your existing probability normalization, but make the post-softmax calibration (uniform mixing) slightly stronger and unconditional at a tiny level to reduce overconfidence, which typically improves KL while keeping semantics intact.'

# 9. Code solution

## === cell 0
"""
Created on Sun Mar  9 17:01:44 2025

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    import subprocess, sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    pass

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"
if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # *** local kaggle *** local training or online testing
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
    NEEDTRAIN = False
    try:
        candidates = [d for d in os.listdir("/kaggle/input/") if d.startswith("models")]
        if len(candidates) > 0:
            candidates = sorted(candidates)
            LOAD_MODELS_FROM = candidates[0]
    except Exception:
        pass

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg

eegs = {}  # preprocessed eegs for training
eegs_test = {}

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import warnings

warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras import optimizers

import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

print(tf.version.VERSION)
print(tf.config.list_physical_devices("GPU"))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # allocate on demand
    except RuntimeError as e:
        print(e)

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

MIX = True
if MIX:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using full precision")

if NEEDTRAIN:
    import itertools

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

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
        if os.path.exists("train.csv") and PLATFORM == "local":
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
            y_data = train[TARGETS].values
            train[TARGETS_RAW] = y_data
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data
else:
    TARGETS_RAW = [t + "_raw" for t in TARGETS]

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
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

            eeg = list()
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

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()




## === cell 1
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False,
        sample_weights=False,
        mode="train",
        eegs=None,
        stage=2,
    ):

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stage = stage
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.dataframe) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        return x, y, sample_weights

    def on_epoch_end(self):
        self.nan = 0
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        x_eeg = np.zeros(
            (len(indexes), EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)),
            dtype="float32",
        )
        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            if self.mode != "test":
                sample_weight = sum(row[TARGETS_RAW].values) / 20

            if self.mode == "test":
                r_eeg = 0
            else:
                rows = df.loc[
                    (df.eeg_id == row.eeg_id)
                    * (df.seizure_vote == row.seizure_vote_raw)
                    * (df.lpd_vote == row.lpd_vote_raw)
                    * (df.gpd_vote == row.gpd_vote_raw)
                    * (df.lrda_vote == row.lrda_vote_raw)
                    * (df.grda_vote == row.grda_vote_raw)
                    * (df.other_vote == row.other_vote_raw),
                    :,
                ].reset_index(drop=True)
                if self.mode == "train":
                    rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                        drop=True
                    )
                    row = rows.loc[0, :]
                elif self.mode == "valid":
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )
                r_eeg = row.eeg_label_offset_seconds
                if self.mode == "train":
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0, r_eeg)
                    r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

            eeg = self.eegs[row.eeg_id][
                :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
            ]
            eeg = np.concatenate(
                (
                    eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                ),
                axis=0,
            )

            eeg = eeg[
                :,
                round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                    (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                ),
            ]

            if self.mode == "train":
                if (self.stage == 2) and (np.random.rand() > 0):
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                else:
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                        0 : round(EEG_CHANNEL_USED / 2), :
                    ][np.random.permutation(8), :]
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                        -round(EEG_CHANNEL_USED / 2) :, :
                    ][np.random.permutation(8), :]

                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]

                    if np.random.rand() > 0.5:
                        eeg = -eeg

                    if np.random.rand() > 0.5:
                        eeg = eeg[:, ::-1]
            else:
                eeg2 = eeg.copy()
                eeg[4:8, :] = eeg2[12:16, :]
                eeg[8:12, :] = eeg2[4:8, :]
                eeg[12:16, :] = eeg2[8:12, :]

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            eeg = eeg + 1024
            eeg = eeg / 2048 * 255

            x_eeg[j] = eeg

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)

                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1

        return x_eeg, y, sample_weights




## === cell 2
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = total_step

        if warmth_rate == 0:
            self.warm_step = 1
        else:
            self.warm_step = int(warmth_rate)

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


class IniToOne(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOne, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        for i in range(filter_count):
            kernel[i % filter_length, 0, i] = 1.0
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOne(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOne, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


class IniToOneAtten(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOneAtten, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
            (filter_length) // 2 + 1 - (filter_length - 1) // 2
        )
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOneAtten(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOneAtten, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}




## === cell 3
def build_model():
    inp_eeg = tf.keras.Input(
        shape=(EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)), name="eeg"
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
        kernel_initializer=IniToOne(),
        kernel_constraint=SumToOne(),
        input_shape=(None, 1),
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
    base_model_eeg.name = "eeg_extractor"

    x_eeg = base_model_eeg(x_eeg)

    x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
        x_eeg
    )

    model = tf.keras.Model(inputs=inp_eeg, outputs=y)

    return model




## === cell 4
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

    print("#" * 25)
    print(f"### Fold {i + 1}")

    model = build_model()
    loss = tf.keras.losses.KLDivergence()

    if stage == 1:
        train_gen_stage = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            eegs=eegs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            eegs=eegs,
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
    elif stage == 2:
        train_gen_stage = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            eegs=eegs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2 * 2,
            mode="valid",
            eegs=eegs,
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1 * 3)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1 * 3,
                    LEARN_RATE * 0.1 * 0.1 * 0.1,
                    0,
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

    model.compile(loss=loss, optimizer=opt)

    if stage == 1:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=EPOCHS,
            callbacks=callbacks_stage,
        )
    elif stage == 2:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=max(round(EPOCHS / 3), 1),
            callbacks=callbacks_stage,
        )

    model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

    loss_hist = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss_hist) + 1)
    plt.figure()
    plt.plot(epochs, loss_hist, "bo", label="loss")
    plt.plot(epochs, val_loss, "b", label="val_loss")
    plt.title(
        f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss), 4)}",
        fontsize=12,
    )
    plt.legend()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
    plt.close()

    if stage == 1:
        valid_stage = df_valid_stage1[TARGETS].values
    elif stage == 2:
        valid_stage = df_valid_stage2[TARGETS].values
    predict_stage = model.predict(valid_gen_stage)

    del train_gen_stage, valid_gen_stage, history, model
    tf.keras.backend.clear_session()
    gc.collect()

    cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
    cm = cm / np.sum(cm, 1, keepdims=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    plt.yticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    thresh = cm.max() / 2.0
    for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[ii, jj] > -0.1:
            plt.text(
                jj,
                ii,
                str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                horizontalalignment="center",
                color="white" if cm[ii, jj] > thresh else "black",
                fontsize=10,
            )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}_cm.svg"))
    plt.close()

    del df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2
    gc.collect()




## === cell 5
if __name__ == "__main__":
    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        mp.set_start_method("spawn")

        gkf = GroupKFold(n_splits=SPLITS)

        for i, (train_index, valid_index) in enumerate(
            gkf.split(train, train.expert_consensus, train.patient_id)
        ):
            print("#" * 25)
            print(f"### Fold {i + 1}")

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[
                np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[
                np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)

            for stage in [1, 2]:
                p = mp.Process(
                    target=train_fold,
                    args=(
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
                    ),
                )
                p.start()
                p.join()

    else:
        def _extract_fold_id(path):
            base = os.path.basename(path)
            try:
                i0 = base.find("fold") + 4
                i1 = base.find("_", i0)
                return int(base[i0:i1])
            except Exception:
                return None

        def _discover_weight_files(root, stage):
            suffix = f"_stage{stage}.weights.h5"
            out = []
            if root and os.path.exists(root):
                for r, _, files in os.walk(root):
                    for fn in files:
                        if fn.startswith("fold") and fn.endswith(suffix):
                            out.append(os.path.join(r, fn))
            return sorted(out)

        def _fold_map(paths):
            m = {}
            for p in paths:
                f = _extract_fold_id(p)
                if f is not None and f not in m:
                    m[f] = p
            return m

        preferred_roots = [
            LOAD_MODELS_FROM,
            os.path.join(LOAD_MODELS_FROM, "models"),
        ]
        stage2_local, stage1_local = [], []
        for root in preferred_roots:
            stage2_local.extend(_discover_weight_files(root, stage=2))
            stage1_local.extend(_discover_weight_files(root, stage=1))
        stage2_local = sorted(list(dict.fromkeys(stage2_local)))
        stage1_local = sorted(list(dict.fromkeys(stage1_local)))

        stage2_map = _fold_map(stage2_local)
        stage1_map = _fold_map(stage1_local)

        def _choose_complete_set(stage_map, desired_splits):
            if len(stage_map) == 0:
                return None
            folds = sorted(stage_map.keys())
            if set(range(desired_splits)).issubset(set(folds)):
                return [stage_map[i] for i in range(desired_splits)]
            for start in folds:
                block = list(range(start, start + desired_splits))
                if set(block).issubset(set(folds)):
                    return [stage_map[i] for i in block]
            if len(folds) >= desired_splits:
                return [stage_map[i] for i in folds[:desired_splits]]
            return [stage_map[i] for i in folds]

        chosen_stage = None
        chosen = None

        chosen2 = _choose_complete_set(stage2_map, SPLITS)
        chosen1 = _choose_complete_set(stage1_map, SPLITS)

        if chosen2 is not None and len(chosen2) >= min(SPLITS, 2):
            chosen_stage, chosen = 2, chosen2
        elif chosen1 is not None and len(chosen1) >= min(SPLITS, 2):
            chosen_stage, chosen = 1, chosen1
        else:
            all2, all1 = [], []
            for r, _, files in os.walk("/kaggle/input"):
                for fn in files:
                    if fn.startswith("fold") and fn.endswith("_stage2.weights.h5"):
                        all2.append(os.path.join(r, fn))
                    elif fn.startswith("fold") and fn.endswith("_stage1.weights.h5"):
                        all1.append(os.path.join(r, fn))
            all2_map = _fold_map(sorted(all2))
            all1_map = _fold_map(sorted(all1))
            chosen2 = _choose_complete_set(all2_map, SPLITS)
            chosen1 = _choose_complete_set(all1_map, SPLITS)
            if chosen2 is not None and len(chosen2) >= min(SPLITS, 2):
                chosen_stage, chosen = 2, chosen2
            elif chosen1 is not None and len(chosen1) >= min(SPLITS, 2):
                chosen_stage, chosen = 1, chosen1
            else:
                chosen_stage, chosen = 2, []  # will trigger uniform fallback

        print(f"Chosen stage={chosen_stage} weight files ({len(chosen)}):")
        for p in chosen:
            print("  ", p)

        try:
            tf.keras.mixed_precision.set_global_policy("float32")
        except Exception:
            pass

        models = []
        for wpath in chosen:
            try:
                print(f"Loading weights: {wpath}")
                model = build_model()
                model.load_weights(wpath)
                models.append(model)
            except Exception as e:
                print(f"WARNING: failed to load {wpath}: {e}")

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        print("Test shape", test.shape)

        if len(models) == 0:
            print(
                f"WARNING: No model weights found/loaded (LOAD_MODELS_FROM={LOAD_MODELS_FROM}). "
                "Writing a safe uniform-probability submission.csv."
            )
            sample = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
            sub = sample.copy()
            uniform = np.full(
                (len(sub), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
            )
            sub.loc[:, TARGETS] = uniform
            sub.to_csv("submission.csv", index=False)
            print("\nSubmission shape", sub.shape)
            print(sub.head())
        else:
            test["sign_id"] = test.index.values

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

            preds_all = None
            n_test = len(test)
            for start in range(0, n_test, TEST_BATCHSIZE):
                end = min(start + TEST_BATCHSIZE, n_test)

                eegs_test = {}
                for idx in range(start, end):
                    if idx % 100 == 0:
                        print(idx, ", ", end="")
                    eeg_id = test.eeg_id.iloc[idx]
                    eeg_default = pd.read_parquet(
                        os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                    )

                    eeg = list()
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

                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                    if filter_range is not None:
                        eeg = signal.filtfilt(b, a, eeg, axis=1)

                    eeg = np.array(eeg, dtype=np.float32)
                    eegs_test[eeg_id] = eeg

                test_batch = test.iloc[start:end].reset_index(drop=True)
                preds = []
                test_gen = DataGenerator(
                    test_batch,
                    shuffle=False,
                    sample_weights=False,
                    batch_size=TEST_BATCHSIZE,
                    mode="test",
                    eegs=eegs_test,
                    stage=chosen_stage,
                )
                for model_i in range(len(models)):
                    pred = models[model_i].predict(test_gen, verbose=0)
                    preds.append(pred)

                pred = np.mean(np.stack(preds, axis=0), axis=0)

                if preds_all is None:
                    preds_all = pred.copy()
                else:
                    preds_all = np.concatenate((preds_all, pred), axis=0)

                del test_gen, preds, pred, eegs_test
                gc.collect()

            preds_all = np.asarray(preds_all, dtype=np.float32)

            preds_all = np.nan_to_num(
                preds_all, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            preds_all = np.clip(preds_all, 1e-8, 1.0)
            preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

            eps_mix = 0.01
            preds_all = (1.0 - eps_mix) * preds_all + eps_mix * (1.0 / len(TARGETS))

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = preds_all

            sample = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
            sub = sub[sample.columns]

            probs = sub[TARGETS].values.astype(np.float64)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.clip(probs, 1e-12, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)
            sub.loc[:, TARGETS] = probs.astype(np.float32)

            sub.to_csv("submission.csv", index=False)
            print("\nSubmission shape", sub.shape)
            print(sub.head())
