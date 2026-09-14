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

0.2867982701776539

# 6. Current score

1.00398

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the protobuf/TensorFlow crash by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` overrides, which cause the `MessageFactory.GetPrototype` error in the Kaggle runtime. I also fix training/inference weight file naming to satisfy modern Keras (`save_weights_only=True` requires `.weights.h5`), so training and loading work without changing the model. For Kaggle, since no pretrained model folder is available and `preprocess/eegs.npy` does not exist, I switch the Kaggle path to a safe inference-only fallback that produces a valid submission by using the class distribution prior from `train.csv` (score-improving vs uniform, and guaranteed correct length). Finally, I enforce submission row alignment and length to exactly match `test.csv` and ensure probabilities sum to 1.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by avoiding the incompatible Keras 3 + TF backend initialization in this environment and, when no trained model weights are available, keep the same “prior-based” fallback but improve it minimally by using a patient-aware prior (computed from `train.csv` grouped by `patient_id`, with a global fallback for unseen patients). This keeps the solution end-to-end, produces a valid `submission.csv`, and should reduce KL divergence versus the current global-prior-only fallback. I also make the imports robust so the script runs even when TensorFlow is unavailable, and ensure probabilities are strictly normalized and aligned to `test.csv`. No model architecture/training logic is changed (it remains present and would run if TF is usable and weights exist), but the Kaggle submission path be stable.'
- What this solution (achieved 0.86104) has done: 'I fix the TensorFlow/protobuf crash by making TensorFlow truly optional and only setting the Keras TensorFlow backend when we are actually going to use TF; otherwise we avoid importing TF entirely and proceed with the existing prior-based fallback. This keeps the core model/training code intact (it still run if TF is available and compatible), but unblocks end-to-end execution in the Kaggle runtime where TF init currently fails. To move the score down toward the target (lower-is-better) without changing model logic, I slightly improve the fallback by using a smoothed blend of patient-specific prior and global prior (with mild Dirichlet/Laplace smoothing), which typically reduces KL vs using a hard patient prior. Finally, I keep the submission-writing guardrails to ensure exact row alignment and probabilities summing to 1.'
- What this solution (achieved 0.97512) has done: 'I fix the immediate crash by preventing the TensorFlow import/initialization that triggers the protobuf `MessageFactory.GetPrototype` AttributeError in this Kaggle runtime, while keeping all original TF/model/training code intact and only executing it when explicitly enabled. Since your current score (0.86104, lower-is-better) is far from the target (~0.2868), I minimally improve the existing inference-only fallback (still “prior-based”, not changing any model logic) by using a spectrogram-aware prior in addition to the patient-aware smoothing you already added. I also add a small, safe row-wise sharpening/softening temperature calibration (bounded) to reduce KL by avoiding overconfident priors, and keep the strict submission alignment + normalization guarantees.'
- What this solution (achieved 0.95238) has done: 'Your current score (0.97512, lower-is-better) is far above the target (0.2868), so we should legitimately reduce KL by making the fallback probabilities better calibrated without changing your core model/training logic (still TF-optional and unchanged). I keep your patient+spectrogram+global blended prior, but add one minimal extra signal that is available at inference: an `eeg_id` prior (from train vote totals grouped by `eeg_id`) and blend it in with small weight plus smoothing. I also replace the fixed temperature with a tiny grid search on the *training labels only* to pick the best temperature for the prior-based predictor under the same KL metric (no leakage from test labels), which typically reduces KL substantially compared to a hardcoded temp. Submission writing, alignment, and probability normalization remain unchanged.'
- What this solution (achieved 0.95238) has done: 'Your current score (0.95238, lower-is-better) is far above the target (0.2868), so the best minimal improvement without changing your model/training logic is to make the fallback prior less overconfident and better matched to label noise. I keep your exact patient/spectrogram/eeg/global blending approach, but add a tiny isotonic-like calibration step via simple “blend with global prior” strength `lambda_mix` and tune it (together with temperature) on train labels only under the same KL metric. This is still legitimate (no test leakage) and usually drops KL substantially for prior-only predictors by preventing rare-id priors from becoming too peaky. Submission alignment/normalization stays strict, and TensorFlow/model code remains untouched and only runs when explicitly enabled and compatible.'
- What this solution (achieved 1.00398) has done: 'I keep your core model/training code untouched and focus only on improving the inference-only fallback (since TF is disabled in this runtime and your score is far worse than the target, lower-is-better). The biggest current issue is that the “priors by patient/spectrogram/eeg_id” can become miscalibrated and overconfident; KL strongly penalizes that, so we add a minimal, legitimate calibration: compute out-of-fold (OOF) priors for patient/spectrogram/eeg_id (to avoid training-row leakage when tuning) and tune the blend weights + global-mix + temperature on OOF KL (same metric). This preserves the same semantics (still just a prior-based predictor when no models exist), but makes the tuned parameters much more reliable and typically lowers KL substantially. Submission writing/alignment/normalization remain strict and unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

cwd_parts = os.getcwd().split(os.sep)
if len(cwd_parts) > 1 and cwd_parts[1] == "home":
    PLATFORM = "local"
    if os.path.exists("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif len(cwd_parts) > 1 and cwd_parts[1] == "kaggle":
    PLATFORM = "kaggle"
else:
    PLATFORM = "kaggle"

if PLATFORM == "kaggle":
    NEEDTRAIN = False
    if os.path.exists("/kaggle/input/"):
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img ***
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
    if not os.path.exists(os.path.join(LOAD_DATA_FROM, "train.csv")):
        alt = "/kaggle/input"
        if os.path.exists(os.path.join(alt, "train.csv")):
            LOAD_DATA_FROM = alt

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100
SPE_WIDE = 256

STFT_LENGTH = 45
STFT_TIME = 0.15
STFT_HIGH = 32
STFT_WIDE = round(STFT_LENGTH / STFT_TIME)

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5
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
    "T5-O1",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
]
TEST_BATCHSIZE = 128

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import warnings

warnings.filterwarnings("ignore")
import io
from PIL import Image
import pandas as pd, numpy as np

from sklearn.metrics import confusion_matrix  # noqa: F401

FORCE_TF = (
    False  # keep False for Kaggle stability; set True only in a compatible TF env.
)

TF_AVAILABLE = False
tf = None
strategy = None
try:
    if FORCE_TF:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        import tensorflow as tf  # noqa: F401

        TF_AVAILABLE = True
        print(tf.version.VERSION)
        print(tf.config.list_physical_devices("GPU"))

        from tensorflow.keras import optimizers
        from tensorflow.keras.models import clone_model
        from tensorflow.python.framework.ops import reset_default_graph
        import tensorflow.keras.backend as K  # noqa: F401

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
    else:
        raise RuntimeError(
            "FORCE_TF is False (skipping TF import to avoid protobuf crash)."
        )
except Exception as e:
    print("TensorFlow unavailable or disabled; will run inference-only fallback.")
    print("TF import/init error:", repr(e))
    TF_AVAILABLE = False

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

HAS_MODELS = (not NEEDTRAIN) and os.path.exists(LOAD_MODELS_FROM) and TF_AVAILABLE



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = []
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
            train.to_csv("train.csv", index=False)

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## === cell 2
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

            if "stft" in DATATYPE:
                eeg2 = signal.filtfilt(b, a, eeg, axis=1)
                nperseg = round(RSFREQ * 0.2)
                ff, tt, ss = signal.spectrogram(
                    eeg2,
                    axis=1,
                    fs=RSFREQ,
                    nperseg=nperseg,
                    noverlap=round(nperseg - RSFREQ * STFT_TIME),
                    nfft=320,
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) * (ff <= 20), :]

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
            if "stft" in DATATYPE:
                stfts[eeg_id] = ss
                stfts[-eeg_id] = tt

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "stft" in DATATYPE:
            np.save("./input/preprocess/stfts.npy", stfts, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        else:
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            f = os.path.join(datapath, "eegs.npy")
            if os.path.exists(f):
                eegs = np.load(f, allow_pickle=True).item()
        if "stft" in DATATYPE:
            f = os.path.join(datapath, "stfts.npy")
            if os.path.exists(f):
                stfts = np.load(f, allow_pickle=True).item()
        if "img" in DATATYPE:
            f = os.path.join(datapath, "imgs.npy")
            if os.path.exists(f):
                imgs = np.load(f, allow_pickle=True).item()



## === cell 3
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH) if os.path.exists(PATH) else []
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
            else:
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()



## === cell 4
if TF_AVAILABLE:

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            dataframe,
            batch_size=32,
            shuffle=False,
            sample_weights=False,
            mode="train",
            eegs=None,
            stfts=None,
            specs=None,
            imgs=None,
        ):

            self.dataframe = dataframe
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.sample_weights = sample_weights
            self.mode = mode
            self.eegs = eegs
            self.stfts = stfts
            self.specs = specs
            self.imgs = imgs
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.dataframe) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        EEG_CHANNEL_USED * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                sign_id = row.sign_id
                if self.mode != "test":
                    sample_weight = min(sum(row[TARGETS_RAW].values) / 10, 1.0)

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                else:
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
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
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = row.eeg_label_offset_seconds

                if "spe" in DATATYPE:
                    spe = []
                    for k in range(4):
                        spe.append(
                            np.reshape(
                                self.specs[row.spectrogram_id][
                                    r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                                ].T,
                                (1, 100, 300),
                            )
                        )
                    spe = np.concatenate(spe, axis=0)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    exp_min, exp_max = -4, 6
                    spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    spe = np.log(spe)
                    spe = spe[
                        :,
                        :,
                        round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                            (spe.shape[2] - SPE_WIDE) / 2
                        ),
                    ]
                    spe = (spe - exp_min) / (exp_max - exp_min) * 255
                    spe = np.clip(spe, a_min=0, a_max=255)
                    x_spe[j] = spe

                if "eeg" in DATATYPE:
                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]
                    eeg_save = np.zeros(
                        (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                    )

                    eeg = np.concatenate(
                        (
                            eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                        ),
                        axis=0,
                    )

                    if self.mode == "train":
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
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

                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]
                    else:
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]
                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                    eeg = np.clip(eeg_save, a_min=-255, a_max=255)
                    eeg = eeg + 255
                    eeg = eeg / 2

                    eeg[0:4, :] = 255 - eeg[0:4, :]
                    eeg[8:12, :] = 255 - eeg[8:12, :]
                    if self.mode == "train":
                        if np.random.rand() > 0.5:
                            eeg = 255 - eeg

                    x_eeg[j] = eeg

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
                    if self.mode == "train":
                        img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                        img[10:18, :, :] = img[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            img = img[::-1, :, :]

                    for ii in range(img.shape[0]):
                        axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                        start_temp = round(
                            max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                        )
                        end_temp = round(
                            min(
                                img_save.shape[0],
                                axis_temp + img_save.shape[1] / img.shape[0],
                            )
                        )
                        temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                        img_save[start_temp:end_temp, :] = (
                            img_save[start_temp:end_temp, :]
                            + img[
                                ii,
                                temp_temp : round(temp_temp + end_temp - start_temp),
                                :,
                            ]
                        )
                    img_save = np.clip(img_save, a_min=0, a_max=1)

                    img = np.reshape(
                        img_save, (img_save.shape[0], img_save.shape[1], 1)
                    )
                    img = np.concatenate((img, img, img), -1)
                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                    x_img[j] = img

                if self.mode != "test":
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                    sample_weights[j] = sample_weight if self.sample_weights else 1.0
                else:
                    sample_weights[j] = 1.0

            x = {}
            if "spe" in DATATYPE:
                x["spe"] = x_spe
            if "eeg" in DATATYPE:
                x["eeg"] = x_eeg
            if "stft" in DATATYPE:
                x["stft"] = (
                    x_stft  # noqa: F821 (kept as original; stft not enabled here)
                )
            if "img" in DATATYPE:
                x["img"] = x_img

            return x, y, sample_weights




## === cell 5
if TF_AVAILABLE:

    from tensorflow.keras import optimizers
    from tensorflow.python.framework.ops import reset_default_graph

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step
            self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
            self.lr_max = lr_max
            self.lr_min = lr_min

        def __call__(self, step):
            step = step + 1
            if step < self.warm_step:
                lr = self.lr_max / self.warm_step * step
            else:
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0
                    + tf.cos(
                        (step - self.warm_step)
                        / (self.total_step - self.warm_step)
                        * np.pi
                    )
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

    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
            self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(ff_dim, activation="gelu"),
                    tf.keras.layers.Dense(feat_dim),
                ]
            )
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output, weights = self.att(
                inputs, inputs, return_attention_scores=True
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output), weights

    def build_model():
        inp = []
        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                name="eeg",
            )
            x_eeg_raw = tf.keras.layers.Reshape(
                (inp_eeg.shape[1], inp_eeg.shape[2], 1)
            )(inp_eeg)

            strides = 10
            if PLATFORM == "local":
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
            else:
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

            base_model_eeg = tf.keras.applications.EfficientNetV2B0(
                include_top=False, weights=None, include_preprocessing=True
            )
            if NEEDTRAIN:
                w_local = f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                w_kaggle = (
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
                if PLATFORM == "local" and os.path.exists(w_local):
                    base_model_eeg.load_weights(w_local)
                if PLATFORM == "kaggle" and os.path.exists(w_kaggle):
                    base_model_eeg.load_weights(w_kaggle)
            base_model_eeg._name = "eeg_extractor"

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = x_eeg[
                :, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :
            ]
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

            inp.append(inp_eeg)
            y_eeg = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_eeg)

        y = y_eeg * 1
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 6
if NEEDTRAIN and TF_AVAILABLE:
    if not os.path.exists("models"):
        os.makedirs("models")

    from sklearn.model_selection import GroupKFold
    import itertools  # noqa: F401

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

        train_gen = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, LEARN_RATE * 0.01, 5)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{1}.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))

        del model, history, train_gen, valid_gen
        tf.keras.backend.clear_session()
        reset_default_graph()
        gc.collect()

        train_gen = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    round(EPOCHS / 3), LEARN_RATE * 0.1, LEARN_RATE * 0.1 * 0.1, 0
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{2}.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=round(EPOCHS / 3),
            callbacks=callbacks_list,
        )
        model.load_weights(os.path.join("models", f"fold{i}_stage2.weights.h5"))

        del model, history, train_gen, valid_gen
        tf.keras.backend.clear_session()
        reset_default_graph()
        gc.collect()



## === cell 7
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test = test.reset_index(drop=True)
test["sign_id"] = test.index.values
print("Test shape", test.shape)


def _write_submission_from_probs(
    test_df: pd.DataFrame, probs: np.ndarray, out_path: str = "submission.csv"
):
    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    sub[TARGETS] = probs

    p = sub[TARGETS].values.astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    sub[TARGETS] = p.astype(np.float32)

    assert len(sub) == len(
        test_df
    ), f"submission length {len(sub)} != test length {len(test_df)}"
    sub.to_csv(out_path, index=False)
    print("Submission shape", sub.shape)
    print(sub.head())


def _apply_temperature(p: np.ndarray, temp: float) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-12, 1.0)
    logp = np.log(p)
    logp = logp / float(temp)
    logp = logp - logp.max(axis=1, keepdims=True)
    p2 = np.exp(logp)
    p2 = p2 / p2.sum(axis=1, keepdims=True)
    return p2


def _kl_divergence_rowwise(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true.astype(np.float64), 1e-12, 1.0)
    y_true = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred = np.clip(y_pred.astype(np.float64), 1e-12, 1.0)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(y_true * (np.log(y_true) - np.log(y_pred)), axis=1)))


def _mix_with_global(p: np.ndarray, prior_global: np.ndarray, lam: float) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    g = np.clip(prior_global.astype(np.float64).reshape(1, -1), 1e-12, 1.0)
    g = g / g.sum(axis=1, keepdims=True)
    pm = (1.0 - float(lam)) * p + float(lam) * g
    pm = np.clip(pm, 1e-12, 1.0)
    pm = pm / pm.sum(axis=1, keepdims=True)
    return pm


def _make_smoothed_priors(
    train_df: pd.DataFrame,
    targets: list[str],
    prior_global: np.ndarray,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_totals = train_df.groupby("patient_id")[targets].sum().astype(np.float64)
    spec_totals = train_df.groupby("spectrogram_id")[targets].sum().astype(np.float64)
    eeg_totals = train_df.groupby("eeg_id")[targets].sum().astype(np.float64)

    def _smooth(totals: pd.DataFrame) -> pd.DataFrame:
        s = totals.sum(axis=1).values.reshape(-1, 1)
        sm = (totals.values + float(alpha) * prior_global.reshape(1, -1)) / (
            s + float(alpha)
        )
        return pd.DataFrame(sm, index=totals.index, columns=targets)

    return _smooth(patient_totals), _smooth(spec_totals), _smooth(eeg_totals)


def _predict_from_priors(
    df_like: pd.DataFrame,
    patient_priors: pd.DataFrame,
    spec_priors: pd.DataFrame,
    eeg_priors: pd.DataFrame,
    prior_global: np.ndarray,
    w_patient: float,
    w_spec: float,
    w_eeg: float,
) -> np.ndarray:
    w_global = 1.0 - (float(w_patient) + float(w_spec) + float(w_eeg))
    probs = np.zeros((len(df_like), len(TARGETS)), dtype=np.float64)
    for i, (pid, sid, eid) in enumerate(
        zip(
            df_like["patient_id"].values,
            df_like["spectrogram_id"].values,
            df_like["eeg_id"].values,
        )
    ):
        p_pat = prior_global
        if pid in patient_priors.index:
            v = patient_priors.loc[pid].values.astype(np.float64)
            if np.isfinite(v).all() and v.sum() > 0:
                p_pat = v

        p_spec = prior_global
        if sid in spec_priors.index:
            v = spec_priors.loc[sid].values.astype(np.float64)
            if np.isfinite(v).all() and v.sum() > 0:
                p_spec = v

        p_eeg = prior_global
        if eid in eeg_priors.index:
            v = eeg_priors.loc[eid].values.astype(np.float64)
            if np.isfinite(v).all() and v.sum() > 0:
                p_eeg = v

        p = (
            float(w_patient) * p_pat
            + float(w_spec) * p_spec
            + float(w_eeg) * p_eeg
            + float(w_global) * prior_global
        )
        probs[i] = p

    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _tune_oof_calibration(
    full_train: pd.DataFrame,
    prior_global: np.ndarray,
    alpha_patient: float,
    alpha_spec: float,
    alpha_eeg: float,
    weights_grid: list[tuple[float, float, float]],
    lambdas: list[float],
    temps: list[float],
    n_splits: int = 5,
    seed: int = 2024,
) -> tuple[float, float, float, float, float]:
    from sklearn.model_selection import GroupKFold

    y_true = full_train[list(TARGETS)].values.astype(np.float64)
    y_true = np.clip(y_true, 1e-12, None)
    y_true = y_true / y_true.sum(axis=1, keepdims=True)

    gkf = GroupKFold(n_splits=n_splits)
    groups = full_train["patient_id"].values

    best = None  # (kl, w_patient, w_spec, w_eeg, lam, temp)

    fold_indices = list(
        gkf.split(
            full_train,
            (
                full_train["expert_consensus"].values
                if "expert_consensus" in full_train.columns
                else None
            ),
            groups,
        )
    )

    for w_patient, w_spec, w_eeg in weights_grid:
        if (w_patient + w_spec + w_eeg) > 1.0:
            continue

        p_oof = np.zeros((len(full_train), len(TARGETS)), dtype=np.float64)

        for tr_idx, va_idx in fold_indices:
            tr = full_train.iloc[tr_idx]
            va = full_train.iloc[va_idx]

            pat_pr, spec_pr, eeg_pr = _make_smoothed_priors(
                tr, targets=list(TARGETS), prior_global=prior_global, alpha=1.0
            )
            pat_pr, _, _ = _make_smoothed_priors(
                tr, list(TARGETS), prior_global, alpha=float(alpha_patient)
            )
            _, spec_pr, _ = _make_smoothed_priors(
                tr, list(TARGETS), prior_global, alpha=float(alpha_spec)
            )
            _, _, eeg_pr = _make_smoothed_priors(
                tr, list(TARGETS), prior_global, alpha=float(alpha_eeg)
            )

            p_oof[va_idx] = _predict_from_priors(
                va, pat_pr, spec_pr, eeg_pr, prior_global, w_patient, w_spec, w_eeg
            )

        for lam in lambdas:
            p_l = _mix_with_global(p_oof, prior_global=prior_global, lam=float(lam))
            for t in temps:
                p_lt = _apply_temperature(p_l, temp=float(t))
                kl = _kl_divergence_rowwise(y_true, p_lt)
                if (best is None) or (kl < best[0]):
                    best = (
                        kl,
                        float(w_patient),
                        float(w_spec),
                        float(w_eeg),
                        float(lam),
                        float(t),
                    )

    assert best is not None
    print(
        f"[OOF-tune] best_OOF_KL={best[0]:.6f} "
        f"w_patient={best[1]:.2f} w_spec={best[2]:.2f} w_eeg={best[3]:.2f} "
        f"lambda={best[4]:.2f} temp={best[5]:.2f}"
    )
    return best[1], best[2], best[3], best[4], best[5]


if HAS_MODELS:
    preds_all = []
    models = []
    model_template = build_model()
    from tensorflow.keras.models import clone_model  # safe here because TF_AVAILABLE

    for model_i in range(SPLITS):
        print(f"Fold {model_i + 1}")
        model = clone_model(model_template)
        w1 = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
        w2 = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
        if os.path.exists(w1):
            model.load_weights(w1)
        else:
            model.load_weights(w2)
        models.append(model)

    PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

    if (
        ("spe" in DATATYPE)
        or ("eeg" in DATATYPE)
        or ("stft" in DATATYPE)
        or ("img" in DATATYPE)
    ):
        b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")

        batch_start = 0

        for i, eeg_id in enumerate(test.eeg_id):
            if i % 100 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(
                os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
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

            if "stft" in DATATYPE:
                eeg2 = signal.filtfilt(b, a, eeg, axis=1)
                nperseg = round(RSFREQ * 0.2)
                ff, tt, ss = signal.spectrogram(
                    eeg2,
                    axis=1,
                    fs=RSFREQ,
                    nperseg=nperseg,
                    noverlap=round(nperseg - RSFREQ * STFT_TIME),
                    nfft=320,
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) * (ff <= 20), :]

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                test_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(test_plot)):
                    eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
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
                            img = np.array(
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]
                        img_save[ii, :, :] = img

                    imgs_test[test_plot.sign_id[j]] = img_save

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            if filter_range is not None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.array(eeg, dtype=np.float32)

            if "eeg" in DATATYPE:
                eegs_test[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts_test[eeg_id] = ss
                stfts_test[-eeg_id] = tt

            if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                batch_end = i + 1
                batch_df = test.iloc[batch_start:batch_end].reset_index(drop=True)

                preds = []
                test_gen = DataGenerator(
                    batch_df,
                    shuffle=False,
                    sample_weights=False,
                    batch_size=TEST_BATCHSIZE,
                    mode="test",
                    specs=spectrograms_test,
                    eegs=eegs_test,
                    stfts=stfts_test,
                    imgs=imgs_test,
                )
                for model_i in range(SPLITS):
                    pred = models[model_i].predict(test_gen, verbose=0)
                    preds.append(pred)
                pred = np.mean(preds, axis=0)

                eegs_test = {}
                stfts_test = {}
                imgs_test = {}
                gc.collect()

                if len(preds_all) == 0:
                    preds_all = pred.copy()
                else:
                    preds_all = np.concatenate((preds_all, pred), axis=0)

                batch_start = batch_end

    preds_all = np.asarray(preds_all)
    if preds_all.shape[0] != len(test):
        if preds_all.shape[0] > len(test):
            preds_all = preds_all[: len(test)]
        else:
            pad = np.tile(
                preds_all.mean(axis=0, keepdims=True),
                (len(test) - preds_all.shape[0], 1),
            )
            preds_all = np.concatenate([preds_all, pad], axis=0)

    _write_submission_from_probs(test, preds_all, out_path="submission.csv")

else:
    vote_totals_global = df[TARGETS].sum(axis=0).values.astype(np.float64)
    prior_global = vote_totals_global / vote_totals_global.sum()
    prior_global = np.clip(prior_global, 1e-12, 1.0)
    prior_global = prior_global / prior_global.sum()

    alpha_patient = 40.0
    alpha_spec = 25.0
    alpha_eeg = 25.0

    weights_grid = [
        (0.60, 0.15, 0.15),
        (0.55, 0.20, 0.15),
        (0.50, 0.20, 0.20),
        (0.50, 0.25, 0.15),
        (0.45, 0.25, 0.20),
        (0.45, 0.30, 0.15),
        (0.40, 0.30, 0.20),
    ]
    best_w_patient, best_w_spec, best_w_eeg, best_lam, best_temp = (
        _tune_oof_calibration(
            full_train=df,
            prior_global=prior_global,
            alpha_patient=alpha_patient,
            alpha_spec=alpha_spec,
            alpha_eeg=alpha_eeg,
            weights_grid=weights_grid,
            lambdas=[0.00, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40],
            temps=[0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40],
            n_splits=5,
            seed=SEED,
        )
    )

    patient_priors, _, _ = _make_smoothed_priors(
        df, targets=list(TARGETS), prior_global=prior_global, alpha=alpha_patient
    )
    _, spec_priors, _ = _make_smoothed_priors(
        df, targets=list(TARGETS), prior_global=prior_global, alpha=alpha_spec
    )
    _, _, eeg_priors = _make_smoothed_priors(
        df, targets=list(TARGETS), prior_global=prior_global, alpha=alpha_eeg
    )

    probs = _predict_from_priors(
        test,
        patient_priors=patient_priors,
        spec_priors=spec_priors,
        eeg_priors=eeg_priors,
        prior_global=prior_global,
        w_patient=best_w_patient,
        w_spec=best_w_spec,
        w_eeg=best_w_eeg,
    )

    probs = _mix_with_global(probs, prior_global=prior_global, lam=best_lam)
    probs = _apply_temperature(probs, temp=best_temp)

    _write_submission_from_probs(
        test, probs.astype(np.float32), out_path="submission.csv"
    )
