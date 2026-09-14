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

0.2831777735807947

# 6. Current score

1.12537

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the crash caused by incompatible protobuf / TensorFlow imports by removing unused heavyweight imports (`torchaudio`, `torch`, `keras_hub`) that trigger the `MessageFactory.GetPrototype` error in this Kaggle runtime. I also make test-time inference run end-to-end on Kaggle even when no pretrained “models*” dataset is available by falling back to a safe uniform-probability submission (this yields a valid CSV and a finite KL score instead of failing). Finally, I fix a slicing/indexing bug in the test batching loop (`len(preds_all)` used as a row index) and ensure predictions are clipped and renormalized to sum to 1 per row to prevent submission-format failures.'
- What this solution (achieved 1.01346) has done: 'I fix the TensorFlow/protobuf crash by defensively forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle runtime. Then, instead of writing a uniform fallback (which explains the very poor 1.40995 KL), I add a minimal, legitimate fallback that uses the training-set mean target distribution per `patient_id` (and global mean if unseen), yielding a much better calibrated probability vector while still producing a valid submission when no weights are present. I also ensure the submission rows align to `test.eeg_id` order, probabilities are strictly positive, and each row sums to 1 to prevent submission-format/KL failures. Core model/data logic remains unchanged; this only affects robustness and the no-weights inference path.'
- What this solution (achieved 1.01346) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf runtime is used *before* TensorFlow is imported, and by safely falling back to a no-TF path if TF still fails to import in this Kaggle image. Then I keep your existing inference logic intact, but make sure a valid `submission.csv` is always written: if no model weights are found (or TF is unavailable), we use the already-implemented patient-mean fallback (which should improve KL vs uniform and move the score toward the target). Finally, I add a small robustness fix in the DataGenerator row-filtering (include `other_vote` in the match) to avoid accidental empty row selections if training is ever enabled again; this is score-neutral for current Kaggle inference.'
- What this solution (achieved 1.01346) has done: 'I fix the hard crash happening before any fallback logic can run by removing the eager TensorFlow import at module import time and instead importing TensorFlow lazily only when it’s actually needed (i.e., when weights are present). This avoids the protobuf `MessageFactory.GetPrototype` issue in this Kaggle image and ensures the script always reaches submission writing. I keep your existing model/inference code intact, but add a lightweight “weights present?” check so we can skip TensorFlow entirely and use the already-implemented patient-mean fallback when no weights exist (score-improving vs uniform and should move KL toward the target). Finally, I keep the probability clipping/renormalization to guarantee valid KL submissions (strictly positive and sum to 1).'
- What this solution (achieved 0.96732) has done: 'Your current score (1.01346, lower-is-better) is far from the target (0.28318), and the main reason is that on Kaggle you are not using the actual model at all: `NEEDTRAIN=False` and no weights are found, so the script always falls back to a patient-mean prior. The smallest legitimate improvement toward the target is to actually run inference with real pretrained weights by pointing `LOAD_MODELS_FROM` to an existing Kaggle input dataset name that contains `*_stage2.weights.h5` files (commonly provided via “Add data”); since we can’t assume such a dataset exists, the next best minimal improvement is to strengthen the fallback without changing your model/training logic by using a spectrogram-level prior instead of only patient_id (more informative than patient_id alone) and then combining patient and spectrogram priors when available. This keeps evaluation semantics (probability vectors) identical, but should reduce KL substantially versus the current fallback while staying safe and fast. I also keep your strict positivity+row-sum-to-1 enforcement and ensure test row order is preserved.'
- What this solution (achieved 0.97262) has done: 'Your current score (0.96732, lower-is-better) is still far from the target (0.28318), and the dominant reason is that you’re almost certainly still not running the actual neural net on Kaggle (no weights found → prior fallback). The smallest legitimate move toward the target is to ensure the script actually discovers and loads any provided `*_stage2.weights.h5` files on Kaggle by searching all `/kaggle/input/**` subfolders (many datasets nest weights one level deeper), rather than only looking at a single guessed folder name. If weights are still unavailable, we keep your strengthened prior fallback but improve it slightly in a metric-aligned way by adding a tiny Dirichlet/Laplace smoothing using global vote totals (keeps probabilities away from 0 and usually reduces KL). All changes preserve the existing model/inference core logic and keep the submission strictly positive and row-normalized.'
- What this solution (achieved 1.11046) has done: 'Your current KL (0.97262, lower-is-better) is far above the target (0.28318), and the biggest likely issue is that you are still not actually using the trained network on Kaggle (no weights discovered/loaded), so you fall back to a weak prior. I make the smallest change that most increases the chance of finding and loading existing `*_stage2.weights.h5` files: broaden the weight search to include common weight filename patterns and make `_weights_exist`/loader use the same patterns (without changing the model itself). If no weights exist, I keep your current fallback but make it slightly more metric-aligned by generating the prior at the **eeg_id** level (train has multiple rows per eeg_id) and blending patient/spec/eeg/global means with the same tiny smoothing; this is still leakage-free and typically lowers KL versus only patient/spec means. Everything else (model architecture, generator, preprocessing, prediction, and CSV format with strictly positive row-normalized probabilities) stays intact.'
- What this solution (achieved 1.13866) has done: 'Your current KL (1.11046, lower-is-better) is still far from the target (0.28318), and the most likely cause is that the script often fails to actually load usable weights (so it falls back to a weak prior). I make a minimal, score-relevant change to (1) discover and load weight files more reliably by searching recursively and accepting common filename patterns, and (2) ensure the fold-loading loop also picks up those discovered patterns (not just `fold{n}_stage2.weights.h5`). I also make the fallback prior slightly more informative but still leakage-free by using **vote-count-weighted** means (instead of simple means) at eeg/patient/spectrogram levels, which is directly aligned with the KL metric and typically reduces KL vs unweighted averaging. Core model/inference logic and submission semantics stay the same: we only improve robustness of weight loading and calibration of the fallback.'
- What this solution (achieved 1.04696) has done: 'Your current KL (1.13866, lower-is-better) is still far above the target (0.28318), and the biggest likely issue is that you’re still not loading usable NN weights on Kaggle, so you fall back to a weak prior. I make the smallest changes that (a) increase the chance that weights are discovered/loaded and (b) improve calibration of the fallback prior in a KL-friendly way without changing the model architecture or training/inference loop. Concretely: fix a bug in the weighted-group-mean computation (it was incorrectly referencing the outer frame inside groupby apply), add a vote-count-weighted “expert_consensus” conditional prior (train-only, leakage-free) blended into the fallback, and (when weights are used) average predictions across multiple 10s windows of each test EEG (test-time augmentation by time shift) which often materially improves KL while preserving the same model. All outputs remain strictly positive and row-normalized so the submission is valid.'
- What this solution (achieved 1.04696) has done: 'Your current KL (1.04696, lower-is-better) is still far from the target (0.28318), and the main likely cause is that you’re either (a) not loading weights at all (falling back to priors) or (b) producing miscalibrated probabilities when weights are loaded. I make minimal, score-relevant changes to (1) ensure the weight-file discovery and the loader use exactly the same discovered file list (so if weights exist, they actually get loaded), and (2) add a tiny, KL-friendly probability calibration at inference time by blending the model prediction with a strong, leakage-free prior (only when NN weights are used), while keeping your model/loop/architecture unchanged. I also fix a small but important input-preprocessing inconsistency: you define a second filter (filter_range2) but never apply it, so I apply it to the test EEG in the same place you already filter, which should move predictions closer to the trained distribution without changing the model. All outputs remain strictly positive and row-normalized, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.04696) has done: 'Your current score is far above the target (lower-is-better), and the most likely reason is that the neural net inference path is either not being used (no weights loaded) or it produces miscalibrated probabilities. I make two minimal, score-relevant fixes without changing the model/training core logic: (1) ensure the “NN inference” branch always produces predictions for all test rows (even if DATATYPE changes) by initializing `preds_all` from the strong prior and only overwriting it when EEG inference runs, and (2) slightly increase the prior-blend weight during NN inference (a pure calibration change) to reduce KL risk and usually improve scores when the model is shaky. I also remove an unintended test-time mutation of the global `imgs_test/stfts_test` dicts inside the EEG loop (it’s a harmless bug but can destabilize inference), keeping everything else identical. The script still always write a valid `submission.csv` with strictly positive row-normalized probabilities.'
- What this solution (achieved 1.125) has done: 'The score gap to the target is still large (current KL 1.04696 vs target 0.28318, lower-is-better), and your code is very likely still not using any neural-net weights on Kaggle (so it falls back to priors). The smallest legitimate change that should move KL down is to make the fallback prior more informative by using **per-(patient_id, spectrogram_id)** and **per-(eeg_id, spectrogram_id)** vote-weighted priors when available (these combinations are more specific than any single key). In the NN branch, I keep your architecture and loops identical but make the prior-blend weight slightly stronger (more conservative, often reduces KL when the model is imperfect) and ensure weight discovery prefers directories that actually contain multiple usable `.weights.h5` files. All changes preserve submission semantics (strictly positive probabilities summing to 1) and keep runtime within limits.'
- What this solution (achieved 1.12537) has done: 'Your current KL (1.125, lower-is-better) is far above the target (0.283), and given the logs/plans this is almost certainly because the neural-net inference path still isn’t being used reliably (no loadable weights → prior fallback). The smallest score-relevant change is to make the fallback prior materially stronger without changing your model/training logic: compute priors on the same “collapsed/unique label” `train` table you already build (less noisy than raw `df`), add a single new train-only key (`patient_id` × `expert_consensus`) and rebalance the blend weights to emphasize the most specific available keys. Additionally, fix a Kaggle runtime issue where `NEEDTRAIN=False` prevents `train` from being created at all, so the fallback is forced to use the noisier `df`; we create the collapsed `train` table even in inference mode (no training happens). These changes keep evaluation semantics identical (probability vectors summing to 1) and only affect the legitimate prior fallback path, which is the active path when weights aren’t found.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
import gc
import numpy as np
import pandas as pd
from PIL import Image

from scipy import signal
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")

TF_AVAILABLE = False
tf = None
optimizers = None
clone_model = None


def _try_import_tensorflow():
    """Lazy TF import so that we can still produce a submission if TF import crashes."""
    global TF_AVAILABLE, tf, optimizers, clone_model
    if TF_AVAILABLE and (tf is not None):
        return True
    try:
        import tensorflow as _tf
        from tensorflow.keras import optimizers as _optimizers
        from tensorflow.keras.models import clone_model as _clone_model

        tf = _tf
        optimizers = _optimizers
        clone_model = _clone_model
        TF_AVAILABLE = True
        return True
    except Exception as e:
        TF_AVAILABLE = False
        tf = None
        optimizers = None
        clone_model = None
        print("TensorFlow import failed; will use fallback submission. Error:")
        print(repr(e))
        return False




## === cell 1
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
print("DATATYPE =", DATATYPE)

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

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

MIX = False  # will be configured after TF import (if used)



## === cell 2
if NEEDTRAIN:
    import itertools

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

TARGETS_RAW = [t + "_raw" for t in TARGETS]

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



## === cell 3
if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

if filter_range2 is not None:
    b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")
else:
    b2, a2 = None, None

if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        pass
    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if os.path.exists(os.path.join(datapath, "eegs.npy")):
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE and os.path.exists(os.path.join(datapath, "stfts.npy")):
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE and os.path.exists(os.path.join(datapath, "imgs.npy")):
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()



## === cell 4
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    if READ_SPE_FILES:
        pass
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




## === cell 5
def _safe_softmax_rows(p, eps=1e-8):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def _fallback_prior_submission(train_df: pd.DataFrame, test_df: pd.DataFrame, targets):
    """
    Score-motivated (fallback path only): vote-count-weighted mean target distributions
    at multiple granularities and blend them to reduce KL when no NN weights are usable.

    Minimal, targeted improvement toward the target:
    - compute priors on the collapsed `train_df` (unique label windows), not raw overlapping `df`
    - add one extra specific, train-only key: patient_id × expert_consensus
    - rebalance blend weights to emphasize the most specific keys when available
    """
    t = list(targets)

    train_votes = train_df[t].values.astype(np.float64)
    row_sum = np.maximum(train_votes.sum(axis=1, keepdims=True), 1.0)
    train_prob = train_votes / row_sum

    tmp = train_df[
        ["eeg_id", "patient_id", "spectrogram_id", "expert_consensus"]
    ].copy()
    for k, col in enumerate(t):
        tmp[col] = train_prob[:, k]
    tmp["_w"] = row_sum[:, 0].astype(np.float64)

    tmp["_pid_sid"] = (
        tmp["patient_id"].astype("int64").astype(str)
        + "_"
        + tmp["spectrogram_id"].astype("int64").astype(str)
    )
    tmp["_eid_sid"] = (
        tmp["eeg_id"].astype("int64").astype(str)
        + "_"
        + tmp["spectrogram_id"].astype("int64").astype(str)
    )
    tmp["_pid_cons"] = (
        tmp["patient_id"].astype("int64").astype(str)
        + "_"
        + tmp["expert_consensus"].astype(str)
    )

    def _weighted_group_mean(frame: pd.DataFrame, key: str):
        g = frame.groupby(key, sort=False)
        num = g.apply(lambda x: (x[t].values * x["_w"].values[:, None]).sum(axis=0))
        num = pd.DataFrame(np.vstack(num.values), index=num.index, columns=t)
        den = g["_w"].sum().replace(0.0, 1.0)
        out = num.div(den, axis=0)
        return out

    eeg_mean = _weighted_group_mean(tmp, "eeg_id")
    patient_mean = _weighted_group_mean(tmp, "patient_id")
    spec_mean = _weighted_group_mean(tmp, "spectrogram_id")
    cons_mean = _weighted_group_mean(tmp, "expert_consensus")

    pid_sid_mean = _weighted_group_mean(tmp, "_pid_sid")
    eid_sid_mean = _weighted_group_mean(tmp, "_eid_sid")
    pid_cons_mean = _weighted_group_mean(tmp, "_pid_cons")

    global_prob = (train_votes.sum(axis=0) / np.maximum(train_votes.sum(), 1.0)).astype(
        np.float64
    )
    alpha = 0.02  # tiny pull toward global prior (stability for rare keys)

    preds = np.zeros((len(test_df), len(t)), dtype=np.float64)

    w_eid_sid = 0.26
    w_pid_sid = 0.20
    w_pid_cons = 0.16
    w_eeg = 0.16
    w_spec = 0.12
    w_pat = 0.06
    w_con = 0.02
    w_glb = 0.02

    eid_values = test_df["eeg_id"].values
    pid_values = test_df["patient_id"].values
    sid_values = test_df["spectrogram_id"].values

    patient_cons = train_df.groupby("patient_id", sort=False)["expert_consensus"].agg(
        lambda x: x.value_counts().index[0]
    )

    for i, (eid, pid, sid) in enumerate(zip(eid_values, pid_values, sid_values)):
        global_base = global_prob

        pe = global_base
        if eid in eeg_mean.index:
            pe = eeg_mean.loc[eid].values.astype(np.float64)

        ps = global_base
        if sid in spec_mean.index:
            ps = spec_mean.loc[sid].values.astype(np.float64)

        pp = global_base
        if pid in patient_mean.index:
            pp = patient_mean.loc[pid].values.astype(np.float64)

        pc = global_base
        c_used = None
        if pid in patient_cons.index:
            c_used = patient_cons.loc[pid]
            if c_used in cons_mean.index:
                pc = cons_mean.loc[c_used].values.astype(np.float64)

        key_pid_sid = str(int(pid)) + "_" + str(int(sid))
        pps = global_base
        if key_pid_sid in pid_sid_mean.index:
            pps = pid_sid_mean.loc[key_pid_sid].values.astype(np.float64)

        key_eid_sid = str(int(eid)) + "_" + str(int(sid))
        pes = global_base
        if key_eid_sid in eid_sid_mean.index:
            pes = eid_sid_mean.loc[key_eid_sid].values.astype(np.float64)

        ppc = global_base
        if c_used is not None:
            key_pid_cons = str(int(pid)) + "_" + str(c_used)
            if key_pid_cons in pid_cons_mean.index:
                ppc = pid_cons_mean.loc[key_pid_cons].values.astype(np.float64)

        pe = (1 - alpha) * pe + alpha * global_base
        ps = (1 - alpha) * ps + alpha * global_base
        pp = (1 - alpha) * pp + alpha * global_base
        pc = (1 - alpha) * pc + alpha * global_base
        pps = (1 - alpha) * pps + alpha * global_base
        pes = (1 - alpha) * pes + alpha * global_base
        ppc = (1 - alpha) * ppc + alpha * global_base

        preds[i] = (
            w_eid_sid * pes
            + w_pid_sid * pps
            + w_pid_cons * ppc
            + w_eeg * pe
            + w_spec * ps
            + w_pat * pp
            + w_con * pc
            + w_glb * global_base
        )

    preds = _safe_softmax_rows(preds, eps=1e-6)
    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    sub[t] = preds
    return sub


_WEIGHT_SUFFIXES = (
    "_stage2.weights.h5",
    "_stage1.weights.h5",
    ".weights.h5",
    ".h5",
)


def _is_weight_file(fname: str) -> bool:
    f = fname.lower()
    if not any(f.endswith(suf) for suf in _WEIGHT_SUFFIXES):
        return False
    if "optimizer" in f:
        return False
    return True


def _weights_exist(models_dir: str) -> bool:
    if (models_dir is None) or (not os.path.isdir(models_dir)):
        return False
    for root, _, files in os.walk(models_dir):
        for fname in files:
            if _is_weight_file(fname):
                return True
    return False


def _discover_models_dir(base="/kaggle/input"):
    """
    Score-motivated: search all Kaggle input subfolders (including nested) for weight files
    and pick the directory containing the most *usable* weight files.
    """
    if not os.path.isdir(base):
        return None

    best_dir = None
    best_count = 0

    for root, _, files in os.walk(base):
        cnt = sum(1 for f in files if _is_weight_file(f))
        if cnt > best_count:
            best_count = cnt
            best_dir = root

    if best_count > 0:
        return best_dir
    return None


def _list_weight_files(models_dir: str):
    """Score-motivated: consistent recursive listing so the loader sees the same files as discovery."""
    out = []
    if (models_dir is None) or (not os.path.isdir(models_dir)):
        return out
    for root, _, files in os.walk(models_dir):
        for f in files:
            if _is_weight_file(f):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def _blend_with_prior(
    nn_pred: np.ndarray, prior_pred: np.ndarray, w_prior: float = 0.12
):
    nn_pred = _safe_softmax_rows(nn_pred, eps=1e-8).astype(np.float64)
    prior_pred = _safe_softmax_rows(prior_pred, eps=1e-8).astype(np.float64)
    w_prior = float(np.clip(w_prior, 0.0, 1.0))
    out = (1.0 - w_prior) * nn_pred + w_prior * prior_pred
    return _safe_softmax_rows(out, eps=1e-8)




## === cell 6
def _define_tf_components():
    global DataGenerator, CosineAnnealingLRScheduler, IniToOne, SumToOne, build_model, MIX

    print(tf.version.VERSION)
    print(tf.config.list_physical_devices("GPU"))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("enable_op_determinism not available:", e)

    MIX = True
    if MIX:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

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
            return int(np.ceil(len(self.dataframe) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

        def on_epoch_end(self):
            self.nan = 0
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

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                else:
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        & (df.seizure_vote == row.seizure_vote_raw)
                        & (df.lpd_vote == row.lpd_vote_raw)
                        & (df.gpd_vote == row.gpd_vote_raw)
                        & (df.lrda_vote == row.lrda_vote_raw)
                        & (df.grda_vote == row.grda_vote_raw)
                        & (df.other_vote == row.other_vote_raw),
                        :,
                    ].reset_index(drop=True)

                    if len(rows) == 0:
                        rows = df.loc[(df.eeg_id == row.eeg_id), :].reset_index(
                            drop=True
                        )

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
                    if self.mode == "train":
                        r_eeg = r_eeg + np.random.random() * 10 - 5
                        r_eeg = max(0, r_eeg)
                        r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

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
                    eeg = np.concatenate(
                        (
                            eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                        ),
                        axis=0,
                    )

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    exp_min, exp_max = -4, 6
                    spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    spe = np.log(spe)

                    if (spe.shape[1] != SPE_HIGH) or (spe.shape[2] != SPE_WIDE):
                        spe2 = np.zeros(
                            (spe.shape[0], SPE_HIGH, SPE_WIDE), dtype=np.float32
                        )
                        for k in range(4):
                            scaled_arr = zoom(
                                spe[k],
                                (SPE_HIGH / spe.shape[1], SPE_WIDE / spe.shape[2]),
                                order=1,
                            )
                            spe2[k, :, :] = scaled_arr
                        spe = spe2.copy()

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

                    if self.mode != "test":
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                    eeg = eeg + 1024
                    eeg = eeg / 2048 * 255
                    x_eeg[j] = eeg

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
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
                    if self.sample_weights:
                        sample_weights[j] = (
                            sum(row[[t + "_raw" for t in TARGETS]].values) / 20
                        )
                    else:
                        sample_weights[j] = 1.0
                else:
                    sample_weights[j] = 1.0

            x = {}
            if "spe" in DATATYPE:
                x["spe"] = x_spe
            if "eeg" in DATATYPE:
                x["eeg"] = x_eeg
            if "img" in DATATYPE:
                x["img"] = x_img

            return x, y, sample_weights

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

    class IniToOne(tf.keras.initializers.Initializer):
        def __call__(self, shape, dtype=None):
            assert len(shape) == 3
            filter_length, input_channel, filter_count = shape
            kernel = np.zeros(shape, dtype=np.float32)
            for i in range(filter_count):
                kernel[i % filter_length, 0, i] = 1.0
            return tf.convert_to_tensor(kernel, dtype=dtype)

        def get_config(self):
            return {}

    class SumToOne(tf.keras.constraints.Constraint):
        def __call__(self, w):
            w = tf.abs(w)
            return w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)

        def get_config(self):
            return {}

    def build_model():
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

            base_model_eeg = tf.keras.applications.EfficientNetV2B3(
                include_top=False, weights=None, include_preprocessing=True
            )

            if NEEDTRAIN:
                w_local = f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                w_kaggle = (
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
                w_path = w_local if PLATFORM == "local" else w_kaggle
                if os.path.exists(w_path):
                    base_model_eeg.load_weights(w_path)

            base_model_eeg.name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)

            x_eeg = x_eeg[
                :, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :
            ]
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

            inp.append(inp_eeg)
            y = x_eeg * 1

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 7
if __name__ == "__main__":
    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")
        print("Training is disabled on Kaggle by default in this script.")
    else:
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        has_weights = _weights_exist(LOAD_MODELS_FROM)
        if (PLATFORM == "kaggle") and (not has_weights):
            discovered = _discover_models_dir("/kaggle/input")
            if discovered is not None:
                print(f"Discovered weights directory: {discovered}")
                LOAD_MODELS_FROM = discovered
                has_weights = _weights_exist(LOAD_MODELS_FROM)

        if not has_weights:
            print(f"No model weights found under: {LOAD_MODELS_FROM}")
            print(
                "Writing a stronger (vote-weighted multi-key prior) fallback submission.csv to improve KL."
            )
            sub = _fallback_prior_submission(train, test, TARGETS)
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
        else:
            if not _try_import_tensorflow():
                print(
                    "TensorFlow unavailable -> writing stronger prior fallback submission.csv"
                )
                sub = _fallback_prior_submission(train, test, TARGETS)
                sub.to_csv("submission.csv", index=False)
                print("Submission shape", sub.shape)
                print(sub.head())
            else:
                _define_tf_components()

                models = []
                model_template = build_model()

                weight_files = []
                for model_i in range(100):
                    w_path = os.path.join(
                        LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5"
                    )
                    if os.path.exists(w_path):
                        weight_files.append(w_path)

                if len(weight_files) == 0:
                    weight_files = _list_weight_files(LOAD_MODELS_FROM)

                print(
                    f"Found {len(weight_files)} candidate weight files under: {LOAD_MODELS_FROM}"
                )

                for w_path in weight_files:
                    try:
                        print(f"Loading weights: {w_path}")
                        model = clone_model(model_template)
                        model.load_weights(w_path)
                        models.append(model)
                    except Exception as e:
                        print(f"Failed to load {w_path}: {repr(e)}")

                if len(models) == 0:
                    print(f"No loadable model weights found under: {LOAD_MODELS_FROM}")
                    print("Writing a stronger prior fallback submission.csv (valid).")
                    sub = _fallback_prior_submission(train, test, TARGETS)
                    sub.to_csv("submission.csv", index=False)
                    print("Submission shape", sub.shape)
                    print(sub.head())
                else:
                    PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

                    prior_sub = _fallback_prior_submission(train, test, TARGETS)
                    prior_preds = prior_sub[TARGETS].values.astype(np.float32)
                    preds_all = prior_preds.copy()

                    if (
                        ("spe" in DATATYPE)
                        or ("eeg" in DATATYPE)
                        or ("stft" in DATATYPE)
                        or ("img" in DATATYPE)
                    ):
                        OFFSETS_SECONDS = (-4.0, -2.0, 0.0, 2.0, 4.0)

                        eegs_full = {}
                        for i, eeg_id in enumerate(test.eeg_id.values):
                            if i % 200 == 0:
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

                            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                            eegshape = eeg.shape[1]
                            eeg = np.concatenate(
                                (eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1
                            )
                            if filter_range is not None:
                                eeg = signal.filtfilt(b, a, eeg, axis=1)
                            eeg = eeg[:, eegshape : eegshape * 2]

                            if (b2 is not None) and (a2 is not None):
                                eeg = signal.filtfilt(b2, a2, eeg, axis=1)

                            eegs_full[eeg_id] = np.array(eeg, dtype=np.float32)

                        pred_offset_list = []
                        for off in OFFSETS_SECONDS:
                            eegs_test_local = {}
                            batch_start = 0
                            preds_all_offset = []

                            for i, eeg_id in enumerate(test.eeg_id.values):
                                eeg = eegs_full[eeg_id]
                                shift = int(round(off * RSFREQ))
                                if shift != 0:
                                    eeg_shifted = np.roll(eeg, shift=shift, axis=1)
                                else:
                                    eeg_shifted = eeg
                                eegs_test_local[eeg_id] = eeg_shifted

                                if ((i + 1) % TEST_BATCHSIZE == 0) or (
                                    (i + 1) == len(test.eeg_id)
                                ):
                                    batch_end = i + 1
                                    df_batch = test.iloc[
                                        batch_start:batch_end, :
                                    ].reset_index(drop=True)

                                    test_gen = DataGenerator(
                                        df_batch,
                                        shuffle=False,
                                        sample_weights=False,
                                        batch_size=TEST_BATCHSIZE,
                                        mode="test",
                                        specs=spectrograms_test,
                                        eegs=eegs_test_local,
                                        stfts=stfts_test,
                                        imgs=imgs_test,
                                    )

                                    preds = []
                                    for m in models:
                                        pred = m.predict(test_gen, verbose=0)
                                        preds.append(pred)
                                    pred = np.mean(preds, axis=0)

                                    pred = _safe_softmax_rows(pred)
                                    preds_all_offset.append(pred)

                                    eegs_test_local = {}
                                    gc.collect()
                                    batch_start = batch_end

                            preds_all_offset = np.concatenate(preds_all_offset, axis=0)
                            preds_all_offset = _safe_softmax_rows(preds_all_offset)
                            pred_offset_list.append(preds_all_offset.astype(np.float64))

                        preds_all = np.mean(np.stack(pred_offset_list, axis=0), axis=0)
                        preds_all = _safe_softmax_rows(preds_all)

                        preds_all = _blend_with_prior(
                            preds_all, prior_preds, w_prior=0.28
                        )

                    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
                    sub[TARGETS] = _safe_softmax_rows(preds_all, eps=1e-8)
                    sub.to_csv("submission.csv", index=False)
                    print("\nSubmission shape", sub.shape)
                    print(sub.head())
