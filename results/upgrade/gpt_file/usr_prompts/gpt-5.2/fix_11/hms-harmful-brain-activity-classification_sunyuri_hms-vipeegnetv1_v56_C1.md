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

3.12

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

0.633040988619371

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the early TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error seen in this environment. Then I make the model-weights loading robust by auto-detecting the actual weights directory and files under `/kaggle/input`, instead of assuming `/kaggle/input/models202402121` exists. Finally, I ensure a valid `submission.csv` is always written: if no weights are found, the script fall back to a safe uniform-probability submission (correct columns, rows, and row-sum=1) so you can submit successfully.'
- What this solution (achieved 1.40995) has done: 'You’re currently crashing before any modeling due to the known TensorFlow+protobuf incompatibility in this Kaggle/Python 3.12 environment; the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` workaround alone isn’t sufficient. I force-install a compatible protobuf version at runtime (via pip from Kaggle’s prebuilt wheels) *before importing TensorFlow*, which resolves the `MessageFactory.GetPrototype` error and lets the rest of your pipeline run unchanged. I also keep your existing robust weight-file auto-discovery and submission writing, only adding small guards to ensure TensorFlow import happens after the protobuf fix and that the submission columns exactly match `TARGETS` and sum to 1. These changes are correctness/stability focused; with weights loading working, the score should move down (better) toward your target versus the uniform-fallback baseline.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.6330), and the biggest likely cause is that you are not actually using the intended pretrained fold weights (or you’re averaging mis-calibrated outputs), so the predictions behave close to a weak baseline. I make two minimal, score-relevant fixes without changing your model or generator logic: (1) ensure we robustly discover and load *all* fold weight files by expanding the filename patterns (some datasets save as `.weights.h5`, `fold0.h5`, etc.), and (2) apply a tiny, competition-safe probability “smoothing” (epsilon floor + renorm) consistently, which reduces KL blow-ups from near-zero probabilities. Everything else (architecture, inputs, generator, loss) stays identical and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.6330), and the most likely cause is that inference is running with mixed precision and/or without the intended pretrained fold weights being reliably loaded, both of which can badly distort probability calibration for KL. I make two minimal, score-relevant changes while preserving your exact architecture and inference flow: (1) disable mixed precision for inference only (keep everything else identical) to avoid float16 numeric issues, and (2) strengthen weight-file discovery to include common Kaggle weight extensions (`.keras`, `.ckpt`, nested subdirs) and load each fold into a fresh model instance to avoid any cross-fold state contamination. I also keep your existing epsilon-floor+renorm (important for KL stability) and still always write a valid `submission.csv` with correct columns and row-sum=1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.6330), and the most likely minimal, score-relevant issue is that the test spectrogram slicing is incorrect in `mode="test"`: it always takes `0:300` time rows instead of the central 300 rows, which makes the spectrogram input distribution mismatch what the model was trained on (where offsets are used). I fix this by computing a per-test-row start index from the spectrogram length (center-crop), keeping the exact same preprocessing, shapes, and model. I also add a tiny safety guard to keep the crop within bounds (and pad if a rare spectrogram is shorter), which prevents silent shape/empty-slice issues that can degrade predictions. Everything else (architecture, inference flow, weight loading, epsilon+renorm, submission writing) stays the same.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-relevant fixes that preserve your model, preprocessing, and inference flow: (1) use a deterministic, vote-weighted ensembling of fold predictions (weights inversely proportional to each fold’s own train-KL computed on the already-built `train` targets, without any retraining) so weak/misloaded folds don’t dominate the mean, and (2) add a tiny temperature calibration (single scalar) selected by minimizing KL on the same train targets using out-of-fold-style approximation (predict train in batches with the same generator logic but `mode="test"`-like deterministic sampling). Both changes directly target the KL metric and should move your score down from 1.40995 toward 0.633 while keeping architecture/loss unchanged and still producing a valid `submission.csv` whose rows sum to 1. The rest of the pipeline (weights discovery, center-crop test spectrograms, epsilon floor+renorm, submission schema) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.6330), so we should improve score with the smallest changes that don’t alter the model or preprocessing logic. The biggest score-relevant issue in your current inference is that `DataGenerator(mode="test")` expects `data` to have a `spec_id` column, but your calibration subset (`calib_df`) is built from `train_spec_ids` and still uses the train column name `spec_id` only by accident; more importantly, `test` uses `spec_id` after renaming, while `calib_df` never gets the same rename/shape as `test` and can silently mis-handle merges/columns across generator modes. I (1) make the generator robust for test-mode by accepting either `spec_id` or `spectrogram_id` without changing any slicing logic, and (2) fix calibration target alignment by rebuilding `calib_y` from `train` *after* filtering `calib_df` with `mask_ok` via a merge on `eeg_id` (so the KL-based fold weighting/temperature selection is computed on correctly matched rows). These are minimal, metric-aligned fixes that should move KL down toward your target while keeping architecture, inputs, and inference flow intact and still writing a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.6330), so the smallest likely score-moving fix is to remove prediction/target misalignment in the calibration logic used for fold-weighting and temperature selection. I (1) rebuild the calibration subset using a deterministic but representative sample across the full `train` (instead of just `iloc[:CALIB_N]`, which can be distribution-skewed), and (2) ensure `calib_df` and `calib_y` are aligned 1:1 by constructing `calib_y` via an index-aligned merge and then reindexing to `calib_df` order. These changes keep your model, generator, preprocessing, and inference flow identical, but make the KL-driven weighting/calibration actually reflect the same rows being predicted, which should move the leaderboard KL down toward the target. Everything still runs end-to-end and writes a valid `submission.csv` with correct columns and row sums of 1.'

# 9. Code solution

## === cell 0
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major < 5:
            return
    except Exception:
        pass

    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]


_ensure_compatible_protobuf()

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402121"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 100

HIGH = 128
LENGTH = 256

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from scipy import signal  # needed by DataGenerator even when NEEDTRAIN=False

print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

MIX = True
if MIX:
    try:
        from tensorflow.keras import mixed_precision

        if NEEDTRAIN:
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision policy set to mixed_float16 (training)")
        else:
            mixed_precision.set_global_policy("float32")
            print(
                "Mixed precision disabled for inference: global policy set to float32"
            )
    except Exception as e:
        print("Mixed precision policy not set; continuing. Details:", repr(e))
else:
    print("Using full precision")




## === cell 1
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)




## === cell 2
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")

    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()




## === cell 3
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")

    if READ_EEG_FILES:
        eegs = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
                list_eeg = list()
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, chan.split("-")[0]]
                            - eeg_default.loc[:, chan.split("-")[1]]
                        ).values

                    eeg[np.isnan(eeg)] = 0

                    if 200 != SFREQ:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs[name] = list_eeg

        if not os.path.exists("./input/brain-eegs"):
            os.makedirs("./input/brain-eegs")
        np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()




## === cell 4
import albumentations as albu

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        data,
        batch_size=32,
        shuffle=False,
        augment=False,
        mode="train",
        specs=None,
        eegs=None,
        df=None,
    ):

        if mode != "test":
            self.df = df.merge(data.iloc[:, :6], on="eeg_id", how="inner").reset_index(
                drop=True
            )

        self.cmin = -4
        self.cmax = 6
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.mode = mode
        self.specs = specs
        self.eegs = eegs
        self.b, self.a = signal.butter(
            3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
        )
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        if self.mode == "test":
            return [X, X_eeg]
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        img = np.ones((HIGH, LENGTH), dtype="float32")
        img_eeg = np.ones((4, round(EEG_LENGTH * SFREQ)), dtype="float32")

        img_map = np.zeros((100 * 300, 3))

        for j, i in enumerate(indexes):
            if self.mode == "test":
                row = self.data.iloc[i]
            else:
                if j < (self.batch_size / 6 * 1):
                    target = "Seizure"
                elif j < (self.batch_size / 6 * 2):
                    target = "GPD"
                elif j < (self.batch_size / 6 * 3):
                    target = "LRDA"
                elif j < (self.batch_size / 6 * 4):
                    target = "Other"
                elif j < (self.batch_size / 6 * 5):
                    target = "GRDA"
                else:
                    target = "LPD"

                rows = self.df[self.df.expert_consensus == target].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row1 = rows.iloc[0]
                row2 = rows.iloc[1]
                mixup = 0.3
                label = (
                    row1[TARGETS].values / sum(row1[TARGETS].values) * (1 - mixup)
                    + row2[TARGETS].values / sum(row2[TARGETS].values) * mixup
                )

            for k in range(4):
                if self.mode == "test":
                    if "spec_id" in row.index:
                        sid = int(row.spec_id)
                    else:
                        sid = int(row.spectrogram_id)

                    spec = self.specs[sid]
                    spec_len = spec.shape[0]
                    start = max(0, (spec_len - 300) // 2)
                    end = start + 300

                    img = spec[start:end, k * 100 : (k + 1) * 100].T

                    if img.shape[1] < 300:
                        pad_w = 300 - img.shape[1]
                        if img.shape[1] == 0:
                            img = np.zeros((img.shape[0], 300), dtype=img.dtype)
                        else:
                            img = np.pad(img, ((0, 0), (0, pad_w)), mode="edge")

                    img_eeg = self.eegs[int(row.eeg_id)][:, :, k]
                    img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                    img = np.log(img)
                else:
                    img1 = self.specs[row1.spectrogram_id][
                        round(row1.spectrogram_label_offset_seconds / 2) : round(
                            row1.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img2 = self.specs[row2.spectrogram_id][
                        round(row2.spectrogram_label_offset_seconds / 2) : round(
                            row2.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img_eeg1 = self.eegs[row1.eeg_id][
                        :,
                        round(row1.eeg_label_offset_seconds * SFREQ) : round(
                            row1.eeg_label_offset_seconds * SFREQ + 50 * SFREQ
                        ),
                        k,
                    ]
                    img_eeg2 = self.eegs[row2.eeg_id][
                        :,
                        round(row2.eeg_label_offset_seconds * SFREQ) : round(
                            row2.eeg_label_offset_seconds * SFREQ + 50 * SFREQ
                        ),
                        k,
                    ]
                    img1 = np.clip(img1, np.exp(self.cmin), np.exp(self.cmax))
                    img1 = np.log(img1)
                    img1 = np.nan_to_num(img1, nan=0.0)
                    img2 = np.clip(img2, np.exp(self.cmin), np.exp(self.cmax))
                    img2 = np.log(img2)
                    img2 = np.nan_to_num(img2, nan=0.0)
                    img = img1 * (1 - mixup) + img2 * mixup
                    img_eeg1 = np.nan_to_num(img_eeg1, nan=0.0)
                    img_eeg2 = np.nan_to_num(img_eeg2, nan=0.0)
                    img_eeg = img_eeg1 * (1 - mixup) + img_eeg2 * mixup

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                img = np.reshape(img, (img.shape[0] * img.shape[1]))
                img = np.array(img, dtype=np.int16)
                img_map = self.cmaps[img - 1]
                img_map = np.reshape(img_map, (100, 300, 3))

                img_map = img_map[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img_map.shape[1]
                    ),
                    :,
                ]
                if HIGH != 100:
                    img_map = np.array(
                        tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                        dtype=np.float32,
                    )
                    X[
                        j,
                        round((HIGH - img_map.shape[0]) / 2) : round(
                            (HIGH + img_map.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = img_map
                else:
                    X[j, :, :, :, k] = img_map

                X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / (0.229**2)
                X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / (0.224**2)
                X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / (0.225**2)

                img_eeg = signal.filtfilt(self.b, self.a, img_eeg, axis=1)
                img_eeg = img_eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = label

        return X, X_eeg, y

    def __random_transform(self, img):
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
            ]
        )
        return composition(image=img)["image"]

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self.__random_transform(img_batch[i,])
        return img_batch




## === cell 5
def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2**i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="tanh",
            dilation_rate=dilation_rate,
        )(x)
        sigm_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            dilation_rate=dilation_rate,
        )(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x


def l2_normalize_layer(axis=-1, epsilon=1e-12):
    return tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=axis, epsilon=epsilon)
    )


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False, weights=None, include_preprocessing=False
    )
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnetv2-s_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnetv2-s_notop.h5"
            )

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = l2_normalize_layer(axis=-1)(x)

    base_model_eeg = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights=None, include_preprocessing=False
    )
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model_eeg.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnetv2-b0_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model_eeg.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnetv2-b0_notop.h5"
            )

    x0_eeg = inp_eeg[:, :, :, :1]
    x1_eeg = inp_eeg[:, :, :, 1:2]
    x2_eeg = inp_eeg[:, :, :, 2:3]
    x3_eeg = inp_eeg[:, :, :, 3:4]
    x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
    x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = l2_normalize_layer(axis=-1)(x_eeg)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 6
import glob


def _safe_softmax_temperature(p, T=1.0, eps=1e-8):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    z = np.log(p) / float(T)
    z = z - z.max(axis=1, keepdims=True)
    p2 = np.exp(z)
    p2 = p2 / p2.sum(axis=1, keepdims=True)
    return p2.astype(np.float32)


def _kl_divergence(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.clip(y_true, eps, 1.0)
    y_true = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred = np.clip(y_pred, eps, 1.0)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(y_true * (np.log(y_true) - np.log(y_pred)), axis=1)))


if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH2 = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 200 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    for i, f in enumerate(files2):
        if i % 200 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = list()
            for region in BRAIN.keys():
                eeg = np.zeros(
                    (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                )
                for chan_i, chan in enumerate(BRAIN[region]):
                    eeg[chan_i, :] = (
                        eeg_default.loc[:, chan.split("-")[0]]
                        - eeg_default.loc[:, chan.split("-")[1]]
                    ).values

                eeg[np.isnan(eeg)] = 0

                if 200 != SFREQ:
                    eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)
            eegs2[name] = list_eeg

    train_spec_ids = train[["eeg_id", "spec_id"]].copy()

    if PLATFORM == "local":
        PATH_TR_SPEC = (
            "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
        )
        PATH_TR_EEG = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    else:
        PATH_TR_SPEC = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
        PATH_TR_EEG = (
            "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
        )

    CALIB_N = min(2048, len(train_spec_ids))
    calib_df = (
        train_spec_ids.sample(n=CALIB_N, random_state=42, replace=False)
        .copy()
        .reset_index(drop=True)
    )

    calib_specs = {}
    for sid in calib_df["spec_id"].unique():
        pth = os.path.join(PATH_TR_SPEC, f"{int(sid)}.parquet")
        if os.path.exists(pth):
            tmp = pd.read_parquet(pth)
            calib_specs[int(sid)] = tmp.iloc[:, 1:].values

    calib_eegs = {}
    for eid in calib_df["eeg_id"].unique():
        pth = os.path.join(PATH_TR_EEG, f"{int(eid)}.parquet")
        if os.path.exists(pth):
            eeg_default = pd.read_parquet(pth)
            list_eeg = []
            for region in BRAIN.keys():
                eeg = np.zeros(
                    (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                )
                for chan_i, chan in enumerate(BRAIN[region]):
                    eeg[chan_i, :] = (
                        eeg_default.loc[:, chan.split("-")[0]]
                        - eeg_default.loc[:, chan.split("-")[1]]
                    ).values
                eeg[np.isnan(eeg)] = 0
                if 200 != SFREQ:
                    eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))
            list_eeg = np.concatenate(list_eeg, 2)
            calib_eegs[int(eid)] = list_eeg

    mask_ok = calib_df["spec_id"].astype(int).isin(calib_specs.keys()) & calib_df[
        "eeg_id"
    ].astype(int).isin(calib_eegs.keys())
    calib_df = calib_df.loc[mask_ok].reset_index(drop=True)

    calib_y_df = train[["eeg_id"] + list(TARGETS)].copy()
    calib_y_df = calib_df[["eeg_id"]].merge(calib_y_df, on="eeg_id", how="left")
    calib_y = calib_y_df[list(TARGETS)].values.astype(np.float32)

    print(f"\nCalibration subset: {len(calib_df)} rows (from requested {CALIB_N}).")

    candidate_dirs = []
    if os.path.isdir(LOAD_MODELS_FROM):
        candidate_dirs.append(LOAD_MODELS_FROM)
    if PLATFORM == "kaggle":
        candidate_dirs.extend(
            [
                "/kaggle/input/models202402121",
                "/kaggle/input/models202402121/models202402121",
            ]
        )

    patterns = [
        f"EB2_v{VER}_f*.h5",
        f"EB2_v{VER}_f*.weights.h5",
        f"EB2_v{VER}_fold*.h5",
        f"EB2_v{VER}_fold*.weights.h5",
        f"EB2_v{VER}_f*.keras",
        f"EB2_v{VER}_fold*.keras",
        f"EB2_v{VER}_f*.ckpt",
        f"EB2_v{VER}_fold*.ckpt",
    ]

    weight_files = []
    for d in candidate_dirs:
        if os.path.isdir(d):
            for pat in patterns:
                weight_files.extend(sorted(glob.glob(os.path.join(d, pat))))

    if PLATFORM == "kaggle" and len(weight_files) == 0:
        for pat in patterns:
            weight_files.extend(
                sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
            )

    seen = set()
    weight_files_unique = []
    for wf in weight_files:
        if wf not in seen:
            weight_files_unique.append(wf)
            seen.add(wf)
    weight_files = weight_files_unique

    print(f"\nFound {len(weight_files)} weight files (expanded patterns).")
    if len(weight_files) > 0:
        print("Example weight file:", weight_files[0])

    if len(weight_files) == 0:
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
        )

        calib_gen = DataGenerator(
            calib_df,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=calib_specs,
            eegs=calib_eegs,
        )

        preds_test = []
        preds_calib = []
        fold_kls = []

        for wf in weight_files:
            print(f"\nLoading weights: {wf}")
            with strategy.scope():
                model = build_model()
            model.load_weights(wf)

            pred_fold_test = model.predict(test_gen, verbose=0)
            preds_test.append(pred_fold_test)

            if len(calib_df) > 0:
                pred_fold_cal = model.predict(calib_gen, verbose=0)
                fold_kl = _kl_divergence(calib_y, pred_fold_cal, eps=1e-8)
                fold_kls.append(fold_kl)
                preds_calib.append(pred_fold_cal)
                print(f"Fold calib KL (approx): {fold_kl:.6f}")
            else:
                fold_kls.append(1.0)
                preds_calib.append(None)
                print("Fold calib KL skipped (no calib data).")

        preds_test = np.array(preds_test, dtype=np.float32)

        if len(calib_df) > 0 and len(fold_kls) == len(weight_files):
            kls = np.asarray(fold_kls, dtype=np.float64)
            w = 1.0 / (kls + 1e-6)
            w = w / w.sum()
            print("Ensemble weights:", np.round(w, 4))
            pred = np.tensordot(w.astype(np.float32), preds_test, axes=(0, 0))
        else:
            pred = np.mean(preds_test, axis=0)

        eps_floor = 1e-5
        pred = np.asarray(pred, dtype=np.float64)
        pred = np.clip(pred, eps_floor, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        if len(calib_df) > 0:
            preds_calib_stack = [p for p in preds_calib if p is not None]
            preds_calib_stack = np.array(preds_calib_stack, dtype=np.float32)
            if preds_calib_stack.shape[0] == len(weight_files) and len(fold_kls) == len(
                weight_files
            ):
                kls = np.asarray(fold_kls, dtype=np.float64)
                w = 1.0 / (kls + 1e-6)
                w = w / w.sum()
                pred_cal = np.tensordot(
                    w.astype(np.float32), preds_calib_stack, axes=(0, 0)
                )
            else:
                pred_cal = np.mean(preds_calib_stack, axis=0)

            temps = [0.9, 1.0, 1.1]
            best_T = 1.0
            best_kl = 1e9
            for T in temps:
                pc = _safe_softmax_temperature(pred_cal, T=T, eps=1e-8)
                k = _kl_divergence(calib_y, pc, eps=1e-8)
                if k < best_kl:
                    best_kl = k
                    best_T = T
            print(f"Chosen temperature (approx KL-min): T={best_T}, KL={best_kl:.6f}")
            pred = _safe_softmax_temperature(pred, T=best_T, eps=1e-8)
        else:
            pred = pred.astype(np.float32)

        pred = np.asarray(pred, dtype=np.float64)
        pred = np.clip(pred, eps_floor, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        pred = pred.astype(np.float32)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[list(TARGETS)] = pred

    row_sums = sub[list(TARGETS)].sum(axis=1).values
    print(
        "Row sum stats:",
        float(row_sums.min()),
        float(row_sums.mean()),
        float(row_sums.max()),
    )

    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print("Submission shape", sub.shape)
    print(sub.head())
