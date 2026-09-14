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

0.312954467843227

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle runtime. Next, I make the model-weight loading robust by auto-detecting the actual checkpoint directory and file extensions inside `/kaggle/input`, and fall back to a valid uniform-probability submission if no weights are found (so you always get a `submission.csv`). I also add small guardrails so the generator won’t KeyError when a particular modality isn’t loaded (e.g., `spe` dictionaries missing), while keeping the core model and preprocessing logic unchanged. Finally, I ensure the submission columns match `sample_submission.csv` and each row sums to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that currently prevents the notebook from running by forcing a compatible protobuf runtime before importing TensorFlow and (as a fallback) uninstalling the compiled `protobuf` wheel so TensorFlow uses the pure-Python implementation. This is a correctness/stability-only change and does not alter your model, preprocessing, or inference logic. I also add small defensive handling so if the TensorFlow import still fails for any reason, the script still write a valid `submission.csv` (uniform probabilities) rather than crashing. Once TensorFlow imports successfully, the rest of your pipeline (data loading, model definition, weight loading, prediction averaging, and submission formatting) remains unchanged.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that currently stops execution by forcing the pure-Python protobuf runtime in a way that actually takes effect before TensorFlow is imported, and by safely removing the compiled protobuf wheel from the runtime if needed. Then I ensure the script always produces a valid `submission.csv` even if TensorFlow still cannot import, but when TensorFlow does import, the original model definition, preprocessing, weight loading, and prediction logic remain unchanged. This should move your score down from the uniform-fallback level (1.40995) toward the target by enabling real model inference with the provided weights. I also add a tiny, score-neutral safeguard to always align prediction columns to `sample_submission.csv` and renormalize probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that stops execution by ensuring the pure-Python protobuf implementation is actually used before importing TensorFlow, and by safely downgrading protobuf to a TF-compatible version if needed (instead of uninstalling it, which can still leave an incompatible compiled runtime in place). This should allow the existing model/weights inference path to run, replacing the uniform fallback that produced the current poor score (1.40995) and moving KLDiv down toward the target. I also add a minimal, score-neutral safeguard to always build the submission columns in the exact order of `sample_submission.csv` and renormalize probabilities to sum to 1. Core model architecture, preprocessing, and prediction averaging remain unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) matches a near-uniform prediction, which strongly suggests your inference is still effectively running without the intended pretrained weights (or the weights don’t match the constructed model). I make minimal changes to (1) correctly locate weight files in typical Kaggle dataset layouts (including nested folders and alternative naming), (2) rebuild the model inside the fold loop so each fold loads cleanly and deterministically, and (3) add a tiny, metric-aligned probability “floor + renorm” to avoid extreme KL penalties without changing core semantics. These changes are directly targeted to move KLDiv down toward your target by ensuring real ensemble predictions are produced instead of the uniform fallback. The submission format/ordering and row-sum-to-1 constraints remain enforced.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is consistent with uniform predictions, so the most direct way to move toward the target is to ensure real pretrained weights are actually being found and loaded. I make a minimal, score-relevant fix to the weight-file discovery to also support common Kaggle artifacts like `.weights.h5` / `.ckpt` style names and to pick the correct stage when only earlier-stage weights exist, without changing your model or preprocessing. I also add a tiny sanity check that prints whether fold predictions differ from uniform (to confirm inference is “live”), while keeping prediction averaging and submission formatting unchanged. If no weights are found even after expanded search, it still safely output a valid `submission.csv` as before.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is consistent with the uniform fallback, so the most direct way to move toward the target is to ensure fold weights are actually discovered and loaded. I make minimal, score-relevant changes to (1) search for weights using more flexible filename patterns (including nested “weights/” directories and `stage{n}_f{fold}` ordering), and (2) load weights with a safe fallback to `tf.train.Checkpoint.restore` when `model.load_weights()` fails (common when checkpoints are saved in TF format rather than H5). I also force `MIX=False` for inference stability (mixed precision can change numerics and occasionally break weight restores) without changing the model or preprocessing. Submission formatting, column order, and per-row normalization remain unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL score (1.40995, lower-is-better) is consistent with the uniform-fallback path, so the smallest score-improving change is to make sure inference can actually load pretrained weights and run. I keep your model, generator, and preprocessing identical, but expand weight discovery to search *all* of `/kaggle/input` (not just top-level folders) and prefer the directory with the most matching fold/stage checkpoints. I also add one minimal fix to ensure TensorFlow uses GPU devices (the `CUDA_VISIBLE_DEVICES` setting must be applied before importing TF), which can be the difference between finishing within the time limit vs timing out and effectively falling back. Finally, I keep your submission formatting/renormalization as-is so the file remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower-is-better) strongly indicates you are still submitting the uniform fallback because no real pretrained weights are being loaded, so the smallest score-improving change is to make weight discovery/loading actually match the common HMS artifact layouts. I keep your model, generator, preprocessing, and inference loop identical, but (1) broaden weight-file discovery to also accept “model”/“best” filenames and non-`stage` naming, and (2) make checkpoint restore stricter by verifying that at least some variables are restored (otherwise we skip that fold instead of silently predicting near-uniform). Finally, I ensure we never accidentally select an empty directory under `/kaggle/input` and print a single, lightweight confirmation of which path pattern matched (to debug why you may still be falling back). These changes are directly aimed at replacing the uniform submission with real ensemble predictions, moving KL down toward your 0.313 target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is consistent with the uniform-fallback path, so the smallest score-improving move toward the 0.313 target is to make weight discovery actually find the correct HMS model artifacts and restore them successfully. I (1) auto-detect the HMS dataset folder under `/kaggle/input` and search it first (instead of scanning all of `/kaggle/input`, which often picks the wrong directory), (2) extend checkpoint discovery to handle TF SavedModel-style `variables/variables` and `checkpoint`-based restores that `model.load_weights()` misses, and (3) add a strict-but-safe restore verification so we only average folds with real restored variables (avoiding near-uniform predictions from mismatched weights). Core model, preprocessing, generator, and inference averaging remain unchanged; the submission formatting and row-sum-to-1 normalization remain enforced.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import sys
import subprocess


def _ensure_tf_compatible_protobuf():
    """
    Change reason (score-relevant): if TF import fails, we fall back to uniform predictions (bad KL score).
    Keep the same approach but make it more reliable by pinning protobuf<5 when detected.
    """
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(v):
        try:
            return int(str(v).split(".")[0])
        except Exception:
            return None

    pb_major = _major(pb_ver)

    if pb_major is not None and pb_major >= 5:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"],
                check=False,
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    sys.modules.pop(m, None)
        except Exception:
            pass


_ensure_tf_compatible_protobuf()

import io
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_OK = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
except Exception as e:
    TF_OK = False
    TF_IMPORT_ERROR = repr(e)
    tf = None

from sklearn.metrics import confusion_matrix  # kept (original import)
import librosa

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models20241022-1"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128
LENGTH = 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024
BATCHSIZE = 16

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

spectrograms = {}
eegs = {}
imgs = {}
stfts = {}

spectrograms2 = {}
eegs2 = {}
imgs2 = {}
stfts2 = {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

if TF_OK:
    print("TensorFlow version =", tf.__version__)

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

    MIX = False
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Mixed precision requested but not available:", repr(e))
    else:
        print("Using full precision")
else:
    strategy = None
    print("TensorFlow import failed; will fall back to uniform submission.")
    print("TF import error:", TF_IMPORT_ERROR)



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
df.head()



## === cell 2
TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}

if TF_OK:

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
            imgs=None,
            stfts=None,
            targets=None,
        ):

            self.targets = targets
            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["cividis"](np.linspace(0, 1, 256))[:, :3]

            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs if specs is not None else {}
            self.eegs = eegs if eegs is not None else {}
            self.imgs = imgs if imgs is not None else {}
            self.stfts = stfts if stfts is not None else {}
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.data) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y = self.__data_generation(indexes)
            return x, y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (len(indexes), 6, round(20 * SFREQ), 3, 4), dtype="float32"
                )
                x_eeg2 = np.zeros(
                    (len(indexes), 4, round(50 * SFREQ), 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros(
                    (len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32"
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                elif self.mode == "valid":
                    r_spe = (
                        round(row["spectrogram_label_offset_seconds"] / 2)
                        if "spectrogram_label_offset_seconds" in row.index
                        else 0
                    )
                    r_eeg = (
                        round(row["eeg_label_offset_seconds"] * SFREQ)
                        if "eeg_label_offset_seconds" in row.index
                        else 0
                    )
                else:
                    rows = df.loc[df.eeg_id == row.eeg_id, :].reset_index(drop=True)
                    for lk in TARGETS:
                        rows = rows.loc[rows[lk] == row[lk + "_raw"], :].reset_index(
                            drop=True
                        )
                    rows_eeg = rows.iloc[np.random.permutation(len(rows))].reset_index(
                        drop=True
                    )

                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(rows_eeg.eeg_label_offset_seconds[0] * SFREQ)

                if self.mode == "train":
                    x1 = np.random.rand() * (LENGTH / 2 - 20)
                    x2 = np.random.rand() * (LENGTH / 2 - 20)
                    if np.random.rand() < 0.5:
                        x1 = x1 + LENGTH / 2
                        x2 = x2 + LENGTH / 2
                    x_spe_min = round(min(x1, x2))
                    x_spe_max = round(max(x1, x2))

                    x1 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                    x2 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                    if np.random.rand() < 0.5:
                        x1 = x1 + EEG_LENGTH * SFREQ / 2
                        x2 = x2 + EEG_LENGTH * SFREQ / 2
                    else:
                        x1 = x1 + 10 * SFREQ / 2
                        x2 = x2 + 10 * SFREQ / 2
                    x_eeg_min = round(min(x1, x2))
                    x_eeg_max = round(max(x1, x2))

                    x1 = np.random.rand() * (LENGTH / 2 - 42)
                    x2 = np.random.rand() * (LENGTH / 2 - 42)
                    if np.random.rand() < 0.5:
                        x1 = x1 + LENGTH / 2
                        x2 = x2 + LENGTH / 2
                    else:
                        x1 = x1 + 42
                        x2 = x2 + 42
                    x_img_min = round(min(x1, x2))
                    x_img_max = round(max(x1, x2))

                for k in range(4):
                    if "spe" in DATATYPE:
                        if row.spectrogram_id in self.specs:
                            spe = self.specs[row.spectrogram_id][
                                r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                            ].T
                            spe = np.nan_to_num(spe, nan=0.0)
                            spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                            spe = np.log(spe)

                            spe = np.round(
                                (spe - self.cmin) / (self.cmax - self.cmin) * 255
                            )
                            spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                            spe = np.array(spe, dtype=np.int16)

                            spe = self.cmaps[spe]
                            spe = np.reshape(spe, (100, 300, 3))
                            spe = spe[
                                :,
                                max(round((600 / 2 - 256) / 2), 0) : min(
                                    (round((600 / 2 - 256) / 2) + LENGTH), spe.shape[1]
                                ),
                                :,
                            ]

                            if self.mode == "train":
                                spe[:, x_spe_min:x_spe_max, :] = 0

                            x_spe[
                                j,
                                round((HIGH - spe.shape[0]) / 2) : round(
                                    (HIGH + spe.shape[0]) / 2
                                ),
                                :,
                                :,
                                k,
                            ] = spe
                        x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                    if "eeg" in DATATYPE:
                        if row.eeg_id in self.eegs:
                            eeg = self.eegs[row.eeg_id][
                                :, r_eeg : r_eeg + round(50 * SFREQ), k
                            ]
                        else:
                            eeg = np.zeros((4, round(50 * SFREQ)), dtype=np.float32)

                        eeg1 = eeg[:, round(10 * SFREQ) : round(30 * SFREQ)]
                        eeg2 = eeg[:, round(15 * SFREQ) : round(35 * SFREQ)]
                        eeg3 = eeg[:, round(20 * SFREQ) : round(40 * SFREQ)]

                        x_eeg[j, 1:5, :, 0, k] = eeg1
                        x_eeg[j, 1:5, :, 1, k] = eeg2
                        x_eeg[j, 1:5, :, 2, k] = eeg3
                        x_eeg[j, :, :, :, k] = (
                            x_eeg[j, :, :, :, k]
                            - np.mean(x_eeg[j, :, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, :, k], 1, keepdims=True) + 1e-6)

                        eeg = (eeg - np.mean(eeg, 1, keepdims=True)) / (
                            np.std(eeg, 1, keepdims=True) + 1e-6
                        )
                        x_eeg2[j, :, :, k] = eeg

                    if "img" in DATATYPE:
                        if row.eeg_id in self.imgs:
                            img = self.imgs[row.eeg_id][:, :, k, :]
                            x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        if row.eeg_id in self.stfts:
                            stft = self.stfts[row.eeg_id][:, :, k]
                        else:
                            stft = np.zeros((HIGH, LENGTH), dtype=np.float32)
                        stft = np.nan_to_num(stft, nan=0.0)
                        cmin = 0
                        cmax = 60
                        stft = np.clip(stft, cmin, cmax)
                        stft = np.round((stft - cmin) / (cmax - cmin) * 255)

                        shape0, shape1 = stft.shape[0], stft.shape[1]
                        stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                        stft = np.array(stft, dtype=np.int16)
                        stft = self.cmaps[stft]
                        stft = np.reshape(stft, (shape0, shape1, 3))

                        x_stft[
                            j,
                            round((HIGH - stft.shape[0]) / 2) : round(
                                (HIGH + stft.shape[0]) / 2
                            ),
                            :,
                            :,
                            k,
                        ] = stft
                        x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                if self.mode != "test":
                    label = row[self.targets].values
                    if self.mode == "train" and sum(label == 1):
                        xx = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx
                        label[label == 1] = 1 - 5 * xx
                    y[j] = label

            alpha = 0
            x = []
            if "spe" in DATATYPE:
                if (self.mode == "train") and (alpha > 0):
                    xx = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                    x_spe = x_spe * (1 - xx) + x_spe[::-1, :, :, :, :] * xx
                x.append(x_spe)

            if "eeg" in DATATYPE:
                if (self.mode == "train") and (alpha > 0):
                    xx = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                    x_eeg = x_eeg * (1 - xx) + x_eeg[::-1, :, :, :] * xx
                x.append(x_eeg)
                x.append(x_eeg2)

            if "img" in DATATYPE:
                if (self.mode == "train") and (alpha > 0):
                    xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1, 1))
                    x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
                if self.mode == "train":
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5
                    ) * 2 - 1
                    x_img = x_img * aug_img
                x.append(x_img)

            if "stft" in DATATYPE:
                if (self.mode == "train") and (alpha > 0):
                    xx = np.reshape(xx, (x_stft.shape[0], 1, 1, 1, 1))
                    x_stft = x_stft * (1 - xx) + x_stft[::-1, :, :, :, :] * xx
                x.append(x_stft)

            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (y.shape[0], 1))
                y = y * (1 - xx) + y[::-1, :] * xx

            return x, y




## === cell 3
if TF_OK:
    from tensorflow.keras.applications import EfficientNetB0

    def build_model(TARGETS_PRETRAIN):
        inp = []
        y = None

        l2norm = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe1 = inp_spe[:, :, :, :, 0]
            x_spe2 = inp_spe[:, :, :, :, 1]
            x_spe3 = inp_spe[:, :, :, :, 2]
            x_spe4 = inp_spe[:, :, :, :, 3]
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_spe"
            )
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = l2norm(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
            x_eeg1 = inp_eeg[:, :, :, :, 0]
            x_eeg2 = inp_eeg[:, :, :, :, 1]
            x_eeg3 = inp_eeg[:, :, :, :, 2]
            x_eeg4 = inp_eeg[:, :, :, :, 3]

            x_eeg1 = tf.keras.layers.Concatenate(axis=1)([x_eeg1, x_eeg2])
            x_eeg2 = tf.keras.layers.Concatenate(axis=1)([x_eeg3, x_eeg4])
            x_eeg = tf.keras.layers.Concatenate(axis=2)([x_eeg1, x_eeg2])

            base_model_eeg = EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_eeg"
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = l2norm(x_eeg)

            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                if y is not None
                else x_eeg
            )

        if "eeg" in DATATYPE:
            inp_eeg2 = tf.keras.Input(shape=(4, round(50 * SFREQ), 4))
            inp.append(inp_eeg2)

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
            x_img1 = inp_img[:, :, :, :, 0]
            x_img2 = inp_img[:, :, :, :, 1]
            x_img3 = inp_img[:, :, :, :, 2]
            x_img4 = inp_img[:, :, :, :, 3]
            x_img = tf.keras.layers.Concatenate(axis=1)(
                [x_img1, x_img2, x_img3, x_img4]
            )

            base_model_img = EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_img"
            )
            base_model_img._name = "img_extractor"
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = l2norm(x_img)

            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_img])
                if y is not None
                else x_img
            )

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_stft1 = inp_stft[:, :, :, :, 0]
            x_stft2 = inp_stft[:, :, :, :, 1]
            x_stft3 = inp_stft[:, :, :, :, 2]
            x_stft4 = inp_stft[:, :, :, :, 3]
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [x_stft1, x_stft2, x_stft3, x_stft4]
            )

            base_model_stft = EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_stft"
            )
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = l2norm(x_stft)

            inp.append(inp_stft)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                if y is not None
                else x_stft
            )

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN), activation="softmax", dtype="float32"
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model

    def my_loss(y_ture, y_pred):
        y_pred1 = y_pred[:, 5:6]
        y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
        y_pred2 = y_pred[:, 0:5]
        y_pred = tf.concat((y_pred2, y_pred1), axis=1)
        return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 4
def _hms_input_root():
    """
    Change reason (score-relevant): searching all of /kaggle/input can select the wrong directory.
    Prefer the official competition dataset folder when present, falling back to /kaggle/input.
    """
    candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return "/kaggle/input"


def _find_model_dir(preferred_path: str):
    """
    Change reason (score-relevant): current score indicates weights likely not found/loaded.
    Make search robust, but search the HMS dataset area first to avoid picking unrelated model folders.
    """
    if os.path.isdir(preferred_path):
        return preferred_path

    root = _hms_input_root()
    if not os.path.isdir(root):
        return preferred_path

    def _is_weight_file(fl: str) -> bool:
        fl = fl.lower()
        return (
            fl.endswith(".h5")
            or fl.endswith(".keras")
            or fl.endswith(".weights.h5")
            or fl.endswith(".index")
            or fl.endswith(".data-00000-of-00001")
            or fl.endswith(".ckpt")
        )

    def _is_likely_hms_weight(fl: str) -> bool:
        fl = fl.lower()
        has_fold = (
            ("f0" in fl)
            or ("fold0" in fl)
            or ("f1" in fl)
            or ("fold1" in fl)
            or ("f2" in fl)
            or ("fold2" in fl)
            or ("f3" in fl)
            or ("fold3" in fl)
            or ("f4" in fl)
            or ("fold4" in fl)
            or ("fold" in fl)
        )
        has_hint = (
            ("stage" in fl)
            or ("best" in fl)
            or ("model" in fl)
            or ("weights" in fl)
            or ("ckpt" in fl)
            or ("checkpoint" in fl)
            or ("variables" in fl)
        )
        return _is_weight_file(fl) and has_fold and has_hint

    best = None
    best_hits = 0

    for r, _, files in os.walk(root):
        hits = sum(1 for f in files if _is_likely_hms_weight(f))
        if os.path.isdir(os.path.join(r, "variables")):
            vv = os.path.join(r, "variables", "variables.index")
            if os.path.exists(vv):
                hits += 2
        if hits > best_hits:
            best_hits = hits
            best = r

    return best if best is not None else preferred_path


def _weight_path(model_dir: str, fold: int, stage: int):
    """
    Change reason (score-relevant): expand matching to common variants, including SavedModel/variables.
    """
    base_variants = [
        f"f{fold}_stage{stage}",
        f"fold{fold}_stage{stage}",
        f"f{fold}-stage{stage}",
        f"fold{fold}-stage{stage}",
        f"stage{stage}_f{fold}",
        f"stage{stage}_fold{fold}",
        f"stage{stage}-f{fold}",
        f"stage{stage}-fold{fold}",
        f"F{fold}_stage{stage}",
        f"FOLD{fold}_stage{stage}",
        f"STAGE{stage}_F{fold}",
        f"STAGE{stage}_FOLD{fold}",
        f"fold{fold}",
        f"f{fold}",
        f"best_fold{fold}",
        f"best_f{fold}",
        f"model_fold{fold}",
        f"model_f{fold}",
    ]
    exts = [".h5", ".keras", ".weights.h5"]

    for base in base_variants:
        for ext in exts:
            p = os.path.join(model_dir, base + ext)
            if os.path.exists(p):
                return p

    subdirs = ["", "weights", "Weight", "ckpt", "checkpoints", "checkpoint", "models"]
    for sd in subdirs:
        sd_path = os.path.join(model_dir, sd) if sd else model_dir
        if not os.path.isdir(sd_path):
            continue
        for base in base_variants:
            for ext in exts:
                p = os.path.join(sd_path, base + ext)
                if os.path.exists(p):
                    return p

    for root, _, files in os.walk(model_dir):
        lower_map = {f.lower(): f for f in files}
        for base in base_variants:
            base_l = base.lower()
            idx_name = base_l + ".index"
            if idx_name in lower_map:
                for f in files:
                    if f.lower().startswith(base_l + ".data-"):
                        return os.path.join(root, lower_map[idx_name]).rsplit(
                            ".index", 1
                        )[0]

        for base in base_variants:
            cand_dir = os.path.join(root, base)
            vv = os.path.join(cand_dir, "variables", "variables.index")
            if os.path.exists(vv):
                return os.path.join(cand_dir, "variables", "variables")

        for base in base_variants:
            cand_dir = os.path.join(root, base)
            ckpt_file = os.path.join(cand_dir, "checkpoint")
            if os.path.exists(ckpt_file):
                return cand_dir

        for f in files:
            fl = f.lower()
            for base in base_variants:
                bl = base.lower()
                if fl.startswith(bl) and (
                    fl.endswith(".h5")
                    or fl.endswith(".keras")
                    or fl.endswith(".weights.h5")
                ):
                    return os.path.join(root, f)

    return None


def _best_available_stage(model_dir: str, fold: int, preferred_stage: int):
    """
    Keep semantics: still uses a single stage, but pick the best available <= preferred.
    """
    for st in [preferred_stage, 2, 1, 0]:
        p = _weight_path(model_dir, fold, st)
        if p is not None:
            return st, p
    p = _weight_path(model_dir, fold, preferred_stage)
    return (preferred_stage, p) if p is not None else (None, None)


def _maybe_checkpoint_prefix_from_dir(d: str):
    """
    Change reason (score-relevant): some artifacts store a 'checkpoint' file and TF checkpoints inside a dir.
    Return a checkpoint prefix path if we can read the latest checkpoint name.
    """
    if not os.path.isdir(d):
        return None
    ckpt_file = os.path.join(d, "checkpoint")
    if not os.path.exists(ckpt_file):
        return None
    try:
        with open(ckpt_file, "r", encoding="utf-8") as f:
            txt = f.read().splitlines()
        for line in txt:
            if "model_checkpoint_path" in line and '"' in line:
                name = line.split('"')[1]
                pref = os.path.join(d, name)
                if os.path.exists(pref + ".index") or os.path.exists(pref):
                    return pref
    except Exception:
        return None
    return None


def _load_weights_robust(model, wpath: str):
    """
    Change reason (score-relevant): handle more real-world TF weight formats so we don't fall back to uniform.
    Also avoid silently accepting a restore that didn't match variables (prevents averaging near-uniform outputs).
    """
    try:
        if os.path.isfile(wpath) and (
            wpath.endswith(".h5")
            or wpath.endswith(".keras")
            or wpath.endswith(".weights.h5")
        ):
            model.load_weights(wpath)
            return True
    except Exception:
        pass

    if not TF_OK:
        return False

    try:
        if os.path.isdir(wpath):
            pref = _maybe_checkpoint_prefix_from_dir(wpath)
            if pref is not None:
                wpath = pref

        is_prefix = (
            (os.path.splitext(wpath)[1] == "")
            or wpath.endswith(".ckpt")
            or os.path.exists(wpath + ".index")
        )
        has_index = os.path.exists(wpath + ".index")
        if is_prefix or has_index:
            ckpt = tf.train.Checkpoint(model=model)
            status = ckpt.restore(wpath)
            status.assert_nontrivial_match()
            try:
                status.expect_partial()
            except Exception:
                pass
            _ = model.trainable_variables
            return True
    except Exception:
        return False

    return False


LOAD_MODELS_FROM = _find_model_dir(LOAD_MODELS_FROM)
print("Resolved LOAD_MODELS_FROM:", LOAD_MODELS_FROM)



## === cell 5
if PLATFORM == "local":
    test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    sample_sub = pd.read_csv(
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    sample_sub = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )

sub_cols = [c for c in sample_sub.columns if c != "eeg_id"]

if (not TF_OK) or NEEDTRAIN:
    pred = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[sub_cols] = pred[:, [list(TARGETS).index(c) for c in sub_cols]]
    sub[sub_cols] = np.clip(sub[sub_cols].values, 1e-9, 1.0)
    sub[sub_cols] = sub[sub_cols].values / sub[sub_cols].values.sum(
        axis=1, keepdims=True
    )
    sub.to_csv("submission.csv", index=False)
    print("Wrote uniform submission.csv due to TF import failure or NEEDTRAIN=True.")
else:
    if "spe" in DATATYPE:
        print("Test shape", test.shape)

        if PLATFORM == "local":
            PATH2 = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = []
            list_img = []
            list_stft = []
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

                eeg = signal.filtfilt(b, a, eeg, axis=1)

                time_temp = 0
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])

                if "stft" in DATATYPE:
                    mel_spec = librosa.feature.melspectrogram(
                        y=eeg[
                            :,
                            round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ),
                        ],
                        sr=SFREQ,
                        hop_length=round(50 * SFREQ / 256),
                        n_fft=512,
                        n_mels=128,
                        fmin=0,
                        fmax=40,
                        win_length=128,
                    )
                    mel_spec = np.mean(mel_spec, 0)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.min).astype(
                        np.float32
                    )
                    list_stft.append(
                        np.reshape(
                            mel_spec_db, (mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
                        )
                    )

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)

            if "stft" in DATATYPE:
                list_stft = np.concatenate(list_stft, 2)
                stfts2[name] = list_stft

            if "eeg" in DATATYPE:
                eegs2[name] = list_eeg

            if "img" in DATATYPE:
                eeg_all_region = np.concatenate(list_img, 0)

                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 200
                for ii in range(eeg_all_region.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-10, eeg_all_region.shape[1] + 10)
                plt.ylim(-amp / 2, eeg_all_region.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img = Image.open(byte_stream)
                img = np.array(img)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img = np.concatenate((img, img, img), 2)
                img = np.array(
                    tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img = img[:, :, 0:1]

                img = np.concatenate(
                    [
                        img[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )

                img[:, :, 0] = -img[:, :, 0]
                img[:, :, 2] = -img[:, :, 2]
                img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], 1))

                eeg_all_region2 = eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 1 / 4) : round(
                        eeg_all_region.shape[1] * 3 / 4
                    ),
                ]
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 150
                for ii in range(eeg_all_region2.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-5, eeg_all_region2.shape[1] + 5)
                plt.ylim(-amp / 2, eeg_all_region2.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img2 = Image.open(byte_stream)
                img2 = np.array(img2)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img2 = np.concatenate((img2, img2, img2), 2)
                img2 = np.array(
                    tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img2 = img2[:, :, 0:1]

                img2 = np.concatenate(
                    [
                        img2[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img2[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img2[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img2[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )

                img2[:, :, 0] = -img2[:, :, 0]
                img2[:, :, 2] = -img2[:, :, 2]
                img2 = np.reshape(
                    img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1)
                )

                eeg_all_region3 = eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 2 / 5) : round(
                        eeg_all_region.shape[1] * 3 / 5
                    ),
                ]
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 100
                for ii in range(eeg_all_region3.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-2, eeg_all_region3.shape[1] + 2)
                plt.ylim(-amp / 2, eeg_all_region3.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img3 = Image.open(byte_stream)
                img3 = np.array(img3)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img3 = np.concatenate((img3, img3, img3), 2)
                img3 = np.array(
                    tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img3 = img3[:, :, 0:1]

                img3 = np.concatenate(
                    [
                        img3[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img3[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img3[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img3[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )

                img3[:, :, 0] = -img3[:, :, 0]
                img3[:, :, 2] = -img3[:, :, 2]
                img3 = np.reshape(
                    img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1)
                )

                img = np.concatenate([img, img2, img3], -1)
                imgs2[name] = img

    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=BATCHSIZE * 2,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
        stfts=stfts2,
        targets=TARGETS,
    )

    preds = []
    loaded_any = False

    for i in range(5):
        print(f"\nFold {i + 1}")

        chosen_stage, wpath = _best_available_stage(LOAD_MODELS_FROM, i, STAGETEST)
        if wpath is None:
            print(
                f"  Missing weights for fold {i} under {LOAD_MODELS_FROM} (searched stage/fold patterns)"
            )
            continue

        with strategy.scope():
            model = build_model(TARGETS)

        print(f"  Restoring/loading (stage {chosen_stage}):", wpath)
        ok = _load_weights_robust(model, wpath)
        if not ok:
            print("  restore/load failed (nontrivial match not found) for:", wpath)
            continue

        pred_i = model.predict(test_gen, verbose=1)
        print("  pred_i mean/std:", float(pred_i.mean()), float(pred_i.std()))

        preds.append(pred_i)
        loaded_any = True

    if loaded_any:
        pred = np.mean(preds, axis=0).astype(np.float32)
        print("\nTest preds shape", pred.shape)

        eps = 1e-6
        pred = np.clip(pred, eps, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
    else:
        print("\nNo weights loaded; writing uniform-probability submission.")
        pred = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[sub_cols] = pred[:, [list(TARGETS).index(c) for c in sub_cols]]
    sub[sub_cols] = np.clip(sub[sub_cols].values, 1e-9, 1.0)
    sub[sub_cols] = sub[sub_cols].values / sub[sub_cols].values.sum(
        axis=1, keepdims=True
    )

    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row prob sum stats:",
        sub[sub_cols].sum(axis=1).min(),
        sub[sub_cols].sum(axis=1).max(),
    )
    sub.head()
