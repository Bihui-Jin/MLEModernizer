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

0.284531242020453

# 6. Current score

1.67996

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting that triggers the `MessageFactory.GetPrototype` error in this environment, while keeping the same TF/Keras core logic. Then I ensure `train` is always defined (even when `NEEDTRAIN` toggles) by constructing it unconditionally from `train.csv` and normalizing targets exactly as the original code intends, which fixes the `NameError: train is not defined`. Finally, I make Kaggle inference stable by avoiding the auto-switch to `NEEDTRAIN=True` when no weights exist (training can’t finish in time), and instead fall back to a valid uniform-probability submission that satisfies the metric/format constraints (rows sum to 1, correct columns, `.csv` suffix).'
- What this solution (achieved 1.67996) has done: 'The timeout is dominated by two avoidable costs in inference: (1) preloading and filtering all 9,850 test EEG parquet files into a huge Python dict, and (2) repeated Python/pandas work inside generators and per-batch creation. I keep the exact same model and preprocessing, but switch test inference to a streaming `Sequence` that reads/parses each EEG parquet on-demand per batch (so memory stays low and we avoid the up-front 9,850-file loop). I also add a small, safe in-process LRU cache for recently-used EEG arrays (equivalent data, fewer disk reads) and speed up the BRAIN montage construction using vectorized column indexing rather than per-channel dict lookups. These changes preserve evaluation semantics and only remove redundant work; training logic remains unchanged.'
- What this solution (achieved 1.67996) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting that triggers `MessageFactory.GetPrototype` errors in this Kaggle runtime. Then I keep the existing model/inference logic intact, but change the “no weights found” branch to write a valid submission and exit cleanly without raising `SystemExit` (so Kaggle notebooks finish normally). Finally, I keep probability post-processing strictly valid (finite, clipped, row-normalized) to avoid submission failures and preserve/very slightly improve the KL metric versus uniform.'
- What this solution (achieved 1.67996) has done: 'I fix the TensorFlow import crash by removing the forced protobuf “cpp” implementation env vars (they break in this Kaggle runtime) and by making the script robust to TensorFlow being unavailable so it can still generate a valid submission. Then I ensure all shared constants used by the baseline inference path (`_BRAIN_PAIRS`, `_NEEDED_COLS`, and the bandpass filter coefficients) are defined before use, so the “no weights found” branch runs end-to-end. Finally, since no score was yielded, I keep the existing core model/training logic intact but default Kaggle to the deterministic EEG-feature baseline submission (valid probabilities summing to 1, correct columns, `submission.csv`) when training/weights aren’t available.'
- What this solution (achieved 1.67996) has done: 'I fix the TensorFlow/protobuf crash that happens at import time by forcing protobuf to use the pure-Python implementation before TensorFlow is imported (this avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle runtime). This is a runtime/stability fix and preserves the existing model/training/inference logic unchanged. With TensorFlow successfully available, the script can load provided fold weights (when present) and run the intended neural-network inference instead of falling back to the weak baseline, which should move the KL score down toward your target. I also keep the existing robust fallback path and submission probability sanitation to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 1.67996) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment override that triggers `MessageFactory.GetPrototype` errors in this Kaggle runtime. This enables TensorFlow to import cleanly so the existing neural-network inference path can run (when weights are present), which should move the KL score down toward your target compared with the weak baseline. I also keep the baseline fallback intact so a valid `submission.csv` is always produced even if TF or fold weights are unavailable. No model architecture, training loop, feature extraction, or loss semantics are changed.'

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

os.environ["KERAS_BACKEND"] = "tensorflow"

PLATFORM = "local"
cwd_parts = os.getcwd().split(os.sep)
if len(cwd_parts) > 1 and cwd_parts[1] == "home":
    PLATFORM = "local"  # local training
    if os.path.isdir("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name.lower().startswith("models"):
                LOAD_MODELS_FROM = dir_name
elif len(cwd_parts) > 1 and cwd_parts[1] == "kaggle":
    PLATFORM = "kaggle"  # kaggle inference by default
    NEEDTRAIN = False
    best = None
    try:
        for dir_name in os.listdir("/kaggle/input/"):
            candidate = os.path.join("/kaggle/input/", dir_name)
            if not os.path.isdir(candidate):
                continue
            files = set(os.listdir(candidate))
            if any(f.startswith("fold") and f.endswith(".weights.h5") for f in files):
                best = dir_name
                break
            models_sub = os.path.join(candidate, "models")
            if os.path.isdir(models_sub):
                subfiles = set(os.listdir(models_sub))
                if any(
                    f.startswith("fold") and f.endswith(".weights.h5") for f in subfiles
                ):
                    best = os.path.join(dir_name, "models")
                    break
    except FileNotFoundError:
        pass
    if best is not None:
        LOAD_MODELS_FROM = best

DATATYPE = ["eeg"]  # using EEG only (as in original)
print("DATATYPE:", DATATYPE)

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
LEARN_RATE = 1e-3 * BATCHSIZE / 16
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import warnings

warnings.filterwarnings("ignore")
import io
from PIL import Image
import pandas as pd, numpy as np

from scipy import signal
from scipy.ndimage import zoom
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

_BRAIN_PAIRS = [ch.split("-") for ch in BRAIN]
_NEEDED_COLS = sorted(set([c for ab in _BRAIN_PAIRS for c in ab]))

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
else:
    b, a = None, None

TF_AVAILABLE = False
tf = None
try:
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
    print("TF:", tf.version.VERSION)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    try:
        if PLATFORM == "kaggle":
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception as e:
        print("Thread config skipped:", e)

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism enable skipped:", e)

    MIX = True
    if MIX:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("Using full precision")

    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model

except Exception as e:
    print("TensorFlow import/config failed; will run baseline submission only.")
    print("TF error:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

TARGETS_RAW = [c + "_raw" for c in TARGETS]

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

y_data = train[TARGETS].values.astype(np.float32)
train[TARGETS_RAW] = y_data
y_sum = y_data.sum(axis=1, keepdims=True)
y_sum[y_sum == 0] = 1.0
train[TARGETS] = y_data / y_sum



## === cell 2
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"

    if PLATFORM == "local":
        datapath = "./" + os.path.join("input", "preprocess")
    else:
        datapath = "/kaggle/" + os.path.join("input", "preprocess")
    pre_eegs_path = os.path.join(datapath, "eegs.npy")

    if (not READ_EEG_FILES) and (not os.path.exists(pre_eegs_path)):
        print(
            "preprocess/eegs.npy not found -> enabling READ_EEG_FILES to build EEG cache."
        )
        READ_EEG_FILES = True

    if READ_EEG_FILES:
        time_start_time = time.time()

        brain_pairs = _BRAIN_PAIRS
        needed_cols = _NEEDED_COLS

        for i, eeg_id in enumerate(train.eeg_id.unique()):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")

            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet")),
                columns=needed_cols,
            )
            arr = eeg_default.to_numpy(dtype=np.float32, copy=False)

            cols = list(eeg_default.columns)
            col_pos = {c: k for k, c in enumerate(cols)}
            a_idx = np.fromiter((col_pos[p[0]] for p in brain_pairs), dtype=np.int64)
            b_idx = np.fromiter((col_pos[p[1]] for p in brain_pairs), dtype=np.int64)

            eeg = arr[:, a_idx].T - arr[:, b_idx].T
            np.nan_to_num(eeg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            if filter_range is not None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.array(eeg, dtype=np.float32, copy=False)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg

        if PLATFORM == "local":
            outdir = "./input/preprocess"
        else:
            outdir = "/kaggle/input/preprocess"
        if PLATFORM == "kaggle":
            outdir = "/kaggle/working/preprocess"

        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        if "eeg" in DATATYPE:
            np.save(os.path.join(outdir, "eegs.npy"), eegs, allow_pickle=True)
    else:
        if "eeg" in DATATYPE and os.path.exists(pre_eegs_path):
            eegs = np.load(pre_eegs_path, allow_pickle=True).item()



## === cell 3
if NEEDTRAIN:
    _group_keys = list(
        zip(
            df.eeg_id.values,
            df.seizure_vote.values,
            df.lpd_vote.values,
            df.gpd_vote.values,
            df.lrda_vote.values,
            df.grda_vote.values,
            df.other_vote.values,
        )
    )
    _df_group_map = {}
    _df_eeg_map = {}
    for idx, key in enumerate(_group_keys):
        _df_group_map.setdefault(key, []).append(idx)
        _df_eeg_map.setdefault(key[0], []).append(idx)
else:
    _df_group_map = {}
    _df_eeg_map = {}

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

            self.dataframe = dataframe.reset_index(drop=True)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.sample_weights = sample_weights
            self.mode = mode
            self.eegs = eegs if eegs is not None else {}
            self.stfts = stfts if stfts is not None else {}
            self.specs = specs if specs is not None else {}
            self.imgs = imgs if imgs is not None else {}

            self._eeg_id = self.dataframe["eeg_id"].to_numpy()
            if self.mode != "test":
                self._eeg_offset = self.dataframe.get(
                    "eeg_label_offset_seconds", pd.Series(np.zeros(len(self.dataframe)))
                ).to_numpy()
                self._y = self.dataframe[TARGETS].to_numpy(dtype=np.float32, copy=False)
                self._yraw = self.dataframe[TARGETS_RAW].to_numpy(
                    dtype=np.float32, copy=False
                )
                self._key_votes = self.dataframe[TARGETS_RAW].to_numpy(
                    dtype=np.float32, copy=False
                )
            else:
                self._eeg_offset = None
                self._y = None
                self._yraw = None
                self._key_votes = None

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
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            bs = len(indexes)
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        bs,
                        EEG_CHANNEL_USED * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )

            if self.mode == "test":
                y = np.zeros((bs, len(TARGETS)), dtype="float32")
                sample_weights = np.ones((bs, 1), dtype="float32")
            else:
                y = np.zeros((bs, len(TARGETS)), dtype="float32")
                sample_weights = np.zeros((bs, 1), dtype="float32")

            for j, i in enumerate(indexes):
                eeg_id = self._eeg_id[i]

                if self.mode == "test":
                    r_eeg = 0
                    picked_df_row = None
                else:
                    key = (
                        eeg_id,
                        float(self._key_votes[i, 0]),
                        float(self._key_votes[i, 1]),
                        float(self._key_votes[i, 2]),
                        float(self._key_votes[i, 3]),
                        float(self._key_votes[i, 4]),
                        float(self._key_votes[i, 5]),
                    )
                    idxs = _df_group_map.get(key)
                    if not idxs:
                        idxs = _df_eeg_map.get(eeg_id, [])

                    if self.mode == "train":
                        pick = idxs[np.random.randint(len(idxs))] if len(idxs) else None
                        if pick is not None:
                            picked_df_row = df.iloc[pick]
                            eeg_id_use = picked_df_row.eeg_id
                            r_eeg = picked_df_row.eeg_label_offset_seconds
                        else:
                            eeg_id_use = eeg_id
                            r_eeg = self._eeg_offset[i]
                    elif self.mode == "valid":
                        if len(idxs):
                            eeg_sub_ids = df.eeg_sub_id.values
                            sorted_idxs = sorted(idxs, key=lambda k: eeg_sub_ids[k])
                            picked_df_row = df.iloc[sorted_idxs[len(sorted_idxs) // 2]]
                            eeg_id_use = picked_df_row.eeg_id
                            r_eeg = picked_df_row.eeg_label_offset_seconds
                        else:
                            eeg_id_use = eeg_id
                            r_eeg = self._eeg_offset[i]
                    else:
                        eeg_id_use = eeg_id
                        r_eeg = self._eeg_offset[i]

                    if self.mode == "train":
                        r_eeg = r_eeg + np.random.random() * 10 - 5
                        r_eeg = max(0, r_eeg)
                        r_eeg = min(r_eeg, self.eegs[eeg_id_use].shape[1] / RSFREQ - 50)

                if "eeg" in DATATYPE:
                    if self.mode == "test":
                        eeg_id_use = eeg_id

                    eeg = self.eegs[eeg_id_use][
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
                        if np.random.rand() > 0.5:
                            eeg = -eeg
                        if np.random.rand() > 0.5:
                            eeg = eeg[:, ::-1]
                    else:
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                    eeg = np.clip(eeg, a_min=-255, a_max=255)
                    eeg = eeg + 255
                    eeg = eeg / 2
                    x_eeg[j] = eeg

                if self.mode != "test":
                    if picked_df_row is not None:
                        yy = picked_df_row[TARGETS].values.astype(np.float32)
                        yy_sum = float(np.sum(yy))
                        y[j] = yy / (yy_sum if yy_sum != 0 else 1.0)
                        if self.sample_weights:
                            sample_weights[j] = (
                                float(np.sum(picked_df_row[TARGETS_RAW].values)) / 20
                            )
                        else:
                            sample_weights[j] = 1.0
                    else:
                        yy = self._y[i]
                        yy_sum = float(np.sum(yy))
                        y[j] = yy / (yy_sum if yy_sum != 0 else 1.0)
                        if self.sample_weights:
                            sample_weights[j] = float(np.sum(self._yraw[i])) / 20
                        else:
                            sample_weights[j] = 1.0

            x = {}
            if "eeg" in DATATYPE:
                x["eeg"] = x_eeg
            return x, y, sample_weights




## === cell 4
if TF_AVAILABLE:

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




## === cell 5
if TF_AVAILABLE:

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
                if PLATFORM == "local":
                    base_model_eeg.load_weights(
                        f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                    )
                if PLATFORM == "kaggle":
                    base_model_eeg.load_weights(
                        f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                    )
            base_model_eeg.name = "eeg_extractor"

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

        model = tf.keras.Model(inputs=inp, outputs=y_eeg)
        return model




## === cell 6
if TF_AVAILABLE:

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
            )
            valid_gen_stage = DataGenerator(
                df_valid_stage1,
                shuffle=False,
                sample_weights=True,
                batch_size=BATCHSIZE * 2,
                mode="valid",
                eegs=eegs,
            )
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            callbacks_stage = [
                tf.keras.callbacks.LearningRateScheduler(
                    CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, LEARN_RATE * 0.01, 5)
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=True,
                    save_best_only=True,
                ),
            ]
        else:
            train_gen_stage = DataGenerator(
                df_train_stage2,
                shuffle=True,
                sample_weights=False,
                batch_size=BATCHSIZE,
                eegs=eegs,
            )
            valid_gen_stage = DataGenerator(
                df_valid_stage2,
                shuffle=False,
                sample_weights=False,
                batch_size=BATCHSIZE * 2,
                mode="valid",
                eegs=eegs,
            )
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
            callbacks_stage = [
                tf.keras.callbacks.LearningRateScheduler(
                    CosineAnnealingLRScheduler(
                        max(round(EPOCHS / 3), 1),
                        LEARN_RATE * 0.1,
                        LEARN_RATE * 0.1 * 0.1,
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
        else:
            history = model.fit(
                train_gen_stage,
                verbose=1,
                validation_data=valid_gen_stage,
                epochs=max(round(EPOCHS / 3), 1),
                callbacks=callbacks_stage,
            )

        model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

        predict_stage = model.predict(valid_gen_stage)

        del train_gen_stage, valid_gen_stage, history, model
        tf.keras.backend.clear_session()
        gc.collect()

        return predict_stage




## === cell 7
def _safe_probs(p, eps=1e-7):
    p = np.asarray(p, dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    p = np.clip(p, eps, None)
    s = p.sum(axis=1, keepdims=True)
    s[~np.isfinite(s)] = 1.0
    s[s == 0] = 1.0
    p = p / s
    return p.astype(np.float32)


from collections import OrderedDict


class _EEGLRUCache:
    def __init__(self, max_items=256):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


_TEST_EEG_CACHE = _EEGLRUCache(max_items=256)


def _load_one_test_eeg(eeg_id, path_test):
    eeg_id = int(eeg_id)
    cached = _TEST_EEG_CACHE.get(eeg_id)
    if cached is not None:
        return cached

    brain_pairs = _BRAIN_PAIRS
    needed_cols = _NEEDED_COLS

    eeg_default = pd.read_parquet(
        os.path.join(path_test, f"{eeg_id}.parquet"),
        columns=needed_cols,
    )
    arr = eeg_default.to_numpy(dtype=np.float32, copy=False)

    cols = list(eeg_default.columns)
    col_pos = {c: k for k, c in enumerate(cols)}
    a_idx = np.fromiter((col_pos[p[0]] for p in brain_pairs), dtype=np.int64)
    b_idx = np.fromiter((col_pos[p[1]] for p in brain_pairs), dtype=np.int64)

    eeg = arr[:, a_idx].T - arr[:, b_idx].T
    np.nan_to_num(eeg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if SFREQ != RSFREQ:
        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

    eeg = np.clip(eeg, a_min=-1024, a_max=1024)
    eegshape = eeg.shape[1]
    eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
    if filter_range is not None:
        eeg = signal.filtfilt(b, a, eeg, axis=1)
    eeg = eeg[:, eegshape : eegshape * 2]
    eeg = np.array(eeg, dtype=np.float32, copy=False)

    _TEST_EEG_CACHE.put(eeg_id, eeg)
    return eeg


def _baseline_predict_from_eeg(eeg_arr):
    x = np.asarray(eeg_arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    sig = np.mean(np.abs(x), axis=0)  # (n_samples,)
    sig = sig - np.mean(sig)
    sig = sig.astype(np.float32)

    freqs, psd = signal.welch(sig, fs=RSFREQ, nperseg=1024, noverlap=512)
    psd = np.maximum(psd, 1e-12)

    def band_power(f_lo, f_hi):
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(m):
            return 1e-12
        return float(np.trapz(psd[m], freqs[m]))

    p_delta = band_power(0.5, 4.0)
    p_theta = band_power(4.0, 8.0)
    p_alpha = band_power(8.0, 12.0)
    p_beta = band_power(12.0, 30.0)
    p_gamma = band_power(30.0, 45.0)

    total = p_delta + p_theta + p_alpha + p_beta + p_gamma
    if total <= 0:
        total = 1.0

    r_delta = p_delta / total
    r_beta_gamma = (p_beta + p_gamma) / total

    logits = np.array(
        [
            0.5 + 3.0 * r_beta_gamma,  # seizure
            0.3 + 1.0 * r_beta_gamma,  # lpd
            0.3 + 0.8 * r_beta_gamma,  # gpd
            0.4 + 1.8 * r_delta,  # lrda
            0.4 + 2.2 * r_delta,  # grda
            1.2,  # other (prior)
        ],
        dtype=np.float64,
    )

    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    return p.astype(np.float32)




## === cell 8
if TF_AVAILABLE:

    class TestEEGSequence(tf.keras.utils.Sequence):
        def __init__(self, eeg_ids, path_test, batch_size):
            self.eeg_ids = np.asarray(eeg_ids, dtype=np.int64)
            self.path_test = path_test
            self.batch_size = int(batch_size)
            self.half = round(EEG_CHANNEL_USED / 2)
            self.n_time = round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY)

            self._start = round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2)
            self._end = round((EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2)

        def __len__(self):
            return int(np.ceil(len(self.eeg_ids) / self.batch_size))

        def __getitem__(self, idx):
            s = idx * self.batch_size
            e = min(s + self.batch_size, len(self.eeg_ids))
            ids = self.eeg_ids[s:e]
            bs = e - s

            x_eeg = np.empty(
                (bs, EEG_CHANNEL_USED * EEG_MULTIPLY, self.n_time), dtype=np.float32
            )

            for j, eeg_id in enumerate(ids):
                eeg_full = _load_one_test_eeg(int(eeg_id), self.path_test)

                eeg = eeg_full[:, : round(50 * RSFREQ)]
                eeg = np.concatenate(
                    (eeg[0 : self.half, :], eeg[-self.half :, :]), axis=0
                )
                eeg = eeg[:, self._start : self._end]

                eeg2 = eeg.copy()
                eeg[4:8, :] = eeg2[12:16, :]
                eeg[8:12, :] = eeg2[4:8, :]
                eeg[12:16, :] = eeg2[8:12, :]

                eeg = np.clip(eeg, a_min=-255, a_max=255)
                eeg = (eeg + 255.0) / 2.0
                x_eeg[j] = eeg

            return {"eeg": x_eeg}




## === cell 9
if __name__ == "__main__":
    if NEEDTRAIN and (not TF_AVAILABLE):
        print(
            "TF not available -> forcing NEEDTRAIN=False and using baseline submission."
        )
        NEEDTRAIN = False

    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=SPLITS)

        for i, (train_index, valid_index) in enumerate(
            gkf.split(train, train.expert_consensus, train.patient_id)
        ):
            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[
                np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[
                np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)

            for stage in [1, 2]:
                _ = train_fold(
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
                )

        LOAD_MODELS_FROM = os.path.abspath("models")

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

    weight_paths = []
    for model_i in range(100):
        wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
        if os.path.exists(wpath):
            weight_paths.append(wpath)
    print("Found", len(weight_paths), "fold weight files")

    if (len(weight_paths) == 0) or (not TF_AVAILABLE):
        print("Using deterministic EEG-feature baseline for submission.")
        out = np.zeros((len(test), len(TARGETS)), dtype=np.float32)
        for i, eeg_id in enumerate(test.eeg_id.values):
            if i % 200 == 0:
                print(i, ", ", end="")
            eeg = _load_one_test_eeg(eeg_id, PATH_test)
            out[i] = _baseline_predict_from_eeg(eeg)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = _safe_probs(out)
        sub.to_csv("submission.csv", index=False)
        print("\nSubmission shape", sub.shape)
        print(sub.head())
    else:
        test_seq = TestEEGSequence(test.eeg_id.values, PATH_test, TEST_BATCHSIZE)

        model = build_model()

        n_test = len(test)
        preds_all = np.zeros((n_test, len(TARGETS)), dtype=np.float32)

        for wi, wpath in enumerate(weight_paths):
            print(
                f"Predicting with weights {wi+1}/{len(weight_paths)}: {os.path.basename(wpath)}"
            )
            model.load_weights(wpath)
            fold_pred = model.predict(test_seq, verbose=0)
            preds_all += fold_pred.astype(np.float32, copy=False)

        preds_all = preds_all / float(len(weight_paths))
        preds_all = _safe_probs(preds_all)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub.to_csv("submission.csv", index=False)
        print("\nSubmission shape", sub.shape)
        print(sub.head())
