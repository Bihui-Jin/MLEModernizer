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

0.2846556071065502

# 6. Current score

1.87934

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash caused by an incompatible `torchaudio/torch` import in this Kaggle/Python 3.13 environment by making those imports optional and only required when `DATATYPE` actually includes `stft`. I also fix a couple of Kaggle-inference path/availability issues that prevent producing a submission: (1) the code currently expects preprocessed `eegs.npy` (not available on Kaggle) even though it can read raw parquet test EEGs, and (2) it may fail to find model weights and still proceed. Finally, I ensure the script always writes a valid `submission.csv` with correct columns and row-wise probabilities summing to 1 (with a safe normalization/clip), so you always get a valid submission file end-to-end.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before TensorFlow/Keras imports, which is a common incompatibility in Python 3.13 Kaggle images. I also ensure inference always has the correct `TARGETS` list even when `NEEDTRAIN=False` by reading it from `sample_submission.csv`, preventing accidental column mismatch and helping score versus uniform fallback. Finally, I make model-weight discovery robust (avoid the dummy `modelsxxxxxxx` path causing “no models found”) by defaulting to a local `models/` directory if present, and I keep the submission probabilities strictly normalized and clipped to valid ranges.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the TensorFlow/protobuf dependency entirely and switching the script to PyTorch (which is available on Kaggle) while keeping the same core “EEG-only EfficientNet-like image backbone” semantics via a simple CNN over the same reshaped EEG “image” representation. I also fix inference-time issues that currently cause a fallback to uniform predictions (missing weights, missing preprocess `.npy`) by training on-the-fly on Kaggle using a patient-group split and then ensembling folds for test predictions. Finally, I ensure the submission matches `sample_submission.csv` column order exactly and that every row is strictly normalized and clipped so it passes Kaggle validation and improves KL divergence toward your target.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by repeatedly reading/parsing thousands of small Parquet files inside `__getitem__` during both training and test inference. The fastest equivalent fix is to cache preprocessed EEG arrays to disk once (as `.npy`) and reuse them across DataLoader workers and across folds, while also limiting Parquet reads to only the required columns. Additionally, we keep the exact same model, loss, epochs, folds, and evaluation semantics, but reduce DataLoader overhead by using `persistent_workers`, `prefetch_factor`, and a faster collation path with contiguous arrays. These changes preserve core logic and accuracy (same preprocessing and tensors), but drastically cut I/O and Python overhead so the pipeline fits in 600 seconds.'
- What this solution (achieved 2.28232) has done: 'I fix the crash caused by concurrent cache writes from multiple DataLoader workers by making cache writes atomic per `eeg_id` (unique temp filename + safe replace, and fall back to recompute if another worker wins). I also make the DataLoader options robust to environments where `persistent_workers/prefetch_factor` can error with `num_workers=0`, and ensure the script always proceeds to inference even if training fails partway. These changes are score-neutral in intent (they preserve the same preprocessing/model/loss), but they unblock full training so you no longer fall back to uniform predictions, which should move KL down toward your target. The submission writing/normalization remains unchanged and valid.'
- What this solution (achieved 2.17492) has done: 'Your score is far above the target (lower-is-better), so the smallest meaningful improvement is to fix a training/inference mismatch that hurts KL: you train on consolidated (unique) EEG-level labels but your test set is EEG-level too, yet your preprocessing implicitly ignores label offsets and your per-sample weight `sw` can become very small/large in a way that destabilizes optimization. I (1) compute `sw` directly from the true vote count per row (sum of raw votes) with a safe lower bound, (2) use the same weighted KL in validation as in training (so “best model” selection matches the training objective), and (3) slightly reduce over-regularization by lowering dropout from 0.5→0.3 (minimal architecture change, same model family) which should move KL down toward your target without changing the pipeline. Submission writing/normalization stays the same and remains strictly valid.'
- What this solution (achieved 1.87934) has done: 'Your current KL (2.17492, lower-is-better) is far from the target (0.2847), so the smallest safe way to move toward the target is to fix an evaluation-semantic mismatch: the competition metric is EEG-level, but your training table is built by “drop_duplicates”, which can still keep multiple rows per `eeg_id` and creates noisy/conflicting supervision. I minimally change the consolidation to aggregate all rows per `eeg_id` into a single target distribution by summing votes (then normalizing), and I derive `sw` from the aggregated total votes (more stable weighting). Everything else (EEG reading, pooling, CNN, KL loss, folds, epochs, submission formatting/normalization) stays the same so runtime and core logic are preserved while KL should drop substantially toward your target.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import gc
import random
import warnings

warnings.filterwarnings("ignore")

NEEDTRAIN = True  # train the model (we enable on Kaggle to avoid "no weights found" uniform fallback)
LOAD_MODELS_FROM = "models"  # kept for compatibility; we write/read local models/

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = True
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    PLATFORM = "unknown"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img ***
print("DATATYPE:", DATATYPE, "PLATFORM:", PLATFORM)

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 3
SPLITS = 3

TEST_BATCHSIZE = 64

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

import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "device:", device)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
_ss_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
_ss = pd.read_csv(_ss_path, nrows=5)
TARGETS = list(_ss.columns[1:])  # exclude eeg_id
print("Train shape:", df.shape)
print("Targets:", TARGETS)

TARGETS_RAW = [t + "_raw" for t in TARGETS]

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

CACHE_DIR = (
    os.path.join("/kaggle/working", "eeg_cache_npy")
    if PLATFORM == "kaggle"
    else os.path.join(".", "eeg_cache_npy")
)
os.makedirs(CACHE_DIR, exist_ok=True)




## === cell 1
def make_train_table(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Score-relevant fix (minimal, semantic): Kaggle evaluation is per eeg_id in test.
    The previous drop_duplicates() can keep multiple rows per eeg_id (different votes),
    which creates inconsistent supervision and hurts KL.
    We aggregate to a single row per eeg_id by summing votes across all overlapping
    labeled segments, then normalize to a probability distribution.
    """
    use_cols = ["eeg_id", "patient_id"] + TARGETS
    d = df_in[use_cols].copy()

    agg_votes = d.groupby("eeg_id", as_index=False)[TARGETS].sum()
    agg_pid = d.groupby("eeg_id", as_index=False)["patient_id"].first()
    train = agg_votes.merge(agg_pid, on="eeg_id", how="left")

    train = train.reset_index(drop=True)
    train["sign_id"] = train.index.values

    y_data = train[TARGETS].values.astype(np.float32)
    train[TARGETS_RAW] = y_data

    y_prob = y_data / np.clip(y_data.sum(axis=1, keepdims=True), 1e-6, None)
    train[TARGETS] = y_prob.astype(np.float32)
    return train


train = make_train_table(df)
print("Consolidated train (eeg-level):", train.shape)


def _fix_length_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Bugfix: some parquet EEGs are not exactly 50s long, causing variable tensor widths after pooling
    and DataLoader stack() failures. Enforce a deterministic fixed window by center-cropping or
    padding with edge values (stable, minimal distortion).
    """
    cur = x.shape[-1]
    if cur == target_len:
        return x
    if cur > target_len:
        start = (cur - target_len) // 2
        return x[..., start : start + target_len]
    pad_total = target_len - cur
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    if cur == 0:
        return np.zeros((x.shape[0], target_len), dtype=x.dtype)
    return np.pad(x, ((0, 0), (pad_left, pad_right)), mode="edge")


_BRAIN_COLS = sorted({c for ch in BRAIN for c in ch.split("-")})


def _eeg_cache_path(eeg_id: int, base_path: str) -> str:
    tag = (
        "train"
        if "train_eegs" in base_path
        else ("test" if "test_eegs" in base_path else "eeg")
    )
    return os.path.join(CACHE_DIR, f"{tag}_{int(eeg_id)}.npy")


def _atomic_save_npy(dst_path: str, arr: np.ndarray) -> None:
    """
    Bugfix for FileNotFoundError in multi-worker DataLoader:
    - use a unique temp filename per process to avoid collisions
    - os.replace for atomic rename
    - if another worker wins the race, we accept and proceed
    """
    tmp = f"{dst_path}.tmp_pid{os.getpid()}"
    np.save(tmp, arr)
    if not tmp.endswith(".npy"):
        tmp_npy = tmp + ".npy"
    else:
        tmp_npy = tmp
    try:
        os.replace(tmp_npy, dst_path)
    except FileNotFoundError:
        if not os.path.exists(dst_path):
            raise
    finally:
        try:
            if os.path.exists(tmp_npy):
                os.remove(tmp_npy)
        except Exception:
            pass


def read_eeg_parquet(eeg_id: int, base_path: str) -> np.ndarray:
    """Return (16, 10000) float32 EEG after montage + filter + clipping."""
    cache_p = _eeg_cache_path(eeg_id, base_path)
    if os.path.exists(cache_p):
        return np.load(cache_p, mmap_mode="r")

    p = os.path.join(base_path, f"{int(eeg_id)}.parquet")
    eeg_default = pd.read_parquet(p, columns=_BRAIN_COLS)

    eeg = []
    for channel in BRAIN:
        a0, a1 = channel.split("-")
        eeg_temp = (eeg_default.loc[:, a0] - eeg_default.loc[:, a1]).to_numpy(
            copy=False
        )
        eeg_temp = np.nan_to_num(eeg_temp, nan=0.0).astype(np.float32, copy=False)
        eeg.append(eeg_temp[None, :])
    eeg = np.concatenate(eeg, axis=0)  # (16, T)

    if SFREQ != RSFREQ:
        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1).astype(np.float32)

    fixed_len = int(RSFREQ * EEG_LENGTH_USED)
    eeg = _fix_length_1d(eeg.astype(np.float32, copy=False), fixed_len).astype(
        np.float32, copy=False
    )

    eeg = np.clip(eeg, a_min=-1024, a_max=1024).astype(np.float32, copy=False)
    if filter_range is not None:
        eeg = signal.filtfilt(b, a, eeg, axis=1).astype(np.float32, copy=False)
    eeg = np.clip(eeg, a_min=-255, a_max=255).astype(np.float32, copy=False)

    eeg = (eeg + 255.0) / 2.0
    eeg = eeg.astype(np.float32, copy=False)

    try:
        _atomic_save_npy(cache_p, eeg)
    except Exception:
        pass

    if os.path.exists(cache_p):
        return np.load(cache_p, mmap_mode="r")
    return eeg




## === cell 2
class EEGDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, eeg_dir: str, mode: str = "train"):
        self.df = meta_df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = int(row.eeg_id)
        x = read_eeg_parquet(
            eeg_id, self.eeg_dir
        )  # (16, 10000) float32, possibly memmap

        x = np.concatenate([x[0:8], x[-8:]], axis=0)  # (16, 10000)

        x = np.ascontiguousarray(x)
        x_t = torch.from_numpy(x).float().unsqueeze(0)  # (1,16,10000)
        x_t = F.avg_pool1d(x_t, kernel_size=10, stride=10)  # (1,16,1000)

        if self.mode != "test":
            y = torch.tensor(row[TARGETS].values.astype(np.float32))
            vote_count = float(np.sum(row[TARGETS_RAW].values))
            sw = max(vote_count / 20.0, 0.1)
            return x_t, y, torch.tensor([sw], dtype=torch.float32)
        else:
            return (
                x_t,
                torch.zeros(len(TARGETS), dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
            )


class SimpleEEGCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))  # W/2
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))  # W/4
        x = F.silu(self.bn3(self.conv3(x)))
        x = x.mean(dim=(2, 3))  # GAP over H,W -> (B,128)
        x = self.drop(x)
        x = self.head(x)  # logits
        return x


def kl_div_loss_per_sample(p_logits, target_probs, eps=1e-7):
    """
    Return per-sample KL so we can apply per-sample weights correctly.
    Core loss remains KLDiv between target distribution and predicted softmax.
    """
    log_p = F.log_softmax(p_logits, dim=1)
    t = torch.clamp(target_probs, eps, 1.0)
    t = t / t.sum(dim=1, keepdim=True)
    return F.kl_div(log_p, t, reduction="none").sum(dim=1)




## === cell 3
from sklearn.model_selection import GroupKFold


def _make_loader(ds, batch_size, shuffle):
    num_workers = 2 if PLATFORM == "kaggle" else 2
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    return DataLoader(ds, **kwargs)


def train_one_fold(fold, tr_idx, va_idx, eeg_dir):
    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)

    tr_ds = EEGDataset(tr_df, eeg_dir=eeg_dir, mode="train")
    va_ds = EEGDataset(va_df, eeg_dir=eeg_dir, mode="valid")

    tr_loader = _make_loader(tr_ds, batch_size=BATCHSIZE, shuffle=True)
    va_loader = _make_loader(va_ds, batch_size=BATCHSIZE * 2, shuffle=False)

    model = SimpleEEGCNN(n_classes=len(TARGETS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)

    best = 1e9
    best_path = os.path.join("models", f"fold{fold}.pt")
    for epoch in range(EPOCHS):
        model.train()
        tr_losses = []
        for xb, yb, sw in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True).view(-1)  # (B,)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            per_sample = kl_div_loss_per_sample(logits, yb)  # (B,)
            loss = (per_sample * sw).sum() / torch.clamp(sw.sum(), min=1e-6)

            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb, sw in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                sw = sw.to(device, non_blocking=True).view(-1)
                logits = model(xb)
                per_sample = kl_div_loss_per_sample(logits, yb)
                loss = (per_sample * sw).sum() / torch.clamp(sw.sum(), min=1e-6)
                va_losses.append(loss.item())

        tr_l = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_l = float(np.mean(va_losses)) if va_losses else float("nan")
        print(f"Fold {fold} epoch {epoch+1}/{EPOCHS} loss {tr_l:.4f} val {va_l:.4f}")

        if va_l < best:
            best = va_l
            torch.save({"model": model.state_dict()}, best_path)

    del tr_ds, va_ds, tr_loader, va_loader, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_path


if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    eeg_dir_train = os.path.join(LOAD_DATA_FROM, "train_eegs")
    gkf = GroupKFold(n_splits=SPLITS)

    fold_paths = []
    if NEEDTRAIN:
        for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(train, groups=train.patient_id)
        ):
            try:
                p = train_one_fold(fold, tr_idx, va_idx, eeg_dir_train)
                fold_paths.append(p)
            except Exception as e:
                print(f"[WARN] Fold {fold} training failed: {repr(e)}")
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print("Saved fold models:", fold_paths)




## === cell 4
def predict_test(test_df: pd.DataFrame, eeg_dir: str, weight_paths):
    test_ds = EEGDataset(test_df, eeg_dir=eeg_dir, mode="test")

    num_workers = 2 if PLATFORM == "kaggle" else 2
    dl_kwargs = dict(
        batch_size=TEST_BATCHSIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    test_loader = DataLoader(test_ds, **dl_kwargs)

    models = []
    for wp in weight_paths:
        m = SimpleEEGCNN(n_classes=len(TARGETS)).to(device)
        ckpt = torch.load(wp, map_location=device)
        m.load_state_dict(ckpt["model"], strict=True)
        m.eval()
        models.append(m)

    all_preds = []
    with torch.no_grad():
        for xb, _, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits_ens = None
            for m in models:
                lg = m(xb)
                logits_ens = lg if logits_ens is None else (logits_ens + lg)
            logits_ens = logits_ens / max(len(models), 1)
            prob = F.softmax(logits_ens, dim=1).detach().cpu().numpy()
            all_preds.append(prob)

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    preds = np.clip(preds, 1e-7, 1.0)
    preds = preds / np.sum(preds, axis=1, keepdims=True)
    return preds


if __name__ == "__main__":
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    eeg_dir_test = os.path.join(LOAD_DATA_FROM, "test_eegs")

    weight_paths = []
    for fold in range(SPLITS):
        p = os.path.join("models", f"fold{fold}.pt")
        if os.path.exists(p):
            weight_paths.append(p)

    if not weight_paths:
        preds = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
    else:
        preds = predict_test(test, eeg_dir_test, weight_paths)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    for i, c in enumerate(TARGETS):
        sub[c] = preds[:, i]

    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.clip(vals, 1e-7, 1.0)
    vals = vals / np.sum(vals, axis=1, keepdims=True)
    sub[TARGETS] = vals.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", sub.shape)
    print(sub.head())
