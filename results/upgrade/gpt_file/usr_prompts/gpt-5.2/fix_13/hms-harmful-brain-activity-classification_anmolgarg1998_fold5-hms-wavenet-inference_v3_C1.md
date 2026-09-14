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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.7561172487850492

# 6. Current score

1.48867

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I fix the missing model weights issue by making inference robust: it try to load the expected fold checkpoints if they exist, otherwise it fall back to a valid, score-safe baseline prediction (the mean class distribution from train) so a submission is always produced. I also fix a couple of runtime/logic bugs in the training block (it incorrectly reuses `batch` inside the validation loop and forces `.cuda()`), while keeping training disabled by default as in your script. Finally, I ensure the submission probabilities are finite, clipped, and row-normalized to sum to one (required by the competition), and that the file is written as `submission.csv`.'
- What this solution (achieved 1.48867) has done: 'Your current score (1.48867, lower-is-better) suggests you’re likely submitting the “mean train distribution” fallback because no checkpoints are being found, which is far from the target (0.7561). The smallest score-improving change that preserves your core model/inference logic is to (1) automatically enable the existing training block when no usable checkpoints are present, (2) train just one fold (not all 5) to stay within the 600s runtime, and (3) then run inference from that saved checkpoint instead of the mean-distribution baseline. This keeps the same architecture, loss (KLDiv), preprocessing, and train loop semantics, but ensures the submission is model-based rather than a constant baseline. I also keep your probability sanitization/normalization as-is to guarantee a valid submission.'
- What this solution (achieved 1.48867) has done: 'Main bottlenecks are (1) per-sample parquet reads + heavy pandas slicing inside `__getitem__`, (2) re-designing the Butterworth filter for every single channel (extremely expensive), and (3) dataloader running single-process with no prefetch/pinned transfer. I keep the exact model/training semantics, but make feature extraction provably equivalent by caching the filter coefficients once and replacing pandas-per-column operations with a single NumPy extraction. I also add a bounded, deterministic in-memory cache for decoded EEG features (helps massively because the test loader is iterated up to 5 times for ensembling) and enable multi-worker DataLoader with persistent workers and prefetch. These changes reduce repeated work and Python overhead without altering the algorithm, targets, or evaluation behavior.'
- What this solution (achieved 1.48867) has done: 'Main bottlenecks are (1) per-sample Parquet reads + Pandas overhead during feature precompute, (2) repeated SciPy `lfilter` calls per channel in Python loops, and (3) expensive validation being run *twice* each epoch (mid-epoch + end-epoch). To fit under 600s without changing the model or training semantics, I (a) switch EEG reads to `pyarrow.parquet` with column projection and zero-copy NumPy conversion, (b) batch-filter all 16 channels at once with `scipy.signal.lfilter(axis=1)` and do downsampling via reshape/mean (exactly equivalent), and (c) keep the same “validate twice per epoch” logic but reuse the already-computed mid-epoch validation loss as the epoch-end validation loss (no change in checkpointing/evaluation semantics; just removes redundant work). I also increase DataLoader throughput safely (workers/prefetch) and ensure memmap precompute is faster via vectorized feature extraction.'
- What this solution (achieved 1.48867) has done: 'Your current score (1.48867, lower-is-better) is far from the target (0.7561), and the most likely cause is that inference is still falling back to the mean-distribution baseline because no usable checkpoints are found. The smallest score-improving change while preserving your model, loss (KLDiv), and overall training semantics is to reliably produce at least one checkpoint by (a) forcing training on fold 0 when no checkpoints exist, and (b) fixing a critical DataLoader bug: your training dataset currently defaults to `mode="Train"` which makes `__getitem__` return **test EEGs** (because it only checks `mode == "Test"`), so you were training on the wrong files. I also keep your probability sanitization/normalization unchanged and add a minimal `test_loader` speedup (safe multi-worker/pin/prefetch) to stay under the 600s limit without changing results. These changes should move the score down toward the target by ensuring you submit model-based predictions instead of a constant baseline.'
- What this solution (achieved 1.48867) has done: 'Your score is far above the target (1.48867 vs 0.7561; lower-is-better), and the most likely reason is that you are still submitting the mean-distribution fallback because no usable fold checkpoints are being found/loaded. The smallest change that should legitimately move you toward the target while preserving your exact model/loss/features is to (1) reliably train fold 0 when no checkpoints exist, and (2) fix a critical training/inference mismatch: your validation dataset was accidentally using `mode="Train"` (reading train EEG files) instead of `mode="Val"`/`mode="Train"` consistently, and the training loop was not moving inputs/labels to device for the backward pass robustly in all cases. I also ensure we always save and then load the fold-0 checkpoint from `/kaggle/working/wavenet_model/` so inference uses the trained model rather than the fallback. All other logic (feature extraction, architecture, KLDiv on log-probs, probability normalization) is kept unchanged.'
- What this solution (achieved 1.48867) has done: 'Your score (1.48867, lower-is-better) is far from the target (0.7561), and the most likely reason is that training is not producing a usable checkpoint because the validation dataset is accidentally reading **train** EEG files (`mode="Train"`) instead of a non-test mode that still reads train EEGs but keeps the split semantics clear. I make the smallest corrective change by setting the validation dataset `mode="Val"` (so it still reads from `train_eegs/`, but fixes the intended train/val separation logic) and ensure fold-0 checkpoint is always found/loaded after training. I also make inference prioritize the newly trained fold-0 checkpoint when training ran in this session, without changing the model, features, or loss. These changes should move the score down toward the target by ensuring you submit model-based predictions rather than effectively training/validating in a mismatched way or falling back to the mean baseline.'
- What this solution (achieved 1.48867) has done: 'Your current score is far worse than the target (1.48867 vs 0.7561; lower is better), and the most likely reason is that inference is effectively near-uniform/weak due to training not producing a strong checkpoint within time. To move the score down toward the target without changing the model or features, I (1) keep your exact architecture/feature extraction/loss, but add a tiny amount of label smoothing during training only (helps KL stability and usually improves generalization on this competition), (2) add lightweight EMA (exponential moving average) of model weights used only for evaluation/checkpointing to reduce variance without changing the training loop structure, and (3) ensure we always run fold-0 training if no checkpoints exist and then load that checkpoint for inference. These are minimal, metric-aligned changes (KL on probabilities) and should reduce the gap toward the target while staying within Kaggle constraints and producing a valid `submission.csv`.'
- What this solution (achieved 1.48867) has done: 'Your current score (1.48867, lower-is-better) is far above the target (0.7561), and the most likely reason is that the model is either not training long enough to learn useful patterns or the saved checkpoint is not the best-performing one. To move the score down toward the target without changing the model, features, or loss, I (1) remove the training-only label smoothing and EMA evaluation (these alter training semantics and can hurt/complicate convergence here), (2) enable mixed, deterministic validation each epoch (once) and save the true best checkpoint, and (3) ensure inference always prefers the freshly trained fold-0 checkpoint when training ran. These are minimal, metric-aligned fixes that keep the same architecture and KLDiv objective while making the produced checkpoint more reliable than the current setup, and they still write a valid `submission.csv` with row-normalized probabilities.'

# 9. Code solution

## === cell 0
import os, gc, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.model_selection import GroupKFold

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass



## === cell 1
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## === cell 2
g = df.groupby("eeg_id", sort=False)

train = g[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    spec_id=("spectrogram_id", "first"),
    min=("spectrogram_label_offset_seconds", "min"),
    max=("spectrogram_label_offset_seconds", "max"),
)

train["patient_id"] = g["patient_id"].first()

votes = g[list(TARGETS)].sum()
y_data = votes.values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[list(TARGETS)] = y_data

consensus = (
    g["expert_consensus"]
    .agg(lambda s: s.value_counts().idxmax())
    .to_frame("expert_consensus")
    .reset_index()
)
tmp2 = (
    df.groupby(["eeg_id", "expert_consensus"], sort=False)[["eeg_sub_id"]]
    .min()
    .reset_index()
)
tmp = pd.merge(consensus, tmp2, on=["eeg_id", "expert_consensus"], how="left")

train["target"] = tmp["expert_consensus"].values
train["eeg_sub_id"] = tmp["eeg_sub_id"].values

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head()



## === cell 3
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


_DENOISE_FS = 200.0
_DENOISE_LOWCUT = 1.0
_DENOISE_HIGHCUT = 25.0
_DENOISE_ORDER = 6
_DENOISE_BA = butter_bandpass(
    _DENOISE_LOWCUT, _DENOISE_HIGHCUT, _DENOISE_FS, order=_DENOISE_ORDER
)


def denoise_filter_batch(x2d: np.ndarray) -> np.ndarray:
    b, a = _DENOISE_BA
    y = lfilter(b, a, x2d, axis=1)
    n = y.shape[1]
    m = (n // 4) * 4
    y = y[:, :m].reshape(y.shape[0], m // 4, 4).mean(axis=2)
    return y.astype(np.float32, copy=False)


def denoise_filter(x: np.ndarray) -> np.ndarray:
    b, a = _DENOISE_BA
    y = lfilter(b, a, x)
    y0 = y[0:-1:4]
    y1 = y[1::4]
    y2 = y[2::4]
    y3 = y[3::4]
    return (y0 + y1 + y2 + y3).astype(np.float32, copy=False) * 0.25




## === cell 4
IS_TRAINING = False

train_eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
test_eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


def load_center_10k_from_parquet(
    parq_path: str, target_len: int = 10_000, columns=None
) -> pd.DataFrame:
    eeg = pd.read_parquet(parq_path, columns=columns)
    rows = len(eeg)
    if rows >= target_len:
        offset = max((rows - target_len) // 2, 0)
        eeg = eeg.iloc[offset : offset + target_len]
    else:
        pad = target_len - rows
        if rows == 0:
            raise ValueError(f"Empty EEG parquet: {parq_path}")
        last = eeg.iloc[[-1]].copy()
        eeg = pd.concat(
            [eeg, pd.concat([last] * pad, ignore_index=True)], ignore_index=True
        )
    return eeg


def load_center_10k_to_numpy(
    parq_path: str, target_len: int, columns: list[str]
) -> np.ndarray:
    if pq is None:
        eeg = load_center_10k_from_parquet(
            parq_path, target_len=target_len, columns=columns
        )
        return eeg[columns].to_numpy(dtype=np.float32, copy=False)

    table = pq.read_table(parq_path, columns=columns)
    rows = table.num_rows
    if rows == 0:
        raise ValueError(f"Empty EEG parquet: {parq_path}")
    if rows >= target_len:
        offset = max((rows - target_len) // 2, 0)
        table = table.slice(offset, target_len)
    else:
        pad = target_len - rows
        last = table.slice(rows - 1, 1)
        pads = [last] * pad
        table = table.append_chunks(pads)
    cols_np = []
    for c in columns:
        arr = table.column(c).to_numpy(zero_copy_only=False)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        cols_np.append(arr)
    return np.stack(cols_np, axis=1)


_ALL_FEAT_COLS = sorted(set([c for grp in FEATS for c in grp]))
_COL_INDEX = {c: i for i, c in enumerate(_ALL_FEAT_COLS)}
_FEAT_IDX = [[_COL_INDEX[c] for c in grp] for grp in FEATS]


def eeg_arr_to_16wave_features(arr: np.ndarray) -> np.ndarray:
    sigs = np.empty((16, arr.shape[0]), dtype=np.float32)
    t = 0
    for k in range(4):
        idxs = _FEAT_IDX[k]
        for j in range(4):
            sigs[t] = arr[:, idxs[j]] - arr[:, idxs[j + 1]]
            t += 1
    sigs = denoise_filter_batch(sigs)  # (16, 2500)
    return sigs


def eeg_to_16wave_features(eeg: pd.DataFrame) -> np.ndarray:
    arr = eeg[_ALL_FEAT_COLS].to_numpy(dtype=np.float32, copy=False)
    return eeg_arr_to_16wave_features(arr)




## === cell 5
class _LRUCache:
    def __init__(self, max_items: int = 4096):
        self.max_items = int(max_items)
        self._d = {}
        self._q = []

    def get(self, k):
        return self._d.get(k, None)

    def put(self, k, v):
        if k in self._d:
            return
        self._d[k] = v
        self._q.append(k)
        if len(self._q) > self.max_items:
            old = self._q.pop(0)
            self._d.pop(old, None)


class CustomDataset(Dataset):
    def __init__(
        self,
        dataframe,
        eegs_data,
        mode="Train",
        transform=None,
        cache_max_items=0,
        precomputed_x=None,
    ):
        self.dataframe = dataframe
        self.mode = mode
        self.eegs_data = eegs_data
        self.precomputed_x = precomputed_x
        self._cache = (
            _LRUCache(cache_max_items)
            if cache_max_items and cache_max_items > 0
            else None
        )

    def __len__(self):
        return len(self.dataframe)

    def _get_signal(self, eeg_id: int, idx: int) -> np.ndarray:
        if self.precomputed_x is not None:
            return self.precomputed_x[idx]

        if self._cache is not None:
            got = self._cache.get(eeg_id)
            if got is not None:
                return got

        if self.mode == "Test":
            parq_path = f"{test_eeg_path}{eeg_id}.parquet"
        else:
            parq_path = f"{train_eeg_path}{eeg_id}.parquet"

        arr = load_center_10k_to_numpy(
            parq_path, target_len=10_000, columns=_ALL_FEAT_COLS
        )
        signal_eeg = eeg_arr_to_16wave_features(arr)

        if self._cache is not None:
            self._cache.put(eeg_id, signal_eeg)
        return signal_eeg

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = int(row["eeg_id"])

        signal_eeg = self._get_signal(eeg_id, idx)

        if self.mode == "Test":
            return torch.from_numpy(signal_eeg)

        labels = row[TARGETS].values.astype(np.float32)
        s = np.sum(labels)
        if s <= 0:
            labels = np.ones_like(labels, dtype=np.float32) / len(labels)
        else:
            labels = labels / s

        return torch.from_numpy(signal_eeg), torch.from_numpy(
            labels.astype(np.float32, copy=False)
        )




## === cell 6
class wave_residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num):
        super(wave_residual_block, self).__init__()
        dilatn = 2 ** (layer_num - 1)
        self.dilatn = dilatn
        self.filter_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            stride=1,
            padding=dilatn,
            dilation=dilatn,
            bias=False,
        )
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            stride=1,
            padding=dilatn,
            dilation=dilatn,
            bias=False,
        )
        self.conv = nn.Conv1d(out_channels, out_channels, 1, 1)

    def forward(self, x):
        y = F.tanh(self.filter_conv(x)) * F.sigmoid(self.gate_conv(x))
        y = y[:, :, : -self.dilatn]
        y = self.conv(y)
        x = x + y
        return x, y


class WaveBlock(nn.Module):
    def __init__(self, in_channels=16, num_layers=6):
        super(WaveBlock, self).__init__()
        self.waveblock_0 = wave_residual_block(16, 16, 1)
        self.waveblocks = nn.ModuleList(
            [wave_residual_block(16, 16, i) for i in range(2, num_layers + 1)]
        )

        self.conv = nn.Conv1d(in_channels, 16, 1, 1)
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(16, 48, 20, 10)
        self.conv2 = nn.Conv1d(48, 32, 10, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        skip_connections = []
        x, y = self.waveblock_0(x)
        skip_connections.append(y)
        for i in range(self.num_layers - 1):
            x, y = self.waveblocks[i](x)
        return x


class WaveClassifier(nn.Module):
    def __init__(self, in_channels=16, num_layers=6):
        super(WaveClassifier, self).__init__()
        self.waveblocks = nn.ModuleList([WaveBlock(4, i) for i in [8, 6, 4, 1]])

        self.conv = nn.Conv1d(in_channels, 16, 1, 1)
        self.num_layers = num_layers

        self.conv0 = nn.Conv1d(64, 16, 1)
        self.conv1 = nn.Conv1d(16, 48, 20, 10)
        self.conv2 = nn.Conv1d(48, 32, 10, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        x = []
        for i in range(4):
            x.append(self.waveblocks[i](inp[:, i : i + 4]))

        x = torch.concat(x, dim=1)

        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.tanh(self.flatten(x))
        x = self.fc1(x)
        x = self.softmax(x)
        return x




## === cell 7
if not os.path.exists("wavenet_model"):
    os.makedirs("wavenet_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 8
def find_checkpoint_path(fold: int) -> str | None:
    candidates = [
        f"/kaggle/input/wavenet-arch2/model_best_fold_{fold}.pt",
        f"/kaggle/working/wavenet_model/model_best_fold_{fold}.pt",
        f"wavenet_model/model_best_fold_{fold}.pt",
        f"./wavenet_model/model_best_fold_{fold}.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


any_ckpt = any(find_checkpoint_path(i) is not None for i in range(5))
if not any_ckpt:
    IS_TRAINING = True
print("Any checkpoint found:", any_ckpt, "| IS_TRAINING:", IS_TRAINING)



## === cell 9
_LOG_EPS = 1e-8

if IS_TRAINING:
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train, train.target, train.patient_id))

    i = 0
    train_index, valid_index = folds[i]

    dataset_train = CustomDataset(
        dataframe=train.iloc[train_index].reset_index(drop=True),
        eegs_data=None,
        mode="Train",
        cache_max_items=1024,
    )
    dataset_val = CustomDataset(
        dataframe=train.iloc[valid_index].reset_index(drop=True),
        eegs_data=None,
        mode="Val",
        cache_max_items=1024,
    )

    nw = min(8, max(2, (os.cpu_count() or 2) // 2))
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=32,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )

    min_val_loss = float("inf")
    epochs = 12
    our_model = WaveClassifier().to(device)

    optimizer = optim.AdamW(our_model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.KLDivLoss(reduction="batchmean").to(device)

    for epoch in range(epochs):
        our_model.train()
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar, start=1):
            inp1, label = batch
            inp1 = inp1.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = our_model(inp1)
            loss = criterion(torch.log(pred + _LOG_EPS), label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch} | batch loss {loss.item():.4f}")

        our_model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for inp1v, labelv in val_loader:
                inp1v = inp1v.to(device, non_blocking=True)
                labelv = labelv.to(device, non_blocking=True)
                predv = our_model(inp1v)
                lossv = criterion(torch.log(predv + _LOG_EPS), labelv)
                val_loss += lossv.item() * inp1v.size(0)
                n += inp1v.size(0)
        val_loss /= max(n, 1)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(our_model.state_dict(), f"wavenet_model/model_best_fold_{i}.pt")
            print("Saved best checkpoint fold,epoch,val loss:", i, epoch, val_loss)

        print("fold,epoch,val loss:", i, epoch, val_loss)

    del our_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



## === cell 10
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape:", test.shape)
test.head()




## === cell 11
def build_or_load_test_memmap(
    test_df: pd.DataFrame, mm_path: str, shape=(9850, 16, 2500), dtype=np.float32
):
    if os.path.exists(mm_path):
        mm = np.memmap(mm_path, mode="r", dtype=dtype, shape=shape)
        return mm

    t0 = time.time()
    mm = np.memmap(mm_path, mode="w+", dtype=dtype, shape=shape)
    eeg_ids = test_df["eeg_id"].values
    for i, eeg_id in enumerate(tqdm(eeg_ids, desc="Precomputing test features")):
        parq_path = f"{test_eeg_path}{int(eeg_id)}.parquet"
        arr = load_center_10k_to_numpy(
            parq_path, target_len=10_000, columns=_ALL_FEAT_COLS
        )
        mm[i] = eeg_arr_to_16wave_features(arr)
    mm.flush()
    print(f"Built memmap {mm_path} in {time.time()-t0:.1f}s")
    mm = np.memmap(mm_path, mode="r", dtype=dtype, shape=shape)
    return mm


TEST_MM_PATH = "test_features_16x2500_float32.memmap"
test_x_mm = build_or_load_test_memmap(
    test, TEST_MM_PATH, shape=(len(test), 16, 2500), dtype=np.float32
)

dataset_test = CustomDataset(
    dataframe=test, mode="Test", eegs_data=None, precomputed_x=test_x_mm
)

nw_test = min(4, max(1, (os.cpu_count() or 2) // 2))
test_loader = DataLoader(
    dataset_test,
    batch_size=64,
    shuffle=False,
    num_workers=nw_test,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw_test > 0),
    prefetch_factor=4 if nw_test > 0 else None,
)



## === cell 12
our_model = WaveClassifier().float().to(device)

preds_all_fold = []
loaded_folds = 0

fold_order = [0, 1, 2, 3, 4]
for i in fold_order:
    ckpt = find_checkpoint_path(i)
    if ckpt is None:
        continue
    state = torch.load(ckpt, map_location=device)
    our_model.load_state_dict(state)
    our_model.eval()

    preds = []
    with torch.inference_mode():
        for batch in test_loader:
            inp1 = batch.to(device, non_blocking=True)
            pred = our_model(inp1)
            preds.append(pred.detach().cpu().numpy())
    preds = np.vstack(preds)
    preds_all_fold.append(preds)
    loaded_folds += 1

if loaded_folds > 0:
    prediction_all_fold = np.mean(preds_all_fold, axis=0)
    print(f"Loaded {loaded_folds} fold(s) checkpoints. Using ensemble mean.")
else:
    base = train[TARGETS].mean(axis=0).values.astype(np.float32)
    base = base / base.sum()
    prediction_all_fold = np.tile(base, (len(test), 1))
    print("No checkpoints found. Falling back to mean train distribution baseline.")



## === cell 13
pred = np.asarray(prediction_all_fold, dtype=np.float64)
pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
pred = np.clip(pred, 1e-8, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = pred

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print("Sub row 0 sums to:", sub.iloc[0, -6:].sum())
sub.head()
