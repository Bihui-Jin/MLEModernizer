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

0.7851158529294608

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.98636) has done: 'I fix the dtype mismatch causing the Conv1d runtime error by ensuring the CNN layers (and the dummy tensor used to infer flatten size) are created in float64 to match your dataset tensors, without changing the model architecture or training loop. I also make sure the model directory exists before saving/loading and add a safe fallback path so inference can still run even if training was skipped due to an existing model file mismatch. Finally, I ensure `preds` is always defined before building the submission and that the submission probabilities are normalized per row to sum to 1, producing a valid `submission.csv`.'
- What this solution (achieved 1.09128) has done: 'Your current score (0.98636, lower-is-better) is worse than the target (0.7851), so we should cautiously improve without changing the model architecture or training loop structure. The biggest minimal-impact issue is that you are training on *row-level overlapping windows* with a sampler, which over-represents the same `eeg_id` many times and hurts generalization; we keep the same dataset/model but collapse the training dataframe to one row per `eeg_id` by averaging the vote targets and using a stable offset selection, reducing leakage/duplication. We also fix the class weighting to be based on the *soft targets* (vote distributions) rather than `expert_consensus`, which is closer to the KL metric and typically improves KL without changing loss/outputs. Finally, we keep all inference/submission logic the same, still normalizing rows to sum to 1.'
- What this solution (achieved 1.09133) has done: 'Your current score (1.09128, lower-is-better) is worse than the target (0.7851), so we should make a small change that tends to reduce KL without changing the model/training loop. The biggest low-risk issue is that you apply `Softmax` inside the model and then feed `KLDivLoss(log(pred), target)`, which is numerically less stable than using logits + `log_softmax`; keeping the exact same architecture, we remove the final softmax from `forward()` and compute KL with `log_softmax` in the training loop. At inference we convert logits to probabilities with `softmax` and keep the existing per-row normalization to guarantee valid submissions. This preserves your feature extraction, conv/linear layers, optimizer, and overall training approach while typically improving KL due to better numerical behavior.'
- What this solution (achieved 1.09133) has done: 'Your score (1.09133, lower-is-better) is worse than the target (0.7851), so we should make a small, low-risk improvement that better matches the KL metric without changing your model or training loop structure. The most direct issue is that `KLDivLoss` expects the *target* to be a proper probability distribution; currently you average per-eeg soft labels but don’t renormalize them before training, which can inflate KL and harm calibration. I add a strict renormalization of the aggregated `train[TARGETS]` after collapsing (and assert finite, nonnegative rows), keeping everything else the same. This should reduce KL modestly and safely while preserving your architecture, feature extraction, optimizer, epochs, and inference/submission logic.'
- What this solution (achieved 1.10914) has done: 'Your current KL (1.09133, lower-is-better) is still far above the target (0.7851), so we make a minimal change that better matches the competition’s “one prediction per eeg_id” requirement without altering your model, loss, or training loop. Right now you train on one window per eeg_id (good), but at inference you also take only the center 10,000 samples; a low-risk improvement is test-time augmentation by averaging predictions from a few deterministic offsets (start/center/end), which typically reduces KL via more stable probabilities. This preserves the same feature extraction, CNN, and KLDiv training semantics, and only changes how test predictions are aggregated. We also keep strict row-wise normalization to guarantee valid submissions.'
- What this solution (achieved 1.18877) has done: 'Your current KL (1.10914, lower-is-better) is still far above the target (0.7851), so we should make a minimal change that usually improves KL without changing your model, loss, or training loop. The most impactful low-risk issue here is the `WeightedRandomSampler`: for a KL-on-soft-labels task it often worsens calibration and overemphasizes rare patterns, increasing KL on the private test. I keep the exact same dataset, feature extraction, CNN, epochs, optimizer, and KLDivLoss, but switch training to standard deterministic shuffling (no reweighting) to better match the real test distribution. Everything else (including your TTA averaging and strict row-normalization in the submission) remains unchanged so it still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.19937) has done: 'We need to move KL down (lower-is-better) from 1.18877 toward 0.7851, so we should make a small change that improves generalization/calibration without changing your CNN, features, optimizer, epochs, or loss semantics. The most likely minimal-impact issue is that when you collapse to one row per `eeg_id`, you take the **median** label offset, which can land between annotated windows and produce mismatched labels vs the extracted 10s EEG segment; switching to a stable *existing* annotated segment (e.g., the first available offset per `eeg_id`) better aligns inputs/targets. I keep everything else identical, only changing the aggregation for `eeg_label_offset_seconds` to `"first"` after sorting, which preserves “one row per eeg_id” and should reduce KL. Submission writing and probability normalization remain unchanged.'
- What this solution (achieved 1.20133) has done: 'Your current KL (1.19937, lower-is-better) is still far above the target (0.7851), so we should make the smallest change that plausibly improves generalization without changing your CNN, features, optimizer, epochs, or KL loss. The most direct low-risk issue is a train/test mismatch: you train on a specific labeled 10s window chosen by `eeg_label_offset_seconds`, but your test-time TTA uses offsets based on file length (start/center/end) rather than the same “labeled-window” logic. I keep your existing 3-offset TTA, but redefine those offsets to be deterministic slices within the *known 50s test window* (0s/20s/40s), which better matches training’s 10s-window extraction and typically lowers KL. Everything else (data reading, denoise, model, training loop, submission normalization) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.08062) has done: 'Your current KL (1.20133, lower-is-better) is still far above the target (0.7851), so we should make a minimal, low-risk change that tends to improve generalization without changing your CNN, feature extraction, or KL training semantics. The most direct issue is that you train for only 1 epoch, which typically underfits badly on this task; increasing to a small fixed number of epochs (e.g., 3) preserves the same training loop and loss, just runs it a bit longer. To keep this stable and deterministic, I keep the same seed/determinism settings and also reuse the already-built `model` for inference (instead of re-instantiating a new model each TTA pass), which avoids any accidental mismatch and reduces overhead without changing outputs. Everything else (data reading, windowing, 3-offset TTA at 0/20/40s, and strict per-row probability normalization) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 1.08062) has done: 'We need to move KL down (lower-is-better) from 1.08062 toward 0.7851, so the smallest safe improvement is to reduce a major train/test mismatch without changing your CNN, features, loss, or training loop structure. Right now you train on label offsets that can start near the beginning of the EEG, while test-time uses fixed 0/20/40s slices; we make training use the same 0/20/40s “phase” by snapping each training offset to the nearest of {0, 20, 40} seconds (then clipping so the 10s window fits), while keeping one-row-per-eeg aggregation and label averaging unchanged. This preserves your model architecture and KLDiv setup, but usually improves generalization because train and test see similarly-positioned windows. I also keep everything deterministic and still write a valid `submission.csv` with strictly normalized probabilities.'
- What this solution (achieved 0.99229) has done: 'Your KL is worse than the target, so we should make a small change that improves generalization/calibration without changing your CNN, features, loss, or training loop structure. The most direct low-risk issue is that you currently train on only one fixed 10s window per `eeg_id`, while your inference averages 3 windows (0/20/40s); this train/test mismatch can inflate KL. I keep the exact same dataset class and windowing logic, but expand the training dataframe to include the same three snapped offsets per `eeg_id` (duplicating each row 3x with identical soft targets), so training sees the same offset distribution as test-time TTA. Everything else (architecture, KLDivLoss on log_softmax, epochs, optimizer, inference averaging, and strict row-normalization) stays the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, gc

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass



## === cell 1
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
print("Train shape", train.shape)
train.head()



## === cell 2
TARGETS = train.columns[-6:]
TARGETS = list(TARGETS)
TARGETS



## === cell 3
_FS = 200.0
_LOWCUT = 1.0
_HIGHCUT = 25.0
_ORDER = 6
_BA = butter(_ORDER, [_LOWCUT, _HIGHCUT], fs=_FS, btype="band")


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def denoise_filter(x):
    b, a = _BA
    y = lfilter(b, a, x)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y




## === cell 4
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"

_NEED_COLS = sorted({c for cols in FEATS for c in cols})


def _snap_offset_to_tta_phase(offset_seconds: float) -> float:
    anchors = np.array([0.0, 20.0, 40.0], dtype=np.float64)
    return float(anchors[np.argmin(np.abs(anchors - float(offset_seconds)))])


def collapse_train_to_eeg_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    votes = df[TARGETS].to_numpy(dtype=np.float64, copy=True)
    rs = votes.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    votes = votes / rs
    for i, c in enumerate(TARGETS):
        df[c] = votes[:, i]

    df = df.sort_values(["eeg_id", "eeg_label_offset_seconds"], kind="mergesort")

    agg = {
        "eeg_label_offset_seconds": "first",  # stable existing window for eeg_id
        "patient_id": "first",
        "spectrogram_id": "first",
        "expert_consensus": "first",
    }
    for c in TARGETS:
        agg[c] = "mean"

    out = df.groupby("eeg_id", as_index=False).agg(agg)

    out["eeg_label_offset_seconds"] = out["eeg_label_offset_seconds"].map(
        _snap_offset_to_tta_phase
    )

    v = out[TARGETS].to_numpy(dtype=np.float64, copy=False)
    v = np.clip(v, 0.0, None)
    rs2 = v.sum(axis=1, keepdims=True)
    rs2[rs2 == 0] = 1.0
    out[TARGETS] = v / rs2

    assert np.isfinite(out[TARGETS].to_numpy()).all()
    return out


train = collapse_train_to_eeg_level(train)
print("Collapsed train shape (one row per eeg_id):", train.shape)
print(
    "Row prob sum stats:",
    train[TARGETS].sum(axis=1).min(),
    train[TARGETS].sum(axis=1).max(),
)
train.head()




## === cell 5
def expand_train_to_tta_offsets(df_eeg_level: pd.DataFrame) -> pd.DataFrame:
    anchors = [0.0, 20.0, 40.0]
    dfs = []
    base = df_eeg_level.copy()
    for a in anchors:
        d = base.copy()
        d["eeg_label_offset_seconds"] = float(a)
        dfs.append(d)
    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


train = expand_train_to_tta_offsets(train)
print("Expanded train shape (3 offsets per eeg_id):", train.shape)
print(
    "Row prob sum stats (should still be 1):",
    train[TARGETS].sum(axis=1).min(),
    train[TARGETS].sum(axis=1).max(),
)




## === cell 6
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = row["eeg_id"]
        parq_path = f"{PATH}{eeg_id}.parquet"

        eeg = pd.read_parquet(parq_path, columns=_NEED_COLS)

        start = int(row["eeg_label_offset_seconds"] * 200)

        max_start = max(0, len(eeg) - 10_000)
        start = int(np.clip(start, 0, max_start))

        eeg = eeg.iloc[start : start + 10_000]
        eeg = eeg.fillna(0)

        eeg_np = eeg.to_numpy(dtype=np.float64, copy=False)
        col_idx = {c: i for i, c in enumerate(eeg.columns)}

        signals = np.empty((4, 2500), dtype=np.float64)
        for k in range(4):
            c0, c1, c2, c3, c4 = FEATS[k]
            i0, i1, i2, i3, i4 = (
                col_idx[c0],
                col_idx[c1],
                col_idx[c2],
                col_idx[c3],
                col_idx[c4],
            )
            x = eeg_np[:, i0] - eeg_np[:, i1]
            x = (
                x
                + (eeg_np[:, i1] - eeg_np[:, i2])
                + (eeg_np[:, i2] - eeg_np[:, i3])
                + (eeg_np[:, i3] - eeg_np[:, i4])
            )
            x /= 4.0
            signals[k] = denoise_filter(x)

        labels = row[TARGETS].values.astype(np.float64, copy=False)
        labels = np.clip(labels, 0.0, None)
        s = float(labels.sum())
        if not np.isfinite(s) or s <= 0:
            labels = np.ones_like(labels, dtype=np.float64) / len(labels)
        else:
            labels = labels / s

        return (
            torch.from_numpy(np.ascontiguousarray(signals)).to(dtype=torch.float64),
            torch.from_numpy(np.ascontiguousarray(labels)).to(dtype=torch.float64),
        )




## === cell 7
dataset = CustomDataset(dataframe=train)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 9
batch_size = 256
num_workers = min(8, (os.cpu_count() or 2))
train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # keep as-is
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

model = CNN1D(in_channels=4).to(device)  # already float64 internally
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)

MODEL_DIR = "/kaggle/working/CNN1D_Model"
MODEL_PATH = os.path.join(MODEL_DIR, "model_nosampler.pt")
os.makedirs(MODEL_DIR, exist_ok=True)

loaded_ok = False
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        loaded_ok = True
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: failed to load existing model at {MODEL_PATH}: {e}")
        loaded_ok = False

if not loaded_ok:
    model.train()
    epochs = 3

    LOSS_SCALE = 1.0 / 3.0

    for epoch in range(epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for eeg_, label in pbar:
            eeg_ = eeg_.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits = model(eeg_)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, label) * LOSS_SCALE

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()))

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3039754398.py in <cell line: 0>()
     11 )
     12 
---> 13 model = CNN1D(in_channels=4).to(device)  # already float64 internally
     14 criterion = nn.KLDivLoss(reduction="batchmean")
     15 optimizer = optim.Adam(model.parameters(), lr=0.001)

NameError: name 'CNN1D' is not defined

## === cell 10
del dataset, train_dataloader
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 11
del train
gc.collect()

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape:", test.shape)
test.head()



## === cell 12
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

_TTA_OFFSETS = ("start", "center", "end")


class CustomDataset_test(Dataset):
    def __init__(self, dataframe, offset_mode: str = "center"):
        self.dataframe = dataframe.reset_index(drop=True)
        if offset_mode not in _TTA_OFFSETS:
            raise ValueError(
                f"offset_mode must be one of {_TTA_OFFSETS}, got {offset_mode}"
            )
        self.offset_mode = offset_mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = row["eeg_id"]
        parq_path = f"{test_path}{eeg_id}.parquet"

        eeg = pd.read_parquet(parq_path, columns=_NEED_COLS).fillna(0)

        rows = len(eeg)
        if rows <= 10_000:
            offset = 0
        else:
            if self.offset_mode == "start":
                offset = 0
            elif self.offset_mode == "center":
                offset = 20 * 200
            else:  # end
                offset = 40 * 200
            offset = int(np.clip(offset, 0, rows - 10_000))

        eeg = eeg.iloc[offset : offset + 10_000]

        eeg_np = eeg.to_numpy(dtype=np.float64, copy=False)
        col_idx = {c: i for i, c in enumerate(eeg.columns)}

        signals = np.empty((4, 2500), dtype=np.float64)
        for k in range(4):
            c0, c1, c2, c3, c4 = FEATS[k]
            i0, i1, i2, i3, i4 = (
                col_idx[c0],
                col_idx[c1],
                col_idx[c2],
                col_idx[c3],
                col_idx[c4],
            )
            x = eeg_np[:, i0] - eeg_np[:, i1]
            x = (
                x
                + (eeg_np[:, i1] - eeg_np[:, i2])
                + (eeg_np[:, i2] - eeg_np[:, i3])
                + (eeg_np[:, i3] - eeg_np[:, i4])
            )
            x /= 4.0
            signals[k] = denoise_filter(x)

        return torch.from_numpy(np.ascontiguousarray(signals)).to(dtype=torch.float64)




## === cell 13
class CNN1D(nn.Module):
    def __init__(self, in_channels):
        super(CNN1D, self).__init__()
        self.hidden_channels = 128

        self.conv1 = nn.Conv1d(in_channels, 64, 20, 10).double()
        self.conv2 = nn.Conv1d(64, 32, 10, 5).double()
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 2500, dtype=torch.float64)
            y = F.relu(self.conv1(dummy))
            y = F.relu(self.conv2(y))
            flat_dim = int(self.flatten(y).shape[1])

        self.fc1 = nn.Linear(flat_dim, 6).double()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.flatten(x))
        x = self.fc1(x)
        return x  # logits




## === cell 14
model.eval()


def predict_probs_for_offset(offset_mode: str) -> np.ndarray:
    dataset_test = CustomDataset_test(dataframe=test, offset_mode=offset_mode)
    num_workers_test = min(8, (os.cpu_count() or 2))
    test_loader = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers_test,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers_test > 0),
        prefetch_factor=4 if num_workers_test > 0 else None,
    )

    preds_local = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Infer({offset_mode})"):
            batch = batch.to(device, non_blocking=True)
            logits = model(batch)
            prob = F.softmax(logits, dim=1)
            preds_local.append(prob.detach().cpu().numpy())
    preds_local = np.vstack(preds_local)
    return preds_local


preds_list = []
for mode in _TTA_OFFSETS:
    preds_list.append(predict_probs_for_offset(mode))

preds = np.mean(preds_list, axis=0)

preds = np.clip(preds, 1e-12, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

print("Preds shape:", preds.shape, "row0 sum:", float(preds[0].sum()))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3002626641.py in <cell line: 0>()
----> 1 model.eval()
      2 
      3 
      4 def predict_probs_for_offset(offset_mode: str) -> np.ndarray:
      5     dataset_test = CustomDataset_test(dataframe=test, offset_mode=offset_mode)

NameError: name 'model' is not defined

## === cell 15
sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = preds

sub[TARGETS] = sub[TARGETS].astype(np.float64)
row_sums = sub[TARGETS].sum(axis=1).values
row_sums[row_sums == 0] = 1.0
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print("Submission shape", sub.shape)
print("Sub row 0 sums to:", float(sub.iloc[0, -6:].sum()))
sub.head()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3111673130.py in <cell line: 0>()
      1 sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
----> 2 sub[TARGETS] = preds
      3 
      4 sub[TARGETS] = sub[TARGETS].astype(np.float64)
      5 row_sums = sub[TARGETS].sum(axis=1).values

NameError: name 'preds' is not defined
