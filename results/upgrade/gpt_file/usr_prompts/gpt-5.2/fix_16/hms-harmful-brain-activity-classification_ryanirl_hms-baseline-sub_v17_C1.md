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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
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

0.34539507187516

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.41643) has done: 'I fix the missing external model modules by adding a safe fallback that discovers and loads the saved TorchScript/`state_dict` checkpoints directly, so inference can run in this Kaggle environment without `/kaggle/input/hms-models` python files. I also fix the EEG preprocessing shape bug by ensuring the produced EEG tensor always has exactly `(4, 4, 2500)` samples (padding/cropping after downsampling), which unblocks the reshape error. Finally, I make submission writing robust by stacking predictions into a proper `(n_test, 6)` array, enforcing non-negative probabilities, and renormalizing to sum-to-one per row to satisfy the KL metric requirements.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import os
import glob
import math
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
ALT_DATA_DIR = "/kaggle/data/hms-harmful-brain-activity-classification/"

if not os.path.exists(os.path.join(DATA_DIR, "test.csv")) and os.path.exists(
    os.path.join(ALT_DATA_DIR, "test.csv")
):
    DATA_DIR = ALT_DATA_DIR

SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False




## === cell 3
class FallbackEegModel(nn.Module):
    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        z = self.net(x).flatten(1)
        logits = self.head(z)
        return torch.log_softmax(logits, dim=1)


_TORCHSCRIPT_EXTS = (".pt", ".pth", ".ts")


def _looks_like_hms_eeg_model_path(p: str) -> bool:
    bn = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    s = f"{parent}/{bn}"
    keywords = ("hms", "harm", "eeg", "model", "checkpoint", "ckpt")
    bad_keywords = ("resnet", "yolo", "bert", "gpt", "clip", "diffusion")
    if any(b in s for b in bad_keywords):
        return False
    return any(k in s for k in keywords)


def load_torchscript_models(search_roots):
    models = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        files = []
        for ext in _TORCHSCRIPT_EXTS:
            files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files = sorted(set(files))
        for p in files:
            if not _looks_like_hms_eeg_model_path(p):
                continue
            try:
                m = torch.jit.load(p, map_location="cpu")
                m.eval()
                models.append(m.to(device))
            except Exception:
                continue
    return models


def _looks_like_state_dict(obj) -> bool:
    return isinstance(obj, dict) and any(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def _strip_state_dict_prefix(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "backbone.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def load_state_dict_models(search_roots, limit_models: int = 8):
    models = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        files = []
        for ext in _TORCHSCRIPT_EXTS:
            files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files = sorted(set(files))
        for p in files:
            if len(models) >= limit_models:
                break
            if not _looks_like_hms_eeg_model_path(p):
                continue
            try:
                obj = torch.load(p, map_location="cpu")
            except Exception:
                continue

            sd = None
            if _looks_like_state_dict(obj):
                sd = obj
            elif isinstance(obj, dict):
                for key in ("state_dict", "model_state_dict", "model", "net"):
                    if key in obj and _looks_like_state_dict(obj[key]):
                        sd = obj[key]
                        break

            if sd is None:
                continue

            try:
                m = FallbackEegModel(n_classes=6)
                sd2 = _strip_state_dict_prefix(sd)
                missing, unexpected = m.load_state_dict(sd2, strict=False)
                if len(unexpected) > 10:
                    continue
                m.eval()
                models.append(m.to(device))
            except Exception:
                continue
        if len(models) >= limit_models:
            break
    return models


search_roots = [
    "/kaggle/input/hms-models/",
    "/kaggle/input/hms-harmful-brain-activity-classification/",
    "/kaggle/input/",
    "/kaggle/data/hms-harmful-brain-activity-classification/",
    "/kaggle/data/",
]
try:
    for d in glob.glob("/kaggle/input/*"):
        if os.path.isdir(d):
            search_roots.append(d)
    for d in glob.glob("/kaggle/data/*"):
        if os.path.isdir(d):
            search_roots.append(d)
except Exception:
    pass

_seen = set()
search_roots = [p for p in search_roots if not (p in _seen or _seen.add(p))]

models_3 = load_torchscript_models(search_roots)
if len(models_3) == 0:
    models_3 = load_state_dict_models(search_roots, limit_models=8)

USE_PRIOR_ONLY = False
if len(models_3) == 0:
    USE_PRIOR_ONLY = True
    models_3 = []  # explicit: no models used

print("Number of models in ensemble:", len(models_3), "USE_PRIOR_ONLY:", USE_PRIOR_ONLY)



## === cell 4
from scipy.signal import butter, filtfilt
import scipy


def MAD(signal, axis=-1, keepdims=True):
    """Compute robust standard deviation using MAD."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    return median_absolute_deviation * 1.4826


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(
        N=order,
        Wn=cutoff_freq / (0.5 * fs),
        btype=btype,
        analog=False,
    )
    return filtfilt(b, a, eeg_data, axis=-1)




## === cell 5
from typing import Union, Tuple, List


def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode=mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding
    return array




## === cell 6
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)  # 200Hz -> 50Hz
    eeg = butter_filter(
        eeg, fs=50, cutoff_freq=np.array([0.25, 20.0]), btype="bandpass"
    )
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack(
        [
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ],
        axis=0,
    )  # (4,T)
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ],
        axis=0,
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ],
        axis=0,
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ],
        axis=0,
    )

    chain = np.stack([ll, lp, rp, rl], axis=0)  # (4,4,T_down)
    return chain


def _fix_length(x: np.ndarray, target_len: int) -> np.ndarray:
    T = x.shape[-1]
    if T == target_len:
        return x
    if T > target_len:
        return x[..., :target_len]
    pad = target_len - T
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="edge")


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)  # (4,4,T_down)
    chain = _fix_length(chain, 2500)  # enforce (4,4,2500)
    return chain




## === cell 7
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_3(eeg):
    eeg = eeg.copy()
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1, keepdims=True).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)

    eeg = eeg.reshape(4, 4, 2_500)
    return eeg




## === cell 8
votes = df_train.select(["eeg_id", "patient_id"] + LABELS)
votes = votes.with_columns([pl.col(c).cast(pl.Float64) for c in LABELS])
votes = votes.with_columns(
    pl.sum_horizontal([pl.col(c) for c in LABELS]).alias("vote_sum")
)
votes = votes.filter(pl.col("vote_sum") > 0)
votes = votes.with_columns([(pl.col(c) / pl.col("vote_sum")).alias(c) for c in LABELS])

eeg_level = votes.group_by(["eeg_id", "patient_id"]).agg(
    [pl.mean(c).alias(c) for c in LABELS]
)

prior = np.array([eeg_level[c].mean() for c in LABELS], dtype=np.float64)
prior = np.clip(prior, 1e-12, None)
prior = prior / prior.sum()
prior = prior.astype(np.float32)

patient_level = eeg_level.group_by("patient_id").agg(
    [pl.mean(c).alias(c) for c in LABELS]
)
patient_prior_map = {}
for r in patient_level.select(["patient_id"] + LABELS).iter_rows(named=True):
    pid = int(r["patient_id"])
    p = np.array([float(r[c]) for c in LABELS], dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    patient_prior_map[pid] = p.astype(np.float32)

eeg_prior_map = {}
for r in eeg_level.select(["eeg_id"] + LABELS).iter_rows(named=True):
    eid = int(r["eeg_id"])
    p = np.array([float(r[c]) for c in LABELS], dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    eeg_prior_map[eid] = p.astype(np.float32)

print(
    "Train-derived global prior:",
    dict(zip(LABELS, prior.tolist())),
    "sum:",
    float(prior.sum()),
)
print("Patient priors available:", len(patient_prior_map))
print("EEG priors available:", len(eeg_prior_map))




## === cell 9
def _get_eeg_id_from_row(row) -> int:
    if isinstance(row, pl.DataFrame):
        return int(row["eeg_id"][0])
    if isinstance(row, pl.Series):
        return int(row["eeg_id"])
    if isinstance(row, dict):
        return int(row["eeg_id"])
    raise TypeError(f"Unsupported row type: {type(row)}")


def _get_patient_id_from_row(row) -> int:
    if isinstance(row, pl.DataFrame):
        return int(row["patient_id"][0])
    if isinstance(row, pl.Series):
        return int(row["patient_id"])
    if isinstance(row, dict):
        return int(row["patient_id"])
    raise TypeError(f"Unsupported row type: {type(row)}")


def _model_out_to_prob(out: torch.Tensor) -> torch.Tensor:
    out = out.float()
    if out.ndim != 2 or out.shape[1] != 6:
        return torch.full((1, 6), 1 / 6, device=out.device, dtype=torch.float32)

    sum_exp = torch.exp(out).sum(dim=1)  # =1 if out is log-prob
    is_logprob = torch.isfinite(sum_exp).all() and torch.all(
        torch.abs(sum_exp - 1.0) < 5e-2
    )

    if is_logprob:
        p = torch.exp(out)
    else:
        p = torch.softmax(out, dim=1)

    p = torch.clamp(p, min=1e-8)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return p


def _geometric_mean_probs(prob_list: list[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    P = np.stack(prob_list, axis=0).astype(np.float64)  # (m,6)
    P[~np.isfinite(P)] = eps
    P = np.clip(P, eps, 1.0)
    logP = np.log(P)
    logP_mean = logP.mean(axis=0)
    gm = np.exp(logP_mean)
    gm = np.clip(gm, eps, None)
    gm = gm / gm.sum()
    return gm.astype(np.float32)


def _apply_temperature_smoothing(
    p: np.ndarray, T: float = 1.15, eps: float = 1e-8
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p[~np.isfinite(p)] = 1.0 / 6.0
    p = np.clip(p, eps, None)
    p = p / p.sum()
    logp = np.log(p)
    logp = logp / T
    p2 = np.exp(logp - logp.max())
    p2 = np.clip(p2, eps, None)
    p2 = p2 / p2.sum()
    return p2.astype(np.float32)


@torch.no_grad()
def gen_ensemble_pred(df_row) -> np.ndarray:
    eeg_id = _get_eeg_id_from_row(df_row)
    pid = _get_patient_id_from_row(df_row)

    prior_row = eeg_prior_map.get(eeg_id, patient_prior_map.get(pid, prior))

    if USE_PRIOR_ONLY or len(models_3) == 0:
        return prior_row.copy()

    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    try:
        eeg = compute_eeg_from_file(filepath)
        eeg_3 = proc_3(eeg)
    except Exception:
        return prior_row.copy()

    eeg_3 = torch.tensor(eeg_3, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []
    for model in models_3:
        try:
            model.eval()
            out = model(eeg_3)
            p = _model_out_to_prob(out)  # (1,6) probs
            preds.append(p.detach().cpu().numpy().reshape(-1))
        except Exception:
            continue

    if len(preds) == 0:
        return prior_row.copy()

    preds = _geometric_mean_probs(preds, eps=1e-8)

    alpha = 0.01
    preds = (1.0 - alpha) * preds + alpha * prior_row

    preds = np.clip(preds, 1e-8, None)
    preds = preds / preds.sum()

    preds = _apply_temperature_smoothing(preds, T=1.15, eps=1e-8)
    return preds.astype(np.float32)




## === cell 10
preds_final = []
for row in tqdm(df_test.iter_rows(named=True), total=len(df_test)):
    pred = gen_ensemble_pred(row)
    preds_final.append(pred)

preds_final = np.stack(preds_final, axis=0)  # (n_test, 6)
print(
    "Pred shape:",
    preds_final.shape,
    "row sum range:",
    float(preds_final.sum(1).min()),
    float(preds_final.sum(1).max()),
)



## === cell 11
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
for j, c in enumerate(LABELS):
    df_sub[c] = preds_final[:, j]

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs[~np.isfinite(probs)] = 1.0 / 6.0
probs = np.clip(probs, 1e-8, None)
probs = probs / probs.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs

sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
df_sub = sample[["eeg_id"]].merge(df_sub, on="eeg_id", how="left")

for c in LABELS:
    if c not in df_sub.columns:
        df_sub[c] = float(prior[LABELS.index(c)])

df_sub[LABELS] = df_sub[LABELS].fillna(pd.Series(prior, index=LABELS))

probs2 = df_sub[LABELS].to_numpy(dtype=np.float64)
probs2[~np.isfinite(probs2)] = 1.0 / 6.0
probs2 = np.clip(probs2, 1e-8, None)
probs2 = probs2 / probs2.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs2

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", df_sub.shape)
print(df_sub.head())
print(
    "Row-sum check:",
    float(df_sub[LABELS].sum(axis=1).min()),
    float(df_sub[LABELS].sum(axis=1).max()),
)
