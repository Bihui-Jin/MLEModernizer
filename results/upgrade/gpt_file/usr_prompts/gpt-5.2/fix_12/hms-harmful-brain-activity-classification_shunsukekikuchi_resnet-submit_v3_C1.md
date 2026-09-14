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

albumentations==2.0.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
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
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
timm==1.0.19
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

0.9764930065789634

# 6. Current score

1.40752

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41214) has done: 'I first make the pipeline produce a valid submission by fixing the cell numbering (your script starts at cell 0) and by ensuring `eeg_id` is read/returned as a 1D integer tensor so batching/catting works reliably. Then I make the probability post-processing both simpler and more numerically safe for the KL metric: instead of “tweaking one entry”, I renormalize each row to sum to 1 and clip away exact zeros with a tiny epsilon (this avoids infinite/huge KL when the target has nonzero mass where you predict 0). These changes keep your model, spectrogram extraction, and inference logic the same, but reduce submission failures and typically improves (lowers) KL in a stable way. The output still be `submission.csv` with the required columns and row sums equal to one.'
- What this solution (achieved 1.41096) has done: 'I fix the immediate runtime failure by making the model weight loading robust to the missing `/kaggle/input/resnet_spec/...` file, falling back to an untrained model only if no weights are found (so the notebook always runs end-to-end and writes `submission.csv`). I also correct the cell numbering to start at 1 (Kaggle “Run all” compatibility) and add a deterministic search for plausible weight locations under `/kaggle/input` to preserve your exact model architecture/inference semantics when weights exist. Finally, I keep your probability post-processing (clip + renorm) to ensure valid KL-safe submissions with row sums exactly 1.'
- What this solution (achieved 1.41043) has done: 'I fix the DataLoader crash by ensuring every EEG sample returned by the dataset has a consistent length (50 seconds = 10,000 samples at 200 Hz) via a minimal pad/center-crop in `__getitem__`, which preserves your model and spectrogram logic but makes batching valid. I also make the spectrogram hop length an integer constant to avoid any implicit type issues and keep inference numerically identical otherwise. Finally, I keep your probability clipping+renormalization so the submission is always KL-safe (no exact zeros) and row-sums are exactly 1, and ensure the notebook writes `submission.csv` successfully.'
- What this solution (achieved 1.40752) has done: 'Your current score (1.41043, lower-is-better) is still far from the target (0.97649), so we should make a small, metric-aligned improvement without changing your model or spectrogram pipeline. The biggest low-risk KL win here is to match the training label semantics: the targets are *vote distributions*, so we should soften overly-confident predictions by blending them slightly with a prior that reflects the average class distribution from `train.csv` (this reduces KL penalty when the model is wrong). This is a pure post-processing calibration step (no architecture/training changes) and keeps valid probability simplex constraints via the existing clip+renorm. I also keep your current batching/length-fix logic intact and only add a fast prior computation and a single blending coefficient.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from scipy import signal

import timm
from torchaudio import transforms as T
from torchvision.transforms import v2

import pyarrow.parquet as pq
import pyarrow as pa
from functools import lru_cache




## === cell 1
def fix_keys(loaded_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in loaded_dict.items()}


def set_seed(seed: int):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




## === cell 2
class CFG:
    model_name = "resnet18"
    seed = 42
    fold = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    img_size = (257, 600)
    train_transform = v2.Resize(img_size)
    valid_transform = v2.Resize(img_size)
    autocast = False  # used for training, not for validation

    dataset_path = "/kaggle/input/hms-harmful-brain-activity-classification"
    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    prior_blend_alpha = (
        0.08  # small smoothing; tuned to be conservative (not "optimize for best")
    )


set_seed(CFG.seed)




## === cell 3
def find_weight_file(model_name: str) -> str | None:
    """
    Keep same logic; not performance critical at runtime, but ensures correct weights when available.
    """
    candidates = [
        f"/kaggle/input/resnet_spec/pytorch/default/1/{model_name}_best_model.pth",
        f"/kaggle/input/resnet_spec/{model_name}_best_model.pth",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    stems = [
        f"{model_name}_best_model",
        f"{model_name}_best",
        f"{model_name}_fold{CFG.fold}_best_model",
        f"{model_name}_fold{CFG.fold}_best",
        f"{model_name}_fold{CFG.fold}",
        model_name,
    ]
    exts = [".pth", ".pt", ".bin"]

    roots = ["/kaggle/input", "/kaggle/data/input"]
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            for stem in stems:
                for ext in exts:
                    fname = stem + ext
                    for p in Path(root).rglob(fname):
                        if p.is_file():
                            found.append((p.stat().st_mtime, str(p)))
        except Exception:
            pass

    if not found:
        return None
    found.sort(key=lambda x: x[0], reverse=True)
    return found[0][1]


model = timm.create_model(
    CFG.model_name, pretrained=False, num_classes=6, in_chans=19
).to(CFG.device)

weight_path = find_weight_file(CFG.model_name)
if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(fix_keys(state), strict=True)

    print("Loaded weights from:", weight_path)
else:
    print(
        "WARNING: No pretrained weights found. Using randomly initialized model; score will be poor."
    )

model.to(CFG.device)



## === cell 4
_SPEC_INPUT_LEN = 50 * 200  # 10000 samples for test EEGs

_HOP_LENGTH = int(_SPEC_INPUT_LEN // 600)

_SPEC_TRANSFORM = T.Spectrogram(
    n_fft=512,
    win_length=64,
    hop_length=_HOP_LENGTH,
    power=1,
)


def eeg2spec(eeg: torch.Tensor) -> torch.Tensor:
    spec = (_SPEC_TRANSFORM(eeg)) ** 0.8
    spec = torch.nan_to_num(spec)
    spec = F.normalize(spec)
    return spec




## === cell 5
train_csv_path = f"{CFG.dataset_path}/train.csv"
train_prior = None
try:
    _train_pl = pl.read_csv(
        train_csv_path,
        columns=CFG.target_cols,
    )
    _votes = _train_pl.select(CFG.target_cols).to_numpy()
    _votes = _votes.astype(np.float64, copy=False)
    _row_sum = _votes.sum(axis=1, keepdims=True)
    _row_sum[_row_sum <= 0] = 1.0
    _dist = _votes / _row_sum
    train_prior = _dist.mean(axis=0)
    train_prior = train_prior / train_prior.sum()
    train_prior = train_prior.astype(np.float32)
    print("Computed train prior:", dict(zip(CFG.target_cols, train_prior.tolist())))
except Exception as e:
    train_prior = np.ones(len(CFG.target_cols), dtype=np.float32) / len(CFG.target_cols)
    print(
        "WARNING: failed to compute train prior, using uniform prior. Error:", repr(e)
    )



## === cell 6
df = (
    pl.read_csv(f"{CFG.dataset_path}/test.csv")
    .with_columns(pl.col("eeg_id").cast(pl.Int64))
    .with_columns(
        pl.concat_str(
            [
                pl.lit(f"{CFG.dataset_path}/test_eegs/"),
                pl.col("eeg_id").cast(pl.String),
                pl.lit(".parquet"),
            ]
        ).alias("path")
    )
)




## === cell 7
@lru_cache(maxsize=8)
def _infer_columns_excluding_ekg(sample_path: str):
    schema = pq.read_schema(sample_path, memory_map=True)
    cols = list(schema.names)
    if "EKG" in cols:
        cols.remove("EKG")
    return tuple(cols)


def _read_eeg_parquet_as_numpy_fast(path: str, cols: tuple[str, ...]) -> np.ndarray:
    table = pq.read_table(path, columns=list(cols), memory_map=True)
    arr = table.to_pandas(types_mapper=None).to_numpy(copy=False)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _fix_eeg_length_ct(
    eeg_ct: torch.Tensor, target_len: int = _SPEC_INPUT_LEN
) -> torch.Tensor:
    """
    Bugfix: test_eegs parquet lengths can vary; DataLoader default_collate requires equal sizes.
    Minimal change: center-crop if too long, pad with zeros if too short.
    """
    c, t = eeg_ct.shape
    if t == target_len:
        return eeg_ct
    if t > target_len:
        start = (t - target_len) // 2
        return eeg_ct[:, start : start + target_len].contiguous()
    pad_right = target_len - t
    return F.pad(eeg_ct, (0, pad_right), mode="constant", value=0.0).contiguous()


class HmsDataset(Dataset):
    def __init__(self, labels_df: pl.DataFrame, train: bool = False, transform=None):
        """
        in train/valid - set train True
        in inference - set train False
        """
        self.labels_df = labels_df
        self.paths = labels_df["path"].to_list()
        self.train = train
        self.eeg_ids_np = labels_df["eeg_id"].to_numpy().astype(np.int64, copy=False)
        self.transform = transform

        self._cols = _infer_columns_excluding_ekg(self.paths[0])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        arr_tc = _read_eeg_parquet_as_numpy_fast(path, self._cols)  # (T, C)
        eeg = torch.from_numpy(arr_tc).transpose(1, 0).contiguous()  # (C, T), float32

        eeg = _fix_eeg_length_ct(eeg, target_len=_SPEC_INPUT_LEN)

        eeg_id = self.eeg_ids_np[idx]
        return eeg, eeg_id




## === cell 8
test_dataset = HmsDataset(df, train=False, transform=None)

cpu_cnt = os.cpu_count() or 1
num_workers = min(8, max(2, cpu_cnt))  # parquet I/O is the bottleneck

try:
    torch.set_num_threads(1)
except Exception:
    pass

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)




## === cell 9
def _fix_probs(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    bad = (row_sums <= 0) | ~np.isfinite(row_sums)
    if np.any(bad):
        probs[bad[:, 0], :] = 1.0 / probs.shape[1]
        row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    return probs.astype(np.float32)




## === cell 10
model.eval()

n = len(test_dataset)
n_classes = len(CFG.target_cols)
out_probs = np.empty((n, n_classes), dtype=np.float32)
out_ids = np.empty((n,), dtype=np.int64)

write_pos = 0

device = CFG.device
use_cuda = torch.cuda.is_available()

spec_transform = _SPEC_TRANSFORM.to(device) if use_cuda else _SPEC_TRANSFORM
resize_transform = (
    CFG.valid_transform.to(device)
    if hasattr(CFG.valid_transform, "to")
    else CFG.valid_transform
)

with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
    for eeg, eeg_id in tqdm(test_loader, desc="Inference", leave=False):
        bs = eeg.shape[0]

        eeg = eeg.to(device, non_blocking=True)

        spec = (spec_transform(eeg)) ** 0.8
        spec = torch.nan_to_num(spec)
        spec = F.normalize(spec)

        if resize_transform is not None:
            spec = resize_transform(spec)

        probs = (
            model(spec)
            .softmax(dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        out_probs[write_pos : write_pos + bs] = probs
        out_ids[write_pos : write_pos + bs] = np.asarray(eeg_id, dtype=np.int64)
        write_pos += bs

alpha = float(CFG.prior_blend_alpha)
if train_prior is None:
    train_prior = np.ones(n_classes, dtype=np.float32) / n_classes
probs_blend = (1.0 - alpha) * out_probs + alpha * train_prior.reshape(1, -1)

probs_np = _fix_probs(probs_blend, eps=1e-7)

submit_df = pl.DataFrame(
    {
        "eeg_id": out_ids.astype("int64", copy=False),
        **{c: probs_np[:, i] for i, c in enumerate(CFG.target_cols)},
    }
)

submit_df = submit_df.select(["eeg_id"] + CFG.target_cols)
submit_df.write_csv("submission.csv")

print("Wrote submission.csv with shape:", submit_df.shape)



## === cell 11
submit_df.head()
