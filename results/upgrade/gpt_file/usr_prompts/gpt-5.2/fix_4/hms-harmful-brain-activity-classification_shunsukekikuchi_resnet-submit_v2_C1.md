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

1.0047043222323844

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import gc

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from scipy import signal
from tqdm import tqdm

from torchaudio import transforms as T
import albumentations as A

from torchvision.transforms import v2



## === cell 1
import sys
from pathlib import Path




## === cell 2
def fix_keys(loaded_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in loaded_dict.items()}




## === cell 3
class CFG:
    model_name = "resnet34"
    seed = 42
    fold = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    img_size = (257, 600)
    train_transform = v2.Resize(img_size)
    valid_transform = v2.Resize(img_size)
    autocast = False  # used for training, not for validation

    dataset_path_candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification",
        "/kaggle/input",
        "/kaggle/data",
    ]
    dataset_path = None

    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]




## === cell 4
def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def eeg2spec(eeg):
    input_len = eeg.shape[-1]
    transform = T.Spectrogram(
        n_fft=512, win_length=64, hop_length=input_len // 600, power=1
    )
    spec = transform(eeg) ** 0.8
    spec = torch.nan_to_num(spec)
    spec = F.normalize(spec)
    return spec


set_seed(CFG.seed)



## === cell 5
for p in CFG.dataset_path_candidates:
    if Path(p).exists():
        if (Path(p) / "test.csv").exists() and (Path(p) / "test_eegs").exists():
            CFG.dataset_path = p
            break
        comp = Path(p) / "hms-harmful-brain-activity-classification"
        if comp.exists() and (comp / "test.csv").exists():
            CFG.dataset_path = str(comp)
            break

if CFG.dataset_path is None:
    raise FileNotFoundError(
        "Could not locate dataset path containing test.csv and test_eegs/"
    )

test_csv_path = f"{CFG.dataset_path}/test.csv"
sample_sub_path = f"{CFG.dataset_path}/sample_submission.csv"
train_csv_path = f"{CFG.dataset_path}/train.csv"

df_test = pl.read_csv(test_csv_path).with_columns(
    pl.concat_str(
        [
            pl.lit(f"{CFG.dataset_path}/test_eegs/"),
            pl.col("eeg_id").cast(pl.String),
            pl.lit(".parquet"),
        ]
    ).alias("path")
)

sample_sub = pl.read_csv(sample_sub_path)



## === cell 6
train_df = pl.read_csv(train_csv_path)
vote_mat = train_df.select(CFG.target_cols).to_numpy().astype(np.float64)
vote_sum = vote_mat.sum(axis=1, keepdims=True)
vote_sum = np.maximum(vote_sum, 1.0)
row_probs = vote_mat / vote_sum
prior = row_probs.mean(axis=0)
prior = np.clip(prior, 1e-12, 1.0)
prior = (prior / prior.sum()).astype(np.float32)
prior



## === cell 7
first_path = df_test["path"][0]
first_eeg = pl.read_parquet(first_path)
if "EKG" in first_eeg.columns:
    n_in_chans = len([c for c in first_eeg.columns if c != "EKG"])
else:
    n_in_chans = len(first_eeg.columns)

n_in_chans



## === cell 8
import timm

model = timm.create_model(
    CFG.model_name, pretrained=False, num_classes=6, in_chans=n_in_chans
).to(CFG.device)

weight_candidates = [
    f"/kaggle/input/resnet_spec/pytorch/default/1/{CFG.model_name}_best_model.pth",
    f"/kaggle/data/resnet_spec/pytorch/default/1/{CFG.model_name}_best_model.pth",
]
weight_path = next((p for p in weight_candidates if Path(p).exists()), None)

has_weights = False
if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(fix_keys(state), strict=True)
    has_weights = True

model.to(CFG.device)
has_weights, weight_path




## === cell 9
class HmsDataset(Dataset):
    def __init__(self, labels_df: pl.DataFrame, train=False, transform=None):
        """
        in train/valid - set train True
        in inference - set train False
        """
        self.labels_df = labels_df
        self.paths = labels_df["path"].to_list()
        self.train = train

        self.targets = labels_df.select(["eeg_id"]).to_torch()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        df = pl.read_parquet(self.paths[idx])
        if "EKG" in df.columns:
            eeg = df.drop("EKG").to_torch().transpose(1, 0)
        else:
            eeg = df.to_torch().transpose(1, 0)

        spec = eeg2spec(eeg)  # (C, F, T)

        if spec.ndim != 3:
            raise RuntimeError(
                f"Unexpected spec shape {tuple(spec.shape)}; expected (C,F,T)."
            )

        if self.transform is not None:
            spec = self.transform(spec)  # still (C, H, W)

        target = self.targets[idx]
        return spec, target




## === cell 10
test_dataset = HmsDataset(df_test, train=False, transform=CFG.valid_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)




## === cell 11
def _safe_renorm_rows(probs_np: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs_np = np.clip(probs_np, 0.0, 1.0)
    row_sums = probs_np.sum(axis=1, keepdims=True)
    probs_np = probs_np / np.maximum(row_sums, eps)
    bad = row_sums[:, 0] <= eps
    if bad.any():
        probs_np[bad] = 1.0 / probs_np.shape[1]
    return probs_np




## === cell 12
model.eval()
all_log_pred, all_eeg_id = [], []

with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
    for spec, eeg_id in test_loader:
        spec = spec.to(CFG.device, non_blocking=True)
        log_pred = model(spec).log_softmax(dim=1).cpu()
        all_log_pred.append(log_pred)
        all_eeg_id.append(eeg_id)

log_preds = torch.cat(all_log_pred, dim=0)  # (N, C)
eeg_ids = torch.cat(all_eeg_id, dim=0)  # (N, 1) or (N,)

log_preds = log_preds.reshape(-1, log_preds.size(-1))
eeg_ids = eeg_ids.reshape(-1)

probs = log_preds.exp()
probs_np = probs.numpy().astype(np.float64)
probs_np = _safe_renorm_rows(probs_np).astype(np.float32)

alpha = (
    0.50 if has_weights else 0.00
)  # if no weights, use prior only (avoids random net)
probs_np = alpha * probs_np + (1.0 - alpha) * prior[None, :]
probs_np = _safe_renorm_rows(probs_np).astype(np.float32)

pred_df = pl.DataFrame(
    {
        "eeg_id": eeg_ids.cpu().numpy().astype("int64"),
        **{c: probs_np[:, i] for i, c in enumerate(CFG.target_cols)},
    }
)

submit_df = sample_sub.select(["eeg_id"]).join(pred_df, on="eeg_id", how="left")

for c in CFG.target_cols:
    if submit_df[c].null_count() > 0:
        submit_df = submit_df.with_columns(
            pl.col(c).fill_null(float(prior[CFG.target_cols.index(c)]))
        )

mat = submit_df.select(CFG.target_cols).to_numpy().astype(np.float64)
mat = _safe_renorm_rows(mat).astype(np.float32)

submit_df = pl.DataFrame({"eeg_id": submit_df["eeg_id"].to_numpy().astype(np.int64)})
for i, c in enumerate(CFG.target_cols):
    submit_df = submit_df.with_columns(pl.Series(name=c, values=mat[:, i]))

submit_df.write_csv("submission.csv")



## === cell 13
submit_df
