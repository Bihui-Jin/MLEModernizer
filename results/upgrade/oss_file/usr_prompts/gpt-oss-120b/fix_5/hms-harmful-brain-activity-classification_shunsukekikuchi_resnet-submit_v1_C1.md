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

0.9546367098266386

# 6. Current score

0.77715

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I make the script robust so it always creates a valid `submission.csv`. The model loading is wrapped in a try‑except; if the pretrained weights are missing the code falls back to a simple prior‑based predictor computed from the training vote distribution. The inference loop uses the model when it exists, otherwise it fills each batch with the class‑prior probabilities. This ensures a complete submission and, by using the empirical class prior, moves the KL score closer to the target without altering the core architecture.'
- What this solution (achieved 1.46771) has done: 'I fixed the Polars grouping call (`groupby` → `group_by`) so patient‑wise priors are correctly computed, and rewrote the dataset class to avoid non‑existent Polars‑to‑torch conversion. The dataset now returns tensors for `eeg_id` and `patient_id`, ensuring the inference loop can concatenate them and the submission file is written correctly.'
- What this solution (achieved 0.77715) has done: 'I blend each patient‑specific prior with the overall class prior (using a modest weight) so the predictions are less extreme and better calibrated, which should lower the KL‑divergence toward the target. The change is limited to the prior construction and inference loop, preserving the original architecture and workflow.'

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
    model_name = "resnet50"
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




## === cell 4
import timm

model = timm.create_model(
    CFG.model_name, pretrained=False, num_classes=6, in_chans=19
).to(CFG.device)

try:
    state_dict = torch.load(
        f"/kaggle/input/resnet_spec/pytorch/default/1/{CFG.model_name}_best_model.pth",
        map_location=CFG.device,
    )
    model.load_state_dict(fix_keys(state_dict))
except Exception as e:
    print(
        f"Warning: could not load pretrained weights ({e}); using untrained model as fallback."
    )
    model = None  # signal that we must use a simpler predictor




## === cell 5
def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def eeg2spec(eeg):
    input_len = eeg.shape[-1]
    transform = T.Spectrogram(
        n_fft=512, win_length=64, hop_length=input_len // 600, power=1  #
    )
    spec = transform(eeg) ** 0.8
    spec = torch.nan_to_num(spec)
    spec = F.normalize(spec)
    return spec


set_seed(CFG.seed)

df = pl.read_csv(f"{CFG.dataset_path}/test.csv").with_columns(
    pl.concat_str(
        [
            pl.lit(f"{CFG.dataset_path}/test_eegs/"),
            pl.col("eeg_id").cast(pl.String),
            pl.lit(".parquet"),
        ],
    ).alias("path")
)

train_df = pl.read_csv(f"{CFG.dataset_path}/train.csv")
vote_cols = CFG.target_cols
train_votes = train_df.select(vote_cols).to_numpy()
row_sums = train_votes.sum(axis=1, keepdims=True)
train_probs = train_votes / np.where(
    row_sums == 0, 1, row_sums
)  # guard against division by zero
CLASS_PRIOR = torch.tensor(
    train_probs.mean(axis=0), dtype=torch.float32, device=CFG.device
)

BLEND_ALPHA = 0.8  # weight for patient prior; rest goes to global prior
patient_stats = train_df.group_by("patient_id").agg(
    [pl.col(col).sum() for col in vote_cols]
)

patient_votes = patient_stats.select(vote_cols).to_numpy()
patient_row_sums = patient_votes.sum(axis=1, keepdims=True)
patient_probs = patient_votes / np.where(patient_row_sums == 0, 1, patient_row_sums)
patient_ids_arr = patient_stats["patient_id"].to_numpy()
PATIENT_PRIOR = {
    int(pid): (
        BLEND_ALPHA * torch.tensor(p, dtype=torch.float32, device=CFG.device)
        + (1 - BLEND_ALPHA) * CLASS_PRIOR
    )
    for pid, p in zip(patient_ids_arr, patient_probs)
}




## === cell 6
class HmsDataset(Dataset):
    def __init__(self, labels_df: pl.DataFrame, train=False, transform=None):
        """
        in train/valid - set train True
        in inference - set train False
        """
        self.labels_df = labels_df
        self.paths = labels_df["path"].to_list()
        self.train = train

        self.eeg_ids = labels_df["eeg_id"].to_list()
        self.patient_ids = labels_df["patient_id"].to_list()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        df = pl.read_parquet(self.paths[idx])
        eeg = df.drop("EKG").to_numpy().transpose(1, 0)  # (channels, samples)
        eeg = torch.from_numpy(eeg).float()
        spec = eeg2spec(eeg)
        if self.transform is not None:
            spec = self.transform(spec)
        eeg_id = torch.tensor(self.eeg_ids[idx], dtype=torch.long)
        patient_id = torch.tensor(self.patient_ids[idx], dtype=torch.long)
        return spec, eeg_id, patient_id




## === cell 7
test_dataset = HmsDataset(df, train=False, transform=CFG.valid_transform)
test_loader = DataLoader(
    test_dataset, batch_size=CFG.batch_size, shuffle=False, drop_last=False
)




## === cell 8
def _fix_row(row: np.ndarray) -> np.ndarray:
    """Return a copy whose elements sum to 1.0 by tweaking one entry."""
    s = row.sum(dtype=np.float64)
    delta = 1.0 - s
    if abs(delta) < 1e-6:  # already OK
        return row

    idx = row.argmin() if delta > 0 else row.argmax()
    row[idx] += delta  # add (or subtract) the difference
    row[idx] = np.clip(row[idx], 0.0, 1.0)
    return row




## === cell 9
model_eval = model.eval() if model is not None else None
all_log_pred, all_eeg_id = [], []

with torch.no_grad():
    for spec, eeg_id, patient_id in test_loader:
        spec = spec.to(CFG.device)

        if model_eval is not None:
            log_pred = model_eval(spec).log_softmax(dim=1).cpu()
        else:
            priors_list = [
                PATIENT_PRIOR.get(int(pid), CLASS_PRIOR) for pid in patient_id.tolist()
            ]
            probs_batch = torch.stack(priors_list, dim=0)  # (batch, 6)
            log_pred = torch.log(probs_batch + 1e-12).cpu()

        all_log_pred.append(log_pred)
        all_eeg_id.append(eeg_id)

log_preds = torch.cat(all_log_pred, dim=0)  # (N, n_classes)
eeg_ids = torch.cat(all_eeg_id, dim=0)  # (N,)

log_preds = log_preds.reshape(-1, log_preds.size(-1))  # (N, C)
eeg_ids = eeg_ids.reshape(-1)  # (N,)

probs = log_preds.exp()
probs_np = probs.cpu().numpy().astype(np.float32)  # shape = (N, C)
probs_np = np.apply_along_axis(_fix_row, 1, probs_np)  # normalise rows

submit_df = pl.DataFrame(
    {
        "eeg_id": eeg_ids.cpu().numpy().astype("int64"),
        **{c: probs_np[:, i] for i, c in enumerate(CFG.target_cols)},
    }
)

submit_df.write_csv("submission.csv")




## === cell 10
submit_df
