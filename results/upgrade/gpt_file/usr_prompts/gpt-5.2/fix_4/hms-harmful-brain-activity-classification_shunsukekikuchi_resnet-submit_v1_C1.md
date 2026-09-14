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

8.62314

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I first make the notebook run end-to-end by removing the hard dependency on an external `/kaggle/input/resnet_spec/...` weights file and instead default to a valid probabilistic baseline if weights are unavailable. Since your metric is KL divergence (lower is better) and you currently have no score due to no valid submission, I generate a stable submission by predicting the global class distribution from `train.csv` (a strong, simple baseline) and ensuring each row sums to exactly 1.0. I also fix the cell numbering to start at 1 and keep your model/dataset code intact (but not required to run for the baseline). The output always be a valid `submission.csv` with the correct columns and row order aligned to `test.csv`.'
- What this solution (achieved 8.70992) has done: 'Your current score (1.41937, lower-is-better) is worse than the target (0.9546), so we should improve the predictions while keeping your existing inference/model code intact. The smallest legitimate gain here is to replace the overly-blunt global prior with a patient-aware prior: use `train.csv` to compute the average class distribution per `patient_id`, and for each test row predict that patient’s distribution (falling back to the global prior for unseen patients). This preserves your “no-weights fallback” core logic (a prior-based probabilistic baseline) but makes it more aligned to the test metadata, which typically reduces KL. I also make the row-normalization fully stable (no negative after adjustment) and keep the submission schema/row order unchanged.'
- What this solution (achieved 8.62314) has done: 'Your current score (8.70992, lower-is-better) is far worse than the target (0.9546), so we should improve the fallback predictions with minimal risk while preserving your existing core logic (patient-aware priors when weights are missing). The biggest issue is that the patient prior is currently an unweighted mean over subsamples, but the competition target is based on *vote distributions*; switching to a vote-weighted patient prior (aggregate votes per patient, then normalize) better matches the label-generating process and typically reduces KL. I also keep the global prior as a vote-sum distribution (already correct) and retain your strict row-normalization and submission alignment logic. No model architecture/training/inference loop changes are made; only the prior estimation math is corrected.'

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

weights_path = (
    f"/kaggle/input/resnet_spec/pytorch/default/1/{CFG.model_name}_best_model.pth"
)
HAS_WEIGHTS = os.path.exists(weights_path)
if HAS_WEIGHTS:
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(fix_keys(state), strict=True)
model.to(CFG.device)




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
        n_fft=512, win_length=64, hop_length=input_len // 600, power=1
    )
    spec = transform(eeg) ** 0.8
    spec = torch.nan_to_num(spec)
    spec = F.normalize(spec)
    return spec


set_seed(CFG.seed)



## === cell 6
df = pl.read_csv(f"{CFG.dataset_path}/test.csv").with_columns(
    pl.concat_str(
        [
            pl.lit(f"{CFG.dataset_path}/test_eegs/"),
            pl.col("eeg_id").cast(pl.String),
            pl.lit(".parquet"),
        ],
    ).alias("path")
)




## === cell 7
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
        dfp = pl.read_parquet(self.paths[idx])
        eeg = dfp.drop("EKG").to_torch().transpose(1, 0)
        spec = eeg2spec(eeg)
        if self.transform is not None:
            spec = self.transform(spec)
        target = self.targets[idx]
        return spec, target




## === cell 8
test_dataset = HmsDataset(df, train=False, transform=CFG.valid_transform)
test_loader = DataLoader(
    test_dataset, batch_size=CFG.batch_size, shuffle=False, drop_last=False
)




## === cell 9
def _normalize_rows(probs_np: np.ndarray) -> np.ndarray:
    probs_np = np.asarray(probs_np, dtype=np.float64)
    probs_np = np.clip(probs_np, 1e-12, 1.0)
    probs_np /= probs_np.sum(axis=1, keepdims=True)

    row_sums = probs_np.sum(axis=1, keepdims=True)
    delta = 1.0 - row_sums
    max_idx = np.argmax(probs_np, axis=1)
    probs_np[np.arange(probs_np.shape[0]), max_idx] += delta.ravel()

    probs_np = np.clip(probs_np, 1e-12, 1.0)
    probs_np /= probs_np.sum(axis=1, keepdims=True)
    return probs_np.astype(np.float32)




## === cell 10
test_df_full = pl.read_csv(f"{CFG.dataset_path}/test.csv").select(
    ["eeg_id", "patient_id"]
)
test_eeg_ids = test_df_full["eeg_id"].to_numpy().astype("int64")

if not HAS_WEIGHTS:
    train_df = pl.read_csv(f"{CFG.dataset_path}/train.csv").select(
        ["patient_id"] + CFG.target_cols
    )

    votes_all = train_df.select(CFG.target_cols).to_numpy().astype(np.float64)
    global_prior = votes_all.sum(axis=0)
    global_prior = np.clip(global_prior, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    patient_votes = train_df.group_by("patient_id").agg(
        [pl.sum(c).alias(c) for c in CFG.target_cols]
    )
    patient_prior_df = (
        patient_votes.with_columns(
            pl.sum_horizontal([pl.col(c) for c in CFG.target_cols]).alias("_sum")
        )
        .with_columns([(pl.col(c) / pl.col("_sum")).alias(c) for c in CFG.target_cols])
        .drop("_sum")
    )

    enriched = (
        test_df_full.join(patient_prior_df, on="patient_id", how="left")
        .with_columns(
            [
                pl.col(c).fill_null(float(global_prior[i])).alias(c)
                for i, c in enumerate(CFG.target_cols)
            ]
        )
        .select(["eeg_id"] + CFG.target_cols)
    )

    probs_np = enriched.select(CFG.target_cols).to_numpy().astype(np.float32)
    probs_np = _normalize_rows(probs_np)

    submit_df = pl.DataFrame(
        {
            "eeg_id": enriched["eeg_id"].to_numpy().astype("int64"),
            **{c: probs_np[:, i] for i, c in enumerate(CFG.target_cols)},
        }
    ).sort("eeg_id")

    if not np.array_equal(submit_df["eeg_id"].to_numpy().astype("int64"), test_eeg_ids):
        pred_map = {
            int(eid): probs_np[i]
            for i, eid in enumerate(submit_df["eeg_id"].to_numpy())
        }
        probs_np = np.stack([pred_map[int(eid)] for eid in test_eeg_ids], axis=0)
        submit_df = pl.DataFrame(
            {
                "eeg_id": test_eeg_ids,
                **{c: probs_np[:, i] for i, c in enumerate(CFG.target_cols)},
            }
        )

    submit_df.write_csv("submission.csv")
else:
    model.eval()
    all_log_pred, all_eeg_id = [], []

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for spec, eeg_id in tqdm(test_loader, desc="Inference", leave=False):
            spec = spec.to(CFG.device)
            log_pred = model(spec).log_softmax(dim=1).cpu()
            all_log_pred.append(log_pred)
            all_eeg_id.append(eeg_id)

    log_preds = torch.cat(all_log_pred, dim=0)  # (N, C)
    eeg_ids = torch.cat(all_eeg_id, dim=0).reshape(-1).cpu().numpy().astype("int64")

    probs_np = log_preds.exp().numpy().astype(np.float32)
    probs_np = _normalize_rows(probs_np)

    if not np.array_equal(eeg_ids, test_eeg_ids):
        pred_map = {int(eid): probs_np[i] for i, eid in enumerate(eeg_ids)}
        probs_np = np.stack([pred_map[int(eid)] for eid in test_eeg_ids], axis=0)

    submit_df = pl.DataFrame(
        {
            "eeg_id": test_eeg_ids,
            **{c: probs_np[:, i] for i, c in enumerate(CFG.target_cols)},
        }
    )
    submit_df.write_csv("submission.csv")

submit_df



## === cell 11
pdf = submit_df.to_pandas()
assert list(pdf.columns) == ["eeg_id"] + CFG.target_cols
row_sums = pdf[CFG.target_cols].sum(axis=1).to_numpy()
assert np.allclose(row_sums, 1.0, atol=1e-5)
assert len(pdf) == 9850
pdf.head()
