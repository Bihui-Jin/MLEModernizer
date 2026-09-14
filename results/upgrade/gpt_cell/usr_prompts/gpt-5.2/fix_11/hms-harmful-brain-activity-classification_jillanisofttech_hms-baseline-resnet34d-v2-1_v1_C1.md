# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.12

# 2. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import pandas as pd  # For handling CSV files
import numpy as np  # For matrix operations
import random

import torch
import torch.nn as nn  # Neural network module
import torch.nn.functional as F  # Neural network functions

import torchvision.transforms as transforms

random.seed(42)
torch.manual_seed(42)

import warnings

warnings.filterwarnings("ignore", category=Warning)




## === cell 1
class Config:
    seed = 2024

    image_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
        ]
    )

    num_folds = 5




## === cell 2
import os
from pathlib import Path
import warnings

candidate_dirs = [
    Path("/kaggle/input/hms-baseline-resnet34d-512-512-training-5-folds"),
    Path("/kaggle/data/hms-baseline-resnet34d-512-512-training-5-folds"),
    Path(
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-baseline-resnet34d-512-512-training-5-folds"
    ),
    Path(
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-baseline-resnet34d-512-512-training-5-folds"
    ),
]

ckpt_dir = next((d for d in candidate_dirs if d.exists()), None)

if ckpt_dir is None:
    checked = "\n".join(str(d) for d in candidate_dirs)
    warnings.warn(
        "Pretrained model checkpoint directory not found; proceeding without loading fold checkpoints.\n"
        f"Checked:\n{checked}\n"
        "Expected files named like 'HMS_resnet_fold{i}.pth'."
    )
    models = []
else:
    models = []
    for i in range(Config.num_folds):
        ckpt_path = ckpt_dir / f"HMS_resnet_fold{i}.pth"
        if ckpt_path.exists():
            models.append(torch.load(str(ckpt_path), map_location="cpu"))
        else:
            warnings.warn(f"Checkpoint not found: {ckpt_path}. Skipping this fold.")




## === cell 3
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disabling this for reproducibility

    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 4
import pyarrow.parquet as pq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for m in models:
    m.to(DEVICE)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)




## === cell 5
def load_and_preprocess_data(path):
    eps = 1e-6

    table = pq.read_table(path)  # read parquet once
    col_names = table.schema.names
    if len(col_names) < 2:
        raise ValueError(f"Unexpected parquet schema in {path}: {col_names}")

    arr = table.select(col_names[1:]).to_numpy(zero_copy_only=False)
    data = np.nan_to_num(arr, nan=-1.0).T  # now shape: (freq, time)

    data = data[:, :300]
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)

    data_mean = data.mean(axis=(0, 1))
    data_std = data.std(axis=(0, 1))
    normalized_data = (data - data_mean) / (data_std + eps)

    data_tensor = torch.unsqueeze(
        torch.tensor(normalized_data, dtype=torch.float32), dim=0
    )  # (1, f, t)
    return Config.image_transform(data_tensor)




## === cell 6
@torch.no_grad()
def predict_batch(models, batch_data):
    batch_data = batch_data.to(DEVICE, non_blocking=True)
    fold_probs = []
    for model in models:
        logits = model(batch_data)
        probs = F.softmax(logits, dim=1)
        fold_probs.append(probs)
    mean_probs = torch.stack(fold_probs, dim=0).mean(dim=0)  # (B, 6)
    return mean_probs.detach().cpu().numpy()




## === cell 7
def prepare_submission(submission_file, test_df, test_preds):
    submission = pd.read_csv(submission_file)
    labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
    for i, label in enumerate(labels):
        submission[f"{label}_vote"] = test_preds[:, i]
    return submission




## === cell 8
test_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)
submission = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

submission = submission.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(
    drop=True
)

test_unique = test_df.drop_duplicates(subset=["eeg_id"], keep="first")[
    ["eeg_id", "spectrogram_id"]
]
submission = submission.merge(
    test_unique, on="eeg_id", how="left", validate="one_to_one"
)

if submission["spectrogram_id"].isnull().any():
    missing = int(submission["spectrogram_id"].isnull().sum())
    raise ValueError(f"Missing spectrogram_id for {missing} eeg_id values after merge.")

submission["path"] = submission["spectrogram_id"].apply(
    lambda x: f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{x}.parquet"
)


## === cell 9
def load_and_preprocess_data(path):
    eps = 1e-6

    table = pq.read_table(path)  # read parquet once
    col_names = table.schema.names
    if len(col_names) < 2:
        raise ValueError(f"Unexpected parquet schema in {path}: {col_names}")

    selected = table.select(col_names[1:])
    selected = selected.combine_chunks()
    arr = selected.to_pandas().to_numpy()

    data = np.nan_to_num(arr, nan=-1.0).T  # now shape: (freq, time)

    data = data[:, :300]
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)

    data_mean = data.mean(axis=(0, 1))
    data_std = data.std(axis=(0, 1))
    normalized_data = (data - data_mean) / (data_std + eps)

    data_tensor = torch.unsqueeze(
        torch.tensor(normalized_data, dtype=torch.float32), dim=0
    )  # (1, f, t)
    return Config.image_transform(data_tensor)


paths = submission["path"].values
if not models:
    test_preds = np.full((len(paths), 6), 1.0 / 6.0, dtype=np.float32)
else:
    BATCH_SIZE = 32  # safe default; does not change results
    all_preds = np.empty((len(paths), 6), dtype=np.float32)

    batch_tensors = []
    batch_indices = []

    for idx, path in enumerate(paths):
        x = load_and_preprocess_data(path)  # CPU tensor (1, 512, 512)
        batch_tensors.append(x)
        batch_indices.append(idx)

        if len(batch_tensors) == BATCH_SIZE:
            batch = torch.stack(batch_tensors, dim=0)  # (B, 1, 512, 512)
            probs = predict_batch(models, batch)
            all_preds[batch_indices, :] = probs
            batch_tensors.clear()
            batch_indices.clear()

    if batch_tensors:
        batch = torch.stack(batch_tensors, dim=0)
        probs = predict_batch(models, batch)
        all_preds[batch_indices, :] = probs

    test_preds = all_preds



## === cell 10
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

test_preds_arr = np.asarray(test_preds, dtype=np.float32)

if test_preds_arr.shape[0] != len(sample_sub):
    raise ValueError(
        f"Prediction rows ({test_preds_arr.shape[0]}) do not match sample submission rows ({len(sample_sub)})."
    )

aligned_preds = test_preds_arr

alpha = 0.015
aligned_preds = aligned_preds * (1.0 - alpha) + alpha * (1.0 / 6.0)

aligned_preds = np.clip(aligned_preds, 1e-7, 1.0)
aligned_preds = aligned_preds / aligned_preds.sum(axis=1, keepdims=True)

final_submission = prepare_submission(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
    test_df,
    aligned_preds,
)



## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2258480829.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m [0;31m# After fixing duplication upstream, this should now match exactly.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;32mif[0m [0mtest_preds_arr[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;34m!=[0m [0mlen[0m[0;34m([0m[0msample_sub[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m     raise ValueError(
[0m[1;32m     10[0m         [0;34mf"Prediction rows ({test_preds_arr.shape[0]}) do not match sample submission rows ({len(sample_sub)})."[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m     )

[0;31mValueError[0m: Prediction rows (1693) do not match sample submission rows (9850).

## === cell 11
final_submission.to_csv("submission.csv", index=None)
final_submission.head()
