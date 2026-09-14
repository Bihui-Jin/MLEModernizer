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
submission = submission.merge(test_df, on="eeg_id", how="left")
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
final_submission = prepare_submission(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
    test_df,
    np.array(test_preds),
)



## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2441815121.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m final_submission = prepare_submission(
[0m[1;32m      2[0m     [0;34m"/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mtest_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mtest_preds[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m )

[0;32m/tmp/ipykernel_11/2831008710.py[0m in [0;36mprepare_submission[0;34m(submission_file, test_df, test_preds)[0m
[1;32m      3[0m     [0mlabels[0m [0;34m=[0m [0;34m[[0m[0;34m"seizure"[0m[0;34m,[0m [0;34m"lpd"[0m[0;34m,[0m [0;34m"gpd"[0m[0;34m,[0m [0;34m"lrda"[0m[0;34m,[0m [0;34m"grda"[0m[0;34m,[0m [0;34m"other"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0;32mfor[0m [0mi[0m[0;34m,[0m [0mlabel[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mlabels[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m         [0msubmission[0m[0;34m[[0m[0;34mf"{label}_vote"[0m[0;34m][0m [0;34m=[0m [0mtest_preds[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0mi[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m     [0;32mreturn[0m [0msubmission[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__setitem__[0;34m(self, key, value)[0m
[1;32m   4309[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4310[0m             [0;31m# set column[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4311[0;31m             [0mself[0m[0;34m.[0m[0m_set_item[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4312[0m [0;34m[0m[0m
[1;32m   4313[0m     [0;32mdef[0m [0m_setitem_slice[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mkey[0m[0;34m:[0m [0mslice[0m[0;34m,[0m [0mvalue[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m_set_item[0;34m(self, key, value)[0m
[1;32m   4522[0m         [0mensure[0m [0mhomogeneity[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4523[0m         """
[0;32m-> 4524[0;31m         [0mvalue[0m[0;34m,[0m [0mrefs[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_sanitize_column[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4525[0m [0;34m[0m[0m
[1;32m   4526[0m         if (

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m_sanitize_column[0;34m(self, value)[0m
[1;32m   5264[0m [0;34m[0m[0m
[1;32m   5265[0m         [0;32mif[0m [0mis_list_like[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 5266[0;31m             [0mcom[0m[0;34m.[0m[0mrequire_length_match[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mindex[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   5267[0m         [0marr[0m [0;34m=[0m [0msanitize_array[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mindex[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mallow_2d[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5268[0m         if (

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/common.py[0m in [0;36mrequire_length_match[0;34m(data, index)[0m
[1;32m    571[0m     """
[1;32m    572[0m     [0;32mif[0m [0mlen[0m[0;34m([0m[0mdata[0m[0;34m)[0m [0;34m!=[0m [0mlen[0m[0;34m([0m[0mindex[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 573[0;31m         raise ValueError(
[0m[1;32m    574[0m             [0;34m"Length of values "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    575[0m             [0;34mf"({len(data)}) "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Length of values (778942) does not match length of index (9850)

## === cell 11
final_submission.to_csv("submission.csv", index=None)
final_submission.head()
