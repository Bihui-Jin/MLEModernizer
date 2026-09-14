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

0.538301

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Diagnosis: The crash happens in `prepare_submission` because `test_preds` has many more rows than the sample submission (778,942 vs 9,850). This mismatch is caused earlier by merging `sample_submission` with `test_df` in cell 8, which duplicates rows when an `eeg_id` appears multiple times in `test.csv` (multiple `spectrogram_id` per `eeg_id`). Cell 9 then generates predictions for the duplicated `paths`, so `test_preds` length matches the expanded merge, not the original submission length.  
Patch summary: Fix cell 10 to align `test_preds` back to the `eeg_id` index expected by `sample_submission.csv` by aggregating predictions per `eeg_id` (mean over duplicates) and ordering them to match the submission file’s `eeg_id` order, then pass the aligned array into `prepare_submission`. This keeps the existing prediction logic intact and only corrects the shape mismatch at submission creation time.  
Updated cells: Only cell 10 is changed.  
Compatibility notes for cell k+1: `final_submission` remains a DataFrame with the same columns as before, so `final_submission.to_csv(...)` in cell 11 works unchanged.  
Assumptions: `submission` from cell 8 has the duplicated rows and contains an `eeg_id` column aligned with `test_preds`; averaging duplicate predictions is an acceptable deterministic consolidation when multiple spectrograms map to the same `eeg_id`.'
- What this solution (achieved 1.40995) has done: 'Your current pipeline likely scores poorly because it (a) normalizes each spectrogram sample independently, which removes absolute scale information that is useful for this KL-divergence target, and (b) produces overconfident probabilities; KL heavily penalizes confident wrong predictions. To move the score down toward the 0.538 target without changing the core model, I keep the same inference but apply a minimal, metric-aligned post-processing: (1) aggregate duplicate `eeg_id` predictions by mean (as you already do) and then (2) apply a light probability smoothing (Dirichlet/Laplace-style) to reduce overconfidence while preserving row-sum-to-1. I also ensure deterministic behavior and keep all paths and submission schema unchanged.'
- What this solution (achieved 1.40995) has done: 'Diagnosis: Cell 10 assumes `test_preds` has one row per `sample_submission` row (9850), but upstream in cell 9 predictions are generated only for the de-duplicated `submission` (one per unique `eeg_id`), resulting in 1693 rows. This mismatch triggers the explicit shape check and raises `ValueError`. The correct alignment is to expand the unique-eeg predictions back to the full `test_df`/`sample_submission` row set by mapping on `eeg_id`.  

Patch summary: In cell 10 only, replace the strict row-count check with an alignment step that merges `test_preds` (indexed by the de-duplicated submission’s `eeg_id`) onto the original `sample_submission` order via `eeg_id`. Keep the existing smoothing (`alpha`), clipping, normalization, and the call to `prepare_submission` unchanged in semantics. Add a safety check to ensure all `eeg_id` values were matched.  

Updated cells: (cell 10 only)  

Compatibility notes for cell k+1: `final_submission` remains a DataFrame with the same columns as `sample_submission` and the same number of rows (9850), so cell 11 (`to_csv` and `head`) works unchanged.  

Assumptions: The de-duplicated `submission` in cell 8/9 contains exactly one row per `eeg_id` and includes all `eeg_id` values present in `sample_submission` (true for this competition data); if not, the added null-check raise a clear error.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.538301), so we should legitimately improve the KL by making predictions less overconfident and better aligned to the label distribution without changing the model itself. The smallest safe lever for KL in this competition is probability calibration/smoothing: increasing the existing uniform-mix `alpha` reduces KL penalties from overly sharp wrong predictions. I keep the exact same preprocessing, model inference, and `eeg_id` alignment logic, and only adjust `alpha` upward to a moderate value that typically lowers KL on this task. The submission format and row order remain identical to `sample_submission.csv`, and probabilities are still clipped and renormalized to sum to 1.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is much worse than the target (0.538301), so we should reduce KL without changing the model or feature pipeline. The smallest, metric-aligned lever here is to increase the existing uniform-mixture smoothing (`alpha`) to make predictions less overconfident; KL heavily penalizes confident mistakes, and this typically improves the public score for this competition. I keep the same `eeg_id` alignment logic and submission formatting, and only adjust `alpha` plus add a tiny numerical safety renormalization (still identical semantics: probabilities sum to 1). Everything else (paths, preprocessing, model inference) remains unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.538301), so we should reduce KL with the smallest metric-aligned change while keeping the exact same model inference and preprocessing. The safest lever here is probability calibration: KL heavily punishes overconfident wrong predictions, so increasing the existing uniform-mixture smoothing (`alpha`) should improve the score without touching architecture, weights, or feature extraction. I keep your existing `eeg_id` alignment logic intact and only adjust `alpha`, plus keep the same clipping/renormalization so each row still sums to 1 (submission-valid). No paths, data loading, or model logic are changed.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.538301), so we should reduce KL with the smallest metric-aligned change while keeping your model inference and preprocessing intact. The safest lever is probability calibration: KL heavily penalizes overconfident wrong predictions, so increasing your existing uniform-mixture smoothing (`alpha`) should move the score downward without changing architecture or data flow. I keep the same `eeg_id` alignment/merge logic and submission formatting, and only adjust `alpha` plus keep the same clipping/renormalization so every row sums to 1 (submission-valid). Everything else remains unchanged.'

# 9. Code solution

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
    if "submission" not in globals():
        raise ValueError(
            "Predictions were generated for de-duplicated eeg_id rows, but the variable "
            "'submission' (with eeg_id order) is not available for alignment."
        )

    if test_preds_arr.shape[0] != len(submission):
        raise ValueError(
            f"Prediction rows ({test_preds_arr.shape[0]}) do not match the de-duplicated "
            f"'submission' rows ({len(submission)}), cannot align to sample_submission."
        )

    labels = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    preds_by_eeg = pd.DataFrame(test_preds_arr, columns=labels)
    preds_by_eeg["eeg_id"] = submission["eeg_id"].values

    aligned = sample_sub[["eeg_id"]].merge(
        preds_by_eeg, on="eeg_id", how="left", validate="many_to_one"
    )
    if aligned[labels].isnull().any().any():
        missing = int(aligned[labels].isnull().any(axis=1).sum())
        raise ValueError(
            f"Missing aligned predictions for {missing} rows after merging on eeg_id."
        )
    aligned_preds = aligned[labels].to_numpy(dtype=np.float32)
else:
    aligned_preds = test_preds_arr

alpha = 0.90
aligned_preds = aligned_preds * (1.0 - alpha) + alpha * (1.0 / 6.0)

aligned_preds = np.clip(aligned_preds, 1e-7, 1.0)
aligned_preds = aligned_preds / np.maximum(
    aligned_preds.sum(axis=1, keepdims=True), 1e-12
)

final_submission = prepare_submission(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
    test_df,
    aligned_preds,
)

final_submission.to_csv("submission.csv", index=None)
final_submission.head()
