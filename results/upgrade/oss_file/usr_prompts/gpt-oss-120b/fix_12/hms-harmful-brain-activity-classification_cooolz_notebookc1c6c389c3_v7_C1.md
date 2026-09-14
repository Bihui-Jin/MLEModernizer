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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.4904182350328406

# 6. Current score

1.38829

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix adds a safe fallback for missing model weight files: the inference loop now catches FileNotFoundError and, if no weights are loaded, computes class‑wise average vote probabilities from the training data and uses these as predictions for every test sample. This guarantees a valid `submission.csv` with correctly‑sized probability columns, eliminating the file‑not‑found and shape‑mismatch errors while keeping the original model code intact.'
- What this solution (achieved 0.77767) has done: 'I replace the naïve global‑average fallback with a patient‑aware baseline: for each test sample we first look up its patient_id in the training data and use that patient’s vote distribution (normalized to a probability vector). If the patient is unseen we fall back to the overall class frequencies. Blending a small portion of the global distribution keeps the probabilities well‑behaved. This change preserves the original model‑inference pipeline while giving a more informative prior, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.68479) has done: 'I adjust the fallback prediction logic so that when a patient appears in the training data we use its exact vote‑distribution without blending with the global prior (the blend was weakening the patient‑specific signal and hurting KL). For unseen patients we keep the global distribution. After obtaining each row I also renormalize to guarantee the probabilities sum to 1, which prevents submission errors and aligns the output more closely with the true label distribution, thereby reducing the KL score toward the target.'
- What this solution (achieved 1.37636) has done: 'I add a missing `timm` import (so the model can be instantiated if weights exist) and replace the raw patient‑wise vote fallback with a lightly smoothed version: each patient’s vote counts are blended with the global class counts using a small Dirichlet‑style weight (`alpha=0.1`). This keeps the patient‑specific signal while avoiding extreme probabilities that hurt KL, and still guarantees rows sum to 1, moving the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I replace the patient‑wise fallback with a simple global‑probability baseline: compute the overall vote counts from the training set, normalize them to a probability vector, and use this same vector for every test row. This removes noisy patient‑specific estimates that were inflating the KL loss, moving the score closer to the lower target while keeping all core logic unchanged.'
- What this solution (achieved 1.26371) has done: 'The fallback prediction is enhanced to use patient‑specific vote distributions when a patient appears in the training set, otherwise it falls back to the overall class probabilities.  Each patient’s votes are normalized to a probability vector and very small epsilons are added then re‑normalized to guarantee rows sum to 1, which typically lowers the KL‑divergence toward the target while keeping the original model‑loading logic unchanged.'
- What this solution (achieved 0.81597) has done: 'I add a minimal placeholder `AlaskaDataIter` so the script never crashes if weight files are present, and I improve the fallback prediction by blending patient‑specific probabilities with the global class distribution (70 % patient, 30 % global). This smooths extreme patient priors, keeps rows summing to 1, and is expected to lower the KL‑divergence toward the target score while preserving the original model logic.'
- What this solution (achieved 1.0413) has done: 'I lower the patient‑specific weight in the fallback prediction so the global class distribution dominates (blend_ratio = 0.3). This reduces the impact of potentially noisy patient priors, yielding smoother probabilities that better match the KL metric and move the score down toward the target while keeping all core logic unchanged.'
- What this solution (achieved 1.41428) has done: 'I adjust the fallback prediction logic (used when model weights are missing) to apply a modest Dirichlet‑style smoothing of the patient‑specific vote counts and increase the blend weight toward patient information (50 % patient + 50 % global). This keeps the original architecture untouched, guarantees valid probability rows, and is expected to reduce the KL‑divergence score, moving it closer to the lower target.'
- What this solution (achieved 1.41916) has done: 'I adjust the fallback prediction logic to rely more on the stable global class distribution and apply stronger Dirichlet‑style smoothing for patient‑specific counts. By increasing the smoothing strength (`alpha`) and reducing the patient‑blend weight (`blend_ratio`), the predictions become less noisy and should lower the KL‑divergence, moving the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 1.38829) has done: 'I adjust the fallback prediction logic to use a much lighter Dirichlet smoothing (α = 0.1) and give the patient‑specific distribution a stronger weight (blend ≈ 0.7). This keeps the overall pipeline unchanged while providing more informative priors, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
import torch
import gc
import os
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm  # ensure model creation works if weights are present



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights": [
        "/kaggle/input/hms-baseline/fold0_epoch_4_val_loss_0.589956.pth",
        "/kaggle/input/hms-baseline/fold1_epoch_4_val_loss_0.519593.pth",
        "/kaggle/input/hms-baseline/fold2_epoch_4_val_loss_0.499242.pth",
        "/kaggle/input/hms-baseline/fold3_epoch_4_val_loss_0.584708.pth",
        "/kaggle/input/hms-baseline/fold4_epoch_4_val_loss_0.655207.pth",
    ],
    "flip": True,
}
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 2
class Net(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 3
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y = model(X)
                y = softmax(y)
                if CFG["flip"]:
                    X_flip = torch.flip(X, [3])
                    y_flip = model(X_flip)
                    y_flip = softmax(y_flip)
                    y = (y + y_flip) / 2.0
                preds.append(y.cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0, len(TARGETS)))




## === cell 4
class AlaskaDataIter(Dataset):
    def __init__(self, df, training_flag=False, shuffle=False):
        self.df = df
        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.zeros(4, 224, 224, dtype=torch.float32)




## === cell 5
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
all_predictions = []

weights_loaded = False
for model_weight in CFG["weights"]:
    if not os.path.isfile(model_weight):
        continue
    weights_loaded = True
    test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
    )
    model = Net()
    state_dict = torch.load(model_weight, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    preds = inference_function(test_loader, model, device)
    all_predictions.append(preds)
    torch.cuda.empty_cache()
    gc.collect()

if weights_loaded:
    predictions = np.mean(np.stack(all_predictions), axis=0)
else:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    train_df = pd.read_csv(train_path)

    eps = 1e-9

    global_counts = train_df[TARGETS].sum()
    global_probs = global_counts / global_counts.sum()
    global_probs = np.clip(global_probs.values.astype(float), eps, None)
    global_probs = global_probs / global_probs.sum()

    alpha = 0.1  # small smoothing to retain patient signal
    patient_counts = train_df.groupby("patient_id")[TARGETS].sum()
    patient_counts_smooth = patient_counts + alpha * global_counts
    patient_probs = patient_counts_smooth.div(patient_counts_smooth.sum(axis=1), axis=0)

    patient_probs_test = patient_probs.reindex(test_df["patient_id"]).values

    mask_missing = np.isnan(patient_probs_test).any(axis=1)
    patient_probs_test[mask_missing] = global_probs

    blend_ratio = 0.7
    predictions = blend_ratio * patient_probs_test + (1 - blend_ratio) * global_probs

    predictions = np.clip(predictions, eps, None)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)



## === cell 7
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
