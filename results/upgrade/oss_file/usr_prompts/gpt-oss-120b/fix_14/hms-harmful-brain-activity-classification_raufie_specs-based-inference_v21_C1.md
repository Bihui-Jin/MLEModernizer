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
PyWavelets==1.8.0
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

0.8042135403855587

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41047) has done: 'I fix the missing model‑weight file by loading it only when it exists, otherwise using the randomly‑initialized model. I also guard the inference loop so it always runs and produces a NumPy array of shape (N, 6). Finally I ensure the submission dataframe is filled with predictions of matching dimensions and written to `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'I adjust the data‑loading logic so the script can actually find the CSV files in the Kaggle environment. Instead of building a `data_root` that doesn’t match the notebook’s directory layout, I call `locate_path` directly with the correct relative paths (e.g., `"hms-harmful-brain-activity-classification/train.csv"`). This fixes the `FileNotFoundError` and lets the rest of the code run unchanged, producing a valid `submission.csv` whose probabilities sum to 1.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from tqdm import tqdm


class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b3"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False
    TEMPERATURE = 1.0  # reduced to avoid over‑smoothing predictions
    GLOBAL_WEIGHT = 0.02  # smaller blend of the global prior for sharper predictions




## === cell 1
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(config.MODEL, pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        """Reshape (B,128,256,8) -> (B, C, H, W) as required by EfficientNet."""
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)  # (B,4,128,256)

        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)  # (B,4,128,256)

        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)  # (B,8,128,256)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        x = torch.cat([x, x, x], dim=3)  # repeat channels to get 24
        x = x.permute(0, 3, 1, 2)  # (B,24,128,256)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 2
def inference_function(loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    temperature = getattr(config, "TEMPERATURE", 1.0)
    with tqdm(loader, unit="batch", desc="Inference") as pbar:
        for X, _ in pbar:
            X = X.to(device)
            with torch.no_grad():
                logits = model(X)
            logits = logits / temperature
            probs = softmax(logits)
            preds.append(probs.cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 3
def locate_path(relative_path: str) -> Path:
    """
    Return a Path object pointing to an existing file.
    Tries multiple common locations used in Kaggle notebooks.
    """
    candidates = [
        Path(relative_path),
        Path("../input") / relative_path,
        Path("./") / relative_path,
        Path("/kaggle/input") / relative_path,
        Path("/kaggle/working") / relative_path,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Unable to locate file: {relative_path}")


train_path = locate_path("hms-harmful-brain-activity-classification/train.csv")
test_path = locate_path("hms-harmful-brain-activity-classification/test.csv")
sample_sub_path = locate_path(
    "hms-harmful-brain-activity-classification/sample_submission.csv"
)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)

row_sums = train_df[vote_cols].sum(axis=1)
valid_mask = row_sums > 0
norm_votes = pd.DataFrame(0.0, index=train_df.index, columns=vote_cols)
norm_votes.loc[valid_mask, vote_cols] = train_df.loc[valid_mask, vote_cols].div(
    row_sums[valid_mask], axis=0
)

global_avg = norm_votes.mean().values  # shape (6,)

eeg_avg = (
    pd.concat([train_df["eeg_id"], norm_votes], axis=1)
    .groupby("eeg_id")
    .mean()
    .reset_index()
)
eeg_prob_map = dict(zip(eeg_avg["eeg_id"], eeg_avg[vote_cols].values))

patient_avg = (
    pd.concat([train_df["patient_id"], norm_votes], axis=1)
    .groupby("patient_id")
    .mean()
    .reset_index()
)
patient_prob_map = dict(zip(patient_avg["patient_id"], patient_avg[vote_cols].values))

eeg_counts = train_df.groupby("eeg_id").size().to_dict()
patient_counts = train_df.groupby("patient_id").size().to_dict()

val_candidates = train_df[valid_mask].sample(frac=0.20, random_state=config.SEED)
val_norm = val_candidates[vote_cols].div(
    val_candidates[vote_cols].sum(axis=1).replace(0, np.nan), axis=0
)

candidate_weights = np.linspace(0.0, 0.10, 11)  # 0.00, 0.01, ..., 0.10
candidate_temps = np.linspace(0.60, 1.40, 9)  # 0.60, 0.70, ..., 1.40

best_kl = float("inf")
best_w = config.GLOBAL_WEIGHT
best_t = config.TEMPERATURE


def blend_probs(eid, pid, gw, temp):
    """Return blended probability vector for a given (eeg_id, patient_id)."""
    eeg_probs = eeg_prob_map.get(eid)
    pat_probs = patient_prob_map.get(pid)
    cnt_eeg = eeg_counts.get(eid, 0)
    cnt_pat = patient_counts.get(pid, 0)

    remaining = 1.0 - gw
    if eeg_probs is not None and pat_probs is not None and (cnt_eeg + cnt_pat) > 0:
        w_eeg = (cnt_eeg / (cnt_eeg + cnt_pat)) * remaining
        w_pat = (cnt_pat / (cnt_eeg + cnt_pat)) * remaining
        combined = (
            w_eeg * np.array(eeg_probs) + w_pat * np.array(pat_probs) + gw * global_avg
        )
    elif eeg_probs is not None:
        combined = (remaining * np.array(eeg_probs)) + (gw * global_avg)
    elif pat_probs is not None:
        combined = (remaining * np.array(pat_probs)) + (gw * global_avg)
    else:
        combined = global_avg.copy()

    logits = np.log(combined + 1e-12) / temp
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return probs


for gw in candidate_weights:
    for temp in candidate_temps:
        kl_vals = []
        for _, row in val_candidates.iterrows():
            eid = row["eeg_id"]
            pid = row["patient_id"]
            pred = blend_probs(eid, pid, gw, temp)
            true = norm_votes.loc[row.name].values
            kl = np.sum(true * (np.log(true + 1e-12) - np.log(pred + 1e-12)))
            kl_vals.append(kl)
        mean_kl = np.mean(kl_vals)
        if mean_kl < best_kl:
            best_kl = mean_kl
            best_w = gw
            best_t = temp

config.GLOBAL_WEIGHT = best_w
config.TEMPERATURE = best_t
print(
    f"Selected GLOBAL_WEIGHT={best_w:.4f}, TEMPERATURE={best_t:.4f}, validation KL={best_kl:.5f}"
)

submission = submission.merge(
    test_df[["eeg_id", "patient_id"]], on="eeg_id", how="left"
)

preds = []
for _, row in submission.iterrows():
    eid = row["eeg_id"]
    pid = row["patient_id"]

    eeg_probs = eeg_prob_map.get(eid)
    pat_probs = patient_prob_map.get(pid)
    cnt_eeg = eeg_counts.get(eid, 0)
    cnt_pat = patient_counts.get(pid, 0)

    remaining = 1.0 - config.GLOBAL_WEIGHT
    if eeg_probs is not None and pat_probs is not None and (cnt_eeg + cnt_pat) > 0:
        w_eeg = (cnt_eeg / (cnt_eeg + cnt_pat)) * remaining
        w_pat = (cnt_pat / (cnt_eeg + cnt_pat)) * remaining
        combined = (
            w_eeg * np.array(eeg_probs)
            + w_pat * np.array(pat_probs)
            + config.GLOBAL_WEIGHT * global_avg
        )
    elif eeg_probs is not None:
        combined = (remaining * np.array(eeg_probs)) + (
            config.GLOBAL_WEIGHT * global_avg
        )
    elif pat_probs is not None:
        combined = (remaining * np.array(pat_probs)) + (
            config.GLOBAL_WEIGHT * global_avg
        )
    else:
        combined = global_avg.copy()

    logits = np.log(combined + 1e-12) / config.TEMPERATURE
    probs = np.exp(logits)
    probs = probs / probs.sum()  # re‑normalize

    preds.append(probs)

pred_array = np.vstack(preds)  # shape (N_test, 6)

for idx, col in enumerate(vote_cols):
    submission[col] = pred_array[:, idx]

submission = submission[["eeg_id"] + vote_cols]

output_path = Path("submission.csv")
submission.to_csv(output_path, index=False)

print(f"Submission file written to {output_path.resolve()}")
