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

0.516782941336992

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the failing model‑loading loop and replace it with a simple baseline that uses the overall class vote distribution from the training set. This avoids the missing directory error, ensures the predictions array has the correct shape, and guarantees that each row sums to 1 so the submission file is valid.'
- What this solution (achieved 1.41937) has done: 'I fixed the probability‑mapping logic that caused a ValueError when filling NaNs with a NumPy array. The new code safely normalizes spectrogram vote sums, replaces any zero‑sum rows with the global class prior, and maps test rows to these probabilities, guaranteeing a valid submission CSV where each row sums to 1.'
- What this solution (achieved 1.41937) has done: 'I add a per‑eeg ID probability source and blend it with the existing spectrogram‑based probabilities. If a spectrogram ID is missing, the code fall back to the corresponding eeg ID distribution, and ultimately to the global prior. This small change provides more detailed class priors without altering the model architecture, and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I add a light blending of each row’s spectrogram/eeg‑based probabilities with the overall class prior (using a small weight) and renormalise, which softens over‑confident predictions and should reduce the KL‑divergence toward the target. The change is confined to the probability‑building cell and keeps the rest of the pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'I reduce the unnecessary blending with the global prior (set BLEND_WEIGHT to 0.0) and add a tiny epsilon smoothing to avoid zero probabilities before renormalising. This keeps the core logic unchanged while making the predictions rely more on the spectrogram/eeg‑specific distributions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I added the missing standard imports (numpy, pandas, os, torch, torch.nn, timm, matplotlib, pywt, librosa) and placed them at the start of the script. This resolves the NameError issues across all cells, allowing the probability‑based baseline to run, produce a correctly shaped submission DataFrame, and write a valid `submission.csv` where each row sums to 1.'
- What this solution (achieved 1.41937) has done: 'I lower the smoothing constant ALPHA from 20.0 to 5.0 so that spectrogram‑ and EEG‑level vote distributions have more influence, and then blend the resulting probabilities slightly toward the global class prior (10 % weight). This modest adjustment keeps the core logic intact while making the predictions less uniform and better calibrated, which should reduce the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I lower the smoothing and blending hyper‑parameters so the predictions rely more on the spectrogram‑ and EEG‑specific vote distributions, which should reduce the KL‑divergence (lower is better). Specifically, I set `ALPHA` to 1.0 (to give higher weight to observed counts) and `BLEND_WEIGHT` to 0.0 (removing the extra pull toward the global prior). The rest of the pipeline remains unchanged.'
- What this solution (achieved 1.41937) has done: 'I make the script locate the correct data directory dynamically (handling the typical Kaggle `/kaggle/input` path as well as the relative `data/…` path) so the CSV files can be read, which removes the FileNotFoundError and the subsequent NameErrors. The rest of the logic (probability calculation, smoothing, and submission writing) is kept unchanged, preserving the core model behavior while ensuring a valid submission file is produced.'
- What this solution (achieved 1.41937) has done: 'I increase the smoothing `ALPHA` so each spectrogram/eeg distribution is blended more toward the global class prior, and add a small `BLEND_WEIGHT` to further pull the final predictions toward that prior. These tiny adjustments keep the original logic intact while making predictions less over‑confident, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I lower the smoothing constant (ALPHA) and remove the final blend toward the global prior, then weight the spectrogram and EEG‑specific probabilities by their sample counts instead of averaging them equally. This makes the predictions rely more on the observed per‑group vote distributions, which should reduce the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path


class Paths:
    def __init__(self, base_dir: str = None):
        possible_paths = [
            "/kaggle/input/hms-harmful-brain-activity-classification",
            "data/hms-harmful-brain-activity-classification",
            "./data/hms-harmful-brain-activity-classification",
        ]
        if base_dir is None:
            for p in possible_paths:
                if Path(p).exists():
                    base_dir = p
                    break
            else:
                raise FileNotFoundError(
                    "Could not locate the dataset directory. Checked: "
                    + ", ".join(possible_paths)
                )
        self.base = Path(base_dir)
        self.train_csv = self.base / "train.csv"
        self.test_csv = self.base / "test.csv"
        self.out = Path(".")


paths = Paths()




## === cell 1
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_df = pd.read_csv(paths.train_csv)
test_df = pd.read_csv(paths.test_csv)

global_sums = train_df[TARGETS].sum().astype(np.float64)
global_prior = (global_sums / global_sums.sum()).values

spec_groups = train_df.groupby("spectrogram_id")[TARGETS].sum()
spec_counts = train_df.groupby("spectrogram_id").size()
spec_probs = spec_groups.div(spec_groups.sum(axis=1), axis=0)

nan_rows = spec_probs.isna().any(axis=1)
if nan_rows.any():
    fill_df = pd.DataFrame(
        np.tile(global_prior, (nan_rows.sum(), 1)),
        index=spec_probs[nan_rows].index,
        columns=TARGETS,
    )
    spec_probs.loc[nan_rows] = fill_df

eeg_groups = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_counts = train_df.groupby("eeg_id").size()
eeg_probs = eeg_groups.div(eeg_groups.sum(axis=1), axis=0)

nan_rows_eeg = eeg_probs.isna().any(axis=1)
if nan_rows_eeg.any():
    fill_df_eeg = pd.DataFrame(
        np.tile(global_prior, (nan_rows_eeg.sum(), 1)),
        index=eeg_probs[nan_rows_eeg].index,
        columns=TARGETS,
    )
    eeg_probs.loc[nan_rows_eeg] = fill_df_eeg

ALPHA = 0.5  # smaller smoothing constant → higher weight to per‑group counts
epsilon = 1e-8  # tiny epsilon to avoid zeros before renormalisation
BLEND_WEIGHT = 0.0  # remove extra pull toward the global prior


def get_prob(sid, eid):
    """
    Produce a probability vector for a test row.
    Use spectrogram and EEG distributions when available, each smoothed toward
    the global prior with Dirichlet‑type smoothing (controlled by ALPHA).
    The two sources are combined proportionally to their sample counts.
    """
    weighted_sum = np.zeros(len(TARGETS))
    total_weight = 0.0

    if sid in spec_probs.index:
        prob = spec_probs.loc[sid].values
        cnt = spec_counts.loc[sid]
        weight = cnt / (cnt + ALPHA) if (cnt + ALPHA) != 0 else 0.0
        smoothed = weight * prob + (1 - weight) * global_prior
        weighted_sum += cnt * smoothed
        total_weight += cnt

    if eid in eeg_probs.index:
        prob = eeg_probs.loc[eid].values
        cnt = eeg_counts.loc[eid]
        weight = cnt / (cnt + ALPHA) if (cnt + ALPHA) != 0 else 0.0
        smoothed = weight * prob + (1 - weight) * global_prior
        weighted_sum += cnt * smoothed
        total_weight += cnt

    if total_weight == 0.0:
        return global_prior

    return weighted_sum / total_weight




## === cell 2
test_probs = np.vstack(
    test_df.apply(
        lambda row: get_prob(row["spectrogram_id"], row["eeg_id"]), axis=1
    ).values
)

assert test_probs.shape == (len(test_df), len(TARGETS)), "Shape mismatch after mapping."

test_probs = np.clip(test_probs, epsilon, None)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

test_probs = (1 - BLEND_WEIGHT) * test_probs + BLEND_WEIGHT * global_prior
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = test_probs

submission_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Submission shape: {sub.shape}")




## === cell 3
row_sums = sub[TARGETS].sum(axis=1)
print(f"Average row sum (should be 1.0): {row_sums.mean():.6f}")
