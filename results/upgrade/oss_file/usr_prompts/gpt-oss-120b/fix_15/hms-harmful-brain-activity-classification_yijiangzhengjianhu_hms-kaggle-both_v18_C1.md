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

0.7023904592507219

# 6. Current score

1.29004

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the failing inference code with a robust fallback that does not depend on external model files. The new version computes class‑wise prior probabilities from the training set and uses these as constant predictions for every test sample, guaranteeing a correctly‑shaped DataFrame and a valid `.csv` submission. This eliminates the glob/model loading error and ensures the submission file is written with rows that sum to 1.'
- What this solution (achieved 1.68479) has done: 'The fix corrects the `fillna` usage when computing per‑patient class priors: instead of passing a NumPy array (which raises a `ValueError`), a Pandas Series indexed by the target columns is supplied so missing values are properly replaced with the global class priors. This allows the script to complete successfully and write a valid `submission.csv` file without altering the core modelling logic.'
- What this solution (achieved 1.33601) has done: 'I replace the overly‑specific patient‑wise priors with a safer fallback: use a patient’s prior only when that patient has contributed a reasonable number of votes (≥ 20); otherwise fall back to the global class priors. This reduces the chance of extreme, over‑confident predictions that drive the KL‑divergence up, moving the score closer to the target while keeping the overall logic unchanged.'
- What this solution (achieved 0.92622) has done: 'I keep the overall prior‑based approach but replace the hard cutoff with a smooth blending of patient‑specific priors and the global prior. The blend weight grows with the number of votes a patient contributed, so patients with many annotations keep their personalized distribution while those with few votes rely more on the global distribution. This small change preserves the core logic, guarantees rows sum to 1, and should reduce overly confident predictions that hurt KL divergence, moving the score closer to the target.'
- What this solution (achieved 0.92622) has done: 'I add a per‑eeg‑id prior and blend it with the existing patient‑wise and global priors using smooth weights based on the number of votes. This keeps the overall prior‑based strategy but gives extra information for recordings that appear often in the training set, which should reduce overly confident errors and move the KL‑divergence down toward the target score.'
- What this solution (achieved 1.41937) has done: 'I simplify the blending logic so that only the global class prior is used for every test sample, removing the patient‑ and EEG‑specific weightings that were causing overly confident predictions. This keeps the core prior‑based approach while making the predictions less extreme, which should lower the KL‑divergence and move the score closer to the target. The rest of the pipeline stays unchanged and a valid `submission.csv` is still written.'
- What this solution (achieved 0.90596) has done: 'I replace the simplistic “global‑only” prediction with a lightweight blending that uses patient‑specific and EEG‑specific priors when enough votes are available, otherwise falls back toward the global prior.  The blend weight grows smoothly with the number of votes, and a tiny smoothing exponent is applied to keep probabilities from becoming overly confident.  This preserves the original prior‑based logic while giving more personalized estimates, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.90762) has done: 'I slightly reduce the influence of patient‑ and EEG‑specific priors and increase the temperature smoothing, which makes the predictions more uniform and therefore less over‑confident. This is done by raising the temperature exponent from 0.9 to 0.7 and by enlarging the vote thresholds used to compute the blending weights (making wp and we smaller). These minimal adjustments keep the original prior‑blending logic intact while nudging the KL‑divergence closer to the target score.'
- What this solution (achieved 0.96822) has done: 'The update makes the predictions less over‑confident by (1) lowering the temperature exponent to 0.5 for stronger probability smoothing and (2) raising the vote‑count thresholds that give full weight to patient‑ and EEG‑specific priors, so most samples rely more on the global prior. These minimal tweaks keep the original blending logic intact but are expected to lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 1.27781) has done: 'I lower the KL‑divergence by making the predictions less dependent on patient‑ and EEG‑specific priors and by applying stronger smoothing. I raise the vote thresholds so most rows use the global prior, scale the blending weights to at most 0.5, and decrease the temperature exponent to 0.3. These small parameter tweaks keep the original prior‑blending logic intact while producing more uniform probabilities, moving the score closer to the lower‑is‑better target.'
- What this solution (achieved 1.3128) has done: 'I slightly reduce the influence of patient‑ and EEG‑specific priors and apply a milder temperature smoothing so the predictions become less extreme. By raising the vote thresholds, increasing the blend scaling factor, and using a higher temperature exponent, the blended probabilities stay closer to the global prior, which should lower the KL‑divergence toward the target while keeping the original blending logic unchanged.'
- What this solution (achieved 1.39588) has done: 'I fixed the pandas rename error and removed the unnecessary heavy EEG/spectrogram loading that caused undefined‑variable crashes. I also tuned the blending hyper‑parameters to rely much more on the global class prior (higher vote thresholds, a smaller BLEND_SCALE, and stronger temperature smoothing) so the predictions become less extreme, which should lower the KL‑divergence toward the target while keeping the original prior‑based logic intact.'
- What this solution (achieved 1.29004) has done: 'I slightly increase the influence of patient‑ and EEG‑specific priors and reduce the aggressive temperature‑based smoothing. By lowering the vote thresholds and raising `BLEND_SCALE` the model can use more personalized distributions, while a higher `TEMP_EXP` keeps probabilities from being overly uniform. These minimal adjustments keep the core prior‑blending logic unchanged but should lower the KL‑divergence toward the target.'

# 9. Code solution

## === cell 0
import glob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
FEATS2 = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]
DATA_TYPE = "both"
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)
test.head()

test = test.rename(columns={"spectrogram_id": "spec_id"})




## === cell 2
def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)), dtype="float32")
    for j, col in enumerate(FEATS2):
        x = eeg[col].values.astype("float32")
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0
        data[:, j] = x
    return data


import librosa
import pywt


def maddest(d, axis=None):
    return np.mean(np.abs(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")


USE_WAVELET = None
NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]
    img = np.zeros((100, 300, 4), dtype="float32")
    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk + 1]].values
            m = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                x1 = np.nan_to_num(x1, nan=m)
            else:
                x1[:] = 0
            m = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                x2 = np.nan_to_num(x2, nan=m)
            else:
                x2[:] = 0
            x = x1 - x2
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 300,
                n_fft=1024,
                n_mels=100,
                fmin=0,
                fmax=20,
                win_length=128,
            )
            width = (mel_spec.shape[1] // 30) * 30
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db
        img[:, :, k] /= 4.0
        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.show()
    return img




## === cell 3
val_transform = A.Compose([A.Resize(p=1.0, height=896, width=896), ToTensorV2(p=1.0)])



## === cell 4
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path)

global_vote_sums = train_df[TARGETS].sum()
global_priors = global_vote_sums / global_vote_sums.sum()
global_priors = global_priors.values.astype("float32")  # shape (6,)

patient_groups = train_df.groupby("patient_id")[TARGETS].sum()
patient_totals = patient_groups.sum(axis=1)  # total votes per patient
patient_priors = patient_groups.div(patient_totals.replace(0, np.nan), axis=0)
global_priors_series = pd.Series(global_priors, index=TARGETS)
patient_priors = patient_priors.fillna(global_priors_series)

eeg_groups = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_totals = eeg_groups.sum(axis=1)  # total votes per eeg_id
eeg_priors = eeg_groups.div(eeg_totals.replace(0, np.nan), axis=0)
eeg_priors = eeg_priors.fillna(global_priors_series)

MIN_VOTES_FOR_PATIENT = 200  # lower threshold to allow patient priors earlier
MAX_VOTES_FOR_FULL_WEIGHT = 800
MIN_VOTES_FOR_EEG = 200  # lower threshold for EEG priors
MAX_VOTES_FOR_EEG_FULL = 800
BLEND_SCALE = 0.2  # increase weight of specific priors
SMOOTH_EPS = 1e-6
TEMP_EXP = 0.7  # milder temperature smoothing (less uniform)

predictions = []

for _, row in test.iterrows():
    pid = row["patient_id"]
    eid = row["eeg_id"]

    blended = global_priors.copy()

    patient_votes = patient_totals.get(pid, 0)
    wp = 0.0
    if patient_votes >= MIN_VOTES_FOR_PATIENT:
        wp = (patient_votes - MIN_VOTES_FOR_PATIENT) / max(
            1, MAX_VOTES_FOR_FULL_WEIGHT - MIN_VOTES_FOR_PATIENT
        )
        wp = min(max(wp, 0.0), 1.0) * BLEND_SCALE

    eeg_votes = eeg_totals.get(eid, 0)
    we = 0.0
    if eeg_votes >= MIN_VOTES_FOR_EEG:
        we = (eeg_votes - MIN_VOTES_FOR_EEG) / max(
            1, MAX_VOTES_FOR_EEG_FULL - MIN_VOTES_FOR_EEG
        )
        we = min(max(we, 0.0), 1.0) * BLEND_SCALE

    if wp > 0.0:
        patient_prior = patient_priors.loc[pid].values.astype("float32")
        blended += wp * (patient_prior - global_priors)

    if we > 0.0:
        eeg_prior = eeg_priors.loc[eid].values.astype("float32")
        blended += we * (eeg_prior - global_priors)

    blended = np.maximum(blended, SMOOTH_EPS)
    blended = blended**TEMP_EXP
    probs = blended / blended.sum()
    predictions.append(probs)

test_pred = np.vstack(predictions).astype("float32")

CLASSES = TARGETS
test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

submission_path = "submission.csv"
test_pred_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(test_pred_df.head())
