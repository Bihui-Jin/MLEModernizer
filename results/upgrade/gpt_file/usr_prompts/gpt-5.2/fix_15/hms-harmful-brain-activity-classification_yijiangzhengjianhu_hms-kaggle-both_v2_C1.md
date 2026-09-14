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

0.4036638285480033

# 6. Current score

0.85059

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The crash happens because the notebook expects external `.pt` model files under `/kaggle/input/model90-both-5model/...`, but that dataset isn’t present in your environment, so inference never runs and no `submission.csv` is written. I keep your preprocessing and DataGenerator logic intact, but add a safe fallback that produces a valid, metric-reasonable submission when no model checkpoints are found. Since the metric is KL divergence (lower is better) and you currently have no score at all, the fallback use the empirical class prior from `train.csv` (normalized) rather than uniform probabilities, which is typically better than uniform without changing core modeling logic. The output always be a correctly formatted `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.43453) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.40366), and the main reason is that you’re effectively submitting a global class-prior fallback rather than real per-sample predictions (because the external model dataset isn’t available). To move the score toward the target with minimal semantic changes, I (1) make the code prefer loading any available `.pt` models from common Kaggle locations (including `/kaggle/working`) without changing your inference logic, and (2) if no models exist, replace the pure global prior with a patient-conditional prior (still derived only from `train.csv`), which typically reduces KL vs a single global distribution. I also keep the submission normalization/safety checks and ensure `submission.csv` is always written.'
- What this solution (achieved 1.06434) has done: 'Your current score is far worse than the target (lower-is-better), and the main bottleneck is that you’re predicting via a patient/global prior fallback rather than true per-sample inference. With minimal semantic changes, I (1) fix a key data bug: your `test` dataframe never defines `spec_id` expected by the generator, and (2) introduce an `offset` value in test that mimics the train-time “center 10s” crop (using the same 300-frame windowing) so the spectrogram slice is aligned similarly to training. I also make the fallback prior slightly better calibrated by using a small Dirichlet/Laplace smoothing on patient/global priors (still only from `train.csv`) to reduce overconfident zeros which can hurt KL. These changes preserve your architecture/inference loop and only adjust metadata alignment and probability smoothing.'
- What this solution (achieved 0.77245) has done: 'Your score is still far from the target (lower-is-better), and the main lever available without changing your model/inference core is improving the fallback predictions when no `.pt` checkpoints are actually used. I keep your spectrogram/EEG feature extraction and DataGenerator logic intact, but (1) fix the test-time offset handling so it matches training semantics (use the provided `test["offset"]=300` instead of forcing `offset=0` in `mode=="test"`), and (2) replace the current patient/global “mean of per-row normalized votes” fallback with a more metric-aligned Dirichlet-multinomial posterior over *vote counts* (patient-conditional, with global backoff), which typically reduces KL by avoiding miscalibrated distributions. I also keep the same submission safety normalization and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.76373) has done: 'Your current score (0.77245, lower-is-better) is still far above the target (0.40366), so we should cautiously improve it. Since no usable `.pt` models are likely being loaded, the only lever without changing your modeling core is improving the fallback probabilities to better match the KL metric. I keep your entire feature/extraction and inference code intact, but replace the fallback from “patient vote-count posterior” to a more label-aligned fallback: compute per-`eeg_id` targets from `train.csv` the same way the competition defines them (normalize votes to probabilities, then average across overlapping segments), then back off to patient/global averages when an `eeg_id` is unseen. This stays strictly within metadata-only information (no leakage from test labels) and typically moves KL substantially toward the target versus a pure patient-count prior.'
- What this solution (achieved 0.76843) has done: 'Your current score (0.76373, lower-is-better) is still far above the target (0.40366), and since you likely aren’t actually loading working `.pt` checkpoints, the only safe lever is improving the fallback distribution while keeping your core inference/feature logic unchanged. I keep all model/inference code intact, but make the fallback more KL-aligned by (1) computing the training “true” target distribution per row using the competition’s vote normalization, then (2) estimating a smoothed Dirichlet-multinomial posterior per `eeg_id` (and backing off to `patient_id` then global) using *vote counts* rather than averaging already-normalized probabilities. This reduces overconfidence and better matches the evaluation target construction, usually lowering KL without any leakage. I also ensure the fallback is applied even if `.pt` files exist but cannot be loaded (state_dict-only), so the pipeline always produces a valid `submission.csv`.'
- What this solution (achieved 0.74211) has done: 'Your current score (0.76843, lower-is-better) is still far above the target (0.40366), and because no usable `.pt` models are being loaded, the only safe lever is to make the metadata-only fallback more KL-aligned without changing your feature/model core. I keep your entire spectrogram/EEG generation and inference loop intact, but improve the fallback by (1) aggregating train targets at the *eeg_id level* the same way the competition target behaves (normalize votes per row, then average across overlapping segments), and (2) backing off eeg_id → patient_id → global with small smoothing and a mix to avoid overconfident spikes that hurt KL. This is a minimal change confined to the “no model loaded” branch and keeps the submission format/normalization guarantees identical. It should move the score downward (better) toward the target while staying stable and within constraints.'
- What this solution (achieved 0.76617) has done: 'Your current score (0.74211, lower-is-better) is still far above the target (0.40366), so we should improve it cautiously without changing your model/inference core. Since you likely still aren’t loading usable `.pt` checkpoints, the biggest safe lever is to make the metadata-only fallback more KL-aligned by matching how targets are constructed: aggregate *vote counts* at the `eeg_id` level first, then convert to probabilities (instead of averaging already-normalized segment probabilities). I keep your eeg_id→patient_id→global backoff, but apply Dirichlet/Laplace smoothing on the aggregated counts (more principled for KL) and use a slightly smaller global-mixing to preserve specificity. Everything else (feature construction, generator, inference loop, submission formatting) stays the same and still always writes a valid `submission.csv`.'
- What this solution (achieved 0.7651) has done: 'Your current score (0.76617, lower-is-better) is still far from the target (0.40366), and since you likely aren’t loading any usable `.pt` checkpoints, the only minimal, core-logic-preserving lever is improving the metadata-only fallback to better match the competition’s target construction. I keep your entire feature generation + inference loop unchanged, but replace the fallback from “vote-count posterior” to a closer proxy of the evaluation target: compute per-row vote probabilities, then aggregate to `eeg_id` and `patient_id` by averaging those probabilities (not summing counts), with light Dirichlet-style smoothing and global backoff. This aligns the fallback with how overlapping train segments relate to a single `eeg_id`, typically lowering KL without any leakage. Submission formatting and per-row normalization stay intact and a valid `submission.csv` is always written.'
- What this solution (achieved 0.79816) has done: 'Your current score (0.7651, lower-is-better) is still far from the target (0.40366), so we should improve it while keeping your model/inference logic intact. Since you likely still aren’t loading any usable `.pt` models, the only safe lever is making the metadata-only fallback closer to the competition target distribution: use the vote-probabilities but *weight/aggregate by number of annotators* per row (more reliable rows contribute more), then back off eeg_id → patient_id → global. I also add a small temperature/power calibration on the fallback probabilities (with mixing) to reduce overconfident peaks that tend to hurt KL, while keeping strict per-row normalization. Everything else (feature extraction, generator, inference loop, submission format) remains unchanged.'
- What this solution (achieved 0.7773) has done: 'Your current score (0.79816, lower-is-better) is still far above the target (0.40366), and since no usable `.pt` checkpoints are being loaded the only minimal lever is improving the metadata-only fallback while keeping your whole feature/inference pipeline intact. I keep your generator, spectrogram building, and model-loading logic unchanged, but make the fallback more KL-aligned by (1) using per-row vote-probabilities averaged at `eeg_id`/`patient_id` (your current approach) while also (2) blending in a *Dirichlet posterior from aggregated vote counts* (adds robustness where row-prob averages are noisy), and (3) replacing the fixed global-mix/power with a small grid search on train OOF (grouped by `eeg_id`) to pick the best `(mix_eeg, mix_patient, power)` for KL. This stays within train-only metadata, doesn’t change the “core model” path, and should reduce KL meaningfully compared to the current hand-tuned fallback. The script still runs end-to-end and always writes a valid `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 0.7773) has done: 'Your current score is worse than the target (lower is better), and since you likely still aren’t loading usable `.pt` checkpoints, the safest way to move toward the target with minimal change is to improve the *fallback* distribution while keeping your feature extraction, generator, and inference loop intact. I replace the in-sample (leaky) “OOF tuning” with a proper group-aware cross-validation on `eeg_id` to pick the fallback calibration/mixing parameters without using the same eeg_id for both fitting and scoring, which should generalize better to the test set. I also fix the weighted group-mean computation (it currently relies on a fragile `groupby.apply` + outer frame indexing) to a deterministic, correct weighted aggregation. Everything else stays the same, and the script still always writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.85059) has done: 'Your current score (0.7773, lower-is-better) is far above the target (0.40366), and since no usable `.pt` checkpoints are being loaded, the only lever that respects your “core logic unchanged” constraint is improving the metadata-only fallback to better match the competition’s target distribution. I keep your entire feature generation and inference code intact, but replace the slow/fragile CV grid search with a deterministic, group-aware CV that (a) evaluates the fallback at the same granularity as the submission (`eeg_id`) and (b) learns *only* lightweight mixing weights between the existing components (eeg-level mean, patient-level mean, and global prior) with small Dirichlet smoothing to avoid KL blowups. This reduces compute (fits in the 600s limit) and usually improves KL because it calibrates the fallback toward the label construction without changing the model path. The script still always write a valid `submission.csv` with rows summing to 1.'

# 9. Code solution

## === cell 0
import glob

import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import torch
import tqdm



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

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

if "offset" not in test.columns:
    test["offset"] = 300

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()




## === cell 2
def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)))
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
import pywt, librosa


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


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


for i, eeg_id in enumerate(EEG_IDS2):
    img = spectrogram_from_eeg(f"{PATH2}{eeg_id}.parquet", i < DISPLAY)
    all_eegs2[eeg_id] = img




## === cell 3
class DataGenerator:
    "Generates data for Keras"

    def __init__(
        self,
        data,
        specs=None,
        eeg_specs=None,
        raw_eegs=None,
        augment=False,
        mode="train",
        data_type=DATA_TYPE,
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.on_epoch_end()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X, y = self.data_generation(index)
        if self.augment:
            X = self.augmentation(X)
        return X, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def data_generation(self, index):
        if self.data_type == "both":
            X, y = self.generate_all_specs(index)
        elif self.data_type == "eeg" or self.data_type == "kaggle":
            X, y = self.generate_specs(index)
        elif self.data_type == "raw":
            X, y = self.generate_raw(index)
        return X, y

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]

        offset = int(getattr(row, "offset", 300) / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]

        offset = int(getattr(row, "offset", 300) / 2)

        if self.data_type == "eeg":
            img = self.eeg_specs[row.eeg_id]
        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]
            img = np.stack(imgs, axis=-1)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 1]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 1]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 3]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 2]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 4
def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_gen_both))):
            batch_data, _ = test_gen_both[i]  # X: (512,512,3)
            batch_data = (
                torch.from_numpy(batch_data).permute(2, 0, 1).unsqueeze(0).to(device)
            )  # (1,3,512,512)
            logits = model(batch_data)
            pred_list.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    pred_arr = np.concatenate(pred_list, axis=0)
    return pred_arr


preds = []

test_gen_both = DataGenerator(
    test, mode="test", data_type="both", specs=spectrograms2, eeg_specs=all_eegs2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_any(model_path, map_location):
    obj = torch.load(model_path, map_location=map_location)
    if isinstance(obj, torch.nn.Module):
        return obj
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        raise ValueError(
            f"Checkpoint at {model_path} contains a state_dict but no model definition was provided."
        )
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        raise ValueError(
            f"Checkpoint at {model_path} appears to be a state_dict; cannot load without model class."
        )
    raise ValueError(f"Unrecognized checkpoint format at {model_path}: {type(obj)}")


CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

candidate_globs = [
    "/kaggle/input/model90-both-5model/model_90/*/*.pt",
    "/kaggle/input/**/**/*.pt",
    "/kaggle/working/**/*.pt",
]
model_paths = []
seen = set()
for pat in candidate_globs:
    for p in glob.glob(pat, recursive=True):
        if p.endswith(".pt") and (p not in seen):
            seen.add(p)
            model_paths.append(p)
model_paths = sorted(model_paths)

loaded_any = False
if len(model_paths) > 0:
    for model_path in model_paths:
        try:
            print("Loading:", model_path)
            model = load_model_any(model_path, map_location=device)
            pred = run_inference_loop(model, test_gen_both, device)
            preds.append(pred)
            loaded_any = True
        except Exception as e:
            print(
                f"Skipping model (could not load/run): {model_path}\n  Reason: {repr(e)}"
            )

if loaded_any:
    test_pred = np.mean(preds, axis=0)
else:
    print(
        "WARNING: No usable .pt models loaded; using eeg_id/patient/global fallback derived from train.csv."
    )
    train = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

    eps = 1e-12
    K = len(CLASSES)

    votes = train[CLASSES].to_numpy(dtype=np.float64)
    votes = np.clip(votes, 0.0, None)
    row_sum = votes.sum(axis=1)
    row_sum_safe = np.maximum(row_sum, eps)

    row_p = votes / row_sum_safe[:, None]
    w = row_sum_safe.astype(np.float64)  # weight by number of annotators

    tmp = pd.DataFrame(row_p, columns=CLASSES)
    tmp["eeg_id"] = train["eeg_id"].values
    tmp["patient_id"] = train["patient_id"].values
    tmp["w"] = w

    def weighted_group_mean(df, key, class_cols, w_col="w"):
        df2 = df[[key, w_col] + class_cols].copy()
        wv = df2[w_col].to_numpy(dtype=np.float64)
        for c in class_cols:
            df2[c] = df2[c].to_numpy(dtype=np.float64) * wv
        num = df2.groupby(key, sort=False)[class_cols].sum()
        den = df2.groupby(key, sort=False)[w_col].sum().to_numpy(dtype=np.float64)
        out = num.to_numpy(dtype=np.float64) / den[:, None]
        return pd.DataFrame(out, index=num.index, columns=class_cols)

    eeg_target = tmp.groupby("eeg_id", sort=False)[CLASSES].mean()  # evaluation-like
    eeg_to_patient = train.groupby("eeg_id", sort=False)["patient_id"].first()

    eeg_p_mean = weighted_group_mean(tmp, "eeg_id", CLASSES, "w")
    patient_p_mean = weighted_group_mean(tmp, "patient_id", CLASSES, "w")
    global_p_mean = (row_p * w[:, None]).sum(axis=0) / w.sum()
    global_p_mean = np.clip(global_p_mean, eps, None)
    global_p_mean = global_p_mean / global_p_mean.sum()

    def kl_divergence(p_true, p_pred):
        p_true = np.clip(p_true, eps, 1.0)
        p_pred = np.clip(p_pred, eps, 1.0)
        p_true = p_true / p_true.sum(axis=1, keepdims=True)
        p_pred = p_pred / p_pred.sum(axis=1, keepdims=True)
        return float(
            np.mean(np.sum(p_true * (np.log(p_true) - np.log(p_pred)), axis=1))
        )

    def dirichlet_smooth(p, alpha=0.02):
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, eps, None)
        p = p / p.sum()
        p = p + alpha
        p = p / p.sum()
        return p

    def get_component_p(eid, pid, eeg_p_mean_local, patient_p_mean_local, global_p):
        if eid in eeg_p_mean_local.index:
            return eeg_p_mean_local.loc[eid].to_numpy(dtype=np.float64)
        if pid in patient_p_mean_local.index:
            return patient_p_mean_local.loc[pid].to_numpy(dtype=np.float64)
        return global_p

    rng = np.random.RandomState(42)
    all_eids = eeg_target.index.to_numpy()
    rng.shuffle(all_eids)
    n_folds = 5
    folds = np.array_split(all_eids, n_folds)

    cand_w_eeg = [0.60, 0.70, 0.80, 0.85]
    cand_w_pat = [0.10, 0.15, 0.20, 0.25, 0.30]
    alpha_smooth_grid = [0.01, 0.02, 0.04]

    best = None
    for alpha_smooth in alpha_smooth_grid:
        for w_eeg_mix in cand_w_eeg:
            for w_pat_mix in cand_w_pat:
                w_glb_mix = 1.0 - w_eeg_mix - w_pat_mix
                if w_glb_mix < 0.02 or w_glb_mix > 0.30:
                    continue

                fold_scores = []
                for f in range(n_folds):
                    val_eids = folds[f]
                    tr_eids = np.concatenate(
                        [folds[j] for j in range(n_folds) if j != f]
                    )

                    tmp_tr = tmp[tmp["eeg_id"].isin(tr_eids)]

                    eeg_p_mean_tr = weighted_group_mean(tmp_tr, "eeg_id", CLASSES, "w")
                    patient_p_mean_tr = weighted_group_mean(
                        tmp_tr, "patient_id", CLASSES, "w"
                    )
                    votes_tr = train.loc[
                        train["eeg_id"].isin(tr_eids), CLASSES
                    ].to_numpy(dtype=np.float64)
                    votes_tr = np.clip(votes_tr, 0.0, None)
                    rs = np.maximum(votes_tr.sum(axis=1), eps)
                    rp = votes_tr / rs[:, None]
                    ww = rs
                    global_p_tr = (rp * ww[:, None]).sum(axis=0) / ww.sum()
                    global_p_tr = np.clip(global_p_tr, eps, None)
                    global_p_tr = global_p_tr / global_p_tr.sum()

                    y_true = eeg_target.loc[val_eids].to_numpy(dtype=np.float64)
                    preds_oof = np.zeros((len(val_eids), K), dtype=np.float64)
                    pids = eeg_to_patient.loc[val_eids].to_numpy()

                    for i, (eid, pid) in enumerate(zip(val_eids, pids)):
                        p_e = get_component_p(
                            eid, pid, eeg_p_mean_tr, patient_p_mean_tr, global_p_tr
                        )
                        p_p = (
                            patient_p_mean_tr.loc[pid].to_numpy(dtype=np.float64)
                            if pid in patient_p_mean_tr.index
                            else global_p_tr
                        )
                        p_g = global_p_tr
                        p_mix = w_eeg_mix * p_e + w_pat_mix * p_p + w_glb_mix * p_g
                        preds_oof[i] = dirichlet_smooth(p_mix, alpha=alpha_smooth)

                    fold_scores.append(kl_divergence(y_true, preds_oof))

                score = float(np.mean(fold_scores))
                if (best is None) or (score < best[0]):
                    best = (score, alpha_smooth, w_eeg_mix, w_pat_mix, w_glb_mix)

    best_score, best_alpha_smooth, best_w_eeg, best_w_pat, best_w_glb = best
    print(
        "Chosen fallback params via eeg_id-group CV (lower KL is better): "
        f"KL={best_score:.6f}, alpha_smooth={best_alpha_smooth}, "
        f"w_eeg={best_w_eeg}, w_pat={best_w_pat}, w_glb={best_w_glb}"
    )

    test_pred = np.zeros((len(test), K), dtype=np.float64)
    for i, (eid, pid) in enumerate(
        zip(test["eeg_id"].values, test["patient_id"].values)
    ):
        p_e = get_component_p(eid, pid, eeg_p_mean, patient_p_mean, global_p_mean)
        p_p = (
            patient_p_mean.loc[pid].to_numpy(dtype=np.float64)
            if pid in patient_p_mean.index
            else global_p_mean
        )
        p_g = global_p_mean
        p_mix = best_w_eeg * p_e + best_w_pat * p_p + best_w_glb * p_g
        test_pred[i] = dirichlet_smooth(p_mix, alpha=best_alpha_smooth)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)

probs = test_pred_df[CLASSES].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_pred_df[CLASSES] = probs

test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df.reset_index(drop=True)],
    axis=1,
)

test_pred_df = test_pred_df[["eeg_id"] + CLASSES]
assert test_pred_df.shape[0] == test.shape[0], "Submission row count mismatch."
assert np.allclose(
    test_pred_df[CLASSES].sum(axis=1).values, 1.0, atol=1e-6
), "Rows do not sum to 1."

test_pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_pred_df.shape)
print(test_pred_df.head())
