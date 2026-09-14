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

0.4750281499538578

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime failure by making the model loading robust to the model directory not existing in your environment, while preserving the rest of the inference pipeline unchanged. If no `.pt` files are found, the script now fall back to a safe uniform-probability submission that matches the required columns and row-wise probability sum constraint, ensuring you always get a valid `submission.csv`. I also add a small safety fix in `denoise()` to return the reconstructed signal (it currently returns `None`, which could break if wavelets are enabled). All other logic (data reading, spectrogram generation, DataGenerator, and ensemble averaging) remains the same.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4750), and the biggest issue is that the test-set “EEG spectrograms” you generate are on a very different log-dB scale than the Kaggle-provided spectrograms, but in `generate_all_specs` you treat both the same (you `np.clip(..., exp(-4), exp(8)); np.log(...)` even for the EEG-derived images). I make a minimal, metric-relevant fix: keep the model and inference loop unchanged, but normalize the EEG-derived spectrogram to the same “log power” domain that the network expects by converting the mel dB back to power (undo dB) before the shared log/clipping pipeline. I also add a tiny safety fallback so missing keys in `spectrograms2`/`all_eegs2` don’t crash and instead yield a neutral image, which stabilizes inference without changing core logic. These changes should materially improve KL because they restore the expected input distribution, moving the score toward your target.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so we need a small but high-impact correctness fix rather than tuning. The biggest issue is that your Kaggle spectrogram half applies the expected `clip(exp(-4), exp(8)) -> log()` preprocessing, but the EEG-derived half does **not**, so the model sees two very different input distributions in the same 512×512 image. I apply the *same* clip+log+minmax-to-255 pipeline to the EEG-derived spectrogram (after a small positive shift to avoid `log(<=0)`), while preserving your model, inference loop, and image layout. I also add a tiny safety normalization at the very end (already present) and keep output schema unchanged so the submission remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is much worse than the target (0.4750), so we need a small but high-impact correctness fix rather than tuning. The biggest issue is that `test.csv` is renamed to `spec_id`, but the model’s expected column is `spectrogram_id` (and your `generate_all_specs` reads `row.spec_id`), causing the Kaggle spectrogram half to be missing and replaced by zeros for every sample—this severely degrades KL. I minimally fix the column naming so the real spectrograms are used, while keeping the model/inference loop identical. I also add a tiny safety normalization to ensure the final submission probabilities are finite and row-normalized (without changing evaluation semantics).'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the biggest correctness issue is that you rename `spectrogram_id` to `spec_id` but then later rely on `row.spec_id`; this mismatch often results in missing Kaggle spectrograms and zero-filled inputs, badly hurting KL. I make the minimal fix to keep `spectrogram_id` intact and have the generator consistently use it, so the real Kaggle spectrograms are actually used. I also add a tiny, metric-relevant safety step to normalize the raw vote targets into probabilities when/if used (it doesn’t affect test inference but keeps semantics correct), and keep the existing probability normalization for submission validity. Everything else (model loading, architecture, inference loop, EEG spec generation, and image construction) remains unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so the most likely cause is an input-format mismatch that makes the pretrained model effectively see garbage and output near-uniform probabilities. I make two minimal, high-impact correctness fixes while keeping your model and inference loop unchanged: (1) ensure `test` contains only the required `eeg_id/spectrogram_id/patient_id` columns in the expected types (so the generator never silently misses `spectrogram_id`), and (2) add the missing `offset` column with the same semantics your generator expects (so slicing into Kaggle spectrograms is consistent and never errors/degenerates). I also add a tiny safeguard in `generate_all_specs` to always slice within bounds (padding/adjusting offset) to avoid zero/empty spectrogram regions that severely harm KL. These changes should move the score toward the target by restoring the intended input distribution without changing the architecture or training approach.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so we need a small but high-impact correctness fix rather than tuning. The most likely issue is an input distribution mismatch: your Kaggle spectrograms are log-scaled via `clip(exp(-4),exp(8)) -> log()`, but your EEG-derived spectrograms are built from `db_to_power(power_to_db(..., ref=max))`, which produces values on a very different scale before the same log pipeline, making the pretrained model effectively see out-of-distribution images. I minimally fix the EEG spectrogram creation to stay in the same “power-like” domain as the Kaggle spectrograms by removing the `power_to_db/db_to_power` round-trip and instead using raw mel power with a tiny epsilon for stability, while keeping the model, inference loop, and image assembly unchanged. I also add a small numerical safety in the per-image min-max scaling to avoid occasional constant-image divisions producing NaNs, which can degrade KL.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4750), so we need a small correctness fix rather than any tuning. The biggest issue is that the Kaggle spectrograms are preprocessed with a global log+clip and then scaled, but your EEG-derived spectrograms are being independently min-max scaled per-sample, which destroys the relative intensity structure the pretrained model expects and can push predictions toward near-uniform. I make a minimal change: apply the same `clip(exp(-4), exp(8)) -> log()` pipeline to the EEG-derived spectrogram too, and remove its per-sample min-max scaling so both halves share consistent intensity scaling semantics. I also ensure the EEG-derived block is safely normalized (finite, bounded) before insertion, without changing the model, inference loop, or submission schema.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so we need a small but high-impact *input correctness* fix rather than any tuning. The pretrained model expects the EEG-derived mel spectrograms to be on a similar “log-power” scale as the Kaggle spectrograms, but right now the EEG block is fed through a different scaling pathway, likely making the combined 512×512 image out-of-distribution and pushing predictions toward poor/uninformative probabilities. I minimally change the EEG block preprocessing inside `generate_all_specs()` to use the same `clip(exp(-4), exp(8)) -> log()` then min-max-to-255 scaling as the Kaggle block (mirroring your `generate_specs()` behavior), while keeping the model, inference loop, image layout, and submission schema unchanged. I also add one small safety clamp to ensure the EEG block is strictly positive before `log()` (to avoid NaNs) without changing the core logic.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so we should focus on a small, high-impact input-correctness fix rather than any tuning. The biggest issue is that your Kaggle spectrogram tiles use `offset` slicing (so they represent the “labeled window”), but your EEG-derived spectrogram is always taken from the *middle* of the 50s recording, ignoring `offset`, which creates a severe time misalignment between the left (Kaggle spec) and right (EEG spec) halves of the 512×512 image. I minimally modify `spectrogram_from_eeg()` to accept an `offset_seconds` and crop the 50s EEG around that offset (same semantics as your other code: `offset/2`), then build `all_eegs2` using `test["offset"]` (0 for test) and, crucially, have `generate_all_specs()` build the EEG spectrogram on-the-fly for train/val rows if it isn’t precomputed (keeps logic identical, but makes alignment correct). This should move KL closer to target by restoring the intended temporal correspondence without changing the model, architecture, or inference loop.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so we should apply a small but high-impact *input correctness* fix rather than any tuning. The biggest issue is that the Kaggle spectrogram crop uses `offset` semantics (10-minute window indexed by `offset/2`), but your EEG-derived spectrogram is generated from the wrong time reference: it uses `offset_seconds` directly in seconds from start, not the intended “centered at labeled time” convention, and for train-style offsets it should be centered at `offset_seconds + 25s` within the 50s clip. I minimally adjust `spectrogram_from_eeg()` to interpret offsets consistently by centering the 50s crop at `(offset_seconds + 25)` seconds (and for test offset=0 this keeps the center at 25s, same as before), which better aligns EEG and Kaggle spectrogram halves and should materially reduce KL. I also ensure `generate_all_specs()` passes the correct `offset_seconds` for non-test rows without changing the model, inference loop, or submission formatting.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), so we should apply a small, high-impact *input correctness* fix rather than any tuning. Right now the EEG-derived spectrogram is independently min-max scaled to 0–255 per sample, but the Kaggle spectrogram half is scaled via a log/clip then min-max; this makes the two halves statistically inconsistent and can push the pretrained model toward uninformative probabilities (high KL). I make the EEG block follow the exact same preprocessing pipeline as the Kaggle block by removing the “shift to positive then clip(exp)->log” step and instead using a safe, global `log(clip(x))` directly on mel-power (with epsilon) so its distribution matches what the model expects. I also fix a bug in `denoise()` where the thresholding generator is never materialized into a list (can silently break if wavelets are enabled), without changing defaults (still off). The model, inference loop, and submission formatting stay unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so we need a minimal but high-impact correctness fix rather than tuning. The biggest issue is that your EEG-derived spectrogram block is being clipped to `exp(-4)..exp(8)` *before* log, but mel-power values are typically far below `exp(-4)`, so most pixels get floored to the same constant and the EEG half becomes nearly uninformative. I minimally rescale the mel-power output from `spectrogram_from_eeg()` into the same approximate numeric domain as the Kaggle spectrograms **before** the shared `clip->log->minmax` pipeline (by applying a constant gain), preserving the same architecture and inference loop. I also make the generator use the correct EEG path depending on `mode` (test vs train) to avoid silent zero-fills that hurt predictions.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4750), so we need a small but high-impact *input correctness* fix rather than any tuning. The largest remaining issue is that the Kaggle spectrogram block is log/clip scaled, but the EEG-derived block is then independently min-max scaled per-sample, which can destroy intensity structure and make the pretrained model’s combined 512×512 input out-of-distribution. I make the EEG block follow the exact same preprocessing semantics as the Kaggle block by using the same global `clip(exp(-4), exp(8)) -> log()` transform **without** introducing an additional constant-gain hack, and I remove the extra gain so we aren’t forcing values into saturation. Finally, I keep the rest of your pipeline identical and add one tiny numerical safeguard so constant images don’t produce NaNs.'

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
test = test[["eeg_id", "spectrogram_id", "patient_id"]].copy()
test["eeg_id"] = test["eeg_id"].astype(np.int64)
test["spectrogram_id"] = test["spectrogram_id"].astype(np.int64)
test["patient_id"] = test["patient_id"].astype(np.int64)

test["offset"] = 0

print("Test shape", test.shape)
print(test.head())

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




## === cell 1
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
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]]

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


def spectrogram_from_eeg(parquet_path, offset_seconds=0.0, display=False):
    """
    Minimal score-relevant fix:
    - Remove the extra constant gain (mel_spec *= 1e6) that can push values into the
      clip(exp(8)) saturation region, flattening contrast and harming KL.
    - Keep the rest (mel-power computation) identical; downstream uses the same
      clip->log->minmax pipeline as the Kaggle spectrograms.
    """
    eeg = pd.read_parquet(parquet_path)

    rows = len(eeg)
    win = 10_000  # 50 seconds at 200 Hz

    center_seconds = float(offset_seconds) + 25.0
    center = int(round(center_seconds * 200.0))

    start = center - win // 2
    if start < 0:
        start = 0
    if start > max(0, rows - win):
        start = max(0, rows - win)
    eeg = eeg.iloc[start : start + win]

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
                y=x.astype(np.float32),
                sr=200,
                hop_length=len(x) // 300,
                n_fft=1024,
                n_mels=100,
                fmin=0,
                fmax=20,
                win_length=128,
                power=2.0,
            ).astype(np.float32)

            width = (mel_spec.shape[1] // 30) * 30
            mel_spec = mel_spec[:, :width]

            mel_spec = np.maximum(mel_spec, 1e-10)

            img[:, :, k] += mel_spec

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
    off_sec = (
        float(test.loc[test.eeg_id == eeg_id, "offset"].iloc[0])
        if "offset" in test.columns
        else 0.0
    )
    img = spectrogram_from_eeg(
        f"{PATH2}{eeg_id}.parquet", offset_seconds=off_sec, display=(i < DISPLAY)
    )
    all_eegs2[eeg_id] = img




## === cell 2
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
        trans=None,
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.on_epoch_end()
        self.trans = trans

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
        if self.mode == "test":
            offset = 0
            offset_seconds = 0.0
            eeg_base = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
            )
        else:
            offset = int(row.offset / 2)
            offset_seconds = float(row.offset) if "offset" in row.index else 0.0
            eeg_base = (
                "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
            )

        eeg = (
            self.eeg_specs.get(row.eeg_id, None) if self.eeg_specs is not None else None
        )
        if eeg is None:
            eeg_path = f"{eeg_base}/{int(row.eeg_id)}.parquet"
            if os.path.exists(eeg_path):
                eeg = spectrogram_from_eeg(
                    eeg_path, offset_seconds=offset_seconds, display=False
                )
            else:
                eeg = np.zeros((100, 300, 4), dtype="float32")

        spec_id = int(row.spectrogram_id) if "spectrogram_id" in row.index else None
        spec = (
            self.specs.get(spec_id, np.zeros((300, 400), dtype="float32"))
            if (spec_id is not None and self.specs is not None)
            else np.zeros((300, 400), dtype="float32")
        )

        max_off = max(0, spec.shape[0] - 300)
        if offset < 0:
            offset = 0
        if offset > max_off:
            offset = max_off

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        img = np.nan_to_num(img, nan=0.0)

        mn = float(img.min())
        mx = float(img.max())
        ep = 1e-5
        if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-12:
            img = np.zeros_like(img, dtype=np.float32)
        else:
            img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_k
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_K

        img = eeg.astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.maximum(img, 1e-10)  # keep strictly positive for log/clip stability

        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        mn = float(img.min())
        mx = float(img.max())
        ep = 1e-5
        if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-12:
            img = np.zeros_like(img, dtype=np.float32)
        else:
            img = 255 * (img - mn) / (mx - mn + ep)

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_e
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_e
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_e

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_e

        X = self.trans(image=X)["image"]

        if self.mode != "test":
            y[:] = row[TARGETS].to_numpy(dtype=np.float32)
            s = float(y.sum())
            if s > 0:
                y /= s
            else:
                y[:] = 1.0 / 6.0

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        if self.data_type == "eeg":
            img = self.eeg_specs.get(
                row.eeg_id, np.zeros((100, 300, 4), dtype="float32")
            )
        elif self.data_type == "kaggle":
            spec_id = int(row.spectrogram_id) if "spectrogram_id" in row.index else None
            spec = (
                self.specs.get(spec_id, np.zeros((300, 400), dtype="float32"))
                if spec_id is not None
                else np.zeros((300, 400), dtype="float32")
            )

            max_off = max(0, spec.shape[0] - 300)
            if offset < 0:
                offset = 0
            if offset > max_off:
                offset = max_off

            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]  # to match kaggle with eeg
            img = np.stack(imgs, axis=-1)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            img = np.nan_to_num(img, nan=0.0)

        mn = float(img.min())
        mx = float(img.max())
        ep = 1e-5
        if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-12:
            img = np.zeros_like(img, dtype=np.float32)
        else:
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
            y[:] = row[TARGETS].to_numpy(dtype=np.float32)
            s = float(y.sum())
            if s > 0:
                y /= s
            else:
                y[:] = 1.0 / 6.0

        return X, y




## === cell 3
val_transform = A.Compose([ToTensorV2(p=1.0)])




## === cell 4
def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm.tqdm(test_gen_both(), total=len(test_gen_both)):
            batch_data = batch_data.unsqueeze(0).to(device, non_blocking=True)
            logits = model(batch_data)
            probs = torch.softmax(logits, dim=1)
            pred_list.append(probs.detach().cpu().numpy())

    pred_arr = (
        np.concatenate(pred_list, axis=0)
        if len(pred_list)
        else np.zeros((0, 6), dtype=np.float32)
    )
    return pred_arr


def _load_model_any(model_path, device):
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        for k in ["model", "state_dict", "model_state_dict", "net"]:
            if k in obj and isinstance(obj[k], torch.nn.Module):
                return obj[k]
        if any(isinstance(v, dict) for v in obj.values()):
            raise ValueError(
                f"Loaded checkpoint from {model_path} looks like a state_dict-only checkpoint; "
                f"this script expects a serialized torch.nn.Module."
            )
    raise ValueError(
        f"Unsupported model checkpoint format at {model_path}: {type(obj)}"
    )


preds = []

test_gen_both = DataGenerator(
    test,
    mode="test",
    data_type="both",
    specs=spectrograms2,
    eeg_specs=all_eegs2,
    trans=val_transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_glob = "/kaggle/input/model101-mixnet-xl-crop-flip/model_101/*/*.pt"
model_paths = sorted(glob.glob(model_glob))
print(f"Found {len(model_paths)} model files with glob: {model_glob}")

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

if len(model_paths) == 0:
    test_pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
else:
    for model_path in model_paths:
        print(model_path)
        model = _load_model_any(model_path, device)
        pred = run_inference_loop(model, test_gen_both, device)
        preds.append(pred)

    test_pred = np.mean(np.stack(preds, axis=0), axis=0)

test_pred = np.nan_to_num(test_pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
test_pred = np.clip(test_pred, 1e-12, None)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

test_pred_df = test_pred_df[["eeg_id"] + CLASSES]

out_path = "submission.csv"
test_pred_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", test_pred_df.shape)
print(test_pred_df.head())
