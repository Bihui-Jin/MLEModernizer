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

0.4320239655414503

# 6. Current score

1.40427

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the inference-time shape/iteration bugs that keep `test_pred` from becoming a valid 2D numpy array, which is why `pd.DataFrame(test_pred, ...)` fails. Specifically: (1) make `DataGenerator.__call__` yield only `X` in test mode and make the inference loop robust to either `(X,y)` or `X`, (2) ensure the test metadata has the expected `spec_id` and a dummy `offset` column used by the generator, and (3) handle missing/empty model glob results by falling back to a uniform-probability submission so a valid `.csv` is always written. These changes are score-neutral aside from actually producing valid probabilities that sum to 1 and a correctly shaped submission.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 score is far from the 0.4320 target (lower is better), so we need a real (but still minimal) improvement rather than just execution fixes. The biggest score issue here is almost certainly a train/test feature mismatch: at inference you currently use the *EEG-derived mel spectrogram* as if it were already in the same scale as the Kaggle spectrogram parquet (you skip the `log/clip` transform for the EEG spectrogram branch), which breaks the distribution your model was trained on and inflates KL. I apply the exact same `clip -> log -> nan_to_num` preprocessing to the EEG spectrogram branch before normalization/tiling, keeping the same architecture and inference loop. I also add a tiny epsilon smoothing to probabilities (legit for KL stability) before renormalizing; this typically reduces catastrophic KL on overconfident wrong predictions without changing core semantics.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4320), so we need a real inference-quality improvement while keeping your model and feature construction intact. The biggest low-risk gain is to fix a distribution mismatch: the Kaggle spectrogram branch is `clip->log`, but the EEG-derived spectrogram should also be brought to a comparable log-power space; we do this by converting the mel dB output back to power and then applying the same `clip->log->nan_to_num` pipeline used elsewhere. Next, we avoid per-sample min/max scaling (which can distort amplitudes at test time) by switching to a fixed scaling consistent with the clip/log bounds, but only in test mode to preserve training semantics. Finally, we keep your small epsilon probability smoothing/renormalization for KL stability.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is much worse than the target (0.4320), so we should make a small but meaningful inference-quality fix rather than tweak ensembling or architecture. The most likely culprit in this code is that you’re applying the Kaggle spectrogram `clip->log` transform, but your cached `spectrograms2` arrays are loaded raw (no preprocessing) and then min/max scaled per-sample, which makes test-time distribution differ from what the model expects and can badly inflate KL. I apply the exact same `clip->log->nan_to_num` preprocessing once at load time for all Kaggle spectrogram parquets (so generator sees consistent log-space inputs), while keeping the generator/model/inference loop identical. I also ensure the final probabilities are smoothed and renormalized (you already do this) and keep everything else unchanged to avoid unnecessary score drift.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4320), so we need a small but meaningful inference-quality fix while keeping your model and generator logic intact. The biggest likely issue is a double `log()` on the Kaggle spectrogram branch: you already `clip->log` when loading parquets into `spectrograms2`, but you apply `clip->log` again inside the generator, creating a severe train/test distribution mismatch and inflating KL. I keep the EEG branch as-is, but remove the redundant `clip->log` for the Kaggle spec inside `generate_all_specs` and `generate_specs`, only keeping `nan_to_num` and scaling. This preserves architecture/training semantics and should improve score by restoring the expected input distribution at inference.'
- What this solution (achieved 1.40899) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest meaningful gain is to reduce systematic inference mismatch rather than change the model. The biggest low-risk issue here is that you run each checkpoint in plain `eval()` mode without enforcing deterministic behavior; enabling inference-time autocast/TF32 and consistent dtype conversion can change outputs slightly but tends to improve stability and often reduces pathological KL spikes. I also add an optional (very small) “prior blending” toward the empirical class distribution from `train.csv` to reduce overconfident wrong predictions, which is legitimate for KL and keeps the core model/feature pipeline unchanged. Finally, I keep your probability epsilon + renormalization, but make it numerically safer by blending before clipping and ensuring exact row-sum to 1.'
- What this solution (achieved 1.40899) has done: 'Your current KL (1.40899, lower-is-better) is far from the 0.4320 target, so we need a small but meaningful inference-quality improvement without changing your model or feature construction. The biggest likely remaining issue is that test-time probabilities are still too peaky/overconfident for KL, even with your current prior blend; a standard, minimal, metric-aligned fix is temperature scaling on logits before softmax (keeps architecture and loop identical, only calibrates probabilities). I add a conservative temperature `T>1` and keep your existing prior blending + epsilon renormalization, which together typically reduce KL spikes without changing the core pipeline. I also make the checkpoint loading more robust (state_dict vs full model object) while preserving the same inference semantics, so you don’t silently fall back to uniform predictions.'
- What this solution (achieved 1.4076) has done: 'Your current KL (1.40899, lower-is-better) is far from the 0.4320 target, so we need a small but meaningful inference-quality improvement without changing your model or feature construction. The most likely remaining issue is test-time distribution shift caused by per-sample min/max scaling in `_scale_to_255` for non-test modes only; your generator is in `mode="test"` already, but it still may be mis-scaled if any path accidentally falls back to the train-style scaling or if NaNs/Infs expand the dynamic range. I make scaling strictly consistent by always applying fixed log-range scaling ([-4,8] -> [0,255]) for both test and non-test to match the earlier clip/log assumptions and reduce unpredictable amplitude distortions (core feature logic unchanged). I also make the prior blending a bit stronger (still conservative) and slightly increase temperature to reduce overconfident spikes in KL, while keeping the exact same inference loop and softmax-based probabilities.'
- What this solution (achieved 1.40666) has done: 'Your current KL (1.4076, lower-is-better) is far from the 0.4320 target, so the smallest meaningful move is to reduce systematic inference mismatch rather than changing the model. The biggest likely issue remaining is that your test generator hardcodes `offset=0` in `mode="test"`, while the model was trained on centrally-labeled windows (non-zero offsets), so test-time crops are misaligned; I change test offset to a centered crop derived from each spectrogram length. I also reset the generator before each checkpoint so every model sees the same full test sequence (avoids subtle iterator state issues affecting ensemble). Finally, I slightly strengthen the existing calibration (temperature + prior blend) in a conservative way to reduce overconfident KL spikes, while keeping your architecture and feature construction unchanged.'
- What this solution (achieved 1.40666) has done: 'Your current KL (1.40666, lower-is-better) is far from the 0.4320 target, so we need a small but meaningful inference-quality fix without touching your model, training loop, or feature design. The most likely remaining systematic issue is that your **EEG-derived spectrograms are never brought into the same log-power space** as the Kaggle spectrograms: you currently `clip->log` assuming raw power, but the EEG images are already “power-like” and can contain zeros, making `log` produce `-inf` and distorted scaling. I add a tiny floor (`+1e-6`) before the EEG `log()` in both generator paths to avoid `-inf`/range blow-ups (keeps the same preprocessing intent), and I also apply the same centered-offset logic for EEG-only test mode (currently hardcoded to 0) to keep temporal alignment consistent. These are minimal, metric-aligned stability fixes that should reduce KL spikes while preserving your core pipeline.'
- What this solution (achieved 1.40666) has done: 'Your current KL (1.40666, lower-is-better) is still far from the 0.432 target, so the most likely remaining “minimal but real” win is fixing a silent input scaling mismatch. Right now the Kaggle spectrograms are pre-log-transformed at load time, but your `_scale_to_255` assumes inputs are already in log-space while your EEG branch still goes through `clip->log` (with a tiny +1e-6) and then the same scaling—these two pipelines are not perfectly aligned numerically because of the extra `+1e-6` inside the log and because Kaggle specs were logged without that term. I make `_safe_log_power` match the Kaggle loading exactly (clip then pure log then nan_to_num), so both branches land in the same log-power space before fixed scaling. I also make the Kaggle parquet loading use the same helper for consistency, and keep your existing temperature scaling + prior blend + epsilon renorm unchanged to avoid destabilizing behavior.'
- What this solution (achieved 1.40666) has done: 'Your current KL (1.40666, lower-is-better) is still far from the 0.4320 target, so we need a small but meaningful inference-quality improvement without changing your model or overall pipeline. The lowest-risk fix here is to correct the **offset semantics mismatch**: in train you use `offset/2` to index spectrogram time bins, but in test you currently derive a centered offset directly in bins (not “seconds/2”), which can shift the crop and hurt predictions. I make the test-time center offset computed in the **same units as training** (i.e., “seconds” then `/2`) and use it consistently for both Kaggle spectrograms and EEG-derived specs. I also fix the EEG offset calculation (it was using `shape[0]` which is the mel-frequency axis, not time), which was silently wrong and can destabilize the EEG half of the “both” input.'
- What this solution (achieved 1.40524) has done: 'Your current KL (1.40666, lower-is-better) is far from the 0.4320 target, so we should make a minimal, metric-aligned inference calibration change rather than touching your model or feature pipeline. The biggest low-risk lever for KL on probabilistic labels is to reduce overconfident predictions by (1) slightly increasing the softmax temperature and (2) slightly strengthening the blend toward the empirical class prior; both keep the same model outputs but make probabilities less peaky. I also apply the epsilon smoothing *after* prior blending (still renormalized) to avoid any accidental “re-peaking” and keep every row summing to 1. All data loading, feature construction, generator logic, and checkpoint ensembling remain unchanged.'
- What this solution (achieved 1.40427) has done: 'Your current KL (1.405, lower-is-better) is far above the 0.432 target, so we need a small, metric-aligned improvement without changing the model or feature design. The most likely low-risk gain left is calibration: for KL, reducing overconfident probabilities typically helps a lot, so I slightly increase the softmax temperature and strengthen the prior blend a bit to pull predictions toward realistic class frequencies. I also apply the blending in logit-space (via a principled convex mix in probability space, then renormalize) while keeping your architecture/inference loop intact, and keep the same CSV schema/paths. No training, no architecture changes, and the submission still be valid (rows sum to 1).'

# 9. Code solution

## === cell 0
import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
import tqdm

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
CLASSES = TARGETS

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
if "offset" not in test.columns:
    test["offset"] = 0

train = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv", usecols=TARGETS
)
train_votes = train[TARGETS].sum(axis=0).values.astype(np.float64)
prior = train_votes / np.clip(train_votes.sum(), 1.0, None)
prior = prior.astype(np.float32)
print("Empirical prior:", dict(zip(TARGETS, prior.round(4))))


def clip_log_nan(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, np.exp(-4), np.exp(8))
    arr = np.log(arr)  # NOTE: no +eps here, to match how Kaggle spectrograms are logged
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


PATH_SPEC = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH_SPEC)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 200 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
    name = int(f.split(".")[0])
    arr = tmp.iloc[:, 1:].values.astype(np.float32)
    arr = clip_log_nan(arr)
    spectrograms2[name] = arr
print()

PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()



## === cell 1
import librosa
import pywt


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
                power=2.0,
            )

            width = (mel_spec.shape[1] // 30) * 30
            mel_spec = mel_spec[:, :width].astype(np.float32)

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
            mel_spec_power_like = librosa.db_to_power(mel_spec_db).astype(np.float32)

            img[:, :, k] += mel_spec_power_like

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
    img = spectrogram_from_eeg(f"{PATH_EEG}{eeg_id}.parquet", i < DISPLAY)
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
            X, y = self.__getitem__(i)
            if self.mode == "test":
                yield X
            else:
                yield X, y
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

    def _scale_to_255(self, img):
        lo, hi = -4.0, 8.0  # clip to exp(-4)..exp(8) then log -> [-4,8]
        img = 255.0 * (img - lo) / (hi - lo)
        return np.clip(img, 0.0, 255.0)

    def _center_offset_seconds_from_specbins(self, t_bins: int) -> int:
        if t_bins <= 300:
            return 0
        center_bins = max(0, (t_bins - 300) // 2)
        return int(center_bins * 2)  # bins -> "seconds" so offset/2 recovers bins

    def _safe_log_power(self, img):
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)  # no +1e-6 to match parquet preprocessing
        return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]

        spec = self.specs[row.spec_id]
        if self.mode == "test":
            offset_seconds = self._center_offset_seconds_from_specbins(
                int(spec.shape[0])
            )
            offset = int(offset_seconds / 2)
        else:
            offset = int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]
        img = np.stack(imgs, axis=-1).astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        img = self._scale_to_255(img)

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

        img = eeg.astype(np.float32)
        img = self._safe_log_power(img)
        img = self._scale_to_255(img)

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

        if self.data_type == "eeg":
            if self.mode == "test":
                img0 = self.eeg_specs[row.eeg_id]
                t_bins = int(img0.shape[1])
                offset_seconds = self._center_offset_seconds_from_specbins(t_bins)
                offset = int(offset_seconds / 2)
            else:
                offset = int(row.offset / 2)

            img = self.eeg_specs[row.eeg_id].astype(np.float32)
            img = self._safe_log_power(img)

            if offset != 0 and img.shape[1] >= offset + 300:
                img = img[:, offset : offset + 300, :]

        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            if self.mode == "test":
                offset_seconds = self._center_offset_seconds_from_specbins(
                    int(spec.shape[0])
                )
                offset = int(offset_seconds / 2)
            else:
                offset = int(row.offset / 2)

            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]
            img = np.stack(imgs, axis=-1).astype(np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        img = self._scale_to_255(img)

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




## === cell 3
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def run_inference_loop(model, test_gen_both, device, temperature=2.60):
    model.to(device)
    model.eval()
    pred_list = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16

    with torch.no_grad():
        for batch in tqdm.tqdm(test_gen_both):
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_data = batch[0]
            else:
                batch_data = batch

            batch_data = np.asarray(batch_data, dtype=np.float32)
            x = torch.from_numpy(batch_data).permute(2, 0, 1).unsqueeze(0).to(device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
            else:
                logits = model(x)

            logits = logits / float(temperature)
            pred_list.append(
                logits.softmax(dim=1).detach().cpu().numpy().astype(np.float32)
            )

    pred_arr = (
        np.concatenate(pred_list, axis=0)
        if len(pred_list)
        else np.zeros((0, 6), dtype=np.float32)
    )
    return pred_arr


preds = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = sorted(
    glob.glob("/kaggle/input/model101-del-horizontalflip/model_101" + "/*/*.pt")
)
print("Found model checkpoints:", len(model_paths))

if len(model_paths) == 0:
    test_pred = np.full((len(test), 6), 1 / 6, dtype=np.float32)
else:
    for model_path in model_paths:
        print(model_path)

        test_gen_both = DataGenerator(
            test,
            mode="test",
            data_type="both",
            specs=spectrograms2,
            eeg_specs=all_eegs2,
        )

        loaded = torch.load(model_path, map_location=device)

        if isinstance(loaded, torch.nn.Module):
            model = loaded
        elif isinstance(loaded, dict) and "state_dict" in loaded:
            raise RuntimeError(
                "Checkpoint contains a state_dict but no model definition is available in this script. "
                "Please provide the model class code or use checkpoints saved with torch.save(model)."
            )
        elif isinstance(loaded, dict) and all(
            isinstance(k, str) for k in loaded.keys()
        ):
            raise RuntimeError(
                "Checkpoint appears to be a raw state_dict without the model definition in this script. "
                "Please provide the model class code or use checkpoints saved with torch.save(model)."
            )
        else:
            raise RuntimeError(f"Unrecognized checkpoint format: {type(loaded)}")

        pred = run_inference_loop(model, test_gen_both, device, temperature=2.60)
        preds.append(pred)

    test_pred = np.mean(preds, axis=0)

test_pred = np.asarray(test_pred, dtype=np.float32)
if test_pred.ndim == 1:
    test_pred = test_pred.reshape(1, -1)
if test_pred.shape[0] != len(test):
    raise RuntimeError(
        f"Prediction rows ({test_pred.shape[0]}) do not match test rows ({len(test)})."
    )

alpha = 0.30
test_pred = (1.0 - alpha) * test_pred + alpha * prior.reshape(1, -1)

eps = 1e-4
test_pred = np.clip(test_pred, eps, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

test_pred_df.to_csv("submission.csv", index=False)
print(test_pred_df.head())
print("Wrote submission.csv with shape:", test_pred_df.shape)
print(
    "Row sums min/max:",
    test_pred_df[CLASSES].sum(axis=1).min(),
    test_pred_df[CLASSES].sum(axis=1).max(),
)
