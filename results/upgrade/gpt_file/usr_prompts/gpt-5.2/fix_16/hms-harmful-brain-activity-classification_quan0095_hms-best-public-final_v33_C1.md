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

No external packages required in the script and installed.

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

0.2833028562330134

# 6. Current score

1.40572

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'Your current notebook likely doesn’t yield a valid Kaggle score because it relies on unavailable inputs (the external model weight datasets) and also contains a few execution blockers (notably the `!rm` shell commands and missing probability normalization/row alignment guarantees). I make the smallest set of changes to (1) ensure the script runs under Python 3.12 as a plain script (no `!` magics), (2) guarantee a valid `submission.csv` with correct row order, exact columns, and probabilities that sum to 1, and (3) add a safe fallback that produces a reasonable baseline prediction (train vote priors) if the external model weights aren’t present—so you can finally get a scored submission and move toward the target. These changes preserve your core modeling/inference logic when weights exist; they only activate the fallback when they don’t. I also remove extremely verbose per-sample printing that can cause timeouts without affecting predictions.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.2833), so we should improve performance with minimal risk. The biggest issue is that when weights are present, you are *summing fold probabilities into dictionaries without averaging by the number of times each eeg_id was added*, which can distort calibration and worsen KL; we fix this by tracking per-eeg counts and averaging at the end for each model group. We also ensure `eeg_id` keys are consistent (always `str`) to avoid silent fallback-to-prior for missing keys, and we normalize each group’s per-eeg vector after averaging to reduce numerical drift. These are minimal changes that preserve your model/inference logic and only correct aggregation/calibration, which should move the score down toward the target.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.2833), so we should reduce KL with minimal, low-risk calibration/aggregation fixes rather than changing any model. The biggest accuracy bug is that each model-group’s per-`eeg_id` vectors are averaged but never re-normalized, so when groups are later averaged together they may not sum to 1 and get distorted by the final row-normalization; we normalize *per eeg_id per group* right after fold-averaging. Second, the final fusion currently falls back to the global prior if *any* one of the five result dicts is missing an `eeg_id`, which can happen due to any transient cache/missing-file issue and badly hurt KL; we instead average over the groups that exist for that id (and only use prior if none exist). These changes preserve your core inference logic and only fix aggregation/calibration to move the score down toward the target.'
- What this solution (achieved 1.41937) has done: 'Your current score is much worse than the target (lower-is-better), so we make minimal, low-risk fixes that directly reduce KL without changing the model architectures or inference loops. The biggest silent bug is that `seed_everything` sets `cudnn.deterministic=True` while also setting `cudnn.benchmark=True`, which makes inference nondeterministic and can hurt calibration; we disable benchmark for stable outputs. Next, we ensure all per-model-group probabilities are *always* explicitly normalized right after softmax (per batch) and again after fold aggregation (per eeg_id) to prevent any drift before fusion. Finally, we add a tiny epsilon-floor normalization right before writing the submission to avoid zero probabilities (which can blow up KL if any true class has nonzero vote share).'
- What this solution (achieved 1.41937) has done: 'Your current score is far worse than the target (lower-is-better), so the safest improvements are calibration/aggregation fixes rather than changing any model. I (1) fix an internal inconsistency where the `raw50seeg_from_eeg` function is silently redefined later with `EEG_LENGTH=20`, which mismatches the pretrained weights’ expected preprocessing and can severely hurt KL; we keep it at 50s for the later stages too. I also (2) stop deleting and regenerating intermediate feature folders between model groups (those `shutil.rmtree(...)` calls), because this can cause missing/corrupted `.npy` inputs and trigger per-id fallbacks that spike KL; instead, we keep the already-generated arrays. Finally, I (3) add a tiny prior-mixing (“epsilon smoothing”) in the final fusion to reduce overconfident wrong predictions (a common KL failure mode) while preserving the core inference logic.'
- What this solution (achieved 1.41937) has done: 'Your current score is much worse than the target (lower-is-better), so we should make small calibration/aggregation fixes rather than changing any model architecture or inference loops. The biggest likely KL spike here is inconsistent preprocessing due to `stft_spec_from_eeg` being redefined multiple times with different shapes/params and being used to overwrite cached `.npy` inputs mid-pipeline; we keep the original (128×568) version for the whole run and stop later redefinitions from taking effect. Next, we make fusion slightly safer for KL by always applying a tiny epsilon floor + renormalization to every per-eeg vector (already done in places, but we enforce it at group finalization and right before fusion), and we keep the “average over available groups” behavior (no prior fallback if only one group is missing). These are minimal, semantics-preserving changes intended to reduce overconfident/invalid distributions and preprocessing mismatch, moving the score down toward the target without changing the trained weights or model logic.'
- What this solution (achieved 1.41937) has done: 'Your current score is far worse than the target (lower-is-better), so we should reduce KL with minimal, low-risk calibration fixes rather than changing any model architecture or inference loops. The biggest issue left is that your final fusion simply averages model-group probabilities; KL is very sensitive to overconfident wrong predictions, so adding a tiny temperature-based “softening” to the fused distribution (then renormalizing) typically reduces KL without changing model outputs themselves. Additionally, we should guard against any rare NaN/Inf coming from intermediate dicts by sanitizing vectors during final fusion, and keep a small epsilon floor before writing. These changes preserve your core logic (same models, same preprocessing, same ensembling), and only adjust post-processing to be more KL-robust, moving the score down toward the target.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.419) is far worse than the target (0.283, lower-is-better), so we should make the smallest changes that reliably reduce overconfident/wrong predictions without changing any model architectures or inference loops. The biggest low-risk improvement here is to stop doing per-sample standardization using the **test sample’s own mean/std** (a distribution shift vs how these models are usually trained), and instead use fixed normalization constants computed once from the training feature cache. We compute channel-wise mean/std for the spectrogram and each EEG-derived input over a small capped subset of cached train items (fast, deterministic) and use those constants in all Dataset variants; this keeps feature extraction identical and only changes normalization. Finally, we keep the existing epsilon-floor + temperature + prior-mix safeguards, but ensure they are applied after this more stable normalization.'
- What this solution (achieved 1.41937) has done: 'We keep your model/inference logic unchanged and only apply two minimal, KL-relevant post-processing fixes to move the score down toward the 0.283 target. First, we apply the same global normalization (mean/std) not only to `spec_img` but also to `eeg_img`, `raw_50s_img`, and the three `raw_10s_*` inputs in all Dataset variants; right now most of those inputs are unnormalized, which commonly causes severe distribution shift vs pretrained weights and can spike KL. Second, we make the final fusion slightly more KL-robust by adding a tiny per-class epsilon floor before temperature/prior-mix and renormalizing (preventing near-zeros that are punished heavily by KL). Everything still runs end-to-end and writes a valid `submission.csv` with row-sum-to-1 guaranteed.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.419, lower-is-better) is far above the target (0.283), so we should reduce KL with minimal, low-risk changes that don’t alter your model architectures or inference loops. The biggest likely KL regression is that `GLOBAL_NORM_STATS` is being computed from the *test cache folders* (because it runs after `DEBUG=False` sets paths to test), causing normalization drift and hurting pretrained-weight inference. I compute global normalization stats from a small, deterministic subset of *train* parquet files (spectrogram + eeg) into temporary arrays and use those fixed mean/std values for all datasets, keeping feature extraction logic intact. Additionally, I add a final “safety” normalization + epsilon floor right before writing submission to prevent any near-zero probabilities that can blow up KL.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.419, lower-is-better) is far above the target (0.283), so we make the smallest post-processing changes that typically *reduce* KL without changing any model architectures, weights, feature extraction, or inference loops. The main adjustment is to tune the final fusion calibration slightly more toward “safer” distributions by increasing the tiny prior-mixing and slightly softening the temperature; this reduces overconfident wrong predictions (a common KL failure mode). We also ensure the final probabilities are strictly valid by applying the epsilon floor + renormalization once at the very end (after all mixing/temperature), which can prevent rare near-zero classes that can spike KL. Everything else (datasets, preprocessing, models, fold aggregation, and submission schema) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.41247) has done: 'We make two minimal, KL-relevant fixes that are very likely hurting your score without changing any model architectures or inference loops. First, we fix a clear preprocessing bug in `ImageFolder2` where `eeg_img` is never resized (it’s expanded to `(H,W,1)` but kept at its original shape), which can silently mismatch what the fold-4 models expect and degrade predictions. Second, we apply the same fusion-time calibration (tiny prior-mix + temperature softening) to the fallback path too (and keep strict epsilon + renorm at the very end), so missing-weight runs don’t produce overly sharp/biased probabilities that inflate KL. Everything else (feature extraction, model forward passes, fold aggregation, and submission format) stays unchanged.'
- What this solution (achieved 1.40961) has done: 'Your current KL (1.412, lower-is-better) is far above the target (0.283), so we should make the smallest post-processing change that usually reduces KL without touching any model/feature logic. The biggest low-risk lever is to slightly increase “safety” calibration at the very end: mix a bit more of the global prior and apply a slightly stronger temperature softening, which reduces overconfident wrong predictions (a common KL failure mode). I keep the same ensembling, preprocessing, and weight usage; only the final fusion constants change, and I keep strict normalization/epsilon floors to guarantee valid probability rows. This should move the score downward toward the target while preserving your core solution.'
- What this solution (achieved 1.40698) has done: 'Your current KL (1.4096, lower-is-better) is still far above the target (0.2833), so the safest way to move toward the target without touching model architectures or inference loops is to make the **final probability distribution less overconfident** (KL heavily punishes near-zero on true classes). I keep your entire feature extraction + model ensembling exactly the same, and only adjust the **final fusion calibration** by (1) mixing a bit more of the global prior and (2) applying slightly stronger temperature softening. I also apply the same “safety” (epsilon floor + renorm) consistently right after fusion to avoid accidental tiny probabilities. These are minimal post-processing-only changes that typically reduce KL while preserving your core logic.'
- What this solution (achieved 1.40572) has done: 'Your current KL (1.40698, lower-is-better) is still far above the target (0.2833), so we should make a minimal, low-risk calibration adjustment that reduces overconfidence without touching any model architecture, feature extraction, or inference loops. The safest lever is the final post-processing: slightly increase the prior-mix and temperature softening, and increase the epsilon floor a bit to avoid near-zero probabilities (which KL punishes heavily). I keep everything else identical, including the fold aggregation logic and per-group normalization, and still guarantee each row sums to 1. This change is intentionally small and only affects the final fused probabilities written to `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

try:
    import librosa  # noqa: F401
except Exception:
    librosa = None



## === cell 3
from scipy import signal




## === cell 4
def stft_spec_from_eeg(parquet_path):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg = eeg.iloc[time_start:time_stop]

    list_eeg = []
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg[COLS[kk + 1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            fs = 200
            nperseg = 70
            noverlap = 0
            f, t, spec = signal.spectrogram(
                new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256
            )

            spec = np.abs(spec)
            spec = np.log1p(spec).astype("float32")

            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img




## === cell 5
NAMES = ["LL", "LP", "RP", "RR"]
SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_c = np.concatenate(list_eeg, 1)
    eeg_c /= 104

    time_temp = 0
    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_l = np.concatenate(list_eeg, 1)
    eeg_l /= 104

    time_temp = 0
    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_r = np.concatenate(list_eeg, 1)
    eeg_r /= 104

    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg(parquet_path):
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            (eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]), 1
        )
        list_eeg.append(eeg)

    eeg = np.concatenate(list_eeg, 1)
    eeg /= 104
    return eeg




## === cell 6
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

print(test.shape)

spec_directory_path = "spec_spectrograms/"
Path(spec_directory_path).mkdir(parents=True, exist_ok=True)

eeg_directory_path = "eeg_spectrograms/"
Path(eeg_directory_path).mkdir(parents=True, exist_ok=True)

raw_10s_directory_path = "eeg_10s_raws/"
Path(raw_10s_directory_path).mkdir(parents=True, exist_ok=True)

raw_50s_directory_path = "eeg_50s_raws/"
Path(raw_50s_directory_path).mkdir(parents=True, exist_ok=True)

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]
    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")

    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", img)


_ = Parallel(n_jobs=4)(delayed(save)(row) for _, row in test.iterrows())




## === cell 7
class Config:
    seed = 2024
    num_folds = 5




## === cell 8
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(Config.seed)



## === cell 9
import timm



## === cell 10
import torch.utils.data as data
from torch.utils.data import DataLoader



## === cell 11
from skimage.transform import resize


def _compute_global_norm_stats_from_parquets(
    train_meta_path="/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    train_spec_path="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/",
    train_eeg_path="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/",
    max_items=64,
    seed=2024,
):
    rng = np.random.RandomState(seed)

    meta = pd.read_csv(train_meta_path, usecols=["eeg_id", "spectrogram_id"])
    uniq = meta.drop_duplicates("eeg_id")[["eeg_id", "spectrogram_id"]].reset_index(
        drop=True
    )
    n = min(max_items, len(uniq))
    idx = rng.choice(len(uniq), size=n, replace=False)
    subset = uniq.iloc[idx].reset_index(drop=True)

    spec_means, spec_stds = [], []
    eeg_means, eeg_stds = [], []
    raw50_means, raw50_stds = [], []
    raw10_means, raw10_stds = [], []

    eps = 1e-6
    test_imgsize = (518, 518)

    for _, row in subset.iterrows():
        eeg_id = str(row["eeg_id"])
        spec_id = str(row["spectrogram_id"])
        try:
            spec = pd.read_parquet(os.path.join(train_spec_path, f"{spec_id}.parquet"))
            spec_arr = spec.values[:, 1:].T.astype("float32")  # (400, 300)
            spec_arr = spec_arr[:, 0:300]

            raw10_l, raw10_c, raw10_r = raw10seeg_from_eeg(
                os.path.join(train_eeg_path, f"{eeg_id}.parquet"), eeg_id
            )
            raw50 = raw50seeg_from_eeg(
                os.path.join(train_eeg_path, f"{eeg_id}.parquet")
            )
            eeg = stft_spec_from_eeg(os.path.join(train_eeg_path, f"{eeg_id}.parquet"))
        except Exception:
            continue

        spec_img = resize(spec_arr, test_imgsize).astype("float32")
        eeg_img = resize(eeg, test_imgsize).astype("float32")
        raw50_img = resize(raw50, test_imgsize).astype("float32")
        raw10_l_img = resize(raw10_l, test_imgsize).astype("float32")
        raw10_c_img = resize(raw10_c, test_imgsize).astype("float32")
        raw10_r_img = resize(raw10_r, test_imgsize).astype("float32")

        spec_img = np.expand_dims(spec_img, -1)
        eeg_img = np.expand_dims(eeg_img, -1)
        raw50_img = np.expand_dims(raw50_img, -1)
        raw10_l_img = np.expand_dims(raw10_l_img, -1)
        raw10_c_img = np.expand_dims(raw10_c_img, -1)
        raw10_r_img = np.expand_dims(raw10_r_img, -1)

        spec_p = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_p = np.log(spec_p)
        spec_p = np.nan_to_num(spec_p, nan=0.0)

        raw10_stack = np.stack([raw10_l_img, raw10_c_img, raw10_r_img], axis=0)

        spec_means.append(float(spec_p.mean()))
        spec_stds.append(float(spec_p.std()))
        eeg_means.append(float(eeg_img.mean()))
        eeg_stds.append(float(eeg_img.std()))
        raw50_means.append(float(raw50_img.mean()))
        raw50_stds.append(float(raw50_img.std()))
        raw10_means.append(float(raw10_stack.mean()))
        raw10_stds.append(float(raw10_stack.std()))

    def _ms(means, stds):
        if len(means) == 0:
            return 0.0, 1.0
        return float(np.mean(means)), max(float(np.mean(stds)), eps)

    sm, ss = _ms(spec_means, spec_stds)
    em, es = _ms(eeg_means, eeg_stds)
    r50m, r50s = _ms(raw50_means, raw50_stds)
    r10m, r10s = _ms(raw10_means, raw10_stds)

    return {
        "spec": {"mean": sm, "std": ss},
        "eeg": {"mean": em, "std": es},
        "raw50": {"mean": r50m, "std": r50s},
        "raw10": {"mean": r10m, "std": r10s},
    }


GLOBAL_NORM_STATS = _compute_global_norm_stats_from_parquets(
    max_items=48, seed=Config.seed
)
print("GLOBAL_NORM_STATS:", GLOBAL_NORM_STATS)


class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        df["eeg_id"] = df["eeg_id"]
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)

        spec_image_path = os.path.join(self.spec_data_path, eeg_id + ".npy")
        eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + ".npy")
        raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        raw_10s_l_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy")
        raw_10s_c_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")
        raw_10s_r_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy")

        spec_img = np.load(spec_image_path).astype("float32")
        raw_50s_img = np.load(raw_50s_image_path).astype("float32")
        raw_10s_l_img = np.load(raw_10s_l_image_path).astype("float32")
        raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32")
        raw_10s_r_img = np.load(raw_10s_r_image_path).astype("float32")
        eeg_img = np.load(eeg_image_path).astype("float32")

        eeg_img = resize(eeg_img, self.test_imgsize)
        spec_img = resize(spec_img, self.test_imgsize)
        raw_10s_l_img = resize(raw_10s_l_img, self.test_imgsize)
        raw_10s_c_img = resize(raw_10s_c_img, self.test_imgsize)
        raw_10s_r_img = resize(raw_10s_r_img, self.test_imgsize)
        raw_50s_img = resize(raw_50s_img, self.test_imgsize)

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
        raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

        eps = 1e-6

        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0)

        spec_m = GLOBAL_NORM_STATS["spec"]["mean"]
        spec_s = GLOBAL_NORM_STATS["spec"]["std"]
        spec_img = (spec_img - spec_m) / (spec_s + eps)

        eeg_m = GLOBAL_NORM_STATS["eeg"]["mean"]
        eeg_s = GLOBAL_NORM_STATS["eeg"]["std"]
        eeg_img = (eeg_img - eeg_m) / (eeg_s + eps)

        raw50_m = GLOBAL_NORM_STATS["raw50"]["mean"]
        raw50_s = GLOBAL_NORM_STATS["raw50"]["std"]
        raw_50s_img = (raw_50s_img - raw50_m) / (raw50_s + eps)

        raw10_m = GLOBAL_NORM_STATS["raw10"]["mean"]
        raw10_s = GLOBAL_NORM_STATS["raw10"]["std"]
        raw_10s_l_img = (raw_10s_l_img - raw10_m) / (raw10_s + eps)
        raw_10s_c_img = (raw_10s_c_img - raw10_m) / (raw10_s + eps)
        raw_10s_r_img = (raw_10s_r_img - raw10_m) / (raw10_s + eps)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 12
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_50s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_10s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )

        self.device_id = device_id

        self.spec_model.fc_norm = nn.Identity()
        self.spec_model.head_drop = nn.Identity()
        self.spec_model.head = nn.Identity()

        self.eeg_model.fc_norm = nn.Identity()
        self.eeg_model.head_drop = nn.Identity()
        self.eeg_model.head = nn.Identity()

        self.raw_50s_model.fc_norm = nn.Identity()
        self.raw_50s_model.head_drop = nn.Identity()
        self.raw_50s_model.head = nn.Identity()

        self.raw_10s_model.fc_norm = nn.Identity()
        self.raw_10s_model.head_drop = nn.Identity()
        self.raw_10s_model.head = nn.Identity()

        self.head = nn.Linear(384 * 4, 6)
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(384, 6)
        self.head4 = nn.Linear(384, 6)

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

        spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

        feature = torch.cat(
            (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
        )
        logits = self.head(feature)
        logits_1 = self.head1(spec_feature)
        logits_2 = self.head2(eeg_feature)
        logits_3 = self.head3(raw_50s_feature)
        logits_4 = self.head4(raw_10s_feature)

        return logits, logits_1, logits_2, logits_3, logits_4




## === cell 13
def _weights_exist(paths):
    return all(Path(p).exists() for p in paths)


def _safe_load_state_dict(model, weight_path, device):
    sd = torch.load(weight_path, map_location=device)
    model.load_state_dict(sd)
    return model


def _normalize_probs(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def _finalize_result_dict(result_dict, count_dict, eps=1e-6):
    for k in list(result_dict.keys()):
        c = max(int(count_dict.get(k, 1)), 1)
        v = (result_dict[k] / c).astype(np.float64)
        v = _normalize_probs(v.reshape(1, -1), eps=eps)[0]
        result_dict[k] = v
    return result_dict


def _apply_temperature(p, t=1.12, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, eps, None)
    p = p ** (1.0 / t)
    p = p / p.sum(axis=1, keepdims=True)
    return p




## === cell 14
train_csv_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_csv_path, usecols=CLASSES)
prior = train_df[CLASSES].sum(axis=0).values.astype(np.float64)
prior = prior / prior.sum()
del train_df
gc.collect()



## === cell 15
device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_models = True

model_weights_7 = [
    "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
]
model_weights_8 = [
    "/kaggle/input/hms-stage2-vitlarge/fold_0_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_1_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_2_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_3_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_4_raw_50_10_bestlb_vitlarge.pth",
]
model_weights_5 = [
    "/kaggle/input/hms-bestlb-vitbase/fold_0_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_1_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_2_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_3_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_4_exp_6_bestlb.pth",
]
model_weights_4 = [
    "/kaggle/input/hms-stage2/fold_0_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_exp_5_bestlb.pth",
]
model_weights_3 = [
    "/kaggle/input/hms-bestlb-vitbase/fold_0_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_1_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_2_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_3_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_4_exp_7_bestlb.pth",
]

if not (
    _weights_exist(model_weights_7)
    and _weights_exist(model_weights_8)
    and _weights_exist(model_weights_5)
    and _weights_exist(model_weights_4)
    and _weights_exist(model_weights_3)
):
    use_models = False
    print(
        "WARNING: One or more model weight files are missing; using train-vote prior fallback to produce a valid submission."
    )



## === cell 16
result_7 = {}
result_8 = {}
result_5 = {}
result_4 = {}
result_3 = {}

count_7, count_8, count_5, count_4, count_3 = {}, {}, {}, {}, {}

if use_models:
    vit_models = []
    device0 = device

    for w in model_weights_7:
        model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device0).to(device0)
        _safe_load_state_dict(model, w, device0)
        model.eval()
        vit_models.append(model)

    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
    )

    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in test_loader:
            spec_imgs = spec_imgs.to(device0).float()
            eeg_imgs = eeg_imgs.to(device0).float()
            raw_50s_imgs = raw_50s_imgs.to(device0).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device0).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device0).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device0).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device0)
            for model in vit_models:
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                probs = (
                    logits_l.softmax(dim=1)
                    + logits_c.softmax(dim=1)
                    + logits_r.softmax(dim=1)
                ) / 3.0
                probs = probs / probs.sum(dim=1, keepdim=True)
                ensemble_probs += probs
            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)

            ensemble_probs = ensemble_probs.detach().cpu().numpy()
            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                if eeg_id not in result_7:
                    result_7[eeg_id] = np.zeros(6, dtype=np.float64)
                    count_7[eeg_id] = 0
                result_7[eeg_id] += ensemble_probs[j].astype(np.float64)
                count_7[eeg_id] += 1

    result_7 = _finalize_result_dict(result_7, count_7, eps=1e-6)

    for model in vit_models:
        del model
    torch.cuda.empty_cache()
    gc.collect()



## === cell 17
if use_models:

    class NetLarge(nn.Module):
        def __init__(self, back_bone, device_id):
            super().__init__()
            self.spec_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.eeg_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.raw_50s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )
            self.raw_10s_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )

            self.device_id = device_id

            self.spec_model.fc_norm = nn.Identity()
            self.spec_model.head_drop = nn.Identity()
            self.spec_model.head = nn.Identity()

            self.eeg_model.fc_norm = nn.Identity()
            self.eeg_model.head_drop = nn.Identity()
            self.eeg_model.head = nn.Identity()

            self.raw_50s_model.fc_norm = nn.Identity()
            self.raw_50s_model.head_drop = nn.Identity()
            self.raw_50s_model.head = nn.Identity()

            self.raw_10s_model.fc_norm = nn.Identity()
            self.raw_10s_model.head_drop = nn.Identity()
            self.raw_10s_model.head = nn.Identity()

            self.head = nn.Linear(384 * 3 + 1024, 6)
            self.head1 = nn.Linear(384, 6)
            self.head2 = nn.Linear(384, 6)
            self.head3 = nn.Linear(1024, 6)
            self.head4 = nn.Linear(384, 6)

        def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
            spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

            spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
            eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
            raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
            raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

            feature = torch.cat(
                (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
            )
            logits = self.head(feature)
            logits_1 = self.head1(spec_feature)
            logits_2 = self.head2(eeg_feature)
            logits_3 = self.head3(raw_50s_feature)
            logits_4 = self.head4(raw_10s_feature)
            return logits, logits_1, logits_2, logits_3, logits_4

    vit_models = []
    for w in model_weights_8:
        model = NetLarge("vit_large_patch14_reg4_dinov2.lvd142m", device).to(device)
        _safe_load_state_dict(model, w, device)
        model.eval()
        vit_models.append(model)

    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=16, pin_memory=False, num_workers=4, drop_last=False
    )

    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in test_loader:
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                probs = logits_c.softmax(dim=1)
                probs = probs / probs.sum(dim=1, keepdim=True)
                ensemble_probs += probs
            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)

            ensemble_probs = ensemble_probs.detach().cpu().numpy()
            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                if eeg_id not in result_8:
                    result_8[eeg_id] = np.zeros(6, dtype=np.float64)
                    count_8[eeg_id] = 0
                result_8[eeg_id] += ensemble_probs[j].astype(np.float64)
                count_8[eeg_id] += 1

    result_8 = _finalize_result_dict(result_8, count_8, eps=1e-6)

    for model in vit_models:
        del model
    torch.cuda.empty_cache()
    gc.collect()



## === cell 18
if use_models:

    class NetBase(nn.Module):
        def __init__(self, back_bone, device_id):
            super().__init__()
            self.spec_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.eeg_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.raw_50s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )
            self.raw_10s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )

            self.device_id = device_id

            self.spec_model.fc_norm = nn.Identity()
            self.spec_model.head_drop = nn.Identity()
            self.spec_model.head = nn.Identity()

            self.eeg_model.fc_norm = nn.Identity()
            self.eeg_model.head_drop = nn.Identity()
            self.eeg_model.head = nn.Identity()

            self.raw_50s_model.fc_norm = nn.Identity()
            self.raw_50s_model.head_drop = nn.Identity()
            self.raw_50s_model.head = nn.Identity()

            self.raw_10s_model.fc_norm = nn.Identity()
            self.raw_10s_model.head_drop = nn.Identity()
            self.raw_10s_model.head = nn.Identity()

            self.head = nn.Linear(384 * 2 + 768 * 2, 6)
            self.head1 = nn.Linear(384, 6)
            self.head2 = nn.Linear(384, 6)
            self.head3 = nn.Linear(768, 6)
            self.head4 = nn.Linear(768, 6)

        def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
            spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

            spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
            eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
            raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
            raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

            feature = torch.cat(
                (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
            )
            logits = self.head(feature)
            logits_1 = self.head1(spec_feature)
            logits_2 = self.head2(eeg_feature)
            logits_3 = self.head3(raw_50s_feature)
            logits_4 = self.head4(raw_10s_feature)
            return logits, logits_1, logits_2, logits_3, logits_4

    vit_models = []
    for w in model_weights_5:
        model = NetBase("vit_base_patch14_reg4_dinov2.lvd142m", device).to(device)
        _safe_load_state_dict(model, w, device)
        model.eval()
        vit_models.append(model)

    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
    )

    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in test_loader:
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                probs = (
                    logits_l.softmax(dim=1)
                    + logits_c.softmax(dim=1)
                    + logits_r.softmax(dim=1)
                ) / 3.0
                probs = probs / probs.sum(dim=1, keepdim=True)
                ensemble_probs += probs
            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)

            ensemble_probs = ensemble_probs.detach().cpu().numpy()
            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                if eeg_id not in result_5:
                    result_5[eeg_id] = np.zeros(6, dtype=np.float64)
                    count_5[eeg_id] = 0
                result_5[eeg_id] += ensemble_probs[j].astype(np.float64)
                count_5[eeg_id] += 1

    result_5 = _finalize_result_dict(result_5, count_5, eps=1e-6)

    for model in vit_models:
        del model
    torch.cuda.empty_cache()
    gc.collect()



## === cell 19
pass



## === cell 20
if use_models:
    pass



## === cell 21
if use_models:

    class ImageFolder2(data.Dataset):
        def __init__(self, df, test_imgsize):
            super().__init__()
            df["eeg_id"] = df["eeg_id"]
            self.spec_data_path = spec_directory_path
            self.eeg_data_path = eeg_directory_path
            self.raw_50s_data_path = raw_50s_directory_path
            self.raw_10s_data_path = raw_10s_directory_path
            self.df = df.reset_index(drop=True)
            self.test_imgsize = test_imgsize

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            row = self.df.loc[index]
            eeg_id = str(row.eeg_id)

            spec_image_path = os.path.join(self.spec_data_path, eeg_id + ".npy")
            eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + ".npy")
            raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
            raw_10s_l_image_path = os.path.join(
                self.raw_10s_data_path, eeg_id + "_l.npy"
            )
            raw_10s_c_image_path = os.path.join(
                self.raw_10s_data_path, eeg_id + "_c.npy"
            )
            raw_10s_r_image_path = os.path.join(
                self.raw_10s_data_path, eeg_id + "_r.npy"
            )

            spec_img = np.load(spec_image_path).astype("float32")
            raw_50s_img = np.load(raw_50s_image_path).astype("float32")
            raw_10s_l_img = np.load(raw_10s_l_image_path).astype("float32")
            raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32")
            raw_10s_r_img = np.load(raw_10s_r_image_path).astype("float32")
            eeg_img = np.load(eeg_image_path).astype("float32")

            eeg_img = resize(eeg_img, self.test_imgsize)
            spec_img = resize(spec_img, self.test_imgsize)
            raw_10s_l_img = resize(raw_10s_l_img, self.test_imgsize)
            raw_10s_c_img = resize(raw_10s_c_img, self.test_imgsize)
            raw_10s_r_img = resize(raw_10s_r_img, self.test_imgsize)
            raw_50s_img = resize(raw_50s_img, self.test_imgsize)

            eeg_img = np.expand_dims(eeg_img, -1)
            spec_img = np.expand_dims(spec_img, -1)
            raw_50s_img = np.expand_dims(raw_50s_img, -1)
            raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
            raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
            raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

            eps = 1e-6
            spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
            spec_img = np.log(spec_img)
            spec_img = np.nan_to_num(spec_img, nan=0.0)

            spec_m = GLOBAL_NORM_STATS["spec"]["mean"]
            spec_s = GLOBAL_NORM_STATS["spec"]["std"]
            spec_img = (spec_img - spec_m) / (spec_s + eps)

            eeg_m = GLOBAL_NORM_STATS["eeg"]["mean"]
            eeg_s = GLOBAL_NORM_STATS["eeg"]["std"]
            eeg_img = (eeg_img - eeg_m) / (eeg_s + eps)

            raw50_m = GLOBAL_NORM_STATS["raw50"]["mean"]
            raw50_s = GLOBAL_NORM_STATS["raw50"]["std"]
            raw_50s_img = (raw_50s_img - raw50_m) / (raw50_s + eps)

            raw10_m = GLOBAL_NORM_STATS["raw10"]["mean"]
            raw10_s = GLOBAL_NORM_STATS["raw10"]["std"]
            raw_10s_l_img = (raw_10s_l_img - raw10_m) / (raw10_s + eps)
            raw_10s_c_img = (raw_10s_c_img - raw10_m) / (raw10_s + eps)
            raw_10s_r_img = (raw_10s_r_img - raw10_m) / (raw10_s + eps)

            return (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                eeg_id,
            )




## === cell 22
if use_models:

    class NetSmallDyn(nn.Module):
        def __init__(self, back_bone, device_id):
            super().__init__()
            self.spec_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )
            self.eeg_model = timm.create_model(
                back_bone,
                num_classes=6,
                pretrained=False,
                in_chans=1,
                dynamic_img_pad=True,
                dynamic_img_size=True,
            )
            self.raw_50s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )
            self.raw_10s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )

            self.device_id = device_id

            self.spec_model.fc_norm = nn.Identity()
            self.spec_model.head_drop = nn.Identity()
            self.spec_model.head = nn.Identity()

            self.eeg_model.fc_norm = nn.Identity()
            self.eeg_model.head_drop = nn.Identity()
            self.eeg_model.head = nn.Identity()

            self.raw_50s_model.fc_norm = nn.Identity()
            self.raw_50s_model.head_drop = nn.Identity()
            self.raw_50s_model.head = nn.Identity()

            self.raw_10s_model.fc_norm = nn.Identity()
            self.raw_10s_model.head_drop = nn.Identity()
            self.raw_10s_model.head = nn.Identity()

            self.head = nn.Linear(384 * 4, 6)
            self.head1 = nn.Linear(384, 6)
            self.head2 = nn.Linear(384, 6)
            self.head3 = nn.Linear(384, 6)
            self.head4 = nn.Linear(384, 6)

        def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
            spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

            spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
            eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
            raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
            raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

            feature = torch.cat(
                (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
            )
            logits = self.head(feature)
            logits_1 = self.head1(spec_feature)
            logits_2 = self.head2(eeg_feature)
            logits_3 = self.head3(raw_50s_feature)
            logits_4 = self.head4(raw_10s_feature)
            return logits, logits_1, logits_2, logits_3, logits_4




## === cell 23
if use_models:
    vit_models = []
    for w in model_weights_4:
        model = NetSmallDyn("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
        _safe_load_state_dict(model, w, device)
        model.eval()
        vit_models.append(model)

    test_data = ImageFolder2(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
    )

    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in test_loader:
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                probs = (
                    logits_l.softmax(dim=1)
                    + logits_c.softmax(dim=1)
                    + logits_r.softmax(dim=1)
                ) / 3.0
                probs = probs / probs.sum(dim=1, keepdim=True)
                ensemble_probs += probs
            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)

            ensemble_probs = ensemble_probs.detach().cpu().numpy()
            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                if eeg_id not in result_4:
                    result_4[eeg_id] = np.zeros(6, dtype=np.float64)
                    count_4[eeg_id] = 0
                result_4[eeg_id] += ensemble_probs[j].astype(np.float64)
                count_4[eeg_id] += 1

    result_4 = _finalize_result_dict(result_4, count_4, eps=1e-6)

    for model in vit_models:
        del model
    torch.cuda.empty_cache()
    gc.collect()



## === cell 24
pass



## === cell 25
if use_models:
    pass



## === cell 26
if use_models:

    class ImageFolder3(data.Dataset):
        def __init__(self, df, test_imgsize):
            super().__init__()
            df["eeg_id"] = df["eeg_id"]
            self.spec_data_path = spec_directory_path
            self.eeg_data_path = eeg_directory_path
            self.raw_50s_data_path = raw_50s_directory_path
            self.raw_10s_data_path = raw_10s_directory_path
            self.df = df.reset_index(drop=True)
            self.test_imgsize = test_imgsize

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            row = self.df.loc[index]
            eeg_id = str(row.eeg_id)
            spec_image_path = os.path.join(self.spec_data_path, eeg_id + ".npy")
            eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + ".npy")
            raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
            raw_10s_l_image_path = os.path.join(
                self.raw_10s_data_path, eeg_id + "_l.npy"
            )
            raw_10s_c_image_path = os.path.join(
                self.raw_10s_data_path, eeg_id + "_c.npy"
            )
            raw_10s_r_image_path = os.path.join(
                self.raw_10s_data_path, eeg_id + "_r.npy"
            )

            spec_img = np.load(spec_image_path).astype("float32")
            raw_50s_img = np.load(raw_50s_image_path).astype("float32")
            raw_10s_l_img = np.load(raw_10s_l_image_path).astype("float32")
            raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32")
            raw_10s_r_img = np.load(raw_10s_r_image_path).astype("float32")
            eeg_img = np.load(eeg_image_path).astype("float32")

            eeg_img = resize(eeg_img, self.test_imgsize)
            spec_img = resize(spec_img, self.test_imgsize)
            raw_10s_l_img = resize(raw_10s_l_img, self.test_imgsize)
            raw_10s_c_img = resize(raw_10s_c_img, self.test_imgsize)
            raw_10s_r_img = resize(raw_10s_r_img, self.test_imgsize)
            raw_50s_img = resize(raw_50s_img, self.test_imgsize)

            eeg_img = np.expand_dims(eeg_img, -1)
            spec_img = np.expand_dims(spec_img, -1)
            raw_50s_img = np.expand_dims(raw_50s_img, -1)
            raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
            raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
            raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

            eps = 1e-6
            spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
            spec_img = np.log(spec_img)
            spec_img = np.nan_to_num(spec_img, nan=0.0)

            eeg_m = GLOBAL_NORM_STATS["eeg"]["mean"]
            eeg_s = GLOBAL_NORM_STATS["eeg"]["std"]
            eeg_img = (eeg_img - eeg_m) / (eeg_s + eps)

            spec_m = GLOBAL_NORM_STATS["spec"]["mean"]
            spec_s = GLOBAL_NORM_STATS["spec"]["std"]
            spec_img = (spec_img - spec_m) / (spec_s + eps)

            raw50_m = GLOBAL_NORM_STATS["raw50"]["mean"]
            raw50_s = GLOBAL_NORM_STATS["raw50"]["std"]
            raw_50s_img = (raw_50s_img - raw50_m) / (raw50_s + eps)

            raw10_m = GLOBAL_NORM_STATS["raw10"]["mean"]
            raw10_s = GLOBAL_NORM_STATS["raw10"]["std"]
            raw_10s_l_img = (raw_10s_l_img - raw10_m) / (raw10_s + eps)
            raw_10s_c_img = (raw_10s_c_img - raw10_m) / (raw10_s + eps)
            raw_10s_r_img = (raw_10s_r_img - raw10_m) / (raw10_s + eps)

            return (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                eeg_id,
            )




## === cell 27
if use_models:

    class NetFinal(nn.Module):
        def __init__(self, back_bone, device_id):
            super().__init__()
            self.spec_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.eeg_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.raw_50s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )
            self.raw_10s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )

            self.device_id = device_id

            self.spec_model.fc_norm = nn.Identity()
            self.spec_model.head_drop = nn.Identity()
            self.spec_model.head = nn.Identity()

            self.eeg_model.fc_norm = nn.Identity()
            self.eeg_model.head_drop = nn.Identity()
            self.eeg_model.head = nn.Identity()

            self.raw_50s_model.fc_norm = nn.Identity()
            self.raw_50s_model.head_drop = nn.Identity()
            self.raw_50s_model.head = nn.Identity()

            self.raw_10s_model.fc_norm = nn.Identity()
            self.raw_10s_model.head_drop = nn.Identity()
            self.raw_10s_model.head = nn.Identity()

            self.head = nn.Linear(768 * 2 + 384 * 2, 6)

        def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
            spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
            raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

            spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
            eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
            raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
            raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

            feature = torch.cat(
                (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
            )
            logits = self.head(feature)
            return logits




## === cell 28
if use_models:
    vit_models = []
    for w in model_weights_3:
        model = NetFinal("vit_base_patch14_reg4_dinov2.lvd142m", device).to(device)
        _safe_load_state_dict(model, w, device)
        model.eval()
        vit_models.append(model)

    test_data = ImageFolder3(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
    )

    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in test_loader:
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits_l = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
                logits_c = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs)
                logits_r = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs)
                probs = (
                    logits_l.softmax(dim=1)
                    + logits_c.softmax(dim=1)
                    + logits_r.softmax(dim=1)
                ) / 3.0
                probs = probs / probs.sum(dim=1, keepdim=True)
                ensemble_probs += probs
            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)

            ensemble_probs = ensemble_probs.detach().cpu().numpy()
            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                if eeg_id not in result_3:
                    result_3[eeg_id] = np.zeros(6, dtype=np.float64)
                    count_3[eeg_id] = 0
                result_3[eeg_id] += ensemble_probs[j].astype(np.float64)
                count_3[eeg_id] += 1

    result_3 = _finalize_result_dict(result_3, count_3, eps=1e-6)

    for model in vit_models:
        del model
    torch.cuda.empty_cache()
    gc.collect()



## === cell 29
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sub = sub[["eeg_id"] + CLASSES].copy()

FINAL_ALPHA = 0.30  # was 0.22
FINAL_T = 1.85  # was 1.60
FINAL_EPS_FLOOR = 4e-4  # was 2e-4

if use_models:
    preds = np.zeros((len(sub), 6), dtype=np.float64)
    groups = [result_8, result_7, result_3, result_4, result_5]
    for i, eeg_id in enumerate(sub["eeg_id"].astype(str).values):
        acc = np.zeros(6, dtype=np.float64)
        n = 0
        for g in groups:
            v = g.get(eeg_id, None)
            if v is not None:
                v = np.asarray(v, dtype=np.float64)
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                s = v.sum()
                if s > 0:
                    v = v / s
                    acc += v
                    n += 1
        if n > 0:
            r = acc / n
        else:
            r = prior.copy()
        preds[i] = r

    preds = _normalize_probs(np.clip(preds, 1e-7, None), eps=1e-12)

    preds = (1.0 - FINAL_ALPHA) * preds + FINAL_ALPHA * prior.reshape(1, -1)
    preds = _apply_temperature(preds, t=FINAL_T, eps=1e-12)

    preds = _normalize_probs(np.clip(preds, FINAL_EPS_FLOOR, None), eps=1e-12)
else:
    preds = np.tile(prior, (len(sub), 1))
    preds = _normalize_probs(np.clip(preds, 1e-7, None), eps=1e-12)

    preds = (1.0 - FINAL_ALPHA) * preds + FINAL_ALPHA * prior.reshape(1, -1)
    preds = _apply_temperature(preds, t=FINAL_T, eps=1e-12)

    preds = _normalize_probs(np.clip(preds, FINAL_EPS_FLOOR, None), eps=1e-12)

sub.loc[:, CLASSES] = preds.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 30
if not DEBUG:
    import shutil

    shutil.rmtree("/kaggle/working/spec_spectrograms", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_spectrograms", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_50s_raws", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_10s_raws", ignore_errors=True)
    shutil.rmtree("/kaggle/working/squeezeformer", ignore_errors=True)
