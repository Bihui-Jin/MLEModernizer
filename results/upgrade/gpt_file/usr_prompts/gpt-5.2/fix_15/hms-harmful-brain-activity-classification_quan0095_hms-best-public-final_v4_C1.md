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

0.284743286928261

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime failures by (1) removing the hard CUDA requirement and running inference on CPU when no GPU is available, and (2) making checkpoint loading device-safe. To ensure the pipeline runs end-to-end in this environment, I also guard optional model ensembles whose weight files may be missing and always produce a valid `submission.csv` with probabilities that sum to 1 for every `eeg_id`. I fix a couple of latent errors in feature generation (missing `signal` import usage in one function, and fragile `fillna(..., inplace=True)` on a view) without changing the core feature/model logic. Finally, I add a safe fallback to the sample submission (uniform distribution) if no models can be loaded, so a valid CSV is always created.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.28474), so we should improve prediction correctness without changing the model/feature core. The biggest likely issue is that `spec_spectrograms/` is being saved under the wrong key (`eeg_id` instead of `spectrogram_id`), so the model is effectively consuming mismatched spectrograms, which can destroy KL score. I make the smallest fix: save spectrogram `.npy` using `spectrogram_id` and load it using the row’s `spectrogram_id`, leaving all model code and inference logic intact. I also add a tiny guard to ensure `spectrogram_id` is returned correctly from the dataset and used consistently, while keeping submission formatting and probability normalization unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.28474), so we should fix a likely data/label alignment issue rather than “tuning” the model. The biggest minimal-impact bug is that the dataset uses `row.spectrogram_id` when loading spectrogram `.npy`, but if the saving key ever mismatches (or if there are duplicates/overwrites), the model effectively see the wrong spectrograms and KL explode. I (1) make the spectrogram saving/loading key explicitly and consistently `spectrogram_id` with a uniqueness-safe filename, and (2) deduplicate work so that each `eeg_id`/`spectrogram_id` pair is processed once, preventing accidental overwrites and reducing I/O errors; model, transforms, and inference remain unchanged. Finally, I keep the same submission post-processing (clip + row-normalize) to ensure valid probabilities.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far worse than the target (0.28474), so the most likely cause is not “tuning” but a data mismatch that makes the model see the wrong inputs. I make a minimal, score-relevant fix: ensure spectrogram `.npy` files are saved and loaded under the correct `spectrogram_id` key (and keep `eeg_id`-based keys for EEG-derived features), plus add a small integrity guard to catch missing/mismatched files early. I also ensure the `DataLoader` iterates over the same unique `(eeg_id, spectrogram_id)` pairs used to generate the cached `.npy` files, preventing accidental overwrites/duplicates that can silently poison predictions. No model architecture, forward pass, or post-processing logic be changed beyond these alignment/integrity fixes, and the script still write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.28474), so the most likely win with minimal change is fixing a subtle inference aggregation bug that can badly misalign predictions. In your `DataLoader` loop, `eeg_ids` comes as a list/tuple of strings per batch, but the code uses `str(eeg_ids[j])`, which turns each id into `"('1001717358',)"` instead of `"1001717358"`, so almost no predictions match `sample_submission` ids and you effectively submit near-uniform probabilities. I minimally fix `eeg_id` extraction so keys match exactly, and (to keep semantics unchanged) keep the same model/feature pipeline and post-processing while ensuring the final per-`eeg_id` aggregation and submission lookup align correctly. This should move the score substantially toward the target without changing architecture, features, or loss.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.28474), so the most likely cause is a prediction–submission alignment bug rather than a modeling/tuning issue. The model loop aggregates predictions using `eeg_id` keys, but `eeg_ids` coming from the DataLoader can be a 0-d/1-d numpy/torch object array, and `str(eeg_ids[j])` can silently produce mismatched strings that never match `sample_submission` ids, leading to widespread uniform-fallback rows and a very poor KL. I make the smallest fix: force `eeg_id` and `spectrogram_id` to be returned as plain Python strings from the Dataset, and add a robust “id unwrapping” helper in inference so aggregation keys always match the `sample_submission` exactly. This preserves the exact model/feature/inference semantics while ensuring the computed predictions are actually used for the correct rows.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.28474), so the most likely issue is that your predictions are not being written to the correct `eeg_id` rows (leading to many uniform fallbacks). I make a minimal, score-relevant fix by ensuring the `DataLoader` returns `eeg_id` as a plain string and by simplifying the dataset output so the inference loop receives exactly the four tensors the model uses plus the id (no duplicated raw inputs). I also average per `eeg_id` by the number of occurrences (instead of summing), which preserves intended semantics when duplicates exist and prevents inflated/unbalanced logits accumulation. These changes keep the same feature generation, same model architecture, same softmaxing, and same submission normalization, but should substantially improve alignment and move KL toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.28474), which strongly suggests a prediction–submission alignment failure rather than a “model quality” issue. The smallest high-impact fix is to ensure the model’s per-sample predictions are aggregated by `eeg_id` and then *expanded back* to the submission rows in the exact `sample_submission` order, without any dtype/object/tuple stringification pitfalls. Concretely, I (1) force `eeg_id`/`spectrogram_id` to be read as integers early, (2) make the Dataset return `eeg_id` as a plain Python `int` (so DataLoader collates to a simple int tensor), and (3) simplify the inference keying to use those ints directly (eliminating the common `"('123',)"`-style mismatch that leads to uniform fallbacks). This keeps the same feature generation, same model forward pass, same ensemble averaging, and the same probability clipping/row-normalization, but should move the score substantially toward the target by actually using the model outputs for the correct rows.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995; lower-is-better) is far from the target (0.28474), so the most likely issue is that the submission is largely falling back to near-uniform probabilities because many `eeg_id` keys don’t match between inference aggregation and the final `sample_submission` merge. I make the smallest score-relevant fix by keeping `eeg_id` as an integer key end-to-end (Dataset → DataLoader → aggregation dict → submission lookup), eliminating any stringification/tuple/object-dtype pitfalls. I also ensure the DataLoader doesn’t try to “tensorize” ids by using a tiny custom `collate_fn` (only for ids), while leaving the model, features, forward pass, and probability post-processing unchanged. This should cause the model outputs to actually populate the correct rows and move KL substantially toward the target.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.28474), which strongly suggests a prediction alignment/aggregation problem rather than a small calibration issue. The biggest minimal-risk fix is to aggregate predictions by `eeg_id` across *all* `(eeg_id, spectrogram_id)` pairs you infer on, instead of averaging within-batch and then re-averaging across ensembles; we accumulate per-`eeg_id` per-ensemble probabilities and only normalize once at the very end. To avoid silent key mismatches that lead to uniform fallbacks, we also enforce `eeg_id` as an `int64` end-to-end (Dataset → collate → dict keys → submission join). Finally, we keep the model/features identical, but fix determinism flags (your current seed function sets `deterministic=True` and `benchmark=True` simultaneously) to ensure reproducible inference and reduce run-to-run noise.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.28474), so the most likely issue is not “model quality” but that the pipeline is not actually using the intended pretrained ensembles. The biggest minimal-impact fix is to correct the weight loading logic for these timm ViT backbones: the `.pth` checkpoints typically store a dict (often with keys like `model`, `state_dict`, etc.), and strict-loading the whole object as a state_dict silently fails by skipping all ensembles (leading to near-uniform predictions). I make `safe_load_weights` robust to common checkpoint formats (still strict on the extracted state_dict) and also ensure the `device_id` argument is passed consistently as a string to avoid edge cases. No model architecture, feature extraction, inference loop, or post-processing change beyond enabling the intended weights to actually load so predictions populate correctly.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.28474), so we should fix a likely inference-time misconfiguration rather than “tune” the model. The biggest minimal-impact issue in your current code is that `Net` hardcodes the feature dimension as `384`, but the `vit_base_patch14_reg4_dinov2` backbone outputs a different embedding size (typically 768), so those checkpoints either fail to load (silently skipping that ensemble) or never contribute correct predictions—both can badly hurt KL. I make `Net` infer the backbone embedding dimension at init-time (keeping the same architecture/heads, just correct input sizes), and I make `safe_load_weights` handle common checkpoint key prefixes (e.g., `model.`) so more of your provided weights actually load. These changes keep feature extraction, forward pass semantics, and submission normalization identical, but should materially move the score toward your target by enabling the intended ensembles to run.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower-is-better) is far worse than the target (0.28474), so we should fix a likely inference mismatch rather than “tune” anything. The minimal high-impact issue here is that your cached spectrograms are transposed and time-cropped on the wrong axis: `spec.values[:, 1:].T` makes the array `(time, freq)` but your comment/intent (and most pretrained pipelines for this competition) expects `(freq, time)` with time cropped to 300. I fix the spectrogram caching to keep it as `(freq, time)` and crop time on axis=1, without changing the model, resizing, or ensemble logic. This should make the model see correctly-oriented spectrogram inputs and move the KL substantially toward the target while preserving the core approach.'
- What this solution (achieved 1.40995) has done: 'We make two minimal, score-relevant fixes that commonly cause very high KL in this competition: (1) correct the spectrogram orientation/cropping so the cached spectrogram is saved as `(freq, time)` and cropped along the time axis (your current code transposes then crops the wrong axis), and (2) ensure the spectrogram cache is normalized the same way as training by applying a safe log/standardization consistently after resize (keeping your existing semantics but preventing degenerate NaNs/zeros). These changes keep your model architecture, weights, and inference loop intact, but should make the inputs match what the pretrained weights expect, moving the score down toward the target. The submission writing and probability normalization remain unchanged and still produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings

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

import librosa  # noqa: F401



## === cell 3
from scipy.signal import butter, lfilter  # noqa: F401
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
            eeg_1 = eeg[COLS[kk]].astype("float32")
            eeg_1 = eeg_1.fillna(eeg_1.mean()).values

            eeg_2 = eeg[COLS[kk + 1]].astype("float32")
            eeg_2 = eeg_2.fillna(eeg_2.mean()).values

            new_eeg = eeg_1 - eeg_2

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
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]].astype("float32")
            eeg_1 = eeg_1.fillna(eeg_1.mean()).values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]].astype("float32")
            eeg_2 = eeg_2.fillna(eeg_2.mean()).values

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
    return eeg_c


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
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]].astype("float32")
            eeg_1 = eeg_1.fillna(eeg_1.mean()).values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]].astype("float32")
            eeg_2 = eeg_2.fillna(eeg_2.mean()).values

            new_eeg = eeg_1 - eeg_2
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

if DEBUG is True:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
        dtype={"eeg_id": "int64", "spectrogram_id": "int64"},
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
        dtype={"eeg_id": "int64", "spectrogram_id": "int64"},
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

print(test.shape)

spec_directory_path = "spec_spectrograms/"
os.makedirs(spec_directory_path, exist_ok=True)

eeg_directory_path = "eeg_spectrograms/"
os.makedirs(eeg_directory_path, exist_ok=True)

raw_10s_directory_path = "eeg_10s_raws/"
os.makedirs(raw_10s_directory_path, exist_ok=True)

raw_50s_directory_path = "eeg_50s_raws/"
os.makedirs(raw_50s_directory_path, exist_ok=True)

from joblib import Parallel, delayed

test_pairs = test[["eeg_id", "spectrogram_id"]].drop_duplicates().reset_index(drop=True)


def save(row):
    eeg_id = int(row["eeg_id"])
    spec_id = int(row["spectrogram_id"])

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].astype("float32")  # (time, freq)
    spec_arr = spec_arr.T  # (freq, time)
    split_spec_arr = spec_arr[:, :300]  # crop time to 300
    np.save(os.path.join(spec_directory_path, f"{spec_id}.npy"), split_spec_arr)

    img_c = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(os.path.join(raw_10s_directory_path, f"{eeg_id}_c.npy"), img_c)

    img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(os.path.join(raw_50s_directory_path, f"{eeg_id}.npy"), img)

    img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(os.path.join(eeg_directory_path, f"{eeg_id}.npy"), img)


_ = Parallel(n_jobs=4)(delayed(save)(row) for _, row in test_pairs.iterrows())




## === cell 7
class Config:
    seed = 2024
    num_folds = 5


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 8
import timm



## === cell 9
import torch.utils.data as data
from torch.utils.data import DataLoader



## === cell 10
from skimage.transform import resize


class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        df = df.copy()
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

        eeg_id = int(row.eeg_id)
        spec_id = int(row.spectrogram_id)

        spec_image_path = os.path.join(self.spec_data_path, f"{spec_id}.npy")
        eeg_image_path = os.path.join(self.eeg_data_path, f"{eeg_id}.npy")
        raw_50s_image_path = os.path.join(self.raw_50s_data_path, f"{eeg_id}.npy")
        raw_10s_c_image_path = os.path.join(self.raw_10s_data_path, f"{eeg_id}_c.npy")

        if not os.path.exists(spec_image_path):
            raise FileNotFoundError(
                f"Missing cached spectrogram for spectrogram_id={spec_id}: {spec_image_path}"
            )
        if not os.path.exists(eeg_image_path):
            raise FileNotFoundError(
                f"Missing cached EEG spec for eeg_id={eeg_id}: {eeg_image_path}"
            )
        if not os.path.exists(raw_50s_image_path):
            raise FileNotFoundError(
                f"Missing cached raw50 for eeg_id={eeg_id}: {raw_50s_image_path}"
            )
        if not os.path.exists(raw_10s_c_image_path):
            raise FileNotFoundError(
                f"Missing cached raw10_c for eeg_id={eeg_id}: {raw_10s_c_image_path}"
            )

        spec_img = np.load(spec_image_path).astype("float32")
        raw_50s_img = np.load(raw_50s_image_path).astype("float32")
        raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32")
        eeg_img = np.load(eeg_image_path).astype("float32")

        eeg_img = resize(eeg_img, self.test_imgsize)
        spec_img = resize(spec_img, self.test_imgsize)
        raw_10s_c_img = resize(raw_10s_c_img, self.test_imgsize)
        raw_50s_img = resize(raw_50s_img, self.test_imgsize)

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)

        eps = 1e-6
        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0, posinf=0.0, neginf=0.0)

        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_c_img,
            np.int64(eeg_id),
        )




## === cell 11
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

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 518, 518)
            feat = self.spec_model.forward_features(dummy)[:, 0]
            embed_dim = int(feat.shape[-1])

        self.head = nn.Linear(embed_dim * 4, 6)
        self.head1 = nn.Linear(embed_dim, 6)
        self.head2 = nn.Linear(embed_dim, 6)
        self.head3 = nn.Linear(embed_dim, 6)
        self.head4 = nn.Linear(embed_dim, 6)

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




## === cell 12
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def safe_load_weights(model, weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unsupported checkpoint format at {weight_path}: {type(ckpt)}"
        )

    prefixes = ("module.", "model.", "net.")
    for p in prefixes:
        if any(k.startswith(p) for k in ckpt.keys()):
            ckpt = {k[len(p) :] if k.startswith(p) else k: v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=True)
    return model


def _collate_keep_ids(batch):
    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs, eeg_ids = zip(*batch)
    spec_imgs = torch.from_numpy(np.stack(spec_imgs, axis=0))
    eeg_imgs = torch.from_numpy(np.stack(eeg_imgs, axis=0))
    raw_50s_imgs = torch.from_numpy(np.stack(raw_50s_imgs, axis=0))
    raw_10s_imgs = torch.from_numpy(np.stack(raw_10s_imgs, axis=0))
    eeg_ids = [int(x) for x in eeg_ids]
    return spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs, eeg_ids


def run_ensemble_inference(model_weights, backbone_name, batch_size=8):
    vit_models = []
    for w in model_weights:
        if not os.path.exists(w):
            print("Missing weights, skipping:", w)
            continue
        model = Net(backbone_name, device_id=str(device)).to(device)
        model = safe_load_weights(model, w, device)
        model.eval()
        vit_models.append(model)

    if len(vit_models) == 0:
        return None

    infer_df = test_pairs.copy()
    test_data = ImageFolder(infer_df, (518, 518))
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0 if device == "cpu" else 2,
        drop_last=False,
        collate_fn=_collate_keep_ids,
    )

    result_sum = {}
    result_cnt = {}

    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_c_imgs,
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
                ensemble_probs += logits_c.softmax(dim=1)
            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs.detach().cpu().numpy().astype(np.float32)

            for j, eeg_id_key in enumerate(eeg_ids):
                eeg_id_key = int(eeg_id_key)
                if eeg_id_key not in result_sum:
                    result_sum[eeg_id_key] = np.zeros(6, dtype=np.float32)
                    result_cnt[eeg_id_key] = 0
                result_sum[eeg_id_key] += ensemble_probs[j]
                result_cnt[eeg_id_key] += 1

    for m in vit_models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {}
    for k in result_sum:
        c = max(1, int(result_cnt[k]))
        result[k] = result_sum[k] / c
    return result




## === cell 13
result_6 = run_ensemble_inference(
    model_weights=[
        "/kaggle/input/hms-stage2/fold_0_spec_raw_50_10_bestlb.pth",
        "/kaggle/input/hms-stage2/fold_1_spec_raw_50_10_bestlb.pth",
        "/kaggle/input/hms-stage2/fold_2_spec_raw_50_10_bestlb.pth",
        "/kaggle/input/hms-stage2/fold_3_spec_raw_50_10_bestlb.pth",
        "/kaggle/input/hms-stage2/fold_4_spec_raw_50_10_bestlb.pth",
    ],
    backbone_name="vit_small_patch14_reg4_dinov2.lvd142m",
    batch_size=8 if device == "cpu" else 32,
)

result_7 = run_ensemble_inference(
    model_weights=[
        "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
        "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
        "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
        "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
        "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
    ],
    backbone_name="vit_small_patch14_reg4_dinov2.lvd142m",
    batch_size=8 if device == "cpu" else 32,
)

result_5 = run_ensemble_inference(
    model_weights=[
        "/kaggle/input/hms-bestlb-vitbase/fold_0_raw_50_10_bestlb_vitbase.pth",
        "/kaggle/input/hms-bestlb-vitbase/fold_1_raw_50_10_bestlb_vitbase.pth",
        "/kaggle/input/hms-bestlb-vitbase/fold_2_raw_50_10_bestlb_vitbase.pth",
        "/kaggle/input/hms-bestlb-vitbase/fold_3_raw_50_10_bestlb_vitbase.pth",
        "/kaggle/input/hms-bestlb-vitbase/fold_4_raw_50_10_bestlb_vitbase.pth",
    ],
    backbone_name="vit_base_patch14_reg4_dinov2.lvd142m",
    batch_size=4 if device == "cpu" else 16,
)

result_3 = None

print(
    "Loaded ensembles:",
    "result_6" if result_6 is not None else "result_6=None",
    "result_7" if result_7 is not None else "result_7=None",
    "result_5" if result_5 is not None else "result_5=None",
    "result_3" if result_3 is not None else "result_3=None",
)



## === cell 14
sample_sub_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sub = pd.read_csv(sample_sub_path)

available_results = [
    r for r in [result_6, result_7, result_5, result_3] if r is not None
]

preds = np.zeros((len(sub), 6), dtype=np.float32)

if len(available_results) == 0:
    preds[:] = 1.0 / 6.0
else:
    sub_eeg_ids = sub["eeg_id"].astype("int64").values
    for i, eeg_id in enumerate(sub_eeg_ids):
        eeg_id = int(eeg_id)
        acc = np.zeros(6, dtype=np.float32)
        cnt = 0
        for res in available_results:
            v = res.get(eeg_id, None)
            if v is not None:
                acc += v.astype(np.float32)
                cnt += 1
        if cnt == 0:
            acc[:] = 1.0 / 6.0
        else:
            acc /= cnt
        preds[i] = acc

preds = np.clip(preds, 1e-8, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

out = pd.DataFrame(
    np.concatenate([sub[["eeg_id"]].values, preds], axis=1),
    columns=["eeg_id"] + CLASSES,
)
out.to_csv("submission.csv", index=False)
print(out.head())
print("Saved submission.csv with shape:", out.shape)



## === cell 15
if DEBUG is False:
    import shutil

    for p in [
        "/kaggle/working/spec_spectrograms/",
        "/kaggle/working/eeg_spectrograms/",
        "/kaggle/working/eeg_50s_raws/",
        "/kaggle/working/eeg_10s_raws/",
        "/kaggle/working/squeezeformer",
    ]:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
