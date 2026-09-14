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

0.2834440389152499

# 6. Current score

0.91622

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Your code currently can fail to yield a score because it depends on external Kaggle input datasets for model weights (e.g. `/kaggle/input/hms-stage2/...`) and also does not guarantee the submission rows match `test.csv` order/coverage; either issue can lead to no valid submission or a misaligned one. I (1) make the pipeline robust: if weights are missing or CUDA is unavailable, it fall back to a valid probabilistic submission (uniform prior) so you always get a scored submission, and (2) ensure the final `submission.csv` is aligned exactly to `sample_submission.csv` (right row count/order) and each row is normalized to sum to 1 (required by the metric checker). I also remove the extremely verbose per-sample printing inside inference loops (it can cause timeouts and prevent completion), without changing model computations. These are minimal, stability-first fixes to get a valid baseline score and allow you to iterate toward the 0.283 target afterward.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.28344), so we should make a small but meaningful improvement without changing the model core. The biggest issue is that your strong model predictions are usually missing because the code points to non-existent external weights (`/kaggle/input/hms-stage2/...`), forcing a near-uniform fallback that scores poorly. I (1) switch weight discovery to look for `.pth` files under `/kaggle/input` and `/kaggle/working` (no architecture/training changes), (2) add a very light “prior blending” fallback using class priors from `train.csv` when no model preds exist (still legitimate, and improves KL vs uniform), and (3) fix determinism settings (your seed function sets `deterministic=True` and `benchmark=True` simultaneously) to reduce run-to-run variance without changing semantics. Submission alignment and normalization to `sample_submission.csv` be preserved.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779; lower-is-better) is far from the target (0.28344), and the most likely reason is that the model predictions are often missing or degraded: weight loading can silently fail due to mismatched checkpoints, and when that happens you fall back to a weak prior. I make a minimal but high-impact robustness fix: load checkpoints with key-stripping/shape-checking so compatible weights actually get used (without changing the model), and ensure the per-eeg_id prediction keys match the sample submission dtype (string) so lookups never miss. Finally, I keep your existing prior-blend/normalization but make the ensemble dictionary keys consistent and guaranteed to populate, which should move KL materially toward the target without changing architecture or inference semantics.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779; lower-is-better) is far above the target (0.28344), so we should make small changes that increase the chance you actually use the intended trained weights and that the inference matches how these ViT backbones expect inputs. I (1) fix a key bug where `device_id` is stored as a string like `"cuda:0"` but later used as if it were an integer GPU id, which can break `load_state_dict` and silently trigger your weak fallback; (2) apply the same log/standardize normalization to the EEG-derived “images” as you already do for the spectrogram image (this keeps the same feature extraction/model, but improves calibration and reduces distribution shift vs training); and (3) make sure weight discovery prefers fold checkpoints deterministically and uses `map_location="cpu"` when needed to avoid partial-load failures. These are minimal, inference-only robustness fixes (no architecture/training/loop changes) and should move KL materially downward toward the target.'
- What this solution (achieved 1.39779) has done: 'Your KL (1.39779, lower-is-better) is still far from the 0.28344 target, and the biggest likely cause is that you’re not actually loading/using the intended fold checkpoints (so you fall back to a weak class-prior for most/all test rows). I make weight discovery/load materially more robust while keeping the exact same model and inference logic: (1) broaden checkpoint discovery to include `.pt` and common “ckpt” naming, (2) properly handle checkpoints that store weights under nested keys and/or with `module.` prefixes, and (3) verify coverage so missing IDs are clearly reported and reduced. This should increase the fraction of test rows getting real model predictions (instead of prior), which is the smallest change most likely to move KL downward toward the target. The submission writing, ordering (matching `sample_submission.csv`), and per-row normalization remain unchanged.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower-is-better) is far above the 0.28344 target, and the most likely reason is that you are still not actually using trained weights (so you mostly submit a weak global prior). I make weight discovery/load robust in a minimal way: (1) also search common checkpoint extensions (including `.ckpt`) and prioritize files that match your backbone name, (2) properly handle checkpoints saved from Lightning/EMA by extracting nested `state_dict` and stripping prefixes, and (3) ensure we don’t silently accept “almost empty” loads by requiring a minimum fraction of matched tensors per checkpoint before using that model. This keeps the exact same model/inference logic but increases the chance real model predictions populate `result_7`, which should reduce KL toward the target. Submission alignment/normalization remains unchanged and we still always produce `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'Your KL is far above target (1.39779 vs 0.28344; lower is better), and the most likely cause is that inference is effectively a weak global prior because the heavy ViT checkpoints are not being found/loaded and/or inference is too slow to finish reliably within time. I make two minimal, score-relevant changes while preserving your model/feature logic: (1) ensure we only keep checkpoints that actually load with a good tensor-match fraction and (2) increase `DataLoader` throughput safely (pin_memory on CUDA + persistent workers) so the model predictions cover more/all test rows instead of falling back. Finally, I slightly increase the prior-blend strength only when predictions are missing/degenerate (still legitimate, improves KL vs near-uniform/prior-only failure cases) while keeping normalization and submission alignment identical.'
- What this solution (achieved 1.39779) has done: 'Your current KL is far above target (1.39779 vs 0.28344; lower is better), and the most likely reason is that you are still submitting mostly the global prior because your fold checkpoints aren’t found/loaded in this environment. I make the smallest changes that increase the probability real model predictions are used: (1) explicitly search the provided competition dataset folders for checkpoints (not just `/kaggle/input`), (2) lower the “matched tensor fraction” gate so partially compatible checkpoints can contribute instead of being discarded, and (3) make the dataloader safer (disable `persistent_workers` when `num_workers=0`) to avoid silent stalls/timeouts that reduce coverage. These changes keep the same model architecture and inference logic, but should materially increase prediction coverage and reduce KL toward the target.'
- What this solution (achieved 1.39779) has done: 'Your score is far worse than the target (KL 1.39779 vs 0.28344; lower is better), so the smallest likely win is to ensure you actually use trained checkpoints instead of falling back to the global prior. I keep your model/data/loops unchanged, but (1) broaden and prioritize checkpoint discovery to include the common “hms-stage2” location and prefer fold checkpoints, and (2) relax the “matched tensor fraction” gate slightly so partially compatible checkpoints aren’t discarded (still using strict shape-matching per tensor). Finally, I add a tiny safety fix to the spectrogram extraction to avoid NaNs/inf propagating into saved .npy files (which can silently degrade predictions).'
- What this solution (achieved 1.39779) has done: 'Your KL (1.39779; lower-is-better) is far above the target (0.28344), and the most likely reason is still that `vit_models` ends up empty (or nearly so) because checkpoints aren’t found/loaded, causing almost all rows to use the weak global prior. I make weight loading more permissive but still safe by (1) detecting and using the correct feature dimension from the backbone instead of hardcoding 384 (this avoids shape-mismatch that prevents loads), and (2) widening the per-tensor match logic to allow the common “head mismatch only” scenario while requiring a higher match fraction for the backbone tensors. Finally, I keep your prediction/normalization/submission format identical, only adding a tiny epsilon-safe softmax and ensuring every `eeg_id` key is a string consistently so lookups never miss.'
- What this solution (achieved 1.39779) has done: 'Your current KL is far above the target (1.39779 vs 0.28344; lower is better), and the most likely reason is that you’re still not actually using meaningful model predictions for most/all test rows (coverage ends up near-zero and you fall back to a weak global prior). I make minimal, inference-only fixes that preserve your architecture and loops: (1) stop discarding the classification head weights during checkpoint filtering (your previous “forgiving load” filtered out `head*`, making the model effectively random), and (2) if your stage2 head is missing/mismatched, automatically fall back to the backbone’s own classifier (`forward()` with `num_classes=6`) so you still get sensible logits from loaded weights. I also make the match-gate logic robust so we accept checkpoints with good backbone+either-head, increasing the chance of real predictions and pushing KL down toward the target. Submission alignment/normalization stays identical.'
- What this solution (achieved 0.73988) has done: 'Your KL (1.39779, lower-is-better) is far above the target (0.28344), and the most likely cause is still that `result_7` is largely empty (or filled with weak predictions) because no compatible checkpoints exist in this environment—so the submission is effectively just the global prior. The smallest legitimate improvement that preserves your whole modeling/inference pipeline is to add a stronger, competition-legal fallback: a per-`patient_id` prior (computed from train vote distributions), blended with the global prior when no model prediction exists for that `eeg_id`. This uses only metadata available at test time and usually improves KL versus a single global prior. I also add a tiny safety normalization at the end of `_get_pred` to handle any accidental NaNs/negatives without changing semantics when predictions are already valid.'
- What this solution (achieved 0.91622) has done: 'Your current KL (0.73988, lower-is-better) is still far above the target (0.28344), and at this point the biggest likely “minimal change” win is to improve the fallback predictions used when `result_7` is missing/weak by using more specific, test-time-legal priors. I keep your model/inference pipeline intact, but strengthen the fallback by adding (a) a `patient_id + expert_consensus` prior when available and (b) an `expert_consensus`-only prior when patient history is sparse, then blend these priors smoothly with the existing global/patient priors. I also add a tiny uniform floor-mix after combining predictions (doesn’t change semantics when outputs are already good) to reduce overconfident near-zero probabilities, which KL strongly penalizes. All output formatting, row order (matching `sample_submission.csv`), and per-row normalization remain unchanged, and the script still always writes `submission.csv`.'

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



## === cell 4
from scipy import signal


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
            spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0).astype(
                "float32"
            )

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

if DEBUG is True:
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
os.makedirs(spec_directory_path, exist_ok=True)

eeg_directory_path = "eeg_spectrograms/"
os.makedirs(eeg_directory_path, exist_ok=True)

raw_10s_directory_path = "eeg_10s_raws/"
os.makedirs(raw_10s_directory_path, exist_ok=True)

raw_50s_directory_path = "eeg_50s_raws/"
os.makedirs(raw_50s_directory_path, exist_ok=True)

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]
    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")

    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
    split_spec_arr = spec_arr[:, 0:300]
    split_spec_arr = np.nan_to_num(
        split_spec_arr, nan=0.0, posinf=0.0, neginf=0.0
    ).astype("float32")
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
import torchvision  # noqa: F401
from torch.utils.data import DataLoader



## === cell 11
from skimage.transform import resize


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

        def _norm_log(img):
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            m = img.mean(axis=(0, 1))
            s = img.std(axis=(0, 1))
            return (img - m) / (s + eps)

        spec_img = _norm_log(spec_img)

        def _norm_eeg(img):
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
            img = img - img.min() + 1e-3  # ensure strictly positive for log
            img = np.clip(img, 1e-6, np.exp(8))
            img = np.log(img)
            m = img.mean(axis=(0, 1))
            s = img.std(axis=(0, 1))
            return (img - m) / (s + eps)

        eeg_img = _norm_eeg(eeg_img)
        raw_50s_img = _norm_eeg(raw_50s_img)
        raw_10s_l_img = _norm_eeg(raw_10s_l_img)
        raw_10s_c_img = _norm_eeg(raw_10s_c_img)
        raw_10s_r_img = _norm_eeg(raw_10s_r_img)

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

        feat_dim = getattr(self.spec_model, "num_features", None)
        if feat_dim is None:
            feat_dim = 384

        self.head = nn.Linear(feat_dim * 4, 6)
        self.head1 = nn.Linear(feat_dim, 6)
        self.head2 = nn.Linear(feat_dim, 6)
        self.head3 = nn.Linear(feat_dim, 6)
        self.head4 = nn.Linear(feat_dim, 6)

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
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def _safe_load_weights(path):
    return os.path.exists(path) and os.path.isfile(path)


def _discover_weights():
    candidates = []
    exts = (".pth", ".pt", ".ckpt")

    for base in (
        "/kaggle/input/hms-stage2",
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    ):
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                for fn in files:
                    if fn.endswith(exts):
                        candidates.append(os.path.join(root, fn))

    def _score_path(p):
        bn = os.path.basename(p).lower()
        parent = os.path.basename(os.path.dirname(p)).lower()
        score = 0
        if (
            "hms-stage2" in p.lower()
            or "hms_stage2" in p.lower()
            or "stage2" in p.lower()
        ):
            score -= 50
        if "raw_50_10_bestlb_twostage" in bn:
            score -= 40
        if "dinov2" in bn or "vit_small_patch14" in bn:
            score -= 10
        if "fold" in bn:
            score -= 2
        if parent.startswith("fold"):
            score -= 1
        return (score, bn)

    def _fold_key(p):
        bn = os.path.basename(p).lower()
        for i in range(10):
            if f"fold_{i}" in bn:
                return (0, i)
        return (1, 999)

    uniq = list(dict.fromkeys(candidates))
    uniq = sorted(uniq, key=lambda p: (_score_path(p), _fold_key(p), p))
    return uniq


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights", "ema"):
            if k in obj:
                v = obj[k]
                if isinstance(v, dict):
                    return v
                if hasattr(v, "state_dict"):
                    return v.state_dict()
        for k in ("module", "student", "teacher"):
            if k in obj and isinstance(obj[k], dict):
                maybe = obj[k]
                for kk in ("state_dict", "model_state_dict", "model", "net", "weights"):
                    if kk in maybe and isinstance(maybe[kk], dict):
                        return maybe[kk]
    return obj


def _strip_prefix(state, prefixes=("module.", "model.", "net.")):
    if not isinstance(state, dict):
        return state
    new_state = {}
    for k, v in state.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_state[nk] = v
    return new_state


def _load_state_dict_forgiving(model, checkpoint_path, map_location):
    obj = torch.load(checkpoint_path, map_location=map_location)
    state = _strip_prefix(_extract_state_dict(obj))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict-like object")

    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape:
            filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    head_prefixes = ("head", "head1", "head2", "head3", "head4")
    total_nonhead = 0
    loaded_nonhead = 0
    total_head = 0
    loaded_head = 0
    for k in model_state.keys():
        is_head = k.split(".")[0] in head_prefixes
        if is_head:
            total_head += 1
            if k in filtered:
                loaded_head += 1
        else:
            total_nonhead += 1
            if k in filtered:
                loaded_nonhead += 1

    return {
        "n_loaded": len(filtered),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "n_total": len(model_state),
        "loaded_nonhead": loaded_nonhead,
        "total_nonhead": total_nonhead,
        "loaded_head": loaded_head,
        "total_head": total_head,
        "filtered_keys": set(filtered.keys()),
    }




## === cell 14
result_7 = {}

model_weights = [
    "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
]
existing = [w for w in model_weights if _safe_load_weights(w)]
if len(existing) == 0:
    discovered = _discover_weights()
    existing = discovered[:16]
    if len(existing) > 0:
        print(
            f"Discovered {len(discovered)} weight files; trying {len(existing)}:",
            existing[:3],
            "..." if len(existing) > 3 else "",
        )
model_weights = existing

vit_models = []
MIN_NONHEAD_MATCH_FRAC = 0.40
MIN_ANY_HEAD_MATCH = 1  # at least some head params should load; otherwise we likely get random head logits


def _timm_classifier_keys(backbone_model):
    keys = set()
    for cand in ("head.weight", "head.bias", "fc.weight", "fc.bias"):
        if cand in backbone_model.state_dict():
            keys.add(cand)
    return keys


for w in model_weights:
    if _safe_load_weights(w):
        device_id = 0 if torch.cuda.is_available() else -1
        m = Net("vit_small_patch14_reg4_dinov2.lvd142m", device_id).to(device)
        try:
            map_loc = device if torch.cuda.is_available() else "cpu"
            info = _load_state_dict_forgiving(m, w, map_location=map_loc)

            nonhead_frac = info["loaded_nonhead"] / max(1, info["total_nonhead"])
            stage2_head_loaded = info["loaded_head"]

            timm_head_keys = _timm_classifier_keys(m.spec_model)
            timm_head_loaded = sum(
                [1 for k in timm_head_keys if k in info["filtered_keys"]]
            )

            if nonhead_frac < MIN_NONHEAD_MATCH_FRAC:
                print(
                    f"Skipping {os.path.basename(w)} due to low NON-HEAD match fraction: "
                    f"matched_nonhead={info['loaded_nonhead']}/{info['total_nonhead']} ({nonhead_frac:.2%}); "
                    f"matched_all={info['n_loaded']}/{info['n_total']}"
                )
                del m
                continue

            if (stage2_head_loaded + timm_head_loaded) < MIN_ANY_HEAD_MATCH:
                print(
                    f"Skipping {os.path.basename(w)} due to no usable head weights: "
                    f"stage2_head_loaded={stage2_head_loaded}/{info['total_head']}, "
                    f"timm_head_loaded={timm_head_loaded}/{len(timm_head_keys)}"
                )
                del m
                continue

            m._use_stage2_head = stage2_head_loaded > 0

            print(
                f"Loaded weights from {os.path.basename(w)}: "
                f"nonhead_matched={info['loaded_nonhead']}/{info['total_nonhead']} ({nonhead_frac:.2%}), "
                f"stage2_head_loaded={stage2_head_loaded}/{info['total_head']}, "
                f"timm_head_loaded={timm_head_loaded}/{len(timm_head_keys)}, "
                f"all_matched={info['n_loaded']}/{info['n_total']}, missing={info['n_missing']}, unexpected={info['n_unexpected']}"
            )
            m.eval()
            vit_models.append(m)
        except Exception as e:
            print(f"Weight load failed for {w}: {e}")
            del m

if len(vit_models) > 0:
    test_data = ImageFolder(test, (518, 518))
    use_cuda = torch.cuda.is_available()

    num_workers = 4 if use_cuda else 0
    persistent_workers = True if num_workers > 0 else False

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        pin_memory=use_cuda,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    def _model_probs(model, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        if getattr(model, "_use_stage2_head", False):
            logits, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs)
            return torch.softmax(torch.nan_to_num(logits, nan=0.0), dim=1)

        def _branch_logits(branch_model, x):
            x = x.transpose(1, 2).transpose(1, 3).contiguous()
            feat = branch_model.forward_features(x)[:, 0]
            if (
                hasattr(branch_model, "head")
                and isinstance(branch_model.head, nn.Module)
                and not isinstance(branch_model.head, nn.Identity)
            ):
                return branch_model.head(feat)
            if (
                hasattr(branch_model, "fc")
                and isinstance(branch_model.fc, nn.Module)
                and not isinstance(branch_model.fc, nn.Identity)
            ):
                return branch_model.fc(feat)
            return torch.zeros((feat.shape[0], 6), device=feat.device, dtype=feat.dtype)

        logits_spec = _branch_logits(model.spec_model, spec_imgs)
        logits_eeg = _branch_logits(model.eeg_model, eeg_imgs)
        logits_raw50 = _branch_logits(model.raw_50s_model, raw_50s_imgs)
        logits_raw10 = _branch_logits(model.raw_10s_model, raw_10s_imgs)

        logits = (logits_spec + logits_eeg + logits_raw50 + logits_raw10) / 4.0
        return torch.softmax(torch.nan_to_num(logits, nan=0.0), dim=1)

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
                probs_l = _model_probs(
                    model, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                probs_c = _model_probs(
                    model, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                probs_r = _model_probs(
                    model, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0

            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs.detach().cpu().numpy()

            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                result_7[eeg_id] = ensemble_probs[j]
else:
    print(
        "Warning: no compatible .pth/.pt/.ckpt weights found; result_7 will be filled by fallback later."
    )

for m in vit_models:
    del m
torch.cuda.empty_cache()
gc.collect()

if DEBUG is False:
    try:
        ss_ids = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )["eeg_id"].astype(str)
        covered = sum([1 for x in ss_ids.values if str(x) in result_7])
        print(f"result_7 coverage: {covered}/{len(ss_ids)}")
    except Exception as e:
        print("Coverage check warning:", e)



## === cell 15
result_8, result_2, result_5, result_4, result_3 = {}, {}, {}, {}, {}



## === cell 16
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
all_ids = sample_sub["eeg_id"].astype(str).values

train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)

train_votes = train_df[CLASSES].astype(np.float64).values
train_probs = train_votes / np.clip(train_votes.sum(axis=1, keepdims=True), 1.0, None)
prior = train_probs.mean(axis=0)
prior = prior / prior.sum()

train_df_pid = train_df[["patient_id"] + CLASSES].copy()
train_df_pid["patient_id"] = train_df_pid["patient_id"].astype(str)
votes_pid = train_df_pid[CLASSES].astype(np.float64).values
probs_pid = votes_pid / np.clip(votes_pid.sum(axis=1, keepdims=True), 1.0, None)
train_df_pid_probs = pd.DataFrame(probs_pid, columns=CLASSES)
train_df_pid_probs.insert(0, "patient_id", train_df_pid["patient_id"].values)
patient_prior_map = (
    train_df_pid_probs.groupby("patient_id")[CLASSES].mean().to_dict(orient="index")
)

test_pid_map = (
    test.set_index(test["eeg_id"].astype(str))["patient_id"].astype(str).to_dict()
)

train_df_ec = train_df[["patient_id", "expert_consensus"] + CLASSES].copy()
train_df_ec["patient_id"] = train_df_ec["patient_id"].astype(str)
train_df_ec["expert_consensus"] = train_df_ec["expert_consensus"].astype(str)

votes_ec = train_df_ec[CLASSES].astype(np.float64).values
probs_ec = votes_ec / np.clip(votes_ec.sum(axis=1, keepdims=True), 1.0, None)
train_df_ec_probs = pd.DataFrame(probs_ec, columns=CLASSES)
train_df_ec_probs.insert(0, "patient_id", train_df_ec["patient_id"].values)
train_df_ec_probs.insert(1, "expert_consensus", train_df_ec["expert_consensus"].values)

patient_consensus_prior_map = (
    train_df_ec_probs.groupby(["patient_id", "expert_consensus"])[CLASSES]
    .mean()
    .to_dict(orient="index")
)
consensus_prior_map = (
    train_df_ec_probs.groupby(["expert_consensus"])[CLASSES]
    .mean()
    .to_dict(orient="index")
)

patient_consensus_mode = (
    train_df_ec.groupby("patient_id")["expert_consensus"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)

patient_train_counts = train_df_pid["patient_id"].value_counts().to_dict()


def _get_pred(eeg_id):
    eeg_id = str(eeg_id)
    preds = []
    for d in (result_8, result_7, result_3, result_4, result_5, result_2):
        if eeg_id in d:
            preds.append(np.asarray(d[eeg_id], dtype=np.float64))

    if len(preds) == 0:
        pid = test_pid_map.get(eeg_id, None)

        if pid is not None:
            pid = str(pid)
            pid_cnt = int(patient_train_counts.get(pid, 0))
            ec = patient_consensus_mode.get(pid, None)

            if (
                ec is not None
                and (pid, str(ec)) in patient_consensus_prior_map
                and pid_cnt >= 3
            ):
                p_pc = np.asarray(
                    [patient_consensus_prior_map[(pid, str(ec))][c] for c in CLASSES],
                    dtype=np.float64,
                )
                p_pc = np.clip(p_pc, 1e-12, 1.0)
                p_pc = p_pc / p_pc.sum()

                beta = 0.10 if pid_cnt >= 10 else 0.20
                p = (1.0 - beta) * p_pc + beta * prior
                return p

            if pid in patient_prior_map and pid_cnt >= 1:
                p_pid = np.asarray(
                    [patient_prior_map[pid][c] for c in CLASSES], dtype=np.float64
                )
                p_pid = np.clip(p_pid, 1e-12, 1.0)
                p_pid = p_pid / p_pid.sum()

                beta = 0.20 if pid_cnt >= 5 else 0.30
                p = (1.0 - beta) * p_pid + beta * prior
                return p

            if ec is not None and str(ec) in consensus_prior_map:
                p_c = np.asarray(
                    [consensus_prior_map[str(ec)][c] for c in CLASSES], dtype=np.float64
                )
                p_c = np.clip(p_c, 1e-12, 1.0)
                p_c = p_c / p_c.sum()

                beta = 0.35
                p = (1.0 - beta) * p_c + beta * prior
                return p

        return prior.copy()

    p = np.mean(preds, axis=0)

    alpha = 0.05
    p = (1.0 - alpha) * p + alpha * prior

    u = np.full((N_CLASSES,), 1.0 / N_CLASSES, dtype=np.float64)
    gamma = 0.005
    p = (1.0 - gamma) * p + gamma * u

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    return p


pred_mat = np.vstack([_get_pred(eeg_id) for eeg_id in all_ids]).astype(np.float64)

eps = 1e-6
pred_mat = np.clip(pred_mat, eps, 1.0)
pred_mat = pred_mat / pred_mat.sum(axis=1, keepdims=True)

submission = pd.DataFrame(pred_mat, columns=CLASSES)
submission.insert(0, "eeg_id", all_ids)
submission.to_csv("submission.csv", index=False)

print(submission.shape)
print(submission.head())



## === cell 17
if DEBUG is False:
    for p in [
        "/kaggle/working/spec_spectrograms/",
        "/kaggle/working/eeg_spectrograms/",
        "/kaggle/working/eeg_50s_raws/",
        "/kaggle/working/eeg_10s_raws/",
        "/kaggle/working/squeezeformer",
    ]:
        if os.path.exists(p):
            try:
                import shutil

                shutil.rmtree(p)
            except Exception as e:
                print("Cleanup warning:", p, e)
