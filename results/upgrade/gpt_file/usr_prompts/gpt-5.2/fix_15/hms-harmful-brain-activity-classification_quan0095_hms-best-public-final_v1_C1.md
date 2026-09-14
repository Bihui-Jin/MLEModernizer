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

0.2843982276553076

# 6. Current score

0.91485

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The crashes come from forcing `cuda:0` in an environment without an NVIDIA driver, plus missing/optional weight files and later referencing result dictionaries that were never created due to earlier failures. I make device selection automatic (CPU fallback) and load model weights safely with `map_location`, so inference runs end-to-end even without GPU or missing external datasets. To keep score reasonable (and valid), I ensemble whatever model groups successfully load; if none load, we fall back to a uniform probability distribution that passes submission checks. Finally, I ensure the submission uses the exact `sample_submission.csv` ordering and that each row sums to 1 (with numerical stabilization).'
- What this solution (achieved 1.43453) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.2844), and the main reason is that none of the referenced weight files exist in your environment, so the pipeline effectively falls back to near-uniform predictions. I make a minimal, metric-aligned improvement by adding a lightweight patient-aware prior computed from `train.csv` vote distributions (no model/training changes), and then blend it with any model predictions if weights do load (otherwise it replaces the uniform fallback). This keeps the core inference logic intact while producing much more realistic class probabilities, which should move KL-divergence substantially toward the target. I also add tiny numerical stabilization and ensure strict row-sum-to-1, without altering required paths or output format.'
- What this solution (achieved 0.79322) has done: 'Your score is far worse than the target (lower-is-better), and the main driver is that none of the external weight files exist, so you’re effectively submitting a (patient/global) prior rather than a learned model. To move the KL score toward the target with minimal change and without altering your model/feature core, I (1) make the prior stronger and more realistic by using a smoothed patient prior with a global fallback that accounts for how many rows each patient has (shrinkage), and (2) use a small “temperature” sharpening on the prior (and on model outputs if they exist) to reduce over-entropy predictions that usually worsen KL here. These changes keep your inference pipeline intact, preserve submission format/order, and only adjust the probability post-processing/blending that directly impacts the KL metric. All outputs are still strictly normalized to sum to 1 per row.'
- What this solution (achieved 0.99233) has done: 'Your current score (0.79322, lower-is-better) is still far above the target (0.2844), and this is most likely because no model weights are available so you’re submitting mostly a patient/global prior. To move the KL divergence closer to the target without changing the model/feature core, I make the prior itself more informative by conditioning not just on `patient_id` but also on the available `spectrogram_id` in test (a strong identifier in this dataset) with shrinkage to global/patient when counts are low. I also fix the prior smoothing math (it currently adds a near-unscaled global vector in a confusing way) and tune the blend so that when no model outputs exist the predictions are sharper and better matched to empirical label distributions (lower KL). All changes are confined to the prior computation and final probability blending/normalization and keep the submission format identical.'
- What this solution (achieved 1.13752) has done: 'Your current KL (0.99233, lower-is-better) is far worse than the target (0.2844), and the biggest lever available without changing the model/training core is fixing the prior so it matches the test label distribution better. I (1) correct the prior math to use proper Dirichlet/Laplace smoothing (add pseudo-counts in vote space, not probability space), (2) compute priors at the same granularity as evaluation (`eeg_id`) by aggregating train votes per `eeg_id` and using those for `spectrogram_id`/`patient_id` mappings, and (3) slightly adjust blend weights/temperature so the prior is less overconfident when identifiers are sparse, while keeping normalization and submission ordering intact. These are minimal, metric-aligned changes confined to the fallback/blending logic and should move KL materially toward the target when weights are missing. The script still runs end-to-end and writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.16833) has done: 'Your current KL (1.1375, lower-is-better) is still far above the target (0.2844), and the most likely reason is that no model weights load so your submission is dominated by a weak prior that doesn’t match the test distribution well. With minimal changes and without touching model/feature/training logic, I make the prior substantially more informative by adding an `eeg_id`-level prior (for cases where the same `eeg_id` appears in train and test) and by using count-adaptive shrinkage (less shrinkage when we have many votes for a patient/spec, more when we have few). I also tune the blend to rely more on the best-available identifier prior (eeg/spec/patient) when models are missing, while keeping strict normalization and exact sample_submission ordering. These changes directly target KL by improving probability calibration without changing the architecture or inference semantics.'
- What this solution (achieved 0.83745) has done: 'Your current KL (1.168, lower-is-better) suggests the pipeline is still effectively “prior-only” and that the prior is poorly matched to the hidden label distribution. With minimal changes and without touching the model/feature extraction core, I (1) fix a key prior bug by computing priors in *vote-count space* with proper Dirichlet smoothing (your current `eeg_prior` mistakenly mixes counts with probabilities), and (2) simplify the prior selection to a single “best available” identifier prior (eeg_id > spectrogram_id > patient_id > global) instead of mixing multiple priors that can wash out signal. Finally, I keep the same normalization/temperature steps but adjust them slightly to avoid overconfident priors that typically worsen KL when you’re missing weights. This should move the score materially toward the target while preserving the overall inference semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 0.99951) has done: 'Your current KL (0.83745, lower-is-better) is still far above the target (0.2844), so we should cautiously improve predictions without changing the model/feature/training core. The biggest easy win here is that your `alpha_model=0.80` heavily trusts model outputs when weights do load, but these models are likely out-of-domain vs the current environment and can hurt calibration; we reduce that weight so the (more stable) identifier prior dominates. Next, we make the fallback prior more informative by using a *mixture* of available identifier priors (eeg/spec/patient) rather than a single “best”, with reliability weights based on available vote mass so it doesn’t wash out signal. Finally, we add a small submission-time smoothing with the global prior (Dirichlet-style) to reduce overconfident spikes that typically increase KL.'
- What this solution (achieved 0.9058) has done: 'Your current KL (0.99951, lower-is-better) is far above the target (0.2844), and given the missing external weight files the only lever that can legitimately improve score without changing the model core is the prior/blending logic. I make the identifier prior less “spiky” and more distribution-matched by (1) computing priors in true vote-count space with an evidence-based Dirichlet posterior (using each identifier’s total vote mass as its confidence), and (2) switching the prior mixer to a single normalized reliability-weighted combination (instead of fixed weights that can over-trust weak identifiers). Finally, I slightly increase global smoothing and soften temperature to reduce overconfident probabilities, which typically reduces KL when you’re essentially prior-only. These are minimal, submission-safe changes confined to post-processing and should move KL down toward the target band.'
- What this solution (achieved 0.7955) has done: 'Your current KL (0.9058, lower-is-better) is far above the target (0.2844), and the most likely cause is that no external model weights are available, so predictions are dominated by a weak identifier prior. To move the score down toward the target with minimal semantic change, I (1) build the prior from the **expert consensus** label distribution in `train.csv` (a stronger signal than raw vote sums for this fallback scenario) and (2) use a reliability-weighted mixture over (eeg_id/spec_id/patient_id/global) in that same consensus space. I keep your model inference and blending intact; if weights load, they are still used, but the improved prior should help substantially when they don’t. I also keep strict normalization and row-sum-to-1 safety to avoid submission failures.'
- What this solution (achieved 0.77816) has done: 'Your current KL (0.7955, lower-is-better) is still far above the target (0.2844), and the main limitation is that the code is effectively “prior-only” because external weight files are missing; so the only legitimate lever is making that prior closer to the hidden test distribution. I keep your feature extraction/model code untouched, but improve the prior by switching from *expert_consensus one-hot counts* to the richer *annotator vote count* targets from `train.csv`, aggregated at `eeg_id` level (same granularity as submission). Then I keep your same identifier-mix logic (eeg/spec/patient/global) but tune the Dirichlet strengths/weights slightly to rely more on high-evidence identifiers and less on the overly-smoothed global component, which should reduce KL toward the target. Finally, I keep the same normalization and safety clipping so the submission remains valid (rows sum to 1) and the pipeline still runs end-to-end.'
- What this solution (achieved 0.92804) has done: 'Your current KL (0.778, lower-is-better) is still far above the target (0.284), and since the external weight files are missing your score is dominated by the identifier-based prior. To move KL down with minimal semantic change, I (1) compute the identifier priors from **per-row vote-probabilities** averaged at each identifier (patient/spec/eeg), instead of summing raw counts (which over-weights rows with more annotators and tends to miscalibrate KL), and (2) use a simple, evidence-weighted Dirichlet shrinkage around the global prior based on the number of contributing rows. I keep your model inference unchanged and only swap the prior computation/mixing so it remains a safe fallback when no weights load. Submission format, ordering, and strict row-normalization stay identical.'
- What this solution (achieved 0.8224) has done: 'Your current KL (0.928, lower-is-better) is far above the target (0.284), and because the external weight files likely don’t load, the submission is dominated by the identifier prior. To move the KL down with minimal semantic change, I revert the prior estimation to **vote-count space** (Dirichlet-smoothed posterior from aggregated votes), because the row-probability averaging tends to wash out the signal and often hurts KL. I keep your identifier mixing/blending structure intact, but change the evidence term from “#rows” to “total vote mass” per identifier (better reflects annotator confidence) and slightly reduce the global smoothing that can over-flatten probabilities. Model inference, architecture, and feature extraction remain unchanged; the patch only affects the fallback prior that directly drives score when weights are missing.'
- What this solution (achieved 0.91485) has done: 'Your current score (0.8224, lower-is-better) is still far above the target (0.2844), and given the missing external weights the submission is dominated by the identifier prior. With minimal changes and no model/feature/training changes, I make the prior more metric-aligned by (1) mapping each test `patient_id` to its **train-time mean vote distribution** (a strong, stable baseline), then (2) shrinking that patient prior toward the global prior based on how much evidence (total vote mass) that patient has in train, and finally (3) slightly increasing the blend weight given to this improved patient prior vs the weaker `spectrogram_id`/`eeg_id` matches (which rarely exist across train/test). This keeps your pipeline identical except for the fallback probability construction, and preserves strict row-normalization so the submission remains valid.'

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



## === cell 3
try:
    import librosa  # noqa: F401
except Exception:
    librosa = None

from scipy import signal  # used by multiple feature extractors



## === cell 4
SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")




## === cell 5
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
            eeg_1 = eeg[COLS[kk]].to_numpy(dtype=np.float32, copy=True)
            if np.isnan(eeg_1).any():
                m = np.nanmean(eeg_1)
                eeg_1 = np.nan_to_num(eeg_1, nan=m)

            eeg_2 = eeg[COLS[kk + 1]].to_numpy(dtype=np.float32, copy=True)
            if np.isnan(eeg_2).any():
                m = np.nanmean(eeg_2)
                eeg_2 = np.nan_to_num(eeg_2, nan=m)

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




## === cell 6
def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)

    def _chunk(time_start, time_stop):
        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )
        list_eeg = []
        for region in RAW_FEATS.keys():
            eeg = np.zeros(
                (len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32
            )
            for chan_i, chan in enumerate(RAW_FEATS[region]):
                c1, c2 = chan.split("-")
                eeg_1 = eeg_default.loc[:, c1].to_numpy(dtype=np.float32, copy=True)
                if np.isnan(eeg_1).any():
                    m = np.nanmean(eeg_1)
                    eeg_1 = np.nan_to_num(eeg_1, nan=m)

                eeg_2 = eeg_default.loc[:, c2].to_numpy(dtype=np.float32, copy=True)
                if np.isnan(eeg_2).any():
                    m = np.nanmean(eeg_2)
                    eeg_2 = np.nan_to_num(eeg_2, nan=m)

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

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_c = _chunk(time_start, time_stop)

    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)
    eeg_l = _chunk(time_start, time_stop)

    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)
    eeg_r = _chunk(time_start, time_stop)

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
            c1, c2 = chan.split("-")
            eeg_1 = eeg_default.loc[:, c1].to_numpy(dtype=np.float32, copy=True)
            if np.isnan(eeg_1).any():
                m = np.nanmean(eeg_1)
                eeg_1 = np.nan_to_num(eeg_1, nan=m)

            eeg_2 = eeg_default.loc[:, c2].to_numpy(dtype=np.float32, copy=True)
            if np.isnan(eeg_2).any():
                m = np.nanmean(eeg_2)
                eeg_2 = np.nan_to_num(eeg_2, nan=m)

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




## === cell 7
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

print("test:", test.shape)

spec_directory_path = "spec_spectrograms/"
eeg_directory_path = "eeg_spectrograms/"
raw_10s_directory_path = "eeg_10s_raws/"
raw_50s_directory_path = "eeg_50s_raws/"

for p in [
    spec_directory_path,
    eeg_directory_path,
    raw_10s_directory_path,
    raw_50s_directory_path,
]:
    os.makedirs(p, exist_ok=True)



## === cell 8
from joblib import Parallel, delayed


def save_full(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time)
    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img50 = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img50)

    imgstft = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", imgstft)


_ = Parallel(n_jobs=4)(delayed(save_full)(row) for _, row in test.iterrows())




## === cell 9
class Config:
    seed = 2024
    num_folds = 5


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 10
import timm



## === cell 11
import torch.utils.data as data
from torch.utils.data import DataLoader
from skimage.transform import resize




## === cell 12
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
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

        spec_img = np.load(os.path.join(self.spec_data_path, eeg_id + ".npy")).astype(
            "float32"
        )
        eeg_img = np.load(os.path.join(self.eeg_data_path, eeg_id + ".npy")).astype(
            "float32"
        )
        raw_50s_img = np.load(
            os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        ).astype("float32")
        raw_10s_l_img = np.load(
            os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy")
        ).astype("float32")
        raw_10s_c_img = np.load(
            os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")
        ).astype("float32")
        raw_10s_r_img = np.load(
            os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy")
        ).astype("float32")

        eeg_img = resize(
            eeg_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        spec_img = resize(
            spec_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_10s_l_img = resize(
            raw_10s_l_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_10s_c_img = resize(
            raw_10s_c_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_10s_r_img = resize(
            raw_10s_r_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_50s_img = resize(
            raw_50s_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")

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

        img_mean = eeg_img.mean(axis=(0, 1))
        img_std = eeg_img.std(axis=(0, 1))
        eeg_img = (eeg_img - img_mean) / (img_std + eps)

        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 13
class NetSmall(nn.Module):
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




## === cell 14
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




## === cell 15
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


def safe_load_state_dict(model, weight_path, device):
    if not os.path.exists(weight_path):
        print(f"[WARN] missing weights: {weight_path}")
        return False
    try:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=True)
        return True
    except Exception as e:
        print(f"[WARN] failed to load {weight_path}: {type(e).__name__}: {e}")
        return False


def predict_with_models(models, loader, device):
    out = {}
    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in loader:
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for m in models:
                logits_l = m(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
                logits_c = m(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs)
                logits_r = m(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs)
                probs_l = logits_l.softmax(dim=1)
                probs_c = logits_c.softmax(dim=1)
                probs_r = logits_r.softmax(dim=1)
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0

            ensemble_probs /= max(len(models), 1)
            ensemble_probs = ensemble_probs.detach().cpu().numpy()

            for j in range(len(eeg_ids)):
                eid = str(eeg_ids[j])
                out[eid] = out.get(eid, np.zeros(6, dtype=np.float32)) + ensemble_probs[
                    j
                ].astype(np.float32)
    return out




## === cell 16
test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data,
    batch_size=(8 if DEVICE == "cpu" else 32),
    pin_memory=False,
    num_workers=0 if DEVICE == "cpu" else 4,
    drop_last=False,
)



## === cell 17
all_result_dicts = []

weights_a = [
    "/kaggle/input/hms-stage2/fold_0_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_spec_raw_50_10_bestlb.pth",
]
models = []
for wp in weights_a:
    m = NetSmall("vit_small_patch14_reg4_dinov2.lvd142m", DEVICE).to(DEVICE)
    ok = safe_load_state_dict(m, wp, DEVICE)
    if ok:
        m.eval()
        models.append(m)
    else:
        del m
if len(models) > 0:
    all_result_dicts.append(predict_with_models(models, test_loader, DEVICE))
for m in models:
    del m
gc.collect()
if DEVICE != "cpu":
    torch.cuda.empty_cache()

weights_b = [
    "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
]
models = []
for wp in weights_b:
    m = NetSmall("vit_small_patch14_reg4_dinov2.lvd142m", DEVICE).to(DEVICE)
    ok = safe_load_state_dict(m, wp, DEVICE)
    if ok:
        m.eval()
        models.append(m)
    else:
        del m
if len(models) > 0:
    all_result_dicts.append(predict_with_models(models, test_loader, DEVICE))
for m in models:
    del m
gc.collect()
if DEVICE != "cpu":
    torch.cuda.empty_cache()

weights_c = [
    "/kaggle/input/hms-stage2-2/fold_0_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_1_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_2_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_3_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_4_raw_20_10_bestlb.pth",
]
models = []
for wp in weights_c:
    m = NetSmall("vit_small_patch14_reg4_dinov2.lvd142m", DEVICE).to(DEVICE)
    ok = safe_load_state_dict(m, wp, DEVICE)
    if ok:
        m.eval()
        models.append(m)
    else:
        del m
if len(models) > 0:
    all_result_dicts.append(predict_with_models(models, test_loader, DEVICE))
for m in models:
    del m
gc.collect()
if DEVICE != "cpu":
    torch.cuda.empty_cache()

weights_d = [
    "/kaggle/input/hms-bestlb-vitbase/fold_0_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_1_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_2_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_3_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_4_raw_50_10_bestlb_vitbase.pth",
]
models = []
for wp in weights_d:
    m = NetBase("vit_base_patch14_reg4_dinov2.lvd142m", DEVICE).to(DEVICE)
    ok = safe_load_state_dict(m, wp, DEVICE)
    if ok:
        m.eval()
        models.append(m)
    else:
        del m
if len(models) > 0:
    all_result_dicts.append(predict_with_models(models, test_loader, DEVICE))
for m in models:
    del m
gc.collect()
if DEVICE != "cpu":
    torch.cuda.empty_cache()

print("Loaded model groups:", len(all_result_dicts))



## === cell 18
train_meta = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)

eps = 1e-12

train_votes = train_meta[["eeg_id", "patient_id", "spectrogram_id"] + CLASSES].copy()
train_votes[CLASSES] = train_votes[CLASSES].astype(np.float64)

eeg_agg = train_votes.groupby("eeg_id", as_index=False).agg(
    {
        **{c: "sum" for c in CLASSES},
        "patient_id": "first",
        "spectrogram_id": "first",
    }
)

global_counts = train_votes[CLASSES].sum(axis=0).to_numpy(dtype=np.float64)
global_counts = np.clip(global_counts, eps, None)
global_prior = (global_counts / global_counts.sum()).astype(np.float32)

eeg_counts_df = eeg_agg.set_index("eeg_id")[CLASSES].astype(np.float64)
eeg_mass_df = eeg_counts_df.sum(axis=1).astype(np.float64)

spec_counts_df = eeg_agg.groupby("spectrogram_id")[CLASSES].sum().astype(np.float64)
spec_mass_df = spec_counts_df.sum(axis=1).astype(np.float64)

eeg_probs = eeg_agg[CLASSES].to_numpy(dtype=np.float64) + eps
eeg_probs = eeg_probs / np.clip(eeg_probs.sum(axis=1, keepdims=True), eps, None)
eeg_probs_df = pd.DataFrame(eeg_probs, columns=CLASSES)
eeg_probs_df["patient_id"] = eeg_agg["patient_id"].values
patient_prob_mean_df = (
    eeg_probs_df.groupby("patient_id")[CLASSES].mean().astype(np.float64)
)

patient_counts_df = eeg_agg.groupby("patient_id")[CLASSES].sum().astype(np.float64)
patient_mass_df = patient_counts_df.sum(axis=1).astype(np.float64)

print(
    "Computed priors: global:",
    global_prior,
    "patients:",
    len(patient_prob_mean_df),
    "specs:",
    len(spec_counts_df),
    "eegs:",
    len(eeg_counts_df),
)




## === cell 19
def apply_temperature(p, t=0.99, eps=1e-12):
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, eps, None)
    p = p / p.sum()
    p = p ** (1.0 / t)
    p = np.clip(p, eps, None)
    return (p / p.sum()).astype(np.float32)


def reliability_weight(mass, scale):
    mass = float(mass)
    return float(1.0 - np.exp(-mass / float(scale)))


def posterior_from_counts(counts, global_prior, strength, eps=1e-12):
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.clip(counts, 0.0, None)
    alpha0 = float(strength) * global_prior.astype(np.float64)
    post = counts + alpha0
    post = np.clip(post, eps, None)
    p = post / np.clip(post.sum(), eps, None)
    p = np.clip(p, eps, None)
    return (p / p.sum()).astype(np.float32)


def patient_prior_from_mean_probs(
    pid, global_prior, patient_prob_mean_df, patient_mass_df, eps=1e-12
):
    if pid is None:
        return global_prior.astype(np.float32), 0.0
    pid_int = int(pid)
    if pid_int not in patient_prob_mean_df.index:
        return global_prior.astype(np.float32), 0.0

    p_mean = patient_prob_mean_df.loc[pid_int].to_numpy(dtype=np.float32)
    p_mean = np.clip(p_mean, eps, None)
    p_mean = p_mean / p_mean.sum()

    mass = (
        float(patient_mass_df.loc[pid_int]) if pid_int in patient_mass_df.index else 0.0
    )
    lam = reliability_weight(
        mass, scale=400.0
    )  # slightly slower saturation than before
    p = lam * p_mean + (1.0 - lam) * global_prior.astype(np.float32)
    p = np.clip(p, eps, None)
    p = p / p.sum()
    return p.astype(np.float32), mass


def mix_identifier_priors(
    eid_int,
    sid,
    pid,
    *,
    global_prior,
    eeg_counts_df,
    eeg_mass_df,
    spec_counts_df,
    spec_mass_df,
    patient_prob_mean_df,
    patient_mass_df,
    eps=1e-12,
):
    priors = [global_prior.astype(np.float32)]
    weights = [0.05]

    if eid_int in eeg_counts_df.index:
        counts = eeg_counts_df.loc[eid_int].values
        mass = float(eeg_mass_df.loc[eid_int]) if eid_int in eeg_mass_df.index else 1.0
        p = posterior_from_counts(counts, global_prior, strength=6.0, eps=eps)
        w = 0.35 * reliability_weight(mass, scale=30.0)
        priors.append(p)
        weights.append(w)

    if sid is not None:
        sid_int = int(sid)
        if sid_int in spec_counts_df.index:
            counts = spec_counts_df.loc[sid_int].values
            mass = (
                float(spec_mass_df.loc[sid_int])
                if sid_int in spec_mass_df.index
                else 1.0
            )
            p = posterior_from_counts(counts, global_prior, strength=8.0, eps=eps)
            w = 0.25 * reliability_weight(mass, scale=60.0)
            priors.append(p)
            weights.append(w)

    p_pat, mass_pat = patient_prior_from_mean_probs(
        pid, global_prior, patient_prob_mean_df, patient_mass_df, eps=eps
    )
    if mass_pat > 0.0:
        w = 0.85 * reliability_weight(mass_pat, scale=400.0)
        priors.append(p_pat)
        weights.append(w)

    w = np.asarray(weights, dtype=np.float32)
    w = w / max(float(w.sum()), 1e-6)

    P = np.zeros((N_CLASSES,), dtype=np.float32)
    for wi, pi in zip(w, priors):
        P += wi * pi.astype(np.float32)

    P = np.clip(P, eps, None)
    P = P / P.sum()
    return P.astype(np.float32)


sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sub = sample_sub.copy()

alpha_model = 0.15

prior_temp = 0.992
model_temp = 0.998

final_global_smooth = 0.015

test_pid_map = test.set_index("eeg_id")["patient_id"].to_dict()
test_spec_map = test.set_index("eeg_id")["spectrogram_id"].to_dict()

for i, eid in enumerate(sub["eeg_id"].values):
    eid_int = int(eid)
    eid_str = str(eid)
    pid = test_pid_map.get(eid, None)
    sid = test_spec_map.get(eid, None)

    prior = mix_identifier_priors(
        eid_int,
        sid,
        pid,
        global_prior=global_prior,
        eeg_counts_df=eeg_counts_df,
        eeg_mass_df=eeg_mass_df,
        spec_counts_df=spec_counts_df,
        spec_mass_df=spec_mass_df,
        patient_prob_mean_df=patient_prob_mean_df,
        patient_mass_df=patient_mass_df,
        eps=eps,
    )
    prior = apply_temperature(prior, t=prior_temp, eps=eps)

    if len(all_result_dicts) == 0:
        p = prior.copy()
    else:
        ps = []
        for d in all_result_dicts:
            if eid_str in d:
                ps.append(d[eid_str])
        if len(ps) == 0:
            p = prior.copy()
        else:
            model_p = np.mean(np.stack(ps, axis=0), axis=0).astype(np.float32)
            model_p = np.clip(model_p, eps, None)
            model_p = model_p / model_p.sum()
            model_p = apply_temperature(model_p, t=model_temp, eps=eps)
            p = (alpha_model * model_p + (1.0 - alpha_model) * prior).astype(np.float32)

    p = np.clip(p, eps, None)
    p = p / p.sum()
    p = (1.0 - final_global_smooth) * p + final_global_smooth * global_prior
    p = np.clip(p, eps, None)
    p = p / p.sum()
    sub.loc[i, CLASSES] = p

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved:", os.path.abspath("submission.csv"), "shape:", sub.shape)



## === cell 20
if not DEBUG:
    for p in [
        "/kaggle/working/spec_spectrograms/",
        "/kaggle/working/eeg_spectrograms/",
        "/kaggle/working/eeg_50s_raws/",
        "/kaggle/working/eeg_10s_raws/",
        "/kaggle/working/squeezeformer",
    ]:
        try:
            if os.path.exists(p):
                for fn in os.listdir(p):
                    fp = os.path.join(p, fn)
                    try:
                        if os.path.isfile(fp) or os.path.islink(fp):
                            os.remove(fp)
                    except Exception:
                        pass
        except Exception:
            pass
    gc.collect()
    if DEVICE != "cpu":
        torch.cuda.empty_cache()
