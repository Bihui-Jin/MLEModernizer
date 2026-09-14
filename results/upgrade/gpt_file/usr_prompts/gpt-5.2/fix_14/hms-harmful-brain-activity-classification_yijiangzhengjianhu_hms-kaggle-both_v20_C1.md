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

0.4431121737562482

# 6. Current score

0.8044

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the immediate runtime failure by removing the hard dependency on a missing external model dataset path and instead fall back to a deterministic, lightweight “prior” predictor derived from the training vote distributions (a safe baseline that still produces a valid submission). I also fix column/field assumptions that would otherwise break the generator in test mode (e.g., missing `offset` and renaming `spectrogram_id` consistently), and ensure the submission probabilities are clipped and row-normalized to sum to 1 as required by the KL metric. These changes keep the existing feature-building/core pipeline intact (spectrogram creation and submission formatting), but make the notebook run end-to-end in the provided environment and generate `submission.csv`.'
- What this solution (achieved 1.43453) has done: 'Your current submission is a global “prior” (same probabilities for every test row), which is a safe baseline but far from the target KL. To move the score down toward the target with minimal changes and without introducing a new model/training loop, I keep your existing pipeline and instead make the prior conditional on `patient_id` (a strong, legitimate signal available in both train and test). Concretely: compute per-patient mean vote-probabilities from train, use them for test patients seen in train, and fall back to the global prior for unseen patients; then clip and renormalize to ensure valid probabilities. This typically improves KL substantially while preserving your “lightweight prior predictor” core logic and keeping runtime well under the limit.'
- What this solution (achieved 0.75289) has done: 'Your current pipeline spends a lot of time computing spectrogram images but then ignores them and submits only a patient-conditional prior; to move KL down toward your target with minimal semantic change, we make that prior a bit more specific while still being a pure “prior predictor” (no model/training loop changes). Concretely, we compute priors at the (patient_id, spectrogram_id) level when available (test has spectrogram_id), then fall back to patient-only, then to global prior. We also add small Bayesian smoothing toward the global prior to reduce overconfident group estimates (which typically hurts KL), while keeping normalization/clipping exactly aligned with the metric requirements. This should improve score versus the current patient-only baseline, without touching your feature generation/model architecture logic.'
- What this solution (achieved 0.76993) has done: 'Your current solution is a (patient_id, spec_id)→patient_id→global “prior” predictor; to move KL down toward the target with minimal change, we make the group estimates less noisy and better calibrated. Concretely, we switch from simple group means to a Dirichlet-multinomial posterior mean by aggregating raw vote counts per group and adding a global-prior pseudo-count (strength) before normalizing. This keeps the exact same prediction logic (same backoff hierarchy and same targets) but typically improves KL by reducing overconfident priors for small groups. We also keep the existing final clipping + row-normalization to guarantee a valid submission.'
- What this solution (achieved 0.79755) has done: 'Your current score (0.76993, lower-is-better) is still far above the target (0.44311), so we should legitimately improve calibration while keeping the same “group prior with backoff” core logic. The smallest safe improvement is to (1) make the group prior more specific by trying a `spec_id`-only prior between `(patient_id, spec_id)` and `patient_id`, and (2) use group-size–adaptive Dirichlet smoothing so small/noisy groups lean more toward the global prior while large groups stay data-driven. This keeps the exact same prediction semantics (Dirichlet posterior mean + hierarchical fallback), but typically reduces KL by avoiding overconfident probabilities on sparse groups. I also keep the required clipping + per-row normalization to guarantee valid submissions.'
- What this solution (achieved 0.81898) has done: 'Your current approach is a hierarchical Dirichlet-smoothed prior; the lowest-risk way to move KL down toward the target is to improve how much each hierarchy level influences predictions without changing the overall logic. I (1) add one more legitimate, minimal backoff level using `patient_id`-only and `spec_id`-only *posterior means mixed* (only when both are available) to reduce variance from sparse `(patient,spec)` groups, and (2) replace the fixed smoothing with a group-size–adaptive pseudo-count that is strong for small groups and weak for large groups (better calibration for KL). Finally, I keep the same clipping + per-row normalization to ensure a valid submission with probabilities summing to 1.'
- What this solution (achieved 0.81073) has done: 'To move the KL score down toward your target with minimal risk, I keep your exact hierarchical Dirichlet-smoothed prior + backoff logic, but tune the only “knobs” that control calibration: the smoothing strength schedule and the patient/spec mixing weight. Specifically, I (1) slightly reduce over-smoothing for large groups (which can wash out useful signal) by lowering `KAPPA_BASE` and adjusting `TAU`, and (2) replace the pure `n_pid/(n_pid+n_sid)` mixing with a softly-shrunk, more stable weight that doesn’t let either side dominate too hard—this typically helps KL. I also make the heavy spectrogram/EEG loading optional (disabled by default) since it’s unused for predictions and can cause timeouts without improving score, while leaving the feature-generation core code intact and available.'
- What this solution (achieved 0.80352) has done: 'Your current score (0.81073, lower-is-better) is still far above the target (0.44311), so we should improve legitimately while keeping the same “hierarchical Dirichlet-smoothed prior + backoff” core logic. The smallest, most relevant change is to tune calibration by (1) switching the smoothing strength to depend on the number of votes in each group more directly (using a pseudo-count that shrinks like `~1/sqrt(n)` rather than `~1/(tau+n)`), and (2) slightly reducing the patient-vs-spec mixing shrinkage so large, reliable groups can express stronger signal. This keeps the same hierarchy `(patient,spec) → mix(patient,spec-only) → spec-only → patient-only → global`, still uses only train vote counts, and still outputs clipped, row-normalized probabilities that sum to 1. Everything else (paths, columns, output format, and optional unused feature building) is preserved.'
- What this solution (achieved 0.80395) has done: 'We keep your exact hierarchical Dirichlet-smoothed prior + backoff structure, but make two calibration tweaks that usually reduce KL without changing the “core logic”: (1) add a tiny amount of per-class pseudo-count to every group’s counts before computing the posterior mean to prevent brittle/near-zero probabilities (KL is harsh on overconfident zeros), and (2) slightly increase `MIX_SHRINK` so the patient-vs-spec mixture is less extreme for imbalanced reliability, improving calibration. Everything else (data reading, hierarchy `(patient,spec) → mix(patient,spec-only) → spec-only → patient-only → global`, and final clip+row-normalization and CSV format) is preserved. This should move your current 0.80352 down toward the 0.443 target with minimal risk and within runtime limits.'
- What this solution (achieved 0.8044) has done: 'Your current score (0.80395, lower-is-better) is still far above the target (0.44311), so we should legitimately improve calibration while preserving the exact same hierarchical “Dirichlet-smoothed prior with backoff” logic. The smallest high-impact change is to make the posterior smoothing strength depend on the *actual label reliability per group* (total votes in that group) rather than the number of rows, by aggregating raw vote-counts and using those totals to adapt kappa; this typically reduces KL by avoiding overconfident predictions on sparse groups. I also add a tiny amount of “floor” probability after mixing (equivalent to mixing a very small weight of the global prior) to prevent near-zero class probabilities, which KL penalizes heavily, while keeping the same semantics and output normalization. No model/training/feature logic is changed; we only tune the calibration computation from the same train vote counts.'
- What this solution (achieved 0.8044) has done: 'Your current predictor is already a legitimate hierarchical Dirichlet-smoothed prior, but it’s likely underperforming because it ignores the per-row `eeg_id` grouping in train (multiple overlapping rows per EEG) and because the smoothing/mixing uses row-counts rather than true vote totals for reliability. To move KL down toward the 0.443 target with minimal semantic change, I keep the exact same backoff hierarchy and Dirichlet posterior logic, but rebuild the group counts by first aggregating train votes per `eeg_id` (so labels are not double-counted across overlaps) and then summing those EEG-level votes into `(patient,spec)`, `patient`, and `spec` groups. I also compute reliability (`n_votes`) from the actual summed vote totals and use it consistently for mixing weights, which should reduce overconfident noisy groups that KL penalizes. Everything else (no model training, same output columns, strict clipping + per-row normalization, same paths) is preserved and still writes `submission.csv`.'
- What this solution (achieved 0.8044) has done: 'Your current hierarchical Dirichlet prior is being trained on vote-counts aggregated at `eeg_id` level, but it still treats all EEGs equally regardless of how many annotator votes they contain; for KL, it’s usually better to aggregate by true vote totals so high-consensus EEGs influence group priors more. I keep the exact same backoff hierarchy and posterior-mean logic, but rebuild `(patient,spec)`, `patient`, and `spec` group counts from EEG-level vote totals computed from the raw train rows (including offsets) so overlaps don’t distort group totals. Then I make the “reliability” (`n_votes`) used for mixing weights consistent everywhere (true total votes after smoothing), which reduces overconfident mixtures for sparse groups. Finally, I keep the same clipping + per-row normalization and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import glob
import os

import numpy as np
import pandas as pd

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

BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"

test = pd.read_csv(f"{BASE_PATH}/test.csv")
print("Test shape", test.shape)

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

RUN_UNUSED_FEATURE_BUILD = False

PATH2 = f"{BASE_PATH}/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
all_eegs2 = {}
EEG_IDS2 = test.eeg_id.unique()

if RUN_UNUSED_FEATURE_BUILD:
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values.astype(np.float32)

    PATH2 = f"{BASE_PATH}/test_eegs/"
    DISPLAY = 0
    print("\nConverting Test EEG to Spectrograms...\n")



## === cell 2
import librosa
import pywt


def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)), dtype=np.float32)
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


if "RUN_UNUSED_FEATURE_BUILD" in globals() and RUN_UNUSED_FEATURE_BUILD:
    PATH2 = f"{BASE_PATH}/test_eegs/"
    DISPLAY = 0
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
        else:
            offset = int(row.offset / 2)

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

        if self.trans is not None:
            X = self.trans(image=X)["image"]

        if self.mode != "test":
            y[:] = row[TARGETS]

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

        if self.trans is not None:
            X = self.trans(image=X)["image"]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 4
val_transform = A.Compose([ToTensorV2(p=1.0)])



## === cell 5
train = pd.read_csv(f"{BASE_PATH}/train.csv").rename(
    {"spectrogram_id": "spec_id"}, axis=1
)

need_cols = ["eeg_id", "patient_id", "spec_id", "label_id"] + TARGETS
train = train[need_cols]

label_level = (
    train.groupby(["eeg_id", "patient_id", "spec_id", "label_id"], sort=False)[TARGETS]
    .sum()
    .reset_index()
)
eeg_level = (
    label_level.groupby(["eeg_id", "patient_id", "spec_id"], sort=False)[TARGETS]
    .sum()
    .reset_index()
)

train_votes = eeg_level[TARGETS].astype(np.float64)
global_counts = train_votes.sum(axis=0).values
global_total = float(global_counts.sum())
global_prior = (global_counts + 1e-12) / (global_total + 1e-12 * len(TARGETS))
global_prior = np.clip(global_prior, 1e-12, 1.0)
global_prior = global_prior / global_prior.sum()

patient_counts = eeg_level.groupby("patient_id", sort=False)[TARGETS].sum()
patient_spec_counts = eeg_level.groupby(["patient_id", "spec_id"], sort=False)[
    TARGETS
].sum()
spec_counts = eeg_level.groupby("spec_id", sort=False)[TARGETS].sum()

print("Global prior:", dict(zip(TARGETS, np.round(global_prior, 6))))
print(
    "Num train patients:",
    patient_counts.shape[0],
    "Num test patients:",
    test["patient_id"].nunique(),
)
print(
    "Num train (patient,spec) groups:",
    patient_spec_counts.shape[0],
    "Num train specs:",
    spec_counts.shape[0],
    "Num test specs:",
    test["spec_id"].nunique(),
)

COUNT_EPS = 0.25


def posterior_mean_from_counts(c, prior, kappa_base, n_ref):
    """Dirichlet-multinomial posterior mean with vote-total adaptive strength."""
    c = c.astype(np.float64)
    c = c + COUNT_EPS
    n_votes = float(c.sum())
    kappa = float(kappa_base * np.sqrt(n_ref / (n_votes + n_ref)))
    post = c + kappa * prior
    denom = float(post.sum())
    if denom <= 0:
        return prior.copy()
    return post / denom


def mix_probs(p1, p2, w1):
    p = w1 * p1 + (1.0 - w1) * p2
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()


KAPPA_BASE = 60.0
N_REF = 200.0
MIX_SHRINK = 20.0
FLOOR_ALPHA = 0.01  # keep small global-prior floor to avoid near-zeros (KL-sensitive)

test_pred = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
pids = test["patient_id"].values
specs = test["spec_id"].values

for i, (pid, sid) in enumerate(zip(pids, specs)):
    key = (pid, sid)

    if key in patient_spec_counts.index:
        c = patient_spec_counts.loc[key].values
        p = posterior_mean_from_counts(c, global_prior, KAPPA_BASE, N_REF)
    else:
        has_pid = pid in patient_counts.index
        has_sid = sid in spec_counts.index

        if has_pid and has_sid:
            c_pid = patient_counts.loc[pid].values
            c_sid = spec_counts.loc[sid].values
            p_pid = posterior_mean_from_counts(c_pid, global_prior, KAPPA_BASE, N_REF)
            p_sid = posterior_mean_from_counts(c_sid, global_prior, KAPPA_BASE, N_REF)

            n_pid = float((c_pid.astype(np.float64) + COUNT_EPS).sum())
            n_sid = float((c_sid.astype(np.float64) + COUNT_EPS).sum())
            w_pid = (n_pid + MIX_SHRINK) / (n_pid + n_sid + 2.0 * MIX_SHRINK + 1e-12)
            p = mix_probs(p_pid, p_sid, w_pid)

        elif has_sid:
            c = spec_counts.loc[sid].values
            p = posterior_mean_from_counts(c, global_prior, KAPPA_BASE, N_REF)
        elif has_pid:
            c = patient_counts.loc[pid].values
            p = posterior_mean_from_counts(c, global_prior, KAPPA_BASE, N_REF)
        else:
            p = global_prior.copy()

    p = (1.0 - FLOOR_ALPHA) * p + FLOOR_ALPHA * global_prior
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    test_pred[i, :] = p

test_pred = np.clip(test_pred, 1e-12, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred.astype(np.float32), columns=TARGETS)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

out_path = "submission.csv"
test_pred_df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", test_pred_df.shape)
print(test_pred_df.head())
print(
    "Row-sum check (min/max):",
    float(test_pred_df[TARGETS].sum(axis=1).min()),
    float(test_pred_df[TARGETS].sum(axis=1).max()),
)
