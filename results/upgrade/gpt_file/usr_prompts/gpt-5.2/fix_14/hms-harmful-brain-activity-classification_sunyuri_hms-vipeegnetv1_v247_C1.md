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

3.13

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

0.3533017934064396

# 6. Current score

1.04988

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I first fix the environment/runtime crash coming from TensorFlow/protobuf incompatibility by disabling the C++ protobuf implementation before importing TensorFlow. Next, because the referenced pretrained weights folder is not present in the provided inputs, I add an automatic fallback that produces a valid submission by outputting a calibrated prior distribution learned from `train.csv` vote totals (this is score-reasonable for KL and avoids random predictions). Finally, I keep the existing model/inference path intact when weights are available, and ensure the submission has the exact required columns, sums to 1 per row, and writes `submission.csv`.'
- What this solution (achieved 1.18699) has done: 'I fix the immediate runtime crash caused by the TensorFlow/protobuf incompatibility by avoiding importing TensorFlow unless the pretrained weights are actually present (so the fallback path never touches TF). This keeps the existing model/inference logic intact when weights exist, but unblocks end-to-end execution in your current environment. To move the KL score down toward the target, I improve the fallback from a global prior to an empirically better “patient-conditioned prior” computed from `train.csv` (still fully label-free with respect to test labels, just using `patient_id` metadata). I also keep the submission formatting strict (correct columns, row-wise sum to 1, `submission.csv` output).'
- What this solution (achieved 0.92674) has done: 'Your current score (1.18699, lower-is-better) is far worse than the target (0.3533), so we should improve the *fallback* predictions (the path you’re actually using when weights are missing) with the smallest, safest change. I keep your existing patient-conditioned prior, but add a minimal shrinkage blend toward the global prior for stability (reduces overconfident patient priors on sparse patients) and also incorporate a spectrogram-conditioned prior (since `spectrogram_id` is available in both train and test and often carries strong signal). Finally, I blend patient- and spectrogram-priors with a small global backoff and keep strict row-normalization so the submission always validates.'
- What this solution (achieved 0.89441) has done: 'Your current score is much worse than the target (lower-is-better), and in your environment you’re clearly using the metadata-fallback path (no weights). To move KL down with minimal risk and without changing the core model logic, I improve only the fallback by (1) switching the group prior computation from “ratio of summed votes” to the better “mean of per-row normalized vote distributions” (reduces bias from samples with many annotators), and (2) adding a tiny “expert_consensus-conditioned” prior (available in train; approximated for test via patient consensus distribution), still shrunk to global for stability. I keep the existing patient/spec/global blend structure and strict row-normalization so the submission always validates. No training loop, architecture, or inference logic is changed when weights exist.'
- What this solution (achieved 0.79664) has done: 'Your current score (0.89441, lower-is-better) is still far above the target (0.3533), so we should keep the existing “no-weights fallback” core idea but make a minimal, metric-aligned improvement that typically reduces KL. I change only the fallback blending: (1) compute priors using a Dirichlet/Laplace-smoothed mean (adds pseudocounts per class) to avoid overconfident group priors that hurt KL, and (2) replace the fixed blend weights with *reliability weights* based on each group’s effective sample size so we trust patient/spectrogram priors only when they’re well-supported. The TF/model path remains untouched and still run identically if weights exist. Submission formatting and per-row normalization are kept strict to avoid invalid submissions.'
- What this solution (achieved 0.85382) has done: 'We keep your core model/inference untouched and only adjust the no-weights fallback path (which you’re using, since weights aren’t found) to reduce KL toward the target. Specifically, we make the fallback less overconfident by adding a small uniform Dirichlet floor and by converting the blend to a “product-of-experts” style combination (log-space averaging) of patient/spec/global priors, which typically improves KL for probability targets versus linear mixing. We also base the reliability on number of rows per group (not total annotator votes) to better reflect how stable the group distribution estimate is. Submission formatting, column order, and row-sum-to-1 constraints remain strict and unchanged.'
- What this solution (achieved 0.90091) has done: 'We’re still far above the target (0.85382 vs 0.3533, lower-is-better), and your run is using the no-weights fallback, so we should only adjust that fallback while leaving the TF/model path untouched. The current product-of-experts blend can become too “peaky”; for KL this often hurts when the true target is a soft vote distribution, so we minimally soften predictions by adding a small temperature (>1) applied to the final fallback probabilities. To further reduce overconfidence without changing the overall approach, we also slightly increase the Dirichlet smoothing strengths (alpha) used to estimate patient/spectrogram group priors. All changes keep the same submission schema and strict row-wise normalization so the CSV remains valid.'
- What this solution (achieved 0.92304) has done: 'Your current score (0.90091, lower-is-better) is still far above the target (0.3533), and since your environment usually lacks the model weights, the only safe place to improve is the metadata-based fallback. I keep the same fallback structure (patient/spec/consensus/global priors + POE combination + temperature), but reduce overconfidence more reliably by (1) adding a small per-group “effective sample size” scaling so we trust group priors less when they are noisy, and (2) slightly increasing the final softening (temperature) while reducing the POE peakiness by increasing the uniform floor a bit. These are minimal calibration changes that preserve semantics (still valid probabilities summing to 1) and typically reduce KL when labels are soft vote distributions. The TF/model path is left untouched and run identically if weights are present.'
- What this solution (achieved 0.9235) has done: 'Your current KL (0.92304, lower-is-better) is still far above the target (0.3533), and since weights are missing you’re on the metadata-fallback path; the most reliable way to move toward the target with minimal change is to make the fallback probabilities less overconfident and more “soft-label-like.” I keep the same fallback structure (patient/spec/cons/global priors + POE + temperature), but add one metric-aligned calibration step: a small per-row blend with a near-uniform distribution whose strength increases when the POE output is too peaky (low entropy). This preserves semantics (valid probabilities summing to 1) and typically reduces KL for soft vote targets without changing any model/training logic. I also keep the existing hard normalization and submission formatting exactly as required.'
- What this solution (achieved 0.93877) has done: 'Your current KL (0.9235, lower-is-better) is far above the target (0.3533), so we should improve only the metadata-based fallback path (since weights are missing) while keeping your TF/model path untouched. The safest minimal change for KL to soft vote distributions is to reduce overconfidence by making the fallback prior estimate closer to the label-generating process: compute patient/spectrogram priors from *Dirichlet-smoothed summed votes* (instead of “mean of per-row normalized votes”), which better matches how the targets are aggregated counts. Then apply the same shrinkage/reliability/POE/temperature logic as you already have, so core semantics remain identical and the submission stays valid (rows sum to 1 with correct columns). This is a small, targeted change that often lowers KL in this competition without altering your architecture, training loop, or inference behavior when weights exist.'
- What this solution (achieved 0.96874) has done: 'Your current KL (0.93877, lower-is-better) is still far above the target (0.3533), and since weights are missing you’re using the metadata-fallback; so we only adjust that fallback calibration while leaving the TF/model path untouched. The most minimal, metric-aligned move is to reduce overconfidence/peakiness: (1) increase Dirichlet smoothing slightly for patient/spectrogram priors, and (2) soften the final distribution a bit more via a slightly higher temperature and a slightly stronger adaptive uniform mix when entropy is too low. These changes preserve the same fallback structure (same priors, shrinkage, POE combination) and keep strict row-normalization so the submission stays valid. Nothing about the model architecture/training/inference changes when weights exist.'
- What this solution (achieved 1.04988) has done: 'Your current KL (0.96874, lower-is-better) is still far above the target (0.3533), and since weights are missing the only safe lever is the metadata fallback calibration. I keep the same fallback structure (patient/spec/cons/global priors + shrinkage + reliability + POE + temperature + adaptive uniform mix), but make it less overconfident in a minimal way that typically reduces KL for soft vote targets: slightly stronger global backoff (shrink_k) and a slightly stronger/earlier entropy-based uniform blending. I also add a tiny final blend with the global prior (still a legitimate probability prior, not label leakage) to prevent extreme peaks that hurt KL. Nothing about the TensorFlow model path is changed, and the script still writes a valid `submission.csv` with rows summing to 1.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0, 1")

import warnings

warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241108c"  # the path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate
EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_MULTIPLY = 5

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / 0.4)  # the width of the STFT (eeg spectrogram)  50 * 5

filter_range = [0.5, 45]  # eeg filtering range
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed
BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg
READ_SPE_FILES = False  # preprocess spectrogram

spectrograms = {}  # preprocessed spectrograms for training
eegs = {}  # preprocessed eegs for training
stfts = {}  # preprocessed short-time fourier transform plots for training
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",  # LL
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",  # LP
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",  # RP
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",  # RL
]

TEST_BATCHSIZE = 128

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    pass




## === cell 2
def _row_normalize_probs(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def compute_global_prior_from_train(train_df: pd.DataFrame, targets) -> np.ndarray:
    votes = train_df[list(targets)].astype(np.float64).values
    votes_sum = votes.sum(axis=1, keepdims=True)
    votes_sum = np.where(votes_sum <= 0, 1.0, votes_sum)
    probs = votes / votes_sum
    prior = probs.mean(axis=0)
    prior = np.clip(prior, 1e-8, 1.0)
    prior = prior / prior.sum()
    return prior


def compute_group_priors_dirichlet(
    train_df: pd.DataFrame, group_col: str, targets, alpha: float
):
    """
    Fallback-only: compute group priors from Dirichlet-smoothed summed votes.
    """
    cols = [group_col] + list(targets)
    tmp = train_df[cols].copy()
    for t in targets:
        tmp[t] = tmp[t].astype(np.float64)

    sum_votes = tmp.groupby(group_col)[list(targets)].sum()
    n_rows = tmp.groupby(group_col).size().astype(np.float64)

    K = len(targets)
    base = (alpha / K) * np.ones((sum_votes.shape[0], K), dtype=np.float64)
    denom = sum_votes.values.sum(axis=1, keepdims=True) + alpha
    denom = np.where(denom <= 0, 1.0, denom)
    probs = (sum_votes.values + base) / denom
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    group_to_prior = {gid: probs[i] for i, gid in enumerate(sum_votes.index.values)}
    group_to_nrows = {gid: float(n_rows.loc[gid]) for gid in sum_votes.index.values}
    return group_to_prior, group_to_nrows


def compute_patient_consensus_prior(train_df: pd.DataFrame, targets):
    if "expert_consensus" not in train_df.columns:
        return {}, {}

    tmp = train_df[["patient_id", "expert_consensus"] + list(targets)].copy()
    for t in targets:
        tmp[t] = tmp[t].astype(np.float64)

    votes = tmp[list(targets)].values
    row_sum = votes.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    row_probs = votes / row_sum
    probs_df = pd.DataFrame(row_probs, columns=list(targets))
    probs_df["patient_id"] = tmp["patient_id"].values

    sum_probs = probs_df.groupby("patient_id")[list(targets)].sum()
    n_rows = probs_df.groupby("patient_id").size().astype(np.float64)
    denom = n_rows.values.reshape(-1, 1)
    denom = np.where(denom <= 0, 1.0, denom)
    pat_prior = sum_probs.values / denom
    pat_prior = np.clip(pat_prior, 1e-8, 1.0)
    pat_prior = pat_prior / pat_prior.sum(axis=1, keepdims=True)

    patient_to_prior = {
        pid: pat_prior[i] for i, pid in enumerate(sum_probs.index.values)
    }
    patient_to_nrows = {pid: float(n_rows.loc[pid]) for pid in sum_probs.index.values}
    return patient_to_prior, patient_to_nrows


def _shrink_to_global(
    prior: np.ndarray, global_prior: np.ndarray, count: float, k: float
) -> np.ndarray:
    w = 0.0 if count <= 0 else (count / (count + k))
    out = w * prior + (1.0 - w) * global_prior
    out = np.clip(out, 1e-8, 1.0)
    out = out / out.sum()
    return out


def _reliability_weight(count: float, k: float) -> float:
    if count <= 0:
        return 0.0
    return float(count / (count + k))


def _poe_combine(
    priors: list[np.ndarray],
    weights: list[float],
    eps: float = 1e-8,
) -> np.ndarray:
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return priors[-1].copy()
    weights = [float(w) / wsum for w in weights]

    logp = np.zeros_like(priors[0], dtype=np.float64)
    for p, w in zip(priors, weights):
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, eps, 1.0)
        logp += w * np.log(p)
    out = np.exp(logp)
    out = np.clip(out, eps, 1.0)
    out = out / out.sum()
    return out


def _apply_temperature(
    p: np.ndarray, temperature: float, eps: float = 1e-8
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    if temperature is None or float(temperature) == 1.0:
        out = p
    else:
        out = p ** (1.0 / float(temperature))
    out = np.clip(out, eps, 1.0)
    out = out / out.sum()
    return out


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def make_fallback_submission(
    train_df: pd.DataFrame, test_df: pd.DataFrame, targets
) -> pd.DataFrame:
    global_prior = compute_global_prior_from_train(train_df, targets)

    patient_priors, patient_nrows = compute_group_priors_dirichlet(
        train_df, "patient_id", targets, alpha=5.0
    )
    spec_priors, spec_nrows = compute_group_priors_dirichlet(
        train_df, "spectrogram_id", targets, alpha=6.0
    )
    cons_patient_priors, cons_patient_nrows = compute_patient_consensus_prior(
        train_df, targets
    )

    shrink_k_patient = 55.0
    shrink_k_spec = 85.0
    shrink_k_cons = 160.0

    base_w_patient = 0.54
    base_w_spec = 0.34
    base_w_cons = 0.02
    base_w_global = 0.10

    rel_k_patient = 25.0
    rel_k_spec = 35.0
    rel_k_cons = 50.0

    K = len(targets)
    uniform = np.full(K, 1.0 / K, dtype=np.float64)

    floor_mix = 0.050
    FALLBACK_TEMPERATURE = 1.55

    extra_pow_patient = 0.80
    extra_pow_spec = 0.75
    extra_pow_cons = 0.70

    ENTROPY_TARGET_FRAC = 0.78
    ENTROPY_BLEND_MAX = 0.14

    max_ent = float(np.log(K))
    ent_target = ENTROPY_TARGET_FRAC * max_ent

    FINAL_GLOBAL_BLEND = 0.06

    preds = np.zeros((len(test_df), len(targets)), dtype=np.float64)
    for i, (pid, sid) in enumerate(
        zip(test_df["patient_id"].values, test_df["spectrogram_id"].values)
    ):
        p_prior = patient_priors.get(pid, global_prior)
        p_cnt = patient_nrows.get(pid, 0.0)
        p_prior = _shrink_to_global(p_prior, global_prior, p_cnt, shrink_k_patient)

        s_prior = spec_priors.get(sid, global_prior)
        s_cnt = spec_nrows.get(sid, 0.0)
        s_prior = _shrink_to_global(s_prior, global_prior, s_cnt, shrink_k_spec)

        c_prior = cons_patient_priors.get(pid, global_prior)
        c_cnt = cons_patient_nrows.get(pid, 0.0)
        c_prior = _shrink_to_global(c_prior, global_prior, c_cnt, shrink_k_cons)

        rp = _reliability_weight(p_cnt, rel_k_patient) ** extra_pow_patient
        rs = _reliability_weight(s_cnt, rel_k_spec) ** extra_pow_spec
        rc = _reliability_weight(c_cnt, rel_k_cons) ** extra_pow_cons

        w_patient = base_w_patient * rp
        w_spec = base_w_spec * rs
        w_cons = base_w_cons * rc
        w_global = base_w_global

        pred = _poe_combine(
            priors=[p_prior, s_prior, c_prior, global_prior],
            weights=[w_patient, w_spec, w_cons, w_global],
        )

        pred = (1.0 - floor_mix) * pred + floor_mix * uniform
        pred = _apply_temperature(pred, FALLBACK_TEMPERATURE)

        ent = _entropy(pred)
        if ent < ent_target:
            strength = (ent_target - ent) / max(1e-12, (ent_target))
            strength = float(np.clip(strength, 0.0, 1.0))
            extra_mix = ENTROPY_BLEND_MAX * strength
            pred = (1.0 - extra_mix) * pred + extra_mix * uniform
            pred = pred / pred.sum()

        pred = (1.0 - FINAL_GLOBAL_BLEND) * pred + FINAL_GLOBAL_BLEND * global_prior
        pred = pred / pred.sum()

        preds[i] = pred

    preds = _row_normalize_probs(preds)
    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    sub[list(targets)] = preds.astype(np.float32)
    return sub




## === cell 3
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
print("Test shape", test.shape)

expected_weights = [
    os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
    for model_i in range(SPLITS)
]
weights_exist = all(os.path.exists(p) for p in expected_weights)

if not weights_exist:
    print(
        "WARNING: Model weights not found. Writing enhanced metadata-conditioned prior submission fallback."
    )
    sub = make_fallback_submission(df, test, TARGETS)

    vals = sub[list(TARGETS)].values.astype(np.float64)
    vals = _row_normalize_probs(vals)
    sub.loc[:, list(TARGETS)] = vals.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
else:
    import tensorflow as tf
    from tensorflow.keras.models import clone_model
    from scipy import signal

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    MIX = True
    if MIX:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled")
        except Exception:
            print("Mixed precision not available, using full precision")
    else:
        print("Using full precision")

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            dataframe,
            batch_size=32,
            shuffle=False,
            sample_weights=False,
            mode="train",
            eegs=None,
            stfts=None,
            specs=None,
            imgs=None,
        ):
            self.dataframe = dataframe.reset_index(drop=True)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.sample_weights = sample_weights
            self.mode = mode
            self.eegs = eegs
            self.stfts = stfts
            self.specs = specs
            self.imgs = imgs
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.dataframe) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        (4 * 4 + 2) * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros(
                    (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]

                if self.mode == "test":
                    r_eeg = 0
                else:
                    r_eeg = row.eeg_label_offset_seconds

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]
                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]
                    eeg_save = np.zeros(
                        (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                    )

                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]

                    eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                        np.std(eeg_save, keepdims=True) + 1e-6
                    )
                    x_eeg[j] = eeg

            x = []
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "stft" in DATATYPE:
                x.append(x_stft)
            if "img" in DATATYPE:
                x.append(x_img)

            return x, y, sample_weights

    def build_model():
        inp = []
        y = None

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    (4 * 4 + 2) * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                name="inp_eeg",
            )
            x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
                inp_eeg
            )
            x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_tensor=None
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)

            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

            inp.append(inp_eeg)
            y = x_eeg

        if y is None:
            raise ValueError("No input modality selected in DATATYPE.")

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model

    models = []
    model_template = build_model()
    for model_i in range(SPLITS):
        print(f"Fold {model_i + 1}")
        model = tf.keras.models.clone_model(model_template)
        weights_path = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
        model.load_weights(weights_path)
        models.append(model)

    test["sign_id"] = test.index.values

    PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

    preds_all = []
    batch_start = 0

    for i, eeg_id in enumerate(test.eeg_id.values):
        if i % 100 == 0:
            print(i, ", ", end="")

        eeg_default = pd.read_parquet(
            os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
        )

        eeg = []
        for channel in BRAIN:
            eeg_temp = (
                eeg_default.loc[:, channel.split("-")[0]]
                - eeg_default.loc[:, channel.split("-")[1]]
            ).values
            eeg_temp[np.isnan(eeg_temp)] = 0
            eeg.append(np.reshape(eeg_temp, (1, -1)))
        eeg = np.concatenate(eeg, axis=0)

        if SFREQ != RSFREQ:
            eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

        eeg = signal.filtfilt(b, a, eeg, axis=1)
        eeg = np.clip(eeg, a_min=-1024, a_max=1024)

        eegs_test[int(eeg_id)] = eeg

        if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test)):
            batch_df = test.iloc[batch_start : i + 1].reset_index(drop=True)
            test_gen = DataGenerator(
                batch_df,
                shuffle=False,
                sample_weights=False,
                batch_size=TEST_BATCHSIZE,
                mode="test",
                specs=spectrograms_test,
                eegs=eegs_test,
                stfts=stfts_test,
                imgs=imgs_test,
            )

            preds = []
            for model_i in range(SPLITS):
                pred = models[model_i].predict(test_gen, verbose=0)
                preds.append(pred)
            pred = np.mean(preds, axis=0)

            preds_all.append(pred)

            eegs_test = {}
            stfts_test = {}
            imgs_test = {}
            gc.collect()
            batch_start = i + 1

    preds_all = np.concatenate(preds_all, axis=0)

    preds_all = np.asarray(preds_all, dtype=np.float64)
    preds_all = np.clip(preds_all, 1e-8, 1.0)
    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = preds_all.astype(np.float32)

    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.clip(vals, 1e-8, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    sub.loc[:, TARGETS] = vals.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("\nSubmission shape", sub.shape)
    print(sub.head())
