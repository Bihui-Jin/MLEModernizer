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

0.3502386564711898

# 6. Current score

1.16668

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf/h5py stack by avoiding TensorFlow/Keras entirely (your current environment can’t import it cleanly), while keeping the evaluation semantics (output is a valid probability distribution over the 6 vote columns). I also fix the broken training-data dependency (`train.csv` and `preprocess/*.npy` files that are not present) by removing those branches and generating predictions directly from `train.csv` priors. Finally, I ensure the submission has the exact required columns/order from `sample_submission.csv`, contains 9850 rows, and each row sums to 1 with safe clipping to avoid submission failure.'
- What this solution (achieved 1.15381) has done: 'Your current submission uses a single global class prior, which is too blunt and yields a high KL. With minimal change and same “train.csv priors only” core logic, I condition the prior on `patient_id` (patient-specific priors) and fall back to the global prior for unseen patients in test. This typically reduces KL because label distributions are patient-dependent in this dataset, while still producing valid per-row probability vectors that sum to 1. I also keep the submission column order exactly as `sample_submission.csv` and add safety normalization/clipping as before.'
- What this solution (achieved 0.81763) has done: 'To move your KL score down toward the target, I keep your “train.csv priors only” approach but make the priors less blunt by (1) conditioning on both `patient_id` and `spectrogram_id` when available (more specific priors), and (2) smoothing these hierarchical priors with the global prior using a simple count-based shrinkage so rare groups don’t overfit. This preserves the same evaluation semantics (still outputs valid per-row probability distributions summing to 1) while typically improving KL vs. patient-only priors. I also ensure robust alignment (no accidental reordering) and keep exact submission column order from `sample_submission.csv`. No model/architecture/training loop is introduced.'
- What this solution (achieved 1.16668) has done: 'Your current approach is already a strong “train-priors only” baseline, but it’s likely leaving easy KL improvements on the table because it averages labels at the row (subsample) level rather than at the evaluation level (`eeg_id`). I keep the same core logic (hierarchical priors + count-based shrinkage) and only change how the priors are *estimated*: first aggregate train votes into a single target distribution per `eeg_id`, then compute patient and (patient,spectrogram) priors from those `eeg_id`-level distributions. This typically improves calibration for the Kaggle metric because test has one row per `eeg_id`, and it also prevents over-counting overlapping subsamples. Everything else (smoothing, fallbacks, safe normalization, submission format) stays the same.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings

warnings.filterwarnings("ignore")

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241106b"  # the path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 100  # resampled EEG sampling rate
EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_MULTIPLY = 4

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



## === cell 1
import numpy as np
import pandas as pd

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
sample_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")

df = pd.read_csv(train_path)
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_sub.shape)

NEEDTRAIN = False



## === cell 2

y_votes = df[TARGETS].astype(np.float64).values
row_sums = y_votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
y_prob = y_votes / row_sums

tmp_row = df[["eeg_id", "patient_id", "spectrogram_id"]].copy()
for i, c in enumerate(TARGETS):
    tmp_row[c] = y_prob[:, i]

agg_dict = {c: "mean" for c in TARGETS}
agg_meta = tmp_row.groupby("eeg_id", sort=False)[
    ["patient_id", "spectrogram_id"]
].first()
agg_probs = tmp_row.groupby("eeg_id", sort=False)[list(TARGETS)].mean()
tmp = pd.concat([agg_meta, agg_probs], axis=1).reset_index()

global_prior = tmp[TARGETS].mean(axis=0).values.astype(np.float64)
global_prior = np.nan_to_num(
    global_prior,
    nan=1.0 / len(TARGETS),
    posinf=1.0 / len(TARGETS),
    neginf=1.0 / len(TARGETS),
)
global_prior = np.clip(global_prior, 1e-8, 1.0)
global_prior = global_prior / global_prior.sum()
print("Global prior:", dict(zip(TARGETS, global_prior.round(6))))


def _safe_norm(mat: np.ndarray) -> np.ndarray:
    mat = np.nan_to_num(
        mat,
        nan=1.0 / len(TARGETS),
        posinf=1.0 / len(TARGETS),
        neginf=1.0 / len(TARGETS),
    )
    mat = np.clip(mat, 1e-8, 1.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


patient_stats = tmp.groupby("patient_id")[list(TARGETS)].mean()
patient_counts = tmp.groupby("patient_id").size().astype(np.float64)

ps_stats = tmp.groupby(["patient_id", "spectrogram_id"])[list(TARGETS)].mean()
ps_counts = tmp.groupby(["patient_id", "spectrogram_id"]).size().astype(np.float64)

patient_stats.loc[:, list(TARGETS)] = _safe_norm(
    patient_stats.values.astype(np.float64)
)
ps_stats.loc[:, list(TARGETS)] = _safe_norm(ps_stats.values.astype(np.float64))

alpha_ps_to_patient = 20.0
alpha_patient_to_global = 50.0

p_mat = patient_stats.values.astype(np.float64)
n_p = patient_counts.reindex(patient_stats.index).values.reshape(-1, 1)
w_p = n_p / (n_p + alpha_patient_to_global)
patient_smooth = w_p * p_mat + (1.0 - w_p) * global_prior.reshape(1, -1)
patient_smooth = _safe_norm(patient_smooth).astype(np.float32)

patient_smooth_df = pd.DataFrame(
    patient_smooth, index=patient_stats.index, columns=TARGETS
)

preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

test_patient = test["patient_id"].values
test_spec = test["spectrogram_id"].values

test_ps_index = pd.MultiIndex.from_arrays(
    [test_patient, test_spec], names=["patient_id", "spectrogram_id"]
)
mask_ps_seen = test_ps_index.isin(ps_stats.index)

if mask_ps_seen.any():
    seen_ps = test_ps_index[mask_ps_seen]
    ps_prior = ps_stats.loc[seen_ps, list(TARGETS)].values.astype(np.float64)
    ps_n = ps_counts.loc[seen_ps].values.astype(np.float64).reshape(-1, 1)

    p_for_ps = patient_smooth_df.loc[
        seen_ps.get_level_values(0), list(TARGETS)
    ].values.astype(np.float64)

    w_ps = ps_n / (ps_n + alpha_ps_to_patient)
    ps_smooth = w_ps * ps_prior + (1.0 - w_ps) * p_for_ps
    ps_smooth = _safe_norm(ps_smooth).astype(np.float32)
    preds_all[mask_ps_seen] = ps_smooth

mask_remaining = ~mask_ps_seen
if mask_remaining.any():
    remaining_patients = test_patient[mask_remaining]
    mask_patient_seen = np.isin(remaining_patients, patient_smooth_df.index.values)

    if mask_patient_seen.any():
        idx = np.flatnonzero(mask_remaining)[mask_patient_seen]
        preds_all[idx] = patient_smooth_df.loc[
            remaining_patients[mask_patient_seen], list(TARGETS)
        ].values.astype(np.float32)

    if (~mask_patient_seen).any():
        idx = np.flatnonzero(mask_remaining)[~mask_patient_seen]
        preds_all[idx] = global_prior.reshape(1, -1).astype(np.float32)

preds_all = np.nan_to_num(
    preds_all,
    nan=1.0 / len(TARGETS),
    posinf=1.0 / len(TARGETS),
    neginf=1.0 / len(TARGETS),
).astype(np.float32)
preds_all = np.clip(preds_all, 1e-8, 1.0)
preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

print(
    "Test groups seen (patient,spectrogram) / patient-only / unseen:",
    int(mask_ps_seen.sum()),
    int(
        (~mask_ps_seen).sum()
        - np.isin(test_patient[~mask_ps_seen], patient_smooth_df.index.values).sum()
    ),
    int((~np.isin(test_patient, patient_smooth_df.index.values)).sum()),
)



## === cell 3
sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for k, col in enumerate(TARGETS):
    sub[col] = preds_all[:, k]

sub = sub[sample_sub.columns.tolist()]

row_sum = sub[TARGETS].sum(axis=1).values
print("Row sum min/max:", float(row_sum.min()), float(row_sum.max()))
print("Submission shape:", sub.shape)
print(sub.head())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
