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

14.0636025834849

# 6. Current score

1.73286

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The submission length mismatch comes from merging against `sample_submission.csv` after you already built predictions for `test_master`, which can change the row count if there are duplicate `eeg_id`s on either side. I fix this by constructing the submission directly in the exact order of `sample_submission.csv` (the file Kaggle uses to validate rows), and mapping predictions by `eeg_id` without a merge that can duplicate rows. I also keep the probabilities normalized and clipped to avoid KL-divergence failures due to zeros/NaNs, while preserving your core approach (prior-based predictions). Finally, I update the validation cell to check against `sample_sub` length (the true required submission length) instead of `test_master`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.39779, lower-is-better) is already far better than the target (14.0636), so to move toward the target we should intentionally *degrade* performance in a controlled, valid way while keeping your core “prior-based predictions” logic intact. The smallest stable change is to replace the empirical class prior with a uniform distribution at inference time; this remains a legitimate probabilistic baseline, preserves your pipeline and submission semantics, and increase KL divergence toward the target. I keep all the existing safety checks (clipping, normalization, row-order via `sample_submission.csv`) to avoid invalid submissions. If this overshoots above the target, we can later blend uniform and prior with a fixed weight to fine-tune.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far better than the target (14.0636), so we should *intentionally worsen* it toward the target while keeping your “prior-based / constant-probability baseline” core logic intact. The smallest controlled way is to apply a strong “sharpening” transform to your constant predictions (raise probabilities to a power > 1), which increases overconfidence and typically increases KL divergence without changing data usage or pipeline structure. I keep your strict submission-row alignment via `sample_submission.csv`, plus clipping/renormalization so the file stays valid. I also leave feature extraction untouched (it’s currently just a sanity check) and only adjust the constant prediction construction.'
- What this solution (achieved 21.6153) has done: 'Your current score (1.40995, lower-is-better) is far better than the target (14.0636), so to move *toward* the target we should deliberately worsen predictions while keeping your constant-probability baseline logic and valid submission constraints. The most controlled minimal change is to make the constant distribution extremely overconfident by concentrating almost all probability mass on a single class (this increases KL substantially when wrong) while still clipping and renormalizing to avoid zeros. I keep the sample_submission row order mapping exactly as you already do (to avoid length/order issues) and leave all feature code untouched. If this overshoots past the target, we can later reduce the concentration strength.'
- What this solution (achieved 1.73286) has done: 'Your current score (21.6153, lower-is-better) is worse than the target (14.0636), so we should improve (decrease) the score but only with minimal changes that keep your constant-probability baseline logic intact. The smallest reliable improvement is to soften the extreme “all mass on seizure” prediction by blending it with the empirical class prior you already computed, which usually reduces KL while staying deterministic and fast. I keep the exact submission row-order logic via `sample_submission.csv`, and retain strict clipping + renormalization so the submission remains valid. This change is confined to how `default_pred` is constructed; no model/feature pipeline changes are required.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from scipy import signal
from sklearn import preprocessing

SEED = 42
np.random.seed(SEED)

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
SUB_COLS = ["eeg_id"] + TARGET_COLS

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print(
    "Python OK; avoiding TensorFlow due to protobuf incompatibility in this environment."
)



## === cell 1
test_master = pd.read_csv(f"{DATA_DIR}/test.csv")
print("test_master:", test_master.shape)
test_master.head()



## === cell 2
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

votes = train_df[TARGET_COLS].to_numpy(dtype=np.float64, copy=True)
votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)
row_sums = votes.sum(axis=1, keepdims=True)
row_sums[row_sums <= 0] = 1.0
row_probs = votes / row_sums

prior = row_probs.mean(axis=0)
prior = np.clip(prior, 1e-12, None)
prior = prior / prior.sum()

print(
    "Computed class prior from train (sum, min, max):",
    float(prior.sum()),
    float(prior.min()),
    float(prior.max()),
)
print(dict(zip(TARGET_COLS, prior.round(6))))



## === cell 3
pairs = [
    ["Fp1", "F7"],
    ["F7", "T3"],
    ["T3", "T5"],
    ["T5", "O1"],
    ["Fp2", "F8"],
    ["F8", "T4"],
    ["T4", "T6"],
    ["T6", "O2"],
    ["Fp1", "F3"],
    ["F3", "C3"],
    ["C3", "P3"],
    ["P3", "O1"],
    ["Fp2", "F4"],
    ["F4", "C4"],
    ["C4", "P4"],
    ["P4", "O2"],
    ["Fz", "Cz"],
    ["Cz", "Pz"],
]


def _safe_col(arr_df: pd.DataFrame, col: str, length: int) -> np.ndarray:
    """Return a numeric column; if missing, return zeros of given length."""
    if col in arr_df.columns:
        x = arr_df[col].to_numpy(dtype=np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x
    return np.zeros(length, dtype=np.float32)


def get_features_for_entry(row: pd.Series, q: int = 2) -> np.ndarray:
    """
    Returns X_eeg with shape (n_channels=18, n_timesteps=1000) for each test eeg_id.
    Uses the same 10-second central window selection (20s..30s of the 50s clip).
    """
    eeg_path = f"{DATA_DIR}/test_eegs/{int(row['eeg_id'])}.parquet"
    eeg = pd.read_parquet(eeg_path)

    start, end = 20 * 200, 30 * 200
    eeg_subsample = eeg.iloc[start:end].reset_index(drop=True)
    L = eeg_subsample.shape[0]  # should be 2000

    output_eeg_X = []
    expected_len = int(np.ceil(L / q))  # for L=2000,q=2 => 1000

    for a, b in pairs:
        xa = _safe_col(eeg_subsample, a, L)
        xb = _safe_col(eeg_subsample, b, L)
        diff = xa - xb
        dec = signal.decimate(diff, q, ftype="iir", zero_phase=True).astype(
            np.float32, copy=False
        )

        if dec.shape[0] < expected_len:
            dec = np.pad(dec, (0, expected_len - dec.shape[0]), mode="constant")
        elif dec.shape[0] > expected_len:
            dec = dec[:expected_len]

        dec_norm = preprocessing.normalize([dec])[0].astype(np.float32, copy=False)
        output_eeg_X.append(dec_norm)

    output_eeg_X = np.stack(output_eeg_X, axis=0)  # (18, 1000)
    return output_eeg_X


x0 = get_features_for_entry(test_master.iloc[0])
print(
    "X shape:",
    x0.shape,
    "dtype:",
    x0.dtype,
    "min/max:",
    float(x0.min()),
    float(x0.max()),
)



## === cell 4
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

DOMINANT_CLASS = "seizure_vote"  # fixed choice for determinism
EPS_FLOOR = 1e-12  # keep strictly positive to avoid submission failure

alpha = 0.50  # weight on the dominant one-hot; (1-alpha) on prior. Tune minimally if needed after a rerun.

onehot = np.full(len(TARGET_COLS), EPS_FLOOR, dtype=np.float64)
onehot[TARGET_COLS.index(DOMINANT_CLASS)] = 1.0
onehot = np.clip(onehot, EPS_FLOOR, None)
onehot = onehot / onehot.sum()

default_pred = alpha * onehot + (1.0 - alpha) * prior
default_pred = np.nan_to_num(default_pred, nan=0.0, posinf=0.0, neginf=0.0)
default_pred = np.clip(default_pred, EPS_FLOOR, None)
default_pred = default_pred / default_pred.sum()

test_eeg_ids = test_master["eeg_id"].to_numpy()
unique_test_ids = pd.unique(test_eeg_ids)

pred_map = {int(eid): default_pred.copy() for eid in unique_test_ids}

preds = np.vstack(
    [pred_map.get(int(eid), default_pred) for eid in sample_sub["eeg_id"].to_numpy()]
)

preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
preds = np.clip(preds, EPS_FLOOR, None)
preds = preds / preds.sum(axis=1, keepdims=True)

submission = pd.DataFrame(preds, columns=TARGET_COLS)
submission.insert(0, "eeg_id", sample_sub["eeg_id"].to_numpy())
submission = submission[SUB_COLS]

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(
    "Row sum stats:",
    float(submission[TARGET_COLS].sum(axis=1).min()),
    float(submission[TARGET_COLS].sum(axis=1).max()),
)
print("alpha:", alpha)
print("Prior:", dict(zip(TARGET_COLS, prior.round(6))))
print("Onehot:", dict(zip(TARGET_COLS, onehot.round(12))))
print("Default prediction:", dict(zip(TARGET_COLS, default_pred.round(12))))
submission.head()



## === cell 5
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")

sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

assert list(sub_check.columns) == SUB_COLS
assert sub_check.shape[0] == sample_sub.shape[0]
assert sub_check["eeg_id"].to_numpy().shape == sample_sub["eeg_id"].to_numpy().shape

sums = sub_check[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(sums))
assert np.allclose(sums, 1.0, atol=1e-5)

vals = sub_check[TARGET_COLS].to_numpy()
assert np.all(vals > 0.0)

print("submission.csv validated.")
