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

0.3227873128869408

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The script was failing because TensorFlow could not be imported in the current Python 3.13 environment, causing an `AttributeError` from the protobuf library. To keep the pipeline runnable we catch the TensorFlow import error and fall back to a lightweight, deterministic baseline that outputs uniform class probabilities for every test sample. This ensures a valid `submission.csv` is written without altering the original training logic when TensorFlow is available.'
- What this solution (achieved 1.39779) has done: 'The fix updates the data‑folder lookup so the script can locate the training and test CSVs regardless of the Kaggle environment, adds safe fall‑backs when files are missing, and ensures the generated `submission.csv` always contains the required columns with probabilities that sum to 1. No core modelling logic is changed, and the uniform‑probability baseline remains, keeping the score behavior unchanged but guaranteeing a valid submission file.'
- What this solution (achieved 1.39779) has done: 'The update fixes the pandas `sum` call that does not support `keepdims` by reshaping the result manually, ensuring `EEG_PROB_DICT` is correctly created. This resolves the runtime errors in the early cells and allows the script to generate a valid `submission.csv` with proper probability normalization.'
- What this solution (achieved 1.39779) has done: 'I smooth the per‑eeg predictions by blending them with the overall class frequencies. This keeps the existing logic but reduces over‑confident, noisy per‑eeg probabilities, which should lower the KL‑divergence score toward the target. The change is only in the prediction loop where the final probability matrix is built.'
- What this solution (achieved 1.39779) has done: 'We add a confidence‑based blending step: compute how many annotator votes each eeg_id had, and use that count to weight the per‑eeg probabilities (more votes → higher trust). This keeps the original baseline but lets confident recordings dominate, which should lower the KL‑divergence toward the target score. The change is limited to storing vote counts and adjusting the blend factor in the prediction loop.'
- What this solution (achieved 1.39779) has done: 'The update reduces the excessive smoothing by removing the vote‑count based down‑weighting of per‑eeg probabilities. The blend factor is set to 1.0, so whenever an EEG ID has recorded votes we use its aggregated class distribution directly (fallback to the global distribution only when no per‑eeg data exists). This sharper, more informed prediction is expected to lower the KL‑divergence toward the target score while keeping all other logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path

TENSORFLOW_OK = False
tf = None  # placeholder to keep references safe

candidate_roots = [
    Path("/kaggle/input/hms-harmful-brain-activity-classification"),
    Path("./data/hms-harmful-brain-activity-classification"),
    Path("./hms-harmful-brain-activity-classification"),
    Path("./data"),  # generic fallback
]
LOAD_DATA_FROM = None
for cand in candidate_roots:
    if (cand / "train.csv").exists():
        LOAD_DATA_FROM = cand
        break
if LOAD_DATA_FROM is None:
    LOAD_DATA_FROM = Path(".")
    print("Warning: train.csv not found; using uniform class probabilities.")
else:
    print(f"Data root resolved to: {LOAD_DATA_FROM}")

train_path = LOAD_DATA_FROM / "train.csv"
if train_path.exists():
    df = pd.read_csv(train_path)
    TARGETS = df.columns[-6:]  # last six columns are the vote counts
    print("Train shape:", df.shape)
    print("Targets:", list(TARGETS))

    row_probs = df[TARGETS].values.astype(np.float32)
    row_sums = row_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    row_probs = row_probs / row_sums

    GLOBAL_PROB = row_probs.mean(axis=0)  # shape (6,)
    print("Global class probabilities:", GLOBAL_PROB)

    eeg_groups = df.groupby("eeg_id")[list(TARGETS)].sum()
    eeg_sums = eeg_groups.sum(axis=1).values[:, None]  # total votes per eeg_id
    eeg_sums[eeg_sums == 0] = 1.0
    eeg_probs = eeg_groups.values / eeg_sums
    EEG_PROB_DICT = {
        int(eid): prob.astype(np.float32)
        for eid, prob in zip(eeg_groups.index, eeg_probs)
    }
    EEG_VOTE_COUNT = {
        int(eid): int(cnt) for eid, cnt in zip(eeg_groups.index, eeg_sums.squeeze())
    }
    MAX_VOTE = max(EEG_VOTE_COUNT.values()) if EEG_VOTE_COUNT else 1
    print(f"Computed per‑eeg_id probabilities for {len(EEG_PROB_DICT)} ids.")
else:
    TARGETS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    GLOBAL_PROB = np.full(len(TARGETS), 1.0 / len(TARGETS), dtype=np.float32)
    EEG_PROB_DICT = {}
    EEG_VOTE_COUNT = {}
    MAX_VOTE = 1
    print("train.csv missing – using uniform GLOBAL_PROB:", GLOBAL_PROB)




## === cell 1
class CosineAnnealingLRScheduler(
    tf.keras.optimizers.schedules.LearningRateSchedule if TENSORFLOW_OK else object
):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = total_step
        self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.begin = 1

    def __call__(self, step):
        if step == self.total_step:
            self.begin = 0
            self.lr_max = self.lr_max * 0.5
            self.lr_min = self.lr_min * 0.1
        step = step % self.total_step + 1
        if (self.begin == 1) and (step < self.warm_step):
            lr = self.lr_max / self.warm_step * step
        else:
            if self.begin == 1:
                if self.total_step == 1:
                    lr = self.lr_max
                else:
                    cos_arg = (
                        (step - self.warm_step)
                        / (self.total_step - self.warm_step)
                        * np.pi
                    )
                    cos_val = np.cos(cos_arg) if not TENSORFLOW_OK else tf.cos(cos_arg)
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0 + cos_val
                    )
            else:
                cos_val = (
                    np.cos(step / 10 * np.pi)
                    if not TENSORFLOW_OK
                    else tf.cos(step / 10 * np.pi)
                )
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos_val)
        return np.float32(lr)




## === cell 2
if __name__ == "__main__":
    test_path = LOAD_DATA_FROM / "test.csv"
    if test_path.exists():
        test = pd.read_csv(test_path)
    else:
        sample_sub_path = LOAD_DATA_FROM / "sample_submission.csv"
        if sample_sub_path.exists():
            test = pd.read_csv(sample_sub_path)[["eeg_id"]].copy()
            print("test.csv missing – using eeg_id from sample_submission.csv")
        else:
            raise FileNotFoundError(
                f"Neither test.csv nor sample_submission.csv found in {LOAD_DATA_FROM}"
            )
    test["sign_id"] = test.index.values
    print("Test shape:", test.shape)

    BASE_BLEND_ALPHA = 1.0  # ensure we rely entirely on per‑eeg data when available
    prob_matrix = np.empty((len(test), len(TARGETS)), dtype=np.float32)
    for i, eid in enumerate(test["eeg_id"].values):
        per_eeg_prob = EEG_PROB_DICT.get(int(eid))
        if per_eeg_prob is not None:
            prob = per_eeg_prob  # full confidence in known EEG IDs
        else:
            prob = GLOBAL_PROB  # fallback for unseen IDs
        prob_matrix[i] = prob

    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prob_matrix = prob_matrix / row_sums

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = prob_matrix

    submission_path = "submission.csv"
    sub.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
    print("Submission shape:", sub.shape)
    print(sub.head())
