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

0.2851469320245234

# 6. Current score

1.40497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.68479) has done: 'I define the missing variables, import the necessary libraries, implement a simple `compute_global_distribution` helper, and simplify the workflow so that it always runs the inference path (no TensorFlow training). This fixes the `NameError` and guarantees a valid `submission.csv` with correctly normalised probabilities.'
- What this solution (achieved 1.41885) has done: 'I replace the pure patient‑level probabilities with a lightly‑smoothed version that blends each patient’s vote counts with the overall training class distribution. This reduces over‑confidence for patients with few annotations, keeping the core logic unchanged while moving the KL‑divergence score closer to the target (lower is better). The script still writes a valid `submission.csv` with rows that sum to 1.'
- What this solution (achieved 1.4053) has done: 'I increase the strength of the global prior (set ALPHA to 1000) so that predictions rely more on the overall class distribution, which tends to lower KL‑divergence for this task. Additionally, I apply a mild temperature scaling (square‑root of probabilities) before the final renormalisation to make the predictions a bit less extreme, further nudging the score toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.4048) has done: 'The fix keeps the same patient‑level smoothing logic but makes the global prior stronger (ALPHA = 5000) and applies a slightly stronger temperature flattening (raising probabilities to the 0.4 power before renormalising). Both tweaks keep the original workflow intact while moving the predictions toward a less‑confident distribution, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.4087) has done: 'I increase the strength of the global prior (set ALPHA to 1 000 000) so each patient’s distribution is essentially the overall class distribution, and I flatten the probabilities even more by using a smaller temperature power (0.05). These minimal constant tweaks keep the original workflow unchanged while moving the predictions toward a near‑uniform distribution, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.41423) has done: 'I lower the smoothing strength and remove the extreme temperature flattening so the predictions rely more on each patient’s own vote counts while still keeping a modest global prior. This small constant change keeps the original workflow intact but should produce a less‑uniform, better‑calibrated distribution and thus move the KL‑divergence score closer to the target (lower is better).'
- What this solution (achieved 1.4053) has done: 'I increase the smoothing strength so each patient’s prediction leans more on the overall class distribution (ALPHA = 1000) and add a mild temperature flattening (power = 0.5) to avoid overly confident predictions. These small adjustments keep the original workflow but should move the KL‑divergence toward the lower target value.'
- What this solution (achieved 1.41423) has done: 'I reduce the smoothing strength back to a modest value (ALPHA = 1.0) so each patient’s own vote counts dominate the prediction, and I remove the temperature flattening (TEMPERATURE_POWER = 1.0). This keeps the overall workflow unchanged while making the predicted distributions more faithful to the observed patient‑level votes, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I replace the patient‑level smoothing with a pure global‑distribution prediction, which makes every row use the overall class proportions from the training data; this typically yields a much lower KL‑divergence when the current approach is far from optimal. The change is limited to the inference block and keeps the rest of the workflow unchanged.'
- What this solution (achieved 0.82785) has done: 'I replace the pure‑global prediction with a simple patient‑level prior: for every test row I use the vote distribution of its patient from the training data (if available) and fall back to the overall class distribution otherwise. A modest smoothing (α = 0.5) blends these two distributions, keeping the core workflow unchanged while providing more informative probabilities that should lower the KL‑divergence toward the target.'
- What this solution (achieved 1.40497) has done: 'We increase the smoothing strength dramatically so the predictions are dominated by the overall class distribution, and apply a stronger temperature flattening (power 0.3) to move the probabilities toward a less‑confident, more uniform shape. This keeps the original workflow unchanged while nudging the KL‑divergence lower, aiming to bring the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

NEEDTRAIN = False
TF_AVAILABLE = False  # TensorFlow is not required for inference here
SPLITS = 5  # placeholder (not used when NEEDTRAIN=False)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
TARGETS_RAW = TARGETS

DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"




## === cell 1
def compute_global_distribution(df: pd.DataFrame, targets: list) -> np.ndarray:
    """
    Compute the overall class distribution from the training data.
    Returns a 1‑D numpy array where the values sum to 1.
    """
    totals = df[targets].sum().values.astype(float)
    if totals.sum() == 0:
        return np.full(len(targets), 1.0 / len(targets))
    return totals / totals.sum()




## === cell 2
if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    test["sign_id"] = test.index.values  # retain original index if needed
    print("Test shape:", test.shape)

    global_dist = compute_global_distribution(df, TARGETS)  # shape (6,)

    patient_votes = df.groupby("patient_id")[TARGETS].sum()
    patient_totals = patient_votes.sum(axis=1).replace(
        0, np.nan
    )  # avoid division by zero
    patient_dist = patient_votes.div(patient_totals, axis=0)

    test_pred = test.merge(
        patient_dist.reset_index(),
        on="patient_id",
        how="left",
        suffixes=("", "_pat"),
    )

    test_pred[TARGETS] = test_pred[TARGETS].fillna(
        pd.Series(global_dist, index=TARGETS)
    )

    ALPHA = 10000.0
    preds_arr = test_pred[TARGETS].values.astype(float)  # patient (or fallback) probs
    global_arr = np.array(global_dist, dtype=float)  # shape (6,)

    blended = (preds_arr + ALPHA * global_arr) / (
        1.0 + ALPHA
    )  # essentially global_dist
    blended = blended / blended.sum(axis=1, keepdims=True)  # ensure rows sum to 1

    TEMPERATURE_POWER = 0.3
    blended = np.power(blended, TEMPERATURE_POWER)
    blended = blended / blended.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = blended
    sub.to_csv(SUBMISSION_PATH, index=False)

    print("Submission saved to:", SUBMISSION_PATH)
    print("Submission shape:", sub.shape)
    print(sub.head())
