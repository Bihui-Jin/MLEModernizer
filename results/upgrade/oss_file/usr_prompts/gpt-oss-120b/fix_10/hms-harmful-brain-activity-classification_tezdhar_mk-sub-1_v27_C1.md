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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.3111068021557354

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the missing‑module import with an explicit list of target columns, simplify the merging logic, and add a fallback that builds a valid submission directly from the training vote statistics when the original prediction files are not available. This fixes the import error, ensures a DataFrame with the required columns is always created, normalizes the probabilities, and writes a correctly‑named `submission.csv` in the working directory.'
- What this solution (achieved 1.41937) has done: 'I replace the generic uniform fallback with a per‑eeg ID probability derived from the training votes, and add a tiny epsilon‑smoothing before the final normalization so that no class receives a zero probability. This keeps the original merging logic intact while giving much more informative predictions for test rows that share an eeg_id with the training set, thereby moving the KL‑divergence score toward the target.'
- What this solution (achieved 1.41937) has done: 'We keep the overall pipeline unchanged but improve the fallback prediction: instead of relying solely on per‑eeg vote ratios (or pure overall frequencies), we blend them. Using a weighted mix of the per‑eeg distribution and the global class distribution smooths noisy per‑eeg estimates and typically lowers KL‑divergence, moving the score closer to the target. The change is confined to the fallback block of `merge_predictions` and retains all existing logic.'
- What this solution (achieved 1.41937) has done: 'I lower the reliance on the per‑eeg fallback distribution (which can be noisy) by reducing its blend weight from 0.7 to 0.3, giving more influence to the stable global class frequencies. This small change keeps all existing pipeline logic unchanged while likely decreasing the KL‑divergence (moving the score closer to the lower target). The adjustment is made in the fallback section of `merge_predictions`.'
- What this solution (achieved 1.41937) has done: 'I lower the reliance on the per‑eeg fallback distribution, which is noisy and inflates KL‑divergence, by setting its blend weight to 0 so the submission uses only the stable global class frequencies (still smoothed and normalized). This minimal change keeps the existing pipeline unchanged while moving the score toward the lower target.'
- What this solution (achieved 1.41937) has done: 'I adjust the fallback blending to include a portion of the per‑eeg vote distribution rather than using only the global class frequencies. Setting `blend_weight` to 0.3 lets each test row benefit from its own EEG‑specific statistics while still keeping the stable overall distribution, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I reduce the fallback blend weight to 0.0 so the submission uses only the stable global class frequencies (still smoothed and normalized). This removes noisy per‑eeg information, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I lower the blend weight from 0.0 to 0.5 so the fallback prediction uses a balanced mix of per‑eeg vote distributions and the stable global class frequencies. This small change keeps the core pipeline unchanged while providing more informative per‑eeg probabilities, which should reduce the KL‑divergence score and move it closer to the target.'
- What this solution (achieved 1.41937) has done: 'I lower the fallback blend weight to 0.0 so the submission relies solely on the stable global class frequencies, removing noisy per‑eeg information that inflates the KL‑divergence. This tiny change keeps the overall pipeline untouched while moving the score toward the lower target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 1
def load_prediction_file(fold: int, version: str) -> pd.DataFrame:
    """
    Safely load a prediction CSV produced by the original pipeline.
    Returns None if the file does not exist.
    """
    file_path = os.path.join(OUT_PATH, f"submission_fold{fold}_{version}.csv")
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    return None




## === cell 2
def _smooth_and_normalize(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """
    Apply epsilon‑smoothing to avoid zero probabilities and renormalize each row.
    """
    probs = df[TARGET_COLS].clip(lower=eps)
    probs = probs.div(probs.sum(axis=1), axis=0)
    df[TARGET_COLS] = probs
    return df


def merge_predictions(
    folds: list[int] = [0, 1, 2, 3, 4],
    versions: list[str] = ["v0", "v2"],
    weights: list[float] = [0.3, 0.7],
) -> pd.DataFrame:
    """
    Combine prediction files across folds and versions using the supplied weights.
    If no files are found, fall back to a per‑eeg_id baseline derived from the training data,
    blended with the overall class frequencies to reduce over‑fitting.
    """
    base_df = None
    weighted_preds = []

    for fold in folds:
        for version, weight in zip(versions, weights):
            df = load_prediction_file(fold, version)
            if df is not None:
                if base_df is None:
                    base_df = df[[col for col in df.columns if col != "eeg_id"]].copy()
                    base_df["eeg_id"] = df["eeg_id"]
                pred = df[TARGET_COLS].values.astype(float) * weight
                weighted_preds.append(pred)

    if weighted_preds:
        summed = np.sum(weighted_preds, axis=0)
        summed /= summed.sum(axis=1, keepdims=True)
        base_df[TARGET_COLS] = summed
        return _smooth_and_normalize(base_df)
    else:
        train_path = os.path.join(DATA_PATH, "train.csv")
        train_df = pd.read_csv(train_path)

        votes = train_df[TARGET_COLS].astype(float)
        overall_dist = votes.sum(axis=0) / votes.values.sum()
        overall_dist = overall_dist / overall_dist.sum()  # ensure sums to 1

        blend_weight = 0.0  # previously 0.5

        test_path = os.path.join(DATA_PATH, "test.csv")
        test_df = pd.read_csv(test_path)

        per_eeg = train_df.groupby("eeg_id")[TARGET_COLS].sum()
        per_eeg_sum = per_eeg.sum(axis=1).replace(0, np.nan)
        per_eeg = per_eeg.div(per_eeg_sum, axis=0).fillna(0)
        per_eeg_aligned = per_eeg.reindex(test_df["eeg_id"]).fillna(overall_dist)

        blended_probs = per_eeg_aligned * blend_weight + overall_dist * (
            1 - blend_weight
        )

        submission = pd.DataFrame(blended_probs.values, columns=TARGET_COLS)
        submission.insert(0, "eeg_id", test_df["eeg_id"])

        return _smooth_and_normalize(submission)




## === cell 3
submission_df = merge_predictions(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v2"],
    weights=[0.3, 0.7],
)




## === cell 4
output_file = os.path.join(OUT_PATH, "submission.csv")
submission_df.to_csv(output_file, index=False)
print(f"Submission written to {output_file}")
