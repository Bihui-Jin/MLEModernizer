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

0.3626664036379633

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I replace the failing import and broken merging logic with a self‑contained implementation: define the target columns directly, load any existing fold prediction files, average them safely, and if none are present fall back to a uniform‑probability baseline using the test set IDs. Finally the script writes a correctly formatted `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑probability fallback with a simple class‑prior baseline computed from the training votes. By averaging the vote counts in *train.csv* we obtain a more realistic probability vector, which should lower the KL‑divergence (the competition metric) and move the score toward the target while keeping the original merging logic unchanged.'
- What this solution (achieved 1.68479) has done: 'I add a small helper that computes per‑patient class‑vote priors from the training data and use them when generating the fallback predictions for the test set. If a test row’s patient_id appears in the training data, we now predict that patient’s empirical vote distribution; otherwise we fall back to the overall class priors. This keeps the original merging logic unchanged while giving each row a more informative probability vector, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.1548) has done: 'I add a per‑eeg ID prior (which is often more specific than the patient prior) and apply a tiny Laplace smoothing before normalising. The fallback now checks for an exact eeg_id match, then patient_id, and finally falls back to the overall class prior, giving more informative probabilities and expectedly lowering the KL‐divergence toward the target score.'
- What this solution (achieved 0.7644) has done: 'I improve the fallback‑prediction step by smoothing the per‑eeg and per‑patient priors. Instead of using the raw empirical distributions (which can be noisy for IDs with few records), I compute how many training rows each eeg_id and patient_id contributed and blend their specific prior with the overall class prior using a simple shrinkage formula (weight = count / (count + 5)). This keeps the original logic—averaging any real fold predictions when they exist—while giving more robust probability estimates, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.80881) has done: 'I increase the shrinkage strength when blending specific per‑eeg or per‑patient priors toward the overall class prior. By raising the default `k` parameter (e.g., to 20) the final predictions rely more on the overall prior, which smooths noisy id‑level estimates and should lower the KL‑divergence, moving the score closer to the target while keeping the core logic unchanged.'
- What this solution (achieved 0.77447) has done: 'The update reduces the heavy shrinkage (k = 20) to a milder value (k = 5) and applies a stronger Laplace smoothing (+1) when computing overall, per‑patient, and per‑eeg priors. These changes give the fallback predictions more informative, class‑specific information while still regularising noisy counts, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.77866) has done: 'I reduced the noise from very low‑count per‑eeg and per‑patient priors by skipping the specific prior when its observation count is below a small threshold (3). This keeps the overall class prior for uncertain IDs, which should lower the KL‑divergence and move the score closer to the target while preserving the original blending logic.'
- What this solution (achieved 0.80401) has done: 'I tighten the fallback prediction logic by using a slightly stronger Laplace smoothing (+5 instead of +1) and by blending both per‑eeg and per‑patient priors sequentially toward the overall class prior (with the same shrinkage k = 5). This keeps the original averaging of real fold predictions unchanged, but gives more robust, less noisy priors when no model files exist, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOTS = [
    Path("./data/hms-harmful-brain-activity-classification"),
    Path("./data/input/hms-harmful-brain-activity-classification"),
    Path("./data"),
]

OUT_PATH = "./working"
os.makedirs(OUT_PATH, exist_ok=True)

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

SMOOTH = 1.0  # Laplace smoothing for priors


def locate_file(filename: str) -> Path:
    """Search known data roots for a given filename."""
    for root in DATA_ROOTS:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    for path in Path(".").rglob(filename):
        if path.is_file():
            return path
    raise FileNotFoundError(f"{filename} not found in any known data directory.")


def load_fold_predictions(fold_ids, version_tag):
    """Read stored prediction CSVs for the given folds and version."""
    preds = []
    for fold in fold_ids:
        file_path = Path(OUT_PATH) / f"submission_fold{fold}_{version_tag}.csv"
        if file_path.is_file():
            df = pd.read_csv(file_path)
            if set(TARGET_COLS).issubset(df.columns):
                preds.append((df["eeg_id"].values, df[TARGET_COLS].values))
    return preds


def compute_class_priors():
    """Overall class prior from training votes with Laplace smoothing."""
    train_path = locate_file("train.csv")
    train_df = pd.read_csv(train_path)
    vote_sums = train_df[TARGET_COLS].sum(axis=0).astype(float) + SMOOTH
    priors = vote_sums / vote_sums.sum()
    return priors


def compute_patient_priors(train_df):
    """Per‑patient class priors with smoothing."""
    patient_votes = train_df.groupby("patient_id")[TARGET_COLS].sum()
    smoothed = patient_votes + SMOOTH
    priors = smoothed.div(smoothed.sum(axis=1), axis=0)
    counts = train_df.groupby("patient_id").size()
    priors_dict = {pid: row.values for pid, row in priors.iterrows()}
    counts_dict = counts.to_dict()
    return priors_dict, counts_dict


def compute_eeg_priors(train_df):
    """Per‑eeg_id class priors with smoothing."""
    eeg_votes = train_df.groupby("eeg_id")[TARGET_COLS].sum()
    smoothed = eeg_votes + SMOOTH
    priors = smoothed.div(smoothed.sum(axis=1), axis=0)
    counts = train_df.groupby("eeg_id").size()
    priors_dict = {eid: row.values for eid, row in priors.iterrows()}
    counts_dict = counts.to_dict()
    return priors_dict, counts_dict


def blend_with_overall(specific_vec, specific_count, overall_vec, k=2, min_count=1):
    """
    Shrink a specific prior toward the overall prior.
    weight = specific_count / (specific_count + k)
    If the specific count is below `min_count`, return the overall prior.
    """
    if specific_vec is None or specific_count is None:
        return overall_vec
    if specific_count < min_count:
        return overall_vec
    weight = specific_count / (specific_count + k)
    return weight * specific_vec + (1 - weight) * overall_vec


def fallback_probs(
    eid, pid, overall, eeg_dict, eeg_counts, patient_dict, patient_counts
):
    """
    Generate a baseline probability vector.
    Simplified to always return the overall prior, avoiding noisy per‑id estimates.
    """
    return overall


def merge_predictions(fold_ids, versions):
    """Average predictions across folds/versions; fall back to priors if none found."""
    all_preds = []
    eeg_ids = None

    for version in versions:
        fold_preds = load_fold_predictions(fold_ids, version)
        for ids, prob in fold_preds:
            if eeg_ids is None:
                eeg_ids = ids
            all_preds.append(prob)

    if not all_preds:
        test_path = locate_file("test.csv")
        train_path = locate_file("train.csv")
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path)

        overall_priors = compute_class_priors()
        patient_priors_dict, patient_counts = compute_patient_priors(train_df)
        eeg_priors_dict, eeg_counts = compute_eeg_priors(train_df)

        merged = []
        for eid, pid in zip(test_df["eeg_id"].values, test_df["patient_id"].values):
            probs = fallback_probs(
                eid,
                pid,
                overall_priors,
                eeg_priors_dict,
                eeg_counts,
                patient_priors_dict,
                patient_counts,
            )
            merged.append(probs)

        merged = np.vstack(merged)  # (n_samples, n_classes)
        eeg_ids = test_df["eeg_id"].values  # Ensure IDs are present for the submission
    else:
        stacked = np.stack(all_preds, axis=0)  # (n_models, n_samples, n_classes)
        merged = np.mean(stacked, axis=0)  # (n_samples, n_classes)

    merged = merged / merged.sum(axis=1, keepdims=True)

    submission = pd.DataFrame(eeg_ids, columns=["eeg_id"])
    for i, col in enumerate(TARGET_COLS):
        submission[col] = merged[:, i]
    return submission




## === cell 1
final_submission = merge_predictions(fold_ids=[0, 1, 2, 3, 4], versions=["v1"])

output_path = f"{OUT_PATH}/submission.csv"
final_submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
