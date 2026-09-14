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
seaborn==0.12.2
sklearn-pandas==2.2.0
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

1.030545

# 6. Current score

1.44021

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'Your code doesn’t yield a score because it currently spends most runtime reading every test parquet to compute NaN ratios and shapes, which is very likely to time out before `submission.csv` is written. I keep your “constant mean vote ratio” core logic, but compute the needed NaN/shape “hypotheses” from a small deterministic sample of files (plus cheap directory/file-count checks) so the notebook finishes quickly and still selects the same branch in practice. I also add a safety normalization step so the probabilities always sum to 1 (required for a valid KL-divergence submission) and ensure the submission rows align exactly to `sample_submission.csv` order.'
- What this solution (achieved 1.41937) has done: 'Your current solution is a constant-prior submission; to move the KL score down toward the target with minimal change, the safest improvement is to compute that prior in a way that better matches the test distribution without changing the modeling approach. I keep the “predict one fixed probability vector for all rows” core logic, but compute the prior at the correct unit (per `eeg_id`, not per row) to avoid overweighting EEGs that have many overlapping labeled windows in `train.csv`. I also add a tiny probability floor + renormalization to prevent any accidental zeros (which can cause extreme KL penalties), while preserving the same semantics (a single global probability vector). The script still runs fast and writes a valid `submission.csv` with rows aligned to `sample_submission.csv`.'
- What this solution (achieved 1.48867) has done: 'Your current submission is a single global prior, so the safest way to improve KL (lower is better) without changing core logic is to compute that prior in a way that better matches the evaluation target distribution. I keep the “one fixed probability vector for all test rows” approach, but I (1) compute the prior from *per-eeg normalized vote distributions* (so EEGs with more overlapping windows don’t dominate) and (2) blend it slightly with a raw per-eeg total prior for stability. I also keep your existing hypothesis branches intact and only apply this improved prior in the fallback `else` branch, plus retain the epsilon+renormalization to guarantee valid probabilities.'
- What this solution (achieved 1.4214) has done: 'I keep your “single global probability vector for all test rows” core logic, but compute that prior in a slightly more label-aligned way for KL: use a per-`label_id` normalized vote distribution (each label set contributes equally), which reduces the distortion from overlapping windows and repeated EEG segments. To avoid regressions and keep behavior stable, I blend this per-label prior with your existing per-eeg equal-weight prior and per-eeg total prior (small, controlled change). I also keep your epsilon floor + renormalization so the submission is always valid for KL (no zeros; rows sum to 1). No changes to the hypothesis branches or submission schema/paths.'
- What this solution (achieved 1.43221) has done: 'To move your KL score down toward the target while preserving the “single global probability vector for all rows” core logic, I keep your hypothesis branches intact and only adjust the fallback prior computation. Specifically, I compute the global prior at the correct unit for this competition (per `eeg_id`) but based on *per-label* vote distributions averaged within each EEG, then average across EEGs; this reduces distortion from overlapping windows and repeated label sets. I blend this new per-eeg-per-label prior with your existing priors using small weights so the change is controlled and stable (aiming to reduce the gap from 1.4214 toward 1.030545 without drastic behavior). I keep the same safety epsilon+renormalization so the submission always sums to 1 and remains valid.'
- What this solution (achieved 1.42968) has done: 'Your current score (1.43221, lower-is-better) is still far above the target (1.030545), so we should cautiously improve KL without changing your “single global probability vector for all rows” core logic. The most directly relevant and minimal lever is better calibration of that single prior to match the evaluation target: use *per-patient* equal-weight averaging of vote distributions (since labels are correlated within patient and test is grouped by patient_id), then blend it lightly with your existing priors for stability. This keeps the exact same prediction semantics (one fixed probability vector for every test eeg_id), but typically reduces KL versus purely per-row/per-eeg priors. I also keep your safety epsilon+renormalization and submission alignment unchanged.'
- What this solution (achieved 1.42768) has done: 'I keep your “single global probability vector for all rows” approach and your hypothesis branches intact, but make one controlled change to the fallback prior: compute a prior based on *test patient_id mixture weights* (using train’s per-patient class distributions, then averaging them weighted by how often each patient appears in test). This is still a single fixed vector (so identical prediction semantics), but it better matches the evaluation population and typically reduces KL compared to an unweighted global prior. I then blend this new prior in with a small weight so the change is stable and incremental, while keeping your epsilon+renormalization and submission alignment exactly the same.'
- What this solution (achieved 1.42937) has done: 'To move your KL score down (lower is better) toward 1.030545 while keeping the “single global probability vector for all rows” core logic unchanged, I make the prior slightly better matched to what’s evaluated: compute it at the `eeg_id` unit using per-label normalized vote distributions (reduces overlap distortion), and then mix it using *test EEG mixture weights* derived from `test.patient_id` (more directly aligned than weighting by test patient row counts). This is a minimal change confined to the fallback `else` branch; all hypothesis branches and the constant-probability submission semantics remain identical. I also keep your epsilon floor + renormalization, and add a tiny guard so the mixture weights always sum correctly even if some test patients are unseen in train (stability, not optimization).'
- What this solution (achieved 1.45475) has done: 'Your current submission is a single fixed probability vector for all test rows, so the most direct way to reduce KL (lower is better) without changing core logic is to better calibrate that one vector using information available in `test.csv`. I keep your hypothesis branches and the “constant prior for every row” semantics, but in the fallback `else` branch I replace the ad-hoc blend with a *test-marginalized prior*: compute per-patient vote distributions from train, then average them weighted by the test patient mix, and separately compute a per-eeg prior; finally blend these two with a small smoothing toward the global prior for stability. I also keep the epsilon floor + renormalization to ensure valid probabilities (no zeros; sums to 1), which avoids catastrophic KL penalties. This should move the score downward toward the target while remaining minimal and fast.'
- What this solution (achieved 1.44563) has done: 'Your current solution is already a constant-prior submission, so the only safe way to move the KL score down toward the target (lower is better) without changing core semantics is to slightly recalibrate that single prior vector. I keep all hypothesis branches intact and only adjust the fallback `else` prior by adding a small, label-unit-correct prior (per `label_id` normalized vote distributions) blended in with a low weight, which typically reduces distortion from overlapping windows and improves KL a bit. I also add a tiny amount of smoothing toward uniform (very small weight) to reduce overconfidence (often helpful for KL) while still keeping a single fixed vector. Submission writing, alignment to `sample_submission.csv`, and the epsilon+renormalization safety remain unchanged.'
- What this solution (achieved 1.4462) has done: 'To move your KL score downward toward the target while preserving the “single fixed probability vector for all test rows” core logic, I only adjust the fallback `else`-branch prior calibration (the hypothesis branches remain untouched). Specifically, I add a small, controlled blend term that estimates a prior using the test distribution over `spectrogram_id` mapped to train `spectrogram_id` class distributions, then mix it in at a low weight by slightly reducing the existing patient-weighted component (net change is small but often helps because test is keyed on `spectrogram_id`/`eeg_id` pairs). I keep your existing epsilon floor + renormalization (mandatory for valid KL submissions) and keep submission alignment exactly matching `sample_submission.csv`. Runtime stays well under the limit because this adds only groupby operations on `train.csv`/`test.csv` (no extra parquet reads).'
- What this solution (achieved 1.45912) has done: 'Your current approach is a single global prior for all test rows, so the only safe way to reduce KL (lower is better) without changing core semantics is to slightly recalibrate that one prior vector using train statistics in a way that better matches the evaluation unit. I keep all hypothesis branches exactly as-is and only adjust the fallback `else` prior by adding a small per-`eeg_id` **median** vote-distribution prior (more robust than the mean to skew/outliers), blended in by slightly reducing the existing uniform smoothing and redistributing weights minimally. I also keep the epsilon floor + renormalization to guarantee valid probabilities and maintain exact submission alignment with `sample_submission.csv`. This should nudge the score downward toward the target with minimal risk and no parquet-reading changes beyond your existing sampled checks.'
- What this solution (achieved 1.447) has done: 'Your current score (1.45912, lower-is-better) is still far above the target (1.030545), so we should make a small, safe calibration-only change while keeping the “single fixed probability vector for all rows” core logic unchanged. The most direct lever for KL is to reduce overconfidence: I slightly increase the uniform-smoothing weight and make the prior-building weights sum to 1 exactly (so the smoothing change behaves predictably). I also replace the robust-but-harsh per-eeg median component with a clipped mean (winsorized) at a small weight; this keeps the same semantics (still one global prior) but avoids spiky class probabilities that can hurt KL when wrong. All hypothesis branches, parquet sampling, submission alignment, and epsilon+renormalization remain intact.'
- What this solution (achieved 1.44554) has done: 'We keep your constant-prior submission logic and all hypothesis branches unchanged, and only make a small calibration tweak in the fallback `else` branch to move KL downward toward the target. Specifically, we reduce the contribution of the more “test-peeking” components (patient/spec weighted) a bit and slightly increase the stabilizing components (global total + per-label + mild uniform smoothing) to reduce overconfident class skews that tend to hurt KL. This is a minimal weight/smoothing-only adjustment, preserving identical prediction semantics (one fixed probability vector for every test row). We keep the existing epsilon floor + renormalization and submission alignment so the CSV remains valid.'
- What this solution (achieved 1.44021) has done: 'Your current score is worse than the target (1.44554 vs 1.030545; lower is better), so we should cautiously lower KL without changing the “single fixed probability vector for all rows” semantics. The smallest reliable lever for KL with a constant prior is *calibration*: reduce overconfident/skewed priors by slightly increasing uniform smoothing and using a safer Bayesian/Dirichlet-style prior computed from total votes with a small pseudocount (prevents extreme probabilities). I keep all hypothesis branches intact and only adjust the fallback `else` prior blend by (1) adding a tiny pseudocount-smoothed `prior_dirichlet_total` component and (2) slightly increasing `smooth` while rebalancing weights minimally. Submission alignment and epsilon+renormalization remain unchanged to guarantee a valid CSV.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
SAMPLE_SUB = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sub = pd.read_csv(SAMPLE_SUB)



## === cell 1
train_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
test_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
test_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

train_spectrogram_files = sorted(os.listdir(train_spectrogram_dir))
test_spectrogram_files = sorted(os.listdir(test_spectrogram_dir))
train_eeg_files = sorted(os.listdir(train_eeg_dir))
test_eeg_files = sorted(os.listdir(test_eeg_dir))




## === cell 2
def get_files_info_sample(files, file_dir, sample_size=20, seed=0):
    if len(files) == 0:
        return np.array([np.nan]), np.array([[np.nan, np.nan]])
    rng = np.random.default_rng(seed)
    k = min(sample_size, len(files))
    idx = rng.choice(len(files), size=k, replace=False)

    nan_ratio = []
    shapes = []
    for i in idx:
        fp = os.path.join(file_dir, files[i])
        df = pd.read_parquet(fp)
        arr = df.to_numpy()
        nan_ratio.append(np.isnan(arr).sum() / arr.size)
        shapes.append(arr.shape)
    return np.array(nan_ratio), np.array(shapes)




## === cell 3
test_spectrogram_nan_ratio, test_spectrogram_shapes = get_files_info_sample(
    test_spectrogram_files, test_spectrogram_dir, sample_size=25, seed=0
)
test_eeg_nan_ratio, test_eeg_shapes = get_files_info_sample(
    test_eeg_files, test_eeg_dir, sample_size=25, seed=1
)



## === cell 4
hypothesis0 = False
hypothesis1 = False

mean_nan = float(np.nanmean(test_spectrogram_nan_ratio))

if mean_nan < 0.05:
    hypothesis0 = True
    hypothesis1 = True
if mean_nan < 0.03:
    hypothesis0 = True
    hypothesis1 = False
if mean_nan < 0.01:
    hypothesis0 = False
    hypothesis1 = True



## === cell 5
hypotheses = []
hypotheses.append(len(test.eeg_id.unique()) == len(test))
hypotheses.append(len(test.spectrogram_id.unique()) == len(test))
hypotheses.append(len(test.patient_id.unique()) != len(test))
hypotheses.append(len(test_eeg_files) == len(test))
hypotheses.append(len(test_spectrogram_files) == len(test))
hypotheses.append(
    len(train.spectrogram_id.unique()) / len(train.patient_id.unique()) < 6
)
hypotheses.append(
    len(train.spectrogram_id.unique()) / len(train.patient_id.unique()) > 5
)
hypotheses.append(
    len(pd.concat([train[["eeg_id"]], test[["eeg_id"]]]).eeg_id.unique())
    == (len(train.eeg_id.unique()) + len(test.eeg_id.unique()))
)
hypotheses.append(
    len(
        pd.concat(
            [train[["spectrogram_id"]], test[["spectrogram_id"]]]
        ).spectrogram_id.unique()
    )
    == (len(train.spectrogram_id.unique()) + len(test.spectrogram_id.unique()))
)
hypotheses.append(
    len(pd.concat([train[["patient_id"]], test[["patient_id"]]]).patient_id.unique())
    == (len(train.patient_id.unique()) + len(test.patient_id.unique()))
)
hypotheses.append(
    np.array_equal(np.unique(test_eeg_shapes, axis=0), np.array([[10000, 20]]))
    or np.array_equal(np.unique(test_eeg_shapes, axis=0), np.array([[20, 10000]]))
    or np.array_equal(np.unique(test_eeg_shapes), np.array([20, 10000]))
)
hypotheses.append(
    np.array_equal(np.unique(test_spectrogram_shapes, axis=0), np.array([[300, 401]]))
    or np.array_equal(np.unique(test_spectrogram_shapes), np.array([300, 401]))
)

hypotheses = all(hypotheses)



## === cell 6
vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
targets = vote_cols

if (hypothesis0 == True) & (hypothesis1 == True) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.196002,
        "gpd_vote": 0.156386,
        "lrda_vote": 0.155805,
        "other_vote": 0.17610,
        "grda_vote": 0.17660,
        "lpd_vote": 0.139101,
    }
elif (hypothesis0 == True) & (hypothesis1 == False) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.174031,
        "lpd_vote": 0.112700,
        "gpd_vote": 0.090854,
        "lrda_vote": 0.071484,
        "grda_vote": 0.136408,
        "other_vote": 0.414523,
    }
elif (hypothesis0 == False) & (hypothesis1 == True) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.152810,
        "lpd_vote": 0.142456,
        "gpd_vote": 0.104062,
        "lrda_vote": 0.065407,
        "grda_vote": 0.114851,
        "other_vote": 0.420414,
    }
elif (hypothesis0 == False) & (hypothesis1 == False) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.310718,
        "lpd_vote": 0.046279,
        "gpd_vote": 0.051885,
        "lrda_vote": 0.081796,
        "grda_vote": 0.231471,
        "other_vote": 0.277851,
    }
else:
    per_patient_votes = (
        train.groupby("patient_id", sort=False)[vote_cols].sum().astype(np.float64)
    )
    per_patient_tot = per_patient_votes.sum(axis=1).to_numpy()
    valid_pat = per_patient_tot > 0
    per_patient_probs = per_patient_votes.to_numpy()
    per_patient_probs[valid_pat] = (
        per_patient_probs[valid_pat] / per_patient_tot[valid_pat, None]
    )

    train_pat_probs = pd.DataFrame(
        per_patient_probs[valid_pat],
        index=per_patient_votes.index[valid_pat],
        columns=vote_cols,
    )

    test_pat_eeg_counts = (
        test[["patient_id", "eeg_id"]].drop_duplicates()["patient_id"].value_counts()
    )
    common_pats = test_pat_eeg_counts.index.intersection(train_pat_probs.index)

    if len(common_pats) > 0:
        w = test_pat_eeg_counts.loc[common_pats].to_numpy(dtype=np.float64)
        wsum = w.sum()
        if wsum <= 0:
            prior_test_weighted_patient = train_pat_probs.to_numpy(
                dtype=np.float64
            ).mean(axis=0)
        else:
            w = w / wsum
            prior_test_weighted_patient = (
                train_pat_probs.loc[common_pats].to_numpy(dtype=np.float64) * w[:, None]
            ).sum(axis=0)
    else:
        prior_test_weighted_patient = train_pat_probs.to_numpy(dtype=np.float64).mean(
            axis=0
        )

    prior_test_weighted_patient = (
        prior_test_weighted_patient / prior_test_weighted_patient.sum()
    )

    per_eeg_votes = (
        train.groupby("eeg_id", sort=False)[vote_cols].sum().astype(np.float64)
    )
    per_eeg_tot = per_eeg_votes.sum(axis=1).to_numpy()
    valid_eeg = per_eeg_tot > 0
    per_eeg_probs = per_eeg_votes.to_numpy()
    per_eeg_probs[valid_eeg] = per_eeg_probs[valid_eeg] / per_eeg_tot[valid_eeg, None]
    prior_equal_eeg = per_eeg_probs[valid_eeg].mean(axis=0)
    prior_equal_eeg = prior_equal_eeg / prior_equal_eeg.sum()

    if valid_eeg.sum() > 0:
        q_lo = np.quantile(per_eeg_probs[valid_eeg], 0.10, axis=0)
        q_hi = np.quantile(per_eeg_probs[valid_eeg], 0.90, axis=0)
        clipped = np.clip(per_eeg_probs[valid_eeg], q_lo[None, :], q_hi[None, :])
        prior_clipmean_eeg = clipped.mean(axis=0)
        prior_clipmean_eeg = prior_clipmean_eeg / prior_clipmean_eeg.sum()
    else:
        prior_clipmean_eeg = prior_equal_eeg.copy()

    prior_total = per_eeg_votes.sum(axis=0).to_numpy()
    prior_total = prior_total / prior_total.sum()

    per_label_votes = (
        train.groupby("label_id", sort=False)[vote_cols].sum().astype(np.float64)
    )
    per_label_tot = per_label_votes.sum(axis=1).to_numpy()
    valid_label = per_label_tot > 0
    per_label_probs = per_label_votes.to_numpy()
    per_label_probs[valid_label] = (
        per_label_probs[valid_label] / per_label_tot[valid_label, None]
    )
    prior_equal_label = per_label_probs[valid_label].mean(axis=0)
    prior_equal_label = prior_equal_label / prior_equal_label.sum()

    per_spec_votes = (
        train.groupby("spectrogram_id", sort=False)[vote_cols].sum().astype(np.float64)
    )
    per_spec_tot = per_spec_votes.sum(axis=1).to_numpy()
    valid_spec = per_spec_tot > 0
    per_spec_probs = per_spec_votes.to_numpy()
    per_spec_probs[valid_spec] = (
        per_spec_probs[valid_spec] / per_spec_tot[valid_spec, None]
    )

    train_spec_probs = pd.DataFrame(
        per_spec_probs[valid_spec],
        index=per_spec_votes.index[valid_spec],
        columns=vote_cols,
    )

    test_spec_counts = test["spectrogram_id"].value_counts()
    common_specs = test_spec_counts.index.intersection(train_spec_probs.index)
    if len(common_specs) > 0:
        ws = test_spec_counts.loc[common_specs].to_numpy(dtype=np.float64)
        wssum = ws.sum()
        if wssum <= 0:
            prior_test_weighted_spec = train_spec_probs.to_numpy(dtype=np.float64).mean(
                axis=0
            )
        else:
            ws = ws / wssum
            prior_test_weighted_spec = (
                train_spec_probs.loc[common_specs].to_numpy(dtype=np.float64)
                * ws[:, None]
            ).sum(axis=0)
    else:
        prior_test_weighted_spec = train_spec_probs.to_numpy(dtype=np.float64).mean(
            axis=0
        )
    prior_test_weighted_spec = prior_test_weighted_spec / prior_test_weighted_spec.sum()

    total_votes = train[vote_cols].sum(axis=0).to_numpy(dtype=np.float64)
    alpha = 1.0  # small pseudocount per class
    prior_dirichlet_total = (total_votes + alpha) / (
        total_votes.sum() + alpha * len(vote_cols)
    )

    w_test_pat = 0.52
    w_spec = 0.02
    w_eeg = 0.23
    w_total = 0.10
    w_label = 0.06
    w_clip = 0.02
    w_dir = 0.05

    wsum = w_test_pat + w_spec + w_eeg + w_total + w_label + w_clip + w_dir
    w_test_pat /= wsum
    w_spec /= wsum
    w_eeg /= wsum
    w_total /= wsum
    w_label /= wsum
    w_clip /= wsum
    w_dir /= wsum

    prior = (
        w_test_pat * prior_test_weighted_patient
        + w_spec * prior_test_weighted_spec
        + w_eeg * prior_equal_eeg
        + w_total * prior_total
        + w_label * prior_equal_label
        + w_clip * prior_clipmean_eeg
        + w_dir * prior_dirichlet_total
    )
    prior = prior / prior.sum()

    u = np.full(len(vote_cols), 1.0 / len(vote_cols), dtype=np.float64)
    smooth = 0.06
    prior = (1.0 - smooth) * prior + smooth * u
    prior = prior / prior.sum()

    mean_vote_ratio = dict(zip(vote_cols, prior))

for t in targets:
    sub[t] = float(mean_vote_ratio[t])

eps = 1e-6
vals = sub[targets].to_numpy(dtype=np.float64)
vals = np.clip(vals, eps, None)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[targets] = vals
sub = sub[["eeg_id"] + targets]



## === cell 7
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", sub.shape)
print(sub.head())
print("Row sum stats:", sub[targets].sum(axis=1).min(), sub[targets].sum(axis=1).max())
