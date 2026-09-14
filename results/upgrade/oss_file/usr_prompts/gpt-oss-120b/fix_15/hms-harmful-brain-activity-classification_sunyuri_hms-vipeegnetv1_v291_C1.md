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

0.3122666935084653

# 6. Current score

1.00109

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fixed the protobuf import issue, removed the faulty TensorFlow model construction, and replaced the whole prediction pipeline with a simple class‑frequency baseline that safely creates a valid “submission.csv”. This eliminates the previous runtime errors, guarantees the required columns and row‑sum‑to‑one constraint, and provides a reasonable score without altering the core competition logic.'
- What this solution (achieved 0.76636) has done: 'Implemented a fix for the Na N‑handling when creating patient‑specific priors and added a light smoothing blend with the global prior to improve calibration.  
The script now correctly builds the `patient_priors` DataFrame, normalizes any missing values, and blends each patient distribution 90 % with the overall class prior (10 %). This resolves the `ValueError` and yields a valid `submission.csv` while nudging the KL‑divergence score toward the target.'
- What this solution (achieved 0.84355) has done: 'I adjust the patient‑specific prior blending so that patients with few annotated samples rely more on the robust global class prior. By weighting each patient’s prior proportionally to its total vote count (plus a smoothing constant), the predictions become better calibrated, which should lower the KL‑divergence score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.03554) has done: 'I increase the smoothing constant used for patient‑specific weighting (SMOOTH_K) from 200 to 2000. This reduces the influence of noisy patient priors and moves predictions closer to the robust global class prior, which should lower the KL‑divergence (the metric is lower‑is‑better) and bring the score nearer the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.83902) has done: 'I lower the smoothing constant (SMOOTH_K) from 2000 to 200 so patient‑specific priors have more influence, and remove the artificial upper‑clip on the weight (allowing it to reach 1). This small adjustment keeps the overall pipeline unchanged while giving more personalized probability estimates, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.75957) has done: 'I lower the smoothing constant **SMOOTH_K** from 200 to 20 so that each patient’s prior receives a larger weight in the blended prediction.  This makes the model rely more on patient‑specific vote distributions (which are informative) and less on the generic global prior, which is expected to reduce the KL‑divergence and move the score closer to the target (lower is better). The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.75957) has done: 'I add an EEG‑level prior computed from the training vote aggregates and use it preferentially when predicting test rows (falling back to the patient prior then the global prior). This gives more specific probability estimates without altering the overall blending logic, and should move the KL‑divergence lower toward the target score.'
- What this solution (achieved 1.39779) has done: 'We replace the patient‑ and EEG‑specific priors with the overall class prior for every test record.  This removes noisy, over‑confident personalized distributions and yields a more uniformly calibrated prediction, which is expected to lower the KL‑divergence (the metric is lower‑is‑better) and move the score closer to the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.75726) has done: 'I fixed the length‑mismatch error when blending patient‑specific priors by computing the global‑prior contribution with NumPy broadcasting instead of trying to multiply a Series by a mismatched array. I also lowered the smoothing constant `SMOOTH_K` to 5 so patient priors have more influence, which should move the KL‑divergence closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.91406) has done: 'I increase the smoothing constant `SMOOTH_K` from 5 to 200 so that patient‑ and EEG‑specific priors are blended more heavily with the robust global class prior. This reduces reliance on noisy personalized distributions, yielding better‑calibrated probabilities and moving the KL‑divergence lower toward the target score while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.00109) has done: 'I slightly increase the smoothing constant so patient‑ and EEG‑specific priors rely more on the robust global prior, and I add a tiny additive smoothing (epsilon) to every blended probability before normalising. This prevents zero probabilities that hurt KL‑divergence and moves the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os, warnings, numpy as np, pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PLATFORM = "kaggle"  # change to "local" when running locally
if PLATFORM == "local":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

warnings.filterwarnings("ignore")
np.random.seed(2024)

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
train_df = pd.read_csv(train_path)

TARGETS = train_df.columns[-6:]

vote_sums = train_df[TARGETS].sum(axis=1).replace(0, np.nan)
proportions = train_df[TARGETS].div(vote_sums, axis=0)
global_class_prior = proportions.mean().fillna(1.0 / len(TARGETS)).values  # 1‑D ndarray
global_series = pd.Series(global_class_prior, index=TARGETS)

patient_votes = train_df.groupby("patient_id")[TARGETS].sum()
patient_vote_sums = patient_votes.sum(axis=1).replace(0, np.nan)
patient_priors_raw = patient_votes.div(patient_vote_sums, axis=0).fillna(global_series)

patient_total_votes = patient_votes.sum(axis=1)  # series indexed by patient_id
SMOOTH_K = 500.0  # increased smoothing to rely more on the global prior
patient_weights = patient_total_votes / (
    patient_total_votes + SMOOTH_K
)  # 0‑1 weight per patient

patient_priors = (
    patient_priors_raw.multiply(patient_weights, axis=0)
    + (1 - patient_weights).values[:, None] * global_series.values
)

eeg_votes = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_vote_sums = eeg_votes.sum(axis=1).replace(0, np.nan)
eeg_priors = eeg_votes.div(eeg_vote_sums, axis=0).fillna(global_series)

eeg_total_votes = eeg_votes.sum(axis=1)
eeg_weights = eeg_total_votes / (eeg_total_votes + SMOOTH_K)  # weight per EEG

test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test_df = pd.read_csv(test_path)

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})

epsilon = 1e-5  # tiny additive smoothing to avoid zero probabilities

for i in range(submission.shape[0]):
    pid = test_df.at[i, "patient_id"]
    eid = test_df.at[i, "eeg_id"]

    if eid in eeg_priors.index:
        eeg_prior = eeg_priors.loc[eid]
        w_eeg = eeg_weights.get(eid, 0.0)
    else:
        eeg_prior = global_series
        w_eeg = 0.0

    if pid in patient_priors.index:
        patient_prior = patient_priors.loc[pid]
        w_pat = patient_weights.get(pid, 0.0)
    else:
        patient_prior = global_series
        w_pat = 0.0

    blended = w_eeg * eeg_prior + (1 - w_eeg) * (
        w_pat * patient_prior + (1 - w_pat) * global_series
    )
    blended = blended + epsilon
    blended = blended / blended.sum()
    submission.loc[i, TARGETS] = blended.values

row_sums = submission[TARGETS].sum(axis=1)
if not np.allclose(row_sums, 1.0):
    submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with shape {submission.shape}")
