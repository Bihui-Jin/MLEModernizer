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
timm==1.0.19
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

0.6090752333501809

# 6. Current score

1.2503

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41564) has done: 'The fix adds a fallback when the pretrained weights file is missing, switches the EfficientNet model to use ImageNet‑pretrained weights, and adjusts the inference loop to continue with the randomly‑initialized model. This prevents the `FileNotFoundError` and ensures a valid prediction array, allowing the script to produce a correctly‑shaped `submission.csv` that satisfies the competition’s format.'
- What this solution (achieved 1.41937) has done: 'The fix defines the missing variables, loads the training data to compute realistic class‑probability priors, and generates a proper Kaggle submission where each row’s probabilities sum to 1. By using the aggregated vote distribution from the training set instead of undefined model weights, the script now runs end‑to‑end and writes a valid `submission.csv` file, moving the score toward the target without altering the original model‑centric logic.'
- What this solution (achieved 1.41937) has done: 'I add a per‑eeg_id probability lookup: for any test `eeg_id` that also appears in the training set we use its observed vote distribution (normalized), otherwise we fall back to the overall class priors. After filling missing values we renormalize each row so the probabilities sum to 1, which keeps the submission format valid and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.68197) has done: 'I fix the `fillna` misuse by assigning the patient‑specific probabilities in one block rather than column‑by‑column with a NumPy array, which caused the TypeError. This ensures missing rows are correctly filled, after which any remaining NaNs are replaced by the overall class priors and rows are renormalized so every probability vector sums to 1, producing a valid submission file.'
- What this solution (achieved 0.77556) has done: 'I keep the overall prior‑lookup logic but blend the per‑eeg / patient probabilities with the overall class priors.  
This slight smoothing reduces over‑confident predictions for rare IDs, which typically lowers the KL‑divergence and moves the score closer to the target. The change is confined to the post‑processing section and retains the original workflow.'
- What this solution (achieved 0.77556) has done: 'I keep the overall lookup‑based approach but improve the blending step so that only rows that actually have a specific per‑eeg or per‑patient estimate are smoothed with the overall class priors. Rows that lack any specific information use the class prior alone, which reduces over‑confident predictions on unseen IDs and is expected to lower the KL‑divergence toward the target. The change is limited to the post‑processing section and preserves all core logic.'
- What this solution (achieved 0.90119) has done: 'I slightly increase the temperature smoothing (making the global class prior flatter) and reduce the blending weight `ALPHA` so the submission relies more on the overall prior rather than the specific per‑eeg/patient estimates. These minimal tweaks keep the core logic unchanged while expectedly lowering the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 1.13178) has done: 'I lower the KL‑divergence by making the global class prior flatter (increase `TEMPERATURE`) and by relying more on that prior instead of the per‑eeg/patient estimates (decrease `ALPHA`). These small parameter tweaks keep the overall workflow unchanged while moving the score closer to the target.'
- What this solution (achieved 1.2503) has done: 'I make three lightweight tweaks to the post‑processing: raise the temperature to flatten the global prior, lower the ALPHA weight so we rely more on that smoother prior, and add a tiny epsilon smoothing before renormalising to avoid zero probabilities. These changes keep the original workflow intact while expectedly lowering the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os, gc, random, numpy as np, pandas as pd, torch


class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False
    TEMPERATURE = 10.0


random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)




## === cell 1
DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_counts = train_df[vote_cols].sum()
class_probs = class_counts / class_counts.sum()
class_probs = np.power(class_probs.values.astype(float), 1.0 / config.TEMPERATURE)
class_probs /= class_probs.sum()  # ensure sum=1
class_probs_series = pd.Series(class_probs, index=vote_cols)

eeg_id_probs = train_df.groupby("eeg_id")[vote_cols].sum().astype(float)
row_sums = eeg_id_probs.sum(axis=1).replace(0, np.nan)
eeg_id_probs = eeg_id_probs.div(row_sums, axis=0)

patient_id_probs = train_df.groupby("patient_id")[vote_cols].sum().astype(float)
row_sums_pat = patient_id_probs.sum(axis=1).replace(0, np.nan)
patient_id_probs = patient_id_probs.div(row_sums_pat, axis=0)

submission_df = test_df[["eeg_id"]].copy()
submission_df = submission_df.merge(
    eeg_id_probs, left_on="eeg_id", right_index=True, how="left"
)

missing_mask = submission_df[vote_cols].isna().any(axis=1)
if missing_mask.any():
    patient_merge = test_df.loc[missing_mask, ["eeg_id", "patient_id"]].merge(
        patient_id_probs, left_on="patient_id", right_index=True, how="left"
    )
    submission_df.loc[missing_mask, vote_cols] = patient_merge[vote_cols].values

has_specific = ~submission_df[vote_cols].isna().any(axis=1)

for col, prob in zip(vote_cols, class_probs):
    if col not in submission_df.columns:
        submission_df[col] = np.nan
    submission_df[col].fillna(prob, inplace=True)

ALPHA = 0.1  # smaller weight for specific estimates → smoother predictions
blended = pd.DataFrame(
    np.tile(class_probs, (len(submission_df), 1)),
    columns=vote_cols,
    index=submission_df.index,
)

blended.loc[has_specific, vote_cols] = (
    ALPHA * submission_df.loc[has_specific, vote_cols].astype(float)
    + (1 - ALPHA) * class_probs_series
)

EPS = 1e-5
blended += EPS

row_sums = blended.sum(axis=1).replace(0, np.nan)
submission_df[vote_cols] = blended.div(row_sums, axis=0)




## === cell 2
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
