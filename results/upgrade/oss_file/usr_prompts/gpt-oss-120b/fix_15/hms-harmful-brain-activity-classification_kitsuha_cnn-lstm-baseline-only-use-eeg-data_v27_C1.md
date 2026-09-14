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

1.166161429369046

# 6. Current score

1.41932

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script was failing because the data paths did not point to the actual Kaggle input directory, causing `FileNotFoundError` and subsequent `NameError`. I updated the base path to the standard Kaggle input location, added a fallback check, and ensured the test CSV is loaded from the corrected path. These small fixes let the code read the data, compute global class probabilities, generate a properly normalized submission, and write it as `submission.csv` in the working directory. No core modeling logic was altered.'
- What this solution (achieved 1.68479) has done: 'I keep the data loading unchanged and replace the uniform global‑probability baseline with a simple patient‑wise probability estimate: for each patient seen in the training set we compute the observed vote distribution and use it for test rows belonging to that patient, falling back to the overall global distribution when a patient is unseen. This small calibration usually lowers the KL‑divergence and moves the score closer to the target without altering any core modeling logic.'
- What this solution (achieved 1.39581) has done: 'I smooth the patient‑wise vote distributions with a small share of the global class frequencies. This keeps the same patient‑based logic but reduces over‑confidence for patients with few samples, which should lower the KL‑divergence and move the score closer to the target. The change is limited to the probability computation in cell 2.'
- What this solution (achieved 1.41328) has done: 'I increase the smoothing of patient‑wise vote counts toward the global class distribution and then blend each patient’s probability vector with the global probabilities (using a weighting factor). This adds a modest amount of regularisation, which should lower the KL‑divergence and move the score closer to the target without changing any core modelling logic.'
- What this solution (achieved 1.41859) has done: 'I slightly increase the smoothing toward the global class distribution (raise ALPHA) and rely less on patient‑specific probabilities (lower BETA). This reduces over‑confidence on sparse patient data, which should lower the KL‑divergence and move the score closer to the target. I also clip very small probabilities before the final renormalisation to avoid extreme KL values.'
- What this solution (achieved 1.41927) has done: 'We tune the smoothing and blending constants to rely more on the stable global class distribution and less on noisy patient‑specific estimates. By increasing `ALPHA` (the weight of global counts in the patient‑wise smoothing) and decreasing `BETA` (the influence of patient probabilities in the final blend), the predictions become better calibrated for KL‑divergence, moving the score closer to the target while keeping the original logic untouched.'
- What this solution (achieved 1.41807) has done: 'I reduce the amount of smoothing toward the global class counts (lower ALPHA) and increase the weight of patient‑specific probabilities in the final blend (higher BETA). This should give more useful patient‑level information while still keeping the distribution calibrated, moving the KL‑divergence lower toward the target. The only changes are the two constants and updated cell numbering.'
- What this solution (achieved 1.41678) has done: 'I add a simple per‑patient confidence weight based on how many training rows each patient has. Patients with few samples rely more on the global distribution, while well‑represented patients keep most of their patient‑specific probabilities. This modest calibration keeps the original blending logic but should lower KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 1.41906) has done: 'We reduce the KL‑divergence by making the model rely more on the stable global class distribution and less on potentially noisy patient‑specific estimates. This is done by increasing the global‑count smoothing factor (ALPHA), lowering the base patient influence (BETA), and using a smoother weighting that grows with patient sample size but never reaches full weight. These small constant adjustments keep the original logic unchanged while moving the score closer to the target.'
- What this solution (achieved 1.41932) has done: 'We increase the global‑count smoothing (ALPHA) and reduce the influence of patient‑specific probabilities (BETA) while also making the patient weight grow more slowly (cnt/(cnt+20)). This relies more on the stable global class distribution, which should lower the KL‑divergence and move the score closer to the target without altering the overall pipeline. The rest of the code remains unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
if not (BASE_PATH + "train.csv").endswith(".csv") or not pd.io.common.file_exists(
    BASE_PATH + "train.csv"
):
    BASE_PATH = "data/hms-harmful-brain-activity-classification/"

PATH = BASE_PATH

df = pd.read_csv(PATH + "train.csv")
TARGETS = df.columns[-6:]  # last six columns are the vote targets
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))
df.head()



## === cell 1
class_votes = df[TARGETS].sum()
class_probs = class_votes / class_votes.sum()
print("Global class probabilities:")
print(class_probs)



## === cell 2
ALPHA = 10.0  # stronger global count influence during patient smoothing
BETA = 0.1  # lower base influence of patient‑specific probabilities

patient_votes = df.groupby("patient_id")[TARGETS].sum()

smoothed_votes = patient_votes + ALPHA * class_votes

patient_probs = smoothed_votes.div(smoothed_votes.sum(axis=1), axis=0)

patient_lookup = patient_probs.to_dict(orient="index")
global_lookup = class_probs.to_dict()

patient_counts = df.groupby("patient_id").size().to_dict()
patient_weight = {pid: cnt / (cnt + 20) for pid, cnt in patient_counts.items()}

test_path = PATH + "test.csv"
test_data = pd.read_csv(test_path)

sub = pd.DataFrame({"eeg_id": test_data["eeg_id"]})

for idx, row in test_data.iterrows():
    pid = row["patient_id"]
    if pid in patient_lookup:
        probs_patient = patient_lookup[pid]
        weight = BETA * patient_weight.get(pid, 0.0)
        blended = {
            col: weight * probs_patient[col] + (1 - weight) * global_lookup[col]
            for col in TARGETS
        }
        probs = blended
    else:
        probs = global_lookup  # fallback to global distribution

    for col in TARGETS:
        sub.at[idx, col] = probs[col]

sub[TARGETS] = sub[TARGETS].clip(lower=1e-6)
sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

submission_file = "/kaggle/working/submission.csv"
sub.to_csv(submission_file, index=False)
print(f"Submission written to {submission_file}")
print("Submission shape:", sub.shape)
sub.head()
