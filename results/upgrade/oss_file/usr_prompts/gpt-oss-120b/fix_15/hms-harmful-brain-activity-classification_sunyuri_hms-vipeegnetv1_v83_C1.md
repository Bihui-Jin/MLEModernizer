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

0.3509331993712218

# 6. Current score

0.77518

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the unused TensorFlow‑related code and any undefined configuration variables, then generate predictions using the class‑frequency baseline so the script runs end‑to‑end and creates a valid `submission.csv` with probabilities that sum to 1 for each row.'
- What this solution (achieved 1.68479) has done: 'I replace the simple global‑frequency baseline with a slightly richer baseline: for each patient_id present in the training set I compute the class‑probability distribution from all its training rows and use that distribution for any test rows with the same patient_id. Test rows whose patient_id has no training data fall back to the original global class frequencies. This per‑patient calibration keeps the overall pipeline unchanged while providing more informative probabilities, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 0.76744) has done: 'I added Laplace smoothing to the per‑patient probability estimates so that patients with very few training samples no longer get overly‑confident (zero‑probability) predictions. The smoothed counts are used to compute the patient‑level distributions, and the rest of the pipeline (fallback to the global baseline, normalization, and CSV output) stays unchanged. This small calibration tweak should lower the KL‑divergence, moving the score toward the target while preserving the original logic.'
- What this solution (achieved 0.77245) has done: 'I keep the existing baseline logic but replace the simple Laplace‑smoothed patient distribution with a blended estimate that combines the patient‑specific class frequencies with the global class frequencies, weighting more heavily toward the global baseline for patients with few training samples. This modest calibration keeps the core pipeline untouched while likely lowering the KL‑divergence score toward the target.'
- What this solution (achieved 1.41939) has done: 'I add a finer‑grained calibration step that uses per‑eeg_id vote distributions when a test row’s eeg_id appears in the training data. This keeps the existing global and patient‑level blending logic unchanged, but adds an extra hierarchy that supplies more informative probabilities for many rows, which should lower the KL‑divergence toward the target. The new code computes smoothed per‑eeg probabilities, derives a weight based on the number of votes for that eeg_id, and applies them before falling back to the patient‑level blend or the global baseline. No core modeling logic is altered, and the submission file format remains the same.'
- What this solution (achieved 1.41712) has done: 'The changes tighten the smoothing (so patient‑ and EEG‑specific distributions have higher influence) and finally blend a small portion of a uniform distribution into every prediction to avoid over‑confident zeros, which together should lower the KL‑divergence toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.75958) has done: 'I add Laplace smoothing to the patient‑ and EEG‑level count tables so no class gets a zero probability, and I compute the patient‑level blend for *all* test rows (not only those that stayed at the global baseline). I also tighten the patient smoothing factor to give a bit more weight to patient‑specific information. These modest calibration tweaks keep the original pipeline intact while making the predictions less over‑confident, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.76749) has done: 'I keep the original baseline‑blending logic but increase the influence of patient‑ and EEG‑specific distributions (by lowering the smoothing constants) and reduce the uniform fallback blend. These small calibration tweaks should make the predictions better reflect known patterns and move the KL‑divergence closer to the target while preserving the overall pipeline.'
- What this solution (achieved 0.76428) has done: 'I keep the original pipeline unchanged and only adjust a few calibration constants to give a slightly stronger smoothing effect, which should reduce over‑confident patient‑ and EEG‑specific probabilities and move the KL‑divergence closer to the target (lower score). Specifically, I increase the Laplace‑smoothing alphas and the weight‑smoothing factors, while keeping the blending and uniform‑mix steps identical.'
- What this solution (achieved 0.80521) has done: 'I reduce the Laplace‑smoothing constants and the smoothing factors so patient‑ and EEG‑specific distributions influence the predictions more strongly, and I drop the tiny uniform‑mix that slightly pulls predictions toward a meaningless uniform guess. These minimal tweaks keep the overall pipeline unchanged while giving sharper, better‑calibrated probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77518) has done: 'I increase the Laplace‑smoothing constants and the weight‑smoothing factors so patient‑ and EEG‑specific distributions are blended more toward the global baseline, and I add a small uniform‑mix (5 %) to keep predictions from being overly confident. These modest calibrations keep the original pipeline unchanged while lowering the KL‑divergence score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # keep as before
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]  # last six columns are the vote targets
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

class_counts = df[TARGETS].sum()
total_counts = class_counts.sum()
BASE_PROBS = (class_counts / total_counts).values.astype(np.float32)
print("Base class probabilities (sum to 1):", BASE_PROBS, BASE_PROBS.sum())

ALPHA_PATIENT = 1.0  # larger smoothing than previous 0.1
patient_counts = df.groupby("patient_id")[TARGETS].sum()
patient_counts_smooth = patient_counts + ALPHA_PATIENT
patient_raw_probs = (
    patient_counts_smooth.div(patient_counts_smooth.sum(axis=1), axis=0)
    .fillna(0)
    .astype(np.float32)
)

patient_totals = patient_counts.sum(axis=1)  # unsmoothed vote totals per patient
SMOOTHING_FACTOR_PATIENT = 10.0
patient_weights = (patient_totals / (patient_totals + SMOOTHING_FACTOR_PATIENT)).astype(
    np.float32
)

print("Number of patients with training data:", patient_raw_probs.shape[0])

ALPHA_EEG = 1.0  # larger smoothing than previous 0.1
eeg_counts = df.groupby("eeg_id")[TARGETS].sum()
eeg_counts_smooth = eeg_counts + ALPHA_EEG
eeg_raw_probs = (
    eeg_counts_smooth.div(eeg_counts_smooth.sum(axis=1), axis=0)
    .fillna(0)
    .astype(np.float32)
)

eeg_totals = eeg_counts.sum(axis=1)  # unsmoothed vote totals per EEG
SMOOTHING_FACTOR_EEG = 10.0
eeg_weights = (eeg_totals / (eeg_totals + SMOOTHING_FACTOR_EEG)).astype(np.float32)

print("Number of eeg_ids with training data:", eeg_raw_probs.shape[0])



## === cell 1
if PLATFORM == "local":
    test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
elif PLATFORM == "kaggle":
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
print("Test shape", test.shape)



## === cell 2
num_test = test.shape[0]

pred = np.tile(BASE_PROBS, (num_test, 1))

for eid, probs in eeg_raw_probs.iterrows():
    mask = test["eeg_id"] == eid
    if not mask.any():
        continue
    w = eeg_weights.get(eid, 0.0)
    blended = w * probs.values + (1.0 - w) * BASE_PROBS
    pred[mask.values] = blended

for pid, probs in patient_raw_probs.iterrows():
    mask = test["patient_id"] == pid
    if not mask.any():
        continue
    w = patient_weights.get(pid, 0.0)
    blended = w * probs.values + (1.0 - w) * pred[mask.values]
    pred[mask.values] = blended

row_sums = pred.sum(axis=1, keepdims=True)
pred = np.clip(pred, 0, None)
pred = pred / np.where(row_sums == 0, 1, row_sums)

UNIFORM_ALPHA = 0.05  # small uniform contribution
if UNIFORM_ALPHA > 0:
    uniform_vec = np.full_like(pred, 1.0 / pred.shape[1], dtype=np.float32)
    pred = (1 - UNIFORM_ALPHA) * pred + UNIFORM_ALPHA * uniform_vec
    row_sums = pred.sum(axis=1, keepdims=True)
    pred = pred / np.where(row_sums == 0, 1, row_sums)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
print("Row sum check (first 5 rows):")
print(sub[TARGETS].sum(axis=1).head())
