# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.4376204139770831

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script failed because it looked for the CSV files under a non‑existent `data/…` directory.  
We now point `find_path` directly to the dataset folder (or the Kaggle `/input/` mount) by removing the leading `data` segment when requesting the files. This restores correct loading of `train.csv` and `test.csv`, allowing the baseline submission to be generated and saved as `submission.csv`.'
- What this solution (achieved 1.68479) has done: 'I replace the simple overall‑class prior with a patient‑specific prior: for each patient present in the training set I compute the normalized vote distribution across the six classes and use it for every test row belonging to that patient. If a patient does not appear in the training data, I fall back to the global class prior. This small, data‑driven adjustment keeps the original workflow unchanged while moving the KL‑divergence closer to the target (lower is better).'
- What this solution (achieved 0.85517) has done: 'I introduce a light smoothing step that falls back from an eeg‑specific prior to a patient‑specific prior and finally to the global prior, and blend the chosen prior with the global prior (α ≈ 0.6). This keeps the original “patient‑specific” idea but protects against over‑fitting for patients with few samples, moving the KL‑divergence toward the target lower score while preserving the overall workflow.'
- What this solution (achieved 1.0413) has done: 'I keep the overall workflow unchanged but reduce the weight given to EEG‑specific (or patient‑specific) priors, which were over‑fitting and caused a higher KL divergence. By lowering `ALPHA` from 0.6 to 0.3 we blend more towards the global class prior, moving the score closer to the target lower value. I also rename the cell header to start at 1 as required.'
- What this solution (achieved 1.19964) has done: 'I lower the blend weight (ALPHA) so the global prior dominates more, and add a tiny “add‑one” smoothing when computing patient‑ and EEG‑specific priors. This reduces over‑fitting of sparse priors, keeping the same workflow while moving the KL divergence closer to the target lower score.'
- What this solution (achieved 0.82219) has done: 'I keep the overall workflow unchanged but increase the weight given to the EEG‑ or patient‑specific priors, which have proven to lower the KL‑divergence. The only modification is to raise `ALPHA` from 0.15 to 0.7 and rename the cell header to start at 1 so the script runs without cell‑index issues.'
- What this solution (achieved 0.9641) has done: 'I keep the overall workflow unchanged but add a modest temperature scaling after blending the EEG‑/patient‑specific prior with the global prior. This flattens overly confident predictions, which typically reduces KL‑divergence, moving the score closer to the target. I also raise the blending weight `ALPHA` slightly (to rely a bit more on the specific priors) while preserving the existing smoothing. These minimal adjustments preserve core logic and still produce a valid submission CSV.'
- What this solution (achieved 1.34594) has done: 'I lower the blending weight so the global class prior dominates (reducing over‑fitting to sparse EEG‑ or patient‑specific priors) and increase the temperature scaling to flatten the predictions further. These small hyper‑parameter tweaks keep the original workflow intact while moving the KL‑divergence down toward the target score.'
- What this solution (achieved 0.86468) has done: 'I raise the blend weight so the EEG‑ or patient‑specific prior contributes more (ALPHA = 0.6) and set the temperature to 1.0 to avoid over‑flattening the predictions. These small hyper‑parameter tweaks keep the original workflow intact while moving the KL‑divergence lower, i.e., toward the target score. I also rename the cell header to start at 1 so the script runs without index issues.'
- What this solution (achieved 1.27355) has done: 'I lower the influence of the EEG/patient‑specific priors (set ALPHA to 0.2), increase additive smoothing (SMOOTH = 5.0) to dampen noisy sparse priors, and apply temperature scaling TEMPERATURE = 2.0 to flatten the blended probabilities. These minimal tweaks keep the original workflow while moving the predictions closer to the global prior and reducing over‑confidence, which should lower the KL‑divergence toward the target score. I also rename the single cell to start at 1 so the script runs without index issues.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def find_path(*relative_parts):
    """
    Locate a file either in a local folder or in the Kaggle
    '/kaggle/input/' directory. If the first component is 'data',
    it is omitted when searching under 'input' because Kaggle mounts
    the dataset directly under '/kaggle/input/<dataset_name>/'.
    """
    candidate = os.path.join(*relative_parts)
    if os.path.exists(candidate):
        return candidate

    parts_for_input = list(relative_parts)
    if parts_for_input and parts_for_input[0] == "data":
        parts_for_input = parts_for_input[1:]  # remove leading "data"
    kaggle_root = os.path.join("input", *parts_for_input)
    if os.path.exists(kaggle_root):
        return kaggle_root

    raise FileNotFoundError(f"Unable to locate {'/'.join(relative_parts)}")


TEST_CSV = find_path("hms‑harmful‑brain‑activity‑classification", "test.csv")
TRAIN_CSV = find_path("hms‑harmful‑brain‑activity‑classification", "train.csv")
SUBMISSION_PATH = "submission.csv"

test_df = pd.read_csv(TEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

global_votes_sum = train_df[TARGETS].sum()
global_prior = (global_votes_sum / global_votes_sum.sum()).values.astype(np.float32)

SMOOTH = 1.0  # add‑one smoothing

patient_votes_sum = train_df.groupby("patient_id")[TARGETS].sum()
patient_probs = {}
for pid, row in patient_votes_sum.iterrows():
    probs = row.values.astype(np.float32) + SMOOTH
    total = probs.sum()
    probs = probs / total if total > 0 else global_prior
    patient_probs[pid] = probs

eeg_votes_sum = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_probs = {}
for eid, row in eeg_votes_sum.iterrows():
    probs = row.values.astype(np.float32) + SMOOTH
    total = probs.sum()
    probs = probs / total if total > 0 else global_prior
    eeg_probs[eid] = probs

ALPHA = 0.7  # larger influence of EEG/patient‑specific priors
TEMPERATURE = 1.0  # no temperature scaling (keeps distribution sharp)


def lookup_probs(eeg_id, patient_id):
    """
    Return a probability vector:
      - Prefer an EEG‑specific prior if available.
      - Otherwise use the patient‑specific prior.
      - Fallback to the global prior.
    Blend the chosen prior with the global prior using ALPHA,
    then apply temperature scaling (if TEMPERATURE != 1) and renormalize.
    """
    specific = eeg_probs.get(eeg_id)
    if specific is None:
        specific = patient_probs.get(patient_id, global_prior)
    blended = ALPHA * specific + (1.0 - ALPHA) * global_prior
    if TEMPERATURE != 1.0:
        blended = np.power(blended, 1.0 / TEMPERATURE)
    blended /= blended.sum()
    return blended


prob_matrix = np.vstack(
    test_df.apply(lambda row: lookup_probs(row["eeg_id"], row["patient_id"]), axis=1)
)

prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
for i, col in enumerate(TARGETS):
    submission[col] = prob_matrix[:, i]

submission[TARGETS] = submission[TARGETS].div(submission[TARGETS].sum(axis=1), axis=0)

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission written to {SUBMISSION_PATH}")
print("Submission head:")
print(submission.head())

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3238875551.py in <cell line: 0>()
     25 
     26 
---> 27 TEST_CSV = find_path("hms‑harmful‑brain‑activity‑classification", "test.csv")
     28 TRAIN_CSV = find_path("hms‑harmful‑brain‑activity‑classification", "train.csv")
     29 SUBMISSION_PATH = "submission.csv"

/tmp/ipykernel_55/3238875551.py in find_path(*relative_parts)
     22         return kaggle_root
     23 
---> 24     raise FileNotFoundError(f"Unable to locate {'/'.join(relative_parts)}")
     25 
     26 

FileNotFoundError: Unable to locate hms‑harmful‑brain‑activity‑classification/test.csv
