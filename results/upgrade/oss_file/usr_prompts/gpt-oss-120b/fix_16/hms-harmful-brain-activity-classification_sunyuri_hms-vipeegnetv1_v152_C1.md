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

0.3372414704793624

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fixed the incorrect NumPy import, computed realistic class‑probability priors from the training data, and made the inference block stop after writing the submission so the heavy file‑reading steps are skipped. This resolves the runtime errors and yields a valid `submission.csv` that uses sensible probabilities, moving the KL‑score toward the target.'
- What this solution (achieved 1.68479) has done: 'The fix corrects the handling of missing patient‑specific priors: instead of trying to fill NaNs with a whole NumPy array (which raises a ValueError), we build a column‑wise dictionary of class priors and use it with `DataFrame.fillna`. This resolves the runtime error, defines `patient_id_to_idx` correctly, and allows the inference block to run and write a valid `submission.csv` with proper probability rows. No core logic or modeling code is altered.'
- What this solution (achieved 0.76744) has done: 'Implemented fixes to correctly compute patient and EEG priors without illegal multi‑dimensional indexing and ensured proper NaN handling by filling with a pandas Series of class priors. Added explanatory comments and structured the calculations to use NumPy arrays for denominators, restoring variable definitions so the inference loop can run and produce a valid `submission.csv`. All other logic remains unchanged.'
- What this solution (achieved 0.76744) has done: 'Implemented a small but effective blend of EEG‑specific and patient‑specific priors during inference. When both identifiers are known, predictions are now the average of the two priors, giving a richer estimate while preserving the original fallback logic. This tweak keeps the core prior‑based strategy unchanged, ensures probabilities stay valid, and is expected to move the KL score closer to the target (lower is better).'
- What this solution (achieved 0.76744) has done: 'Implemented a weighted blending of EEG‑specific and patient‑specific priors based on the amount of annotation votes each source provides. This uses the existing prior calculations but adds vote‑counts to determine a data‑driven mixing ratio, aiming to lower the KL‑score toward the target while preserving the original logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

PLATFORM = "local"  # change to "kaggle" if needed
possible_roots = [
    "./data/hms-harmful-brain-activity-classification",
    "./working/hms-harmful-brain-activity-classification",
    "/kaggle/input/hms-harmful-brain-activity-classification",
]
DATA_ROOT = next((p for p in possible_roots if os.path.isdir(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate the dataset directory.")

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_path = os.path.join(DATA_ROOT, "train.csv")
train = pd.read_csv(train_path)

global_votes = train[TARGETS].sum().values.astype(np.float64)
class_priors = global_votes / global_votes.sum()

eeg_groups = train.groupby("eeg_id")[TARGETS].sum()
eeg_totals = eeg_groups.sum(axis=1).replace(0, np.nan)  # avoid div‑by‑zero
eeg_priors = eeg_groups.div(eeg_totals, axis=0).fillna(class_priors)

pat_groups = train.groupby("patient_id")[TARGETS].sum()
pat_totals = pat_groups.sum(axis=1).replace(0, np.nan)
patient_priors = pat_groups.div(pat_totals, axis=0).fillna(class_priors)

eeg_id_to_priors = eeg_priors.to_dict(orient="index")  # eeg_id -> dict of class probs
patient_id_to_priors = patient_priors.to_dict(
    orient="index"
)  # patient_id -> dict of class probs

eeg_vote_counts = eeg_groups.sum(axis=1).replace(0, np.nan)
patient_vote_counts = pat_groups.sum(axis=1).replace(0, np.nan)

if PLATFORM == "local":
    test_path = os.path.join(DATA_ROOT, "test.csv")
else:  # kaggle
    test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"

test = pd.read_csv(test_path)
num_test = test.shape[0]

probs = np.empty((num_test, len(TARGETS)), dtype=np.float32)

epsilon = 1e-12  # small value to avoid division by zero
smoothing_alpha = 0.2  # blend a bit of the global prior to reduce over‑fitting

for i in range(num_test):
    eid = test["eeg_id"].iloc[i]
    pid = test["patient_id"].iloc[i]

    has_eeg = eid in eeg_id_to_priors
    has_pat = pid in patient_id_to_priors

    if has_eeg and has_pat:
        eeg_p = np.array(list(eeg_id_to_priors[eid].values()), dtype=np.float64)
        pat_p = np.array(list(patient_id_to_priors[pid].values()), dtype=np.float64)

        eeg_votes = eeg_vote_counts.get(eid, np.nan)
        pat_votes = patient_vote_counts.get(pid, np.nan)

        if np.isnan(eeg_votes) or np.isnan(pat_votes):
            w_eeg, w_pat = 0.5, 0.5
        else:
            total = eeg_votes + pat_votes
            w_eeg = eeg_votes / total
            w_pat = pat_votes / total

        combined = w_eeg * eeg_p + w_pat * pat_p
        probs[i] = (1 - smoothing_alpha) * combined + smoothing_alpha * class_priors
    elif has_eeg:
        probs[i] = np.array(list(eeg_id_to_priors[eid].values()), dtype=np.float64)
    elif has_pat:
        probs[i] = np.array(list(patient_id_to_priors[pid].values()), dtype=np.float64)
    else:
        probs[i] = class_priors

probs = np.clip(probs, epsilon, None)
probs /= probs.sum(axis=1, keepdims=True)

submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for idx, col in enumerate(TARGETS):
    submission[col] = probs[:, idx]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Submission written to", submission_path)
print("Shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3945007851.py in <cell line: 0>()
     34 eeg_groups = train.groupby("eeg_id")[TARGETS].sum()
     35 eeg_totals = eeg_groups.sum(axis=1).replace(0, np.nan)  # avoid div‑by‑zero
---> 36 eeg_priors = eeg_groups.div(eeg_totals, axis=0).fillna(class_priors)
     37 
     38 # Patient‑specific priors

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7438                 new_data = self.where(self.notna(), value)._mgr
   7439             else:
-> 7440                 raise ValueError(f"invalid fill value with a {type(value)}")
   7441 
   7442         result = self._constructor_from_mgr(new_data, axes=new_data.axes)

ValueError: invalid fill value with a <class 'numpy.ndarray'>
