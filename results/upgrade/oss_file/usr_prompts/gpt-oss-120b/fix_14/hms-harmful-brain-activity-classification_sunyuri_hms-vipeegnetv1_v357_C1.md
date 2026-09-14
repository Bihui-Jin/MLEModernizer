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

0.2874542214373286

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I replace the unsupported `keepdims` argument with a standard Pandas sum, compute row‑wise sums correctly, and ensure the division broadcasts properly. After normalising the global class probabilities they are used to fill the submission so that every row sums to 1 and all required columns are present.'
- What this solution (achieved 1.39779) has done: 'The changes fix the NaN‑filling logic after normalising per‑eeg probabilities (using a Series instead of a raw ndarray), ensure the `eeg_probs` variable is correctly created, and safely map those probabilities onto the test set while falling back to the global class distribution. After mapping we normalise each row so the probabilities sum to 1, guaranteeing a valid submission file with the required columns.'
- What this solution (achieved 1.39779) has done: 'I add a lightweight “majority‑label” boost: for each eeg_id we take the most common expert_consensus label in the training set and, when that eeg_id appears in the test set, assign 0.9 probability to the corresponding class and split the remaining 0.1 uniformly among the other five classes. This keeps the original logic but nudges predictions toward a plausible class, reducing the KL‑divergence toward the target score while preserving the required submission format.'
- What this solution (achieved 1.39779) has done: 'I tone‑down the majority‑label boost, which was set to 0.9 confidence and likely caused overly confident predictions that increase KL‑divergence. By introducing a modest `boost_weight` (e.g., 0.6) and recomputing the complementary probability for the other classes, the predictions stay closer to the learned global distribution while still using the useful majority‑label information. This minor change keeps the original pipeline intact, ensures rows still sum to 1, and should lower the score toward the target.'
- What this solution (achieved 1.39779) has done: 'I blend the per‑EEG probabilities with the overall class distribution to avoid overly confident predictions, and reduce the majority‑label boost weight from 0.6 to 0.3. This smooths the predictions toward realistic values, keeping rows normalized and preserving the original workflow while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 1.39779) has done: 'I raise the reliance on per‑EEG statistics and on the majority‑label cue, which are the most informative signals available, by (1) increasing the blend weight `blend_alpha` from 0.7 to 0.9 so that per‑EEG probabilities dominate the global baseline, and (2) strengthening the majority‑label boost to 0.5 (raising `boost_weight`) while recomputing the complementary probability for the other classes. These small, targeted adjustments keep the overall pipeline unchanged but should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I make the script robust to the actual data location by checking several possible base directories, and fall back to the plain “data” folder if needed. This fixes the FileNotFoundError so the notebook can read the CSV files, produce a properly‑normalized submission, and write it to submission.csv. No other logic is altered, preserving the original modeling approach.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

possible_base_dirs = [
    "data/hms-harmful-brain-activity-classification",  # original assumption
    "data",  # flat layout
    "/kaggle/input/hms-harmful-brain-activity-classification",  # typical Kaggle path
    "/kaggle/input",  # fallback to any Kaggle input root
    ".",  # current working directory
]

BASE_DIR = None
for d in possible_base_dirs:
    if os.path.isdir(d):
        if os.path.isfile(os.path.join(d, "train.csv")) and os.path.isfile(
            os.path.join(d, "test.csv")
        ):
            BASE_DIR = d
            break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate the dataset. Checked directories:\n"
        + "\n".join(possible_base_dirs)
    )

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
OUTPUT_PATH = "submission.csv"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 1
df_train = pd.read_csv(TRAIN_PATH)

eeg_votes_sum = df_train.groupby("eeg_id")[TARGET_COLS].sum()

row_totals = eeg_votes_sum.sum(axis=1).replace(0, np.nan)
eeg_probs_df = eeg_votes_sum.div(row_totals, axis=0)  # indexed by eeg_id

global_votes = df_train[TARGET_COLS].sum()
global_total = global_votes.sum()
global_probs = global_votes / global_total  # Series indexed by class name

eeg_majority_label = eeg_votes_sum.idxmax(axis=1)  # Series indexed by eeg_id



## === cell 2
df_test = pd.read_csv(TEST_PATH)

submission = pd.DataFrame({"eeg_id": df_test["eeg_id"]})

per_eeg_probs = eeg_probs_df.reindex(submission["eeg_id"]).fillna(0)

alpha = 0.6
blended_probs = per_eeg_probs * alpha + global_probs * (1 - alpha)

epsilon = 1e-8
blended_probs = blended_probs + epsilon

row_sum = blended_probs.sum(axis=1)
submission[TARGET_COLS] = blended_probs.div(row_sum, axis=0)

submission.to_csv(OUTPUT_PATH, index=False)

print(f"Submission written to {OUTPUT_PATH} with shape {submission.shape}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/653998857.py in <cell line: 0>()
     19 # Normalise each row so probabilities sum to 1
     20 row_sum = blended_probs.sum(axis=1)
---> 21 submission[TARGET_COLS] = blended_probs.div(row_sum, axis=0)
     22 
     23 submission.to_csv(OUTPUT_PATH, index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4297             self._setitem_frame(key, value)
   4298         elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
   4301             self._set_item_frame_value(key, value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _setitem_array(self, key, value)
   4341                 check_key_length(self.columns, key, value)
   4342                 for k1, k2 in zip(key, value.columns):
-> 4343                     self[k1] = value[k2]
   4344 
   4345             elif not is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5261             if not isinstance(value, Series):
   5262                 value = Series(value)
-> 5263             return _reindex_for_setitem(value, self.index)
   5264 
   5265         if is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12690         if not value.index.is_unique:
  12691             # duplicate axis
> 12692             raise err
  12693 
  12694         raise TypeError(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12685     # GH#4107
  12686     try:
> 12687         reindexed_value = value.reindex(index)._values
  12688     except ValueError as err:
  12689         # raised in MultiIndex.from_tuples, see test_insert_error_msmgs

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5151         tolerance=None,
   5152     ) -> Series:
-> 5153         return super().reindex(
   5154             index=index,
   5155             method=method,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5608 
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy
   5612         ).__finalize__(self, method="reindex")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5631 
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method
   5635             )

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4427                 elif not self.is_unique:
   4428                     # GH#42568
-> 4429                     raise ValueError("cannot reindex on an axis with duplicate labels")
   4430                 else:
   4431                     indexer, _ = self.get_indexer_non_unique(target)

ValueError: cannot reindex on an axis with duplicate labels
