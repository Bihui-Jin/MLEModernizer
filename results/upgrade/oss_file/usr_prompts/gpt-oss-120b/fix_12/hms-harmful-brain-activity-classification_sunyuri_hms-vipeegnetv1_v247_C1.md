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

0.3533017934064396

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I guard all TensorFlow imports and related code so they are only executed when TensorFlow is available, and replace the inference block with a simple baseline that predicts the overall class distribution from the training data. This fixes the `AttributeError` caused by missing TensorFlow/protobuf and ensures a valid `submission.csv` is written, giving a deterministic score without altering the core training logic.'
- What this solution (achieved 1.39779) has done: 'I replace the naive global‑distribution baseline with a per‑`eeg_id` average distribution: for each `eeg_id` present in the training data we compute the mean normalized vote vector and use it for the matching test rows; unseen ids fall back to the overall class distribution. This keeps the original inference‑only design, avoids any TensorFlow use, and yields probabilities that better reflect the training data, moving the KL‑divergence score toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 1.64506) has done: 'The update keeps the original inference‑only design but adds a patient‑level fallback: if a test `eeg_id` was never seen, we now try to use the mean distribution of its `patient_id` before falling back to the overall class distribution. This extra signal brings the predicted probabilities closer to the true label distribution, reducing the KL‑divergence and moving the score toward the target while preserving all existing logic and ensuring a valid `submission.csv` is written.'
- What this solution (achieved 1.64506) has done: 'I fixed the TensorFlow import so any protobuf errors are caught and the code continues with TF disabled, and I added a spectrogram‑level fallback (eeg_id → patient_id → spectrogram_id → global) to give more informative probabilities, which should lower the KL‑divergence toward the target while keeping the original inference‑only design unchanged.'
- What this solution (achieved 1.77933) has done: 'I fixed the TensorFlow import error handling (which already prevents crashes) and rewrote the inference logic to use **weighted averages** of the vote counts when computing per‑eeg, per‑patient and per‑spectrogram distributions. This makes the fallback probabilities reflect the true amount of annotator votes, which better matches the KL‑divergence target. I also combined patient‑ and spectrogram‑level information when both are available, giving a more informative prediction while preserving the original fallback hierarchy and ensuring the final CSV is correctly written.'
- What this solution (achieved 1.64506) has done: 'I bypass all TensorFlow imports to prevent the protobuf‑related crash and simplify the fallback logic by using the average of normalized vote probabilities (instead of a raw‑count weighted mean). This keeps the original hierarchy (eeg_id → patient_id + spectrogram_id → patient_id → spectrogram_id → global) while making the predictions more representative of the true label distribution, which should lower the KL‑divergence score toward the target. The script now always write a valid `submission.csv`.'
- What this solution (achieved 1.47425) has done: 'I fixed the import typo (`numpy`), ensured all constants are defined before they are used, and wrapped optional heavy imports in try/except blocks to avoid crashes if they are missing. The core hierarchical fallback logic (eeg_id → patient_id & spectrogram_id → patient_id → spectrogram_id → global) is retained, and the script now reliably writes a valid `submission.csv` with rows summing to one.'
- What this solution (achieved 1.47425) has done: 'I improve the fallback probability calculation by weighting the patient‑level and spectrogram‑level distributions according to the total number of votes they contain, rather than averaging them equally. This uses the same hierarchical logic but produces a more representative combined distribution, which should reduce the KL‑divergence and move the score closer to the target while keeping the overall structure unchanged.'
- What this solution (achieved 1.47425) has done: 'I keep the hierarchical fallback logic but add a small global‑distribution smoothing term when both patient and spectrogram information are available. By blending the patient‑level, spectrogram‑level and overall class probabilities (weighted by their total vote counts), the predictions become less noisy for rare IDs, which should lower the KL‑divergence and move the score toward the target without changing the core approach.'

# 9. Code solution

## === cell 0
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six columns are the vote counts
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2281049136.py in <cell line: 0>()
----> 1 df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
      2 TARGETS = df.columns[-6:]  # last six columns are the vote counts
      3 print("Train shape:", df.shape)
      4 print("Targets:", list(TARGETS))
      5 

NameError: name 'pd' is not defined

## === cell 1
if not NEEDTRAIN:

    def group_prob_sum(group):
        vote_sum = group[TARGETS].sum().astype(np.float32)
        total = vote_sum.sum()
        if total == 0:
            return pd.Series(np.full(len(TARGETS), 1.0 / len(TARGETS)), index=TARGETS)
        return vote_sum / total

    eeg_id_probs = df.groupby("eeg_id").apply(group_prob_sum)
    patient_probs = df.groupby("patient_id").apply(group_prob_sum)
    spectrogram_probs = df.groupby("spectrogram_id").apply(group_prob_sum)

    patient_vote_sum = df.groupby("patient_id")[TARGETS].sum().astype(np.float32)
    patient_totals = patient_vote_sum.sum(axis=1)  # total votes per patient
    spectrogram_vote_sum = (
        df.groupby("spectrogram_id")[TARGETS].sum().astype(np.float32)
    )
    spectrogram_totals = spectrogram_vote_sum.sum(axis=1)  # total votes per spectrogram

    global_vote_sum = df[TARGETS].sum().astype(np.float32)
    global_total = global_vote_sum.sum()
    if global_total == 0:
        global_prob = np.full(len(TARGETS), 1.0 / len(TARGETS))
    else:
        global_prob = (global_vote_sum / global_total).values

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

    probs = []
    for idx, eid in enumerate(sub["eeg_id"]):
        if eid in eeg_id_probs.index:
            probs.append(eeg_id_probs.loc[eid].values)
        else:
            patient_id = test.loc[idx, "patient_id"]
            spectrogram_id = test.loc[idx, "spectrogram_id"]
            has_patient = patient_id in patient_probs.index
            has_spec = spectrogram_id in spectrogram_probs.index

            if has_patient and has_spec:
                patient_weight = patient_totals.loc[patient_id]
                spec_weight = spectrogram_totals.loc[spectrogram_id]
                global_weight = global_total * 1e-3
                combined = (
                    patient_probs.loc[patient_id].values * patient_weight
                    + spectrogram_probs.loc[spectrogram_id].values * spec_weight
                    + global_prob * global_weight
                ) / (patient_weight + spec_weight + global_weight)
                probs.append(combined)
            elif has_patient:
                probs.append(patient_probs.loc[patient_id].values)
            elif has_spec:
                probs.append(spectrogram_probs.loc[spectrogram_id].values)
            else:
                probs.append(global_prob)

    probs = np.vstack(probs)

    blending_factor = 0.9
    probs = blending_factor * probs + (1 - blending_factor) * global_prob
    probs = probs / probs.sum(axis=1, keepdims=True)

    eps = 1e-12
    probs = probs + eps
    probs = probs / probs.sum(axis=1, keepdims=True)

    for col, prob_col in zip(TARGETS, probs.T):
        sub[col] = prob_col

    sub.to_csv("submission.csv", index=False)
    print("Submission shape:", sub.shape)
    print(sub.head())

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1298021380.py in <cell line: 0>()
----> 1 if not NEEDTRAIN:
      2 
      3     def group_prob_sum(group):
      4         vote_sum = group[TARGETS].sum().astype(np.float32)
      5         total = vote_sum.sum()

NameError: name 'NEEDTRAIN' is not defined
