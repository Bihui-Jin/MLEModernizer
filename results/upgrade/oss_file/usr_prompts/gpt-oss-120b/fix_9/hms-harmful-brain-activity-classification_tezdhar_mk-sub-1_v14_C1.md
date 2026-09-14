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

0.3391724675284024

# 6. Current score

0.76992

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I replace the missing `src` import with a direct definition of the target columns, guard the heavy conversion/inference steps so they don’t crash if the code or checkpoints are unavailable, and add a fallback that creates a valid submission using uniform probabilities (or averages any existing fold predictions). This ensures the notebook runs end‑to‑end, produces `submission.csv` with the correct columns that sum to 1, and moves the pipeline from “no output” to a usable submission while keeping the original logic minimal.'
- What this solution (achieved 1.41937) has done: 'I compute class‑prior probabilities from the training labels and use them as a fallback prediction (instead of a uniform distribution). When fold predictions are available they are still averaged, but if they are missing the prior‑based predictions give a more realistic distribution and should reduce the KL‑divergence toward the target score.'
- What this solution (achieved 1.68479) has done: 'I improve the fallback predictions by using patient‑specific class priors instead of a single global prior. For each test row we look up the patient’s vote distribution from the training set (if the patient appears there) and use that normalized distribution; otherwise we fall back to the overall class prior. This small change keeps the original workflow intact while providing more informative probabilities, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 0.76992) has done: 'I smooth the patient‑specific priors with the global class prior (using a small Dirichlet‑type α) so that predictions for patients with few annotations become more reliable, while keeping the overall fallback logic unchanged. This minor calibration tweak should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.79979) has done: 'I reduce the Dirichlet smoothing strength by changing `ALPHA` from 10.0 to 1.0 so the fallback predictions rely more on patient‑specific vote distributions, which are usually closer to the true test distribution and should lower the KL‑divergence toward the target score. This minor tweak preserves all existing logic and keeps the output format unchanged.'
- What this solution (achieved 0.81696) has done: 'I lowered the Dirichlet α to 5.0 (a moderate smoothing between the previous extremes) and added a simple temperature‑scaling step (T = 2) after the probability vectors are formed; this smooths overly‑confident predictions and should move the KL‑divergence closer to the target without changing the overall workflow.'
- What this solution (achieved 0.81714) has done: 'I lower the Dirichlet smoothing strength (ALPHA = 0.5) so the fallback relies more on patient‑specific vote distributions, and remove the temperature‑scaling step (set TEMP = 1.0) to keep the probabilities as‑is rather than flattening them. These minimal tweaks keep the original workflow intact while making the predictions more confident and better calibrated, which should move the KL‑divergence score down toward the target.'
- What this solution (achieved 0.76992) has done: 'I increase the Dirichlet smoothing strength back to a higher value (ALPHA = 10.0). Higher smoothing gives more reliable patient‑specific priors, which has been shown to reduce the KL‑divergence in earlier attempts. No other logic is changed, preserving the original workflow while moving the score closer to the lower‑is‑better target.'

# 9. Code solution

## === cell 0
import os, glob
import pandas as pd
import numpy as np

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"




## === cell 1
def safe_run(command: str):
    try:
        import subprocess, shlex

        subprocess.run(
            shlex.split(command),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # Silently continue if the external code or files are unavailable.


safe_run(
    f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
)

fold_ckpts = [
    "fold0_pseudo_log.ckpt",
    "fold1_pseudo_log.ckpt",
    "fold2_pseudo_log.ckpt",
    "fold3_pseudo_log.ckpt",
    "fold4_pseudo_log.ckpt",
]
for i, ckpt in enumerate(fold_ckpts):
    safe_run(
        f"python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path=/kaggle/input/hms-mk-data/{ckpt} "
        f"hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_pseudo +model.net.pretrained=False"
    )
    src_file = os.path.join(OUT_PATH, "submission.csv")
    dst_file = os.path.join(OUT_PATH, f"submission_fold{i}.csv")
    if os.path.exists(src_file):
        os.replace(src_file, dst_file)



## === cell 2
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
eeg_ids = test_df["eeg_id"].values
test_patient_ids = test_df["patient_id"].values

train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
class_counts = train_df[TARGET_COLS].sum()
class_prior = class_counts / class_counts.sum()  # shape (6,)

patient_counts = train_df.groupby("patient_id")[TARGET_COLS].sum()
patient_total = patient_counts.sum(axis=1)  # total votes per patient

ALPHA = 10.0

patient_prior_smooth = (patient_counts + ALPHA * class_prior.values) / (
    patient_total.values[:, None] + ALPHA
)

patient_prior_smooth = pd.DataFrame(
    patient_prior_smooth,
    index=patient_counts.index,
    columns=TARGET_COLS,
)

fold_files = sorted(glob.glob(os.path.join(OUT_PATH, "submission_fold*.csv")))
if fold_files:
    pred_arrays = []
    for f in fold_files:
        df = pd.read_csv(f)
        if set(TARGET_COLS).issubset(df.columns):
            pred_arrays.append(df[TARGET_COLS].values)
    if pred_arrays:
        preds = np.mean(np.stack(pred_arrays, axis=0), axis=0)
        preds = preds / preds.sum(axis=1, keepdims=True)
    else:
        preds = np.empty((len(eeg_ids), len(TARGET_COLS)), dtype=np.float64)
        for idx, pid in enumerate(test_patient_ids):
            if pid in patient_prior_smooth.index:
                preds[idx] = patient_prior_smooth.loc[pid].values
            else:
                preds[idx] = class_prior.values
else:
    preds = np.empty((len(eeg_ids), len(TARGET_COLS)), dtype=np.float64)
    for idx, pid in enumerate(test_patient_ids):
        if pid in patient_prior_smooth.index:
            preds[idx] = patient_prior_smooth.loc[pid].values
        else:
            preds[idx] = class_prior.values

preds = preds / preds.sum(axis=1, keepdims=True)

TEMP = 1.0
preds = np.clip(preds, 1e-12, 1.0)  # avoid zeros before power
preds = np.power(preds, 1.0 / TEMP)
preds = preds / preds.sum(axis=1, keepdims=True)

submission = pd.DataFrame(preds, columns=TARGET_COLS)
submission.insert(0, "eeg_id", eeg_ids)



## === cell 3
submission_path = os.path.join(OUT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")
