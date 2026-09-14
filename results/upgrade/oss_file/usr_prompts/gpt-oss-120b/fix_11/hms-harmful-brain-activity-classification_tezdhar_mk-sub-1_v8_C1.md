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

0.3839901784373968

# 6. Current score

1.05339

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I replace the failing external‑package calls with safe no‑op guards and add a small fallback that always creates a valid submission. The script now (1) sets up paths, (2) optionally runs the original conversion/testing steps only if the required modules exist, and (3) builds a uniform‑probability submission for every `eeg_id` in the test set, guaranteeing a correctly formatted CSV and a deterministic KL‑score that moves toward the target without altering the core modeling logic.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑probability baseline with class‑prior probabilities derived from the training vote counts. Computing the overall vote distribution from `train.csv` and using those fractions as predictions for every test row yields a more realistic baseline, which should lower the KL divergence and move the score closer to the target while preserving the original pipeline structure.'
- What this solution (achieved 1.68479) has done: 'I replace the uniform‑prior baseline with a simple patient‑wise prior: for each patient present in the training set I compute the normalized vote distribution across the six classes and use that distribution for all of that patient’s test rows. Test rows whose patient_id does not appear in the training data fall back to the global class‑prior. This small, data‑driven adjustment keeps the original pipeline intact while providing more realistic probabilities, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 1.68479) has done: 'I replace the patient‑wise prior with a more specific per‑eeg‑id prior, falling back to the patient prior and finally to the global class prior when needed. This finer‑grained prior better reflects the training distribution for each exact recording, so the predicted probabilities should be closer to the true targets and lower the KL‑divergence toward the target score. The rest of the pipeline and file handling stay unchanged.'
- What this solution (achieved 0.81597) has done: 'I blend the per‑eeg and per‑patient priors with the overall class‑prior instead of using them alone, which reduces over‑confidence and is expected to lower the KL‑divergence toward the target score. The change is minimal: after filling missing values, each class probability is weighted 0.7 toward the specific prior and 0.3 toward the global prior, then the row is re‑normalized before writing the submission.'
- What this solution (achieved 0.9785) has done: 'I reduce the reliance on the per‑eeg/patient priors (which can be over‑confident) and increase the influence of the global class prior, then smooth the priors with a Laplace add‑one term before blending. This modest adjustment is expected to pull the KL‑divergence down toward the target score without changing the overall pipeline.'
- What this solution (achieved 1.26166) has done: 'I lower the KL‑divergence by making the predictions less over‑confident: increase the influence of the global class‑prior and reduce the weight given to the per‑eeg/patient specific priors. This small adjustment keeps the overall pipeline intact while moving the score closer to the target (lower is better).'
- What this solution (achieved 0.86468) has done: 'I increase the influence of the data‑driven priors (eeg‑ and patient‑wise) by raising `specific_weight` to 0.6 and lowering `global_weight` to 0.4. This keeps the same fallback logic and smoothing but gives more weight to the information that best reflects the training distribution, which should reduce the KL‑divergence and move the score closer to the target (lower is better) without altering any core modeling steps.'
- What this solution (achieved 0.78876) has done: 'I lower the KL‑divergence by giving the data‑driven priors (per‑eeg / per‑patient) more influence. The only change is to increase `specific_weight` to 0.8 and reduce `global_weight` to 0.2, keeping the same blending and renormalisation logic so the pipeline and output format stay unchanged.'
- What this solution (achieved 1.05339) has done: 'I lower the influence of the highly specific EEG/ patient priors and increase the contribution of the global class prior, because the current over‑confident blending (specific_weight = 0.8) is yielding a KL score that is far above the target. Reducing `specific_weight` to 0.3 (and `global_weight` to 0.7) makes the predictions smoother and more similar to the overall class distribution, which is expected to lower the KL divergence toward the target while keeping the original pipeline unchanged.'

# 9. Code solution

## === cell 0
import sys, os, subprocess

sys.path.append("/kaggle/input/hms-mk-codes/")




## === cell 1
def safe_pip_install(wheel_path):
    if os.path.isfile(wheel_path):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                wheel_path,
                "--no-index",
                "--no-deps",
                "--force-reinstall",
            ],
            check=False,
        )


safe_pip_install(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl"
)
safe_pip_install("/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl")
safe_pip_install("/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl")
safe_pip_install("/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
os.makedirs(OUT_PATH, exist_ok=True)




## === cell 3
def run_cmd_if_exists(cmd, work_dir):
    if os.path.isdir(work_dir):
        subprocess.run(cmd, shell=True, cwd=work_dir, check=False)


run_cmd_if_exists(
    "python -m src.convert_parquet_to_npy --data_dir=$DATA_PATH --out_dir=$OUT_PATH",
    "/kaggle/input/hms-mk-codes",
)

run_cmd_if_exists(
    "python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH "
    "ckpt_path=/kaggle/input/hms-mk-data/epoch_014_val_loss_0.4944.ckpt "
    "hydra=test +model.test_output_dir=$OUT_PATH +model.net.pretrained=False",
    "/kaggle/input/hms-mk-codes",
)



## === cell 4
import pandas as pd
import numpy as np

test_csv = os.path.join(DATA_PATH, "test.csv")
submission_path = os.path.join(OUT_PATH, "submission.csv")

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_csv = os.path.join(DATA_PATH, "train.csv")
if os.path.exists(train_csv):
    train_df = pd.read_csv(train_csv, usecols=["patient_id", "eeg_id"] + vote_cols)

    class_totals = train_df[vote_cols].sum()
    global_prior = (class_totals / class_totals.sum()).values

    patient_votes = train_df.groupby("patient_id")[vote_cols].sum()
    patient_votes += 1  # add‑one smoothing
    patient_totals = patient_votes.sum(axis=1)
    patient_probs = patient_votes.div(patient_totals, axis=0)

    eeg_votes = train_df.groupby("eeg_id")[vote_cols].sum()
    eeg_votes += 1
    eeg_totals = eeg_votes.sum(axis=1)
    eeg_probs = eeg_votes.div(eeg_totals, axis=0)
else:
    global_prior = np.full(len(vote_cols), 1.0 / len(vote_cols))
    patient_probs = pd.DataFrame(columns=vote_cols)
    eeg_probs = pd.DataFrame(columns=vote_cols)

test_df = pd.read_csv(test_csv, usecols=["eeg_id", "patient_id"])

merged = test_df.copy()

merged = merged.merge(
    eeg_probs, left_on="eeg_id", right_index=True, how="left", suffixes=("", "_eeg")
)

merged = merged.merge(
    patient_probs,
    left_on="patient_id",
    right_index=True,
    how="left",
    suffixes=("", "_patient"),
)

for col in vote_cols:
    merged[col] = merged[col].fillna(merged[col + "_patient"])
    merged.drop(columns=[col + "_patient"], inplace=True, errors="ignore")
    merged[col] = merged[col].fillna(global_prior[vote_cols.index(col)])

specific_weight = 0.3  # lowered from 0.8
global_weight = 0.7  # raised from 0.2; still sums to 1

for idx, col in enumerate(vote_cols):
    merged[col] = specific_weight * merged[col] + global_weight * global_prior[idx]

epsilon = 1e-12
merged[vote_cols] = merged[vote_cols] + epsilon
merged[vote_cols] = merged[vote_cols].div(merged[vote_cols].sum(axis=1), axis=0)

submission_df = merged[["eeg_id"] + vote_cols]

submission_df.to_csv(submission_path, index=False)

print(submission_df.head())
