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

0.4683979233755924

# 6. Current score

1.4053

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I prevent the TensorFlow import (which crashes due to a protobuf mismatch) and replace the model‑based inference with a simple baseline that uses the overall class distribution from the training data. This keeps the data handling logic, produces a valid .csv submission, and gives a reasonable KL‑Divergence score without altering the core modelling approach when training is disabled.'
- What this solution (achieved 1.41937) has done: 'I add a per‑eeg ID baseline: for each eeg_id in the train set I compute the normalized vote distribution (with a tiny smoothing term to avoid zeros) and use it for matching test rows, falling back to the global class distribution when an ID is unseen. This simple refinement is expected to reduce the KL‑divergence and move the score closer to the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'I add a lightweight validation step that searches for a simple blending factor α between the per‑eeg baseline and the global class distribution. By evaluating KL‑divergence on a held‑out slice of the training data, the script picks the α that yields the lowest divergence and then uses this α to smooth the test‑time predictions. This keeps the original baseline logic while making the predictions less over‑confident, moving the score toward the target.'
- What this solution (achieved 1.41937) has done: 'I add Laplace smoothing to both the per‑eeg and global class distributions, use a finer α grid for blending, and clip the blended probabilities before normalising. This reduces zero‑probability issues that inflate KL‑divergence, bringing the validation (and thus test) score closer to the target while preserving the existing workflow.'
- What this solution (achieved 1.41937) has done: 'I reduce the Laplace smoothing from 1.0 to 0.5 to keep class distributions sharper and then compare the validation KL‑divergence of the pure per‑eeg baseline (α = 1) against the best blended α found on the validation split. If the pure per‑eeg model performs better, I set α to 1.0, ensuring the test predictions rely on the more specific per‑eeg probabilities, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 1.41937) has done: 'I reduced the Laplace smoothing to make the class distributions sharper and added a lightweight temperature‑scaling step that is tuned on the validation split. The optimal temperature together with the previously‑found blending factor α is then applied to the test predictions, keeping the same overall workflow while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I reduced the Laplace smoothing constant from 0.1 to 0.01 so that per‑EEG and global class distributions stay sharper and closer to the observed vote counts. This small hyper‑parameter tweak keeps the overall workflow unchanged but yields less‑biased probability estimates, which should lower the KL‑divergence and move the score nearer the target.'
- What this solution (achieved 1.41937) has done: 'I add a patient‑level fallback and tune a three‑way blend (eeg‑specific, patient‑specific, global) on a validation split, then use those optimal weights (plus the existing temperature scaling) when generating the final submission. This hierarchical smoothing should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'The script now uses a slightly larger Laplace smoothing ( 0.01 ) to keep per‑eeg and per‑patient probabilities from becoming over‑confident, and it refines the hierarchical blending weights with a two‑stage grid search (coarse 0.05 step then fine 0.01 step around the best) plus a denser temperature sweep (0.5–2.0 in 0.05 increments). These adjustments stay within the original workflow while aiming to lower the validation KL‑divergence and move the score closer to the target. Cells have been renumbered to start at 1 as required.'
- What this solution (achieved 1.41937) has done: 'I lower the Laplace smoothing constant to 0.001 so the per‑eeg and per‑patient distributions stay sharper, and I broaden the temperature‑scaling search to start at 0.2 (which can further sharpen predictions). These small tweaks keep the original workflow intact while aiming to reduce the KL‑divergence, moving the score closer to the target. I also renumber the notebook cells to start at 1 as required.'
- What this solution (achieved 1.41937) has done: 'I keep the original workflow (hierarchical blending, temperature scaling) but add a tiny grid‑search for the Laplace smoothing constant.  
A slightly larger smoothing (e.g., 0.01 instead of 0.001) reduces over‑confident per‑eeg / per‑patient probabilities, which typically lowers KL‑divergence and moves the score nearer the target.  
The script now (1) computes the initial probabilities with a minimal smoothing (used to tune weights w_eeg, w_pat and temperature T), (2) evaluates a few smoothing candidates on the validation split, picks the best one, and (3) recomputes the final per‑eeg and per‑patient dictionaries with that smoothing before generating the submission.'
- What this solution (achieved 1.4053) has done: 'I keep the overall workflow but replace the hierarchical blending with a pure‑global baseline and apply a stronger temperature (> 1) to smooth the probabilities. This reduces over‑confident per‑eeg/per‑patient estimates that were inflating the KL‑divergence, moving the score closer to the lower‑is‑better target while preserving the existing data handling and submission steps.'
- What this solution (achieved 1.41937) has done: 'I keep the existing workflow but stop overwriting the best hierarchical weights and temperature that were found during validation. By retaining `best_w_eeg`, `best_w_pat`, and `best_temp` instead of resetting them to neutral values, the final predictions use the tuned blending and scaling, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I expand the temperature and Laplace‑smoothing searches to explore a wider range that is more likely to lower the KL‑divergence, and I renumber the notebook cells so they start at 1 as required. These small, focused changes keep the original workflow intact while giving the model a better chance to approach the target score.'
- What this solution (achieved 1.4053) has done: 'I keep the overall workflow but steer predictions toward a more uniform distribution, which tends to lower KL‑divergence when the current score is far above the target. After the validation‑based search I overwrite the hierarchical weights to rely only on the global baseline, increase the temperature to 2.0 (stronger smoothing), and keep the best Laplace smoothing found earlier. This minimal change preserves all data handling and submission steps while moving the score closer to the lower‑is‑better target.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np

PLATFORM = "kaggle"  # change to 'local' if running locally
NEEDTRAIN = False  # training disabled to avoid TF import
DATA_ROOT = (
    "./input/hms-harmful-brain-activity-classification/"
    if PLATFORM == "local"
    else "/kaggle/input/hms-harmful-brain-activity-classification/"
)

train_path = os.path.join(DATA_ROOT, "train.csv")
df = pd.read_csv(train_path)

TARGETS = df.columns[-6:]  # seizure, lpd, gpd, lrda, grda, other
print("Train shape:", df.shape)
print("Target columns:", list(TARGETS))

SMOOTH_INIT = 0.001
epsilon = 1e-12  # tiny value to avoid division by zero

global_counts = df[TARGETS].sum().astype(np.float64) + SMOOTH_INIT
overall_probs = global_counts / (global_counts.sum() + len(TARGETS) * SMOOTH_INIT)
print("\nOverall class probabilities (smoothed, init):")
print(overall_probs)

eeg_groups = df.groupby("eeg_id")[TARGETS].sum()
eeg_counts = eeg_groups + SMOOTH_INIT
eeg_probs = (eeg_counts + epsilon).div(
    eeg_counts.sum(axis=1) + epsilon * len(TARGETS), axis=0
)
eeg_prob_dict = {eid: row.values for eid, row in eeg_probs.iterrows()}

patient_groups = df.groupby("patient_id")[TARGETS].sum()
patient_counts = patient_groups + SMOOTH_INIT
patient_probs = (patient_counts + epsilon).div(
    patient_counts.sum(axis=1) + epsilon * len(TARGETS), axis=0
)
patient_prob_dict = {pid: row.values for pid, row in patient_probs.iterrows()}


def kl_divergence(true_counts, pred_probs, eps=1e-12):
    """KL(p_true || p_pred) for one observation."""
    true_sum = true_counts.sum()
    if true_sum == 0:
        return 0.0
    p_true = true_counts / true_sum
    p_pred = np.clip(pred_probs, eps, 1.0)
    return np.sum(p_true * np.log(p_true / p_pred))




## === cell 1
np.random.seed(42)
mask = np.random.rand(len(df)) < 0.9  # 90 % train, 10 % validation
train_df = df[mask]
val_df = df[~mask]

eeg_groups_val = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_counts_val = eeg_groups_val + SMOOTH_INIT
eeg_probs_val = (eeg_counts_val + epsilon).div(
    eeg_counts_val.sum(axis=1) + epsilon * len(TARGETS), axis=0
)
eeg_prob_dict_val = {eid: row.values for eid, row in eeg_probs_val.iterrows()}

patient_groups_val = train_df.groupby("patient_id")[TARGETS].sum()
patient_counts_val = patient_groups_val + SMOOTH_INIT
patient_probs_val = (patient_counts_val + epsilon).div(
    patient_counts_val.sum(axis=1) + epsilon * len(TARGETS), axis=0
)
patient_prob_dict_val = {pid: row.values for pid, row in patient_probs_val.iterrows()}

global_counts_val = train_df[TARGETS].sum().astype(np.float64) + SMOOTH_INIT
overall_probs_val = global_counts_val / (
    global_counts_val.sum() + len(TARGETS) * SMOOTH_INIT
)

coarse_step = 0.05
best_w_eeg, best_w_pat = 0.0, 0.0
best_kl = float("inf")

for w_eeg in np.arange(0, 1 + coarse_step, coarse_step):
    for w_pat in np.arange(0, 1 - w_eeg + coarse_step, coarse_step):
        w_glob = 1.0 - w_eeg - w_pat
        if w_glob < 0:
            continue
        total_kl = 0.0
        for _, row in val_df.iterrows():
            eid = row["eeg_id"]
            pid = row["patient_id"]
            true_counts = row[TARGETS].values.astype(np.float64)

            per_eeg = eeg_prob_dict_val.get(eid, overall_probs_val.values)
            per_pat = patient_prob_dict_val.get(pid, overall_probs_val.values)

            blended = (
                w_eeg * per_eeg + w_pat * per_pat + w_glob * overall_probs_val.values
            )
            blended = np.clip(blended, 1e-8, None)
            blended = blended / blended.sum()
            total_kl += kl_divergence(true_counts, blended)

        avg_kl = total_kl / len(val_df)
        if avg_kl < best_kl:
            best_kl = avg_kl
            best_w_eeg, best_w_pat = w_eeg, w_pat

fine_step = 0.01
search_range = 0.05  # +/- around the coarse best
w_eeg_start = max(0.0, best_w_eeg - search_range)
w_eeg_end = min(1.0, best_w_eeg + search_range)
w_pat_start = max(0.0, best_w_pat - search_range)
w_pat_end = min(1.0 - w_eeg_start, best_w_pat + search_range)

for w_eeg in np.arange(w_eeg_start, w_eeg_end + fine_step, fine_step):
    for w_pat in np.arange(
        w_pat_start, min(1.0 - w_eeg, w_pat_end) + fine_step, fine_step
    ):
        w_glob = 1.0 - w_eeg - w_pat
        if w_glob < 0:
            continue
        total_kl = 0.0
        for _, row in val_df.iterrows():
            eid = row["eeg_id"]
            pid = row["patient_id"]
            true_counts = row[TARGETS].values.astype(np.float64)

            per_eeg = eeg_prob_dict_val.get(eid, overall_probs_val.values)
            per_pat = patient_prob_dict_val.get(pid, overall_probs_val.values)

            blended = (
                w_eeg * per_eeg + w_pat * per_pat + w_glob * overall_probs_val.values
            )
            blended = np.clip(blended, 1e-8, None)
            blended = blended / blended.sum()
            total_kl += kl_divergence(true_counts, blended)

        avg_kl = total_kl / len(val_df)
        if avg_kl < best_kl:
            best_kl = avg_kl
            best_w_eeg, best_w_pat = w_eeg, w_pat

print(
    f"\nBest hierarchical weights after refinement: w_eeg={best_w_eeg:.2f}, "
    f"w_pat={best_w_pat:.2f}, w_glob={1-best_w_eeg-best_w_pat:.2f} "
    f"(validation KL ≈ {best_kl:.5f})"
)




## === cell 2
def apply_temperature(probs, temp):
    """Scale probabilities with temperature T (temp > 0)."""
    if abs(temp - 1.0) < 1e-12:
        return probs
    scaled = np.clip(probs, 1e-12, 1.0) ** (1.0 / temp)
    return scaled / scaled.sum()


temps = np.arange(0.5, 3.05, 0.05)
best_temp = 1.0
best_kl_temp = best_kl

for t in temps:
    total_kl = 0.0
    for _, row in val_df.iterrows():
        eid = row["eeg_id"]
        pid = row["patient_id"]
        true_counts = row[TARGETS].values.astype(np.float64)

        per_eeg = eeg_prob_dict_val.get(eid, overall_probs_val.values)
        per_pat = patient_prob_dict_val.get(pid, overall_probs_val.values)

        blended = (
            best_w_eeg * per_eeg
            + best_w_pat * per_pat
            + (1 - best_w_eeg - best_w_pat) * overall_probs_val.values
        )
        blended = apply_temperature(blended, t)
        blended = np.clip(blended, 1e-8, None)
        blended = blended / blended.sum()
        total_kl += kl_divergence(true_counts, blended)

    avg_kl = total_kl / len(val_df)
    if avg_kl < best_kl_temp:
        best_kl_temp = avg_kl
        best_temp = t

print(f"\nChosen temperature T = {best_temp:.2f} (validation KL ≈ {best_kl_temp:.5f})")




## === cell 3
smooth_candidates = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
]  # added larger smoothing options
best_smooth = SMOOTH_INIT
best_smooth_kl = best_kl_temp

for s in smooth_candidates:
    glob_cnt = train_df[TARGETS].sum().astype(np.float64) + s
    glob_probs = glob_cnt / (glob_cnt.sum() + len(TARGETS) * s)

    eeg_grp = train_df.groupby("eeg_id")[TARGETS].sum()
    eeg_cnt = eeg_grp + s
    eeg_pr = (eeg_cnt + epsilon).div(
        eeg_cnt.sum(axis=1) + epsilon * len(TARGETS), axis=0
    )
    eeg_dict = {eid: row.values for eid, row in eeg_pr.iterrows()}

    pat_grp = train_df.groupby("patient_id")[TARGETS].sum()
    pat_cnt = pat_grp + s
    pat_pr = (pat_cnt + epsilon).div(
        pat_cnt.sum(axis=1) + epsilon * len(TARGETS), axis=0
    )
    pat_dict = {pid: row.values for pid, row in pat_pr.iterrows()}

    total_kl = 0.0
    for _, row in val_df.iterrows():
        eid = row["eeg_id"]
        pid = row["patient_id"]
        true_counts = row[TARGETS].values.astype(np.float64)

        per_eeg = eeg_dict.get(eid, glob_probs.values)
        per_pat = pat_dict.get(pid, glob_probs.values)

        blended = (
            best_w_eeg * per_eeg
            + best_w_pat * per_pat
            + (1 - best_w_eeg - best_w_pat) * glob_probs.values
        )
        blended = apply_temperature(blended, best_temp)
        blended = np.clip(blended, 1e-8, None)
        blended = blended / blended.sum()
        total_kl += kl_divergence(true_counts, blended)

    avg_kl = total_kl / len(val_df)
    if avg_kl < best_smooth_kl:
        best_smooth_kl = avg_kl
        best_smooth = s

print(
    f"\nChosen Laplace smoothing = {best_smooth:.5f} (validation KL ≈ {best_smooth_kl:.5f})"
)

best_w_eeg = 0.0  # ignore per‑eeg specific distribution
best_w_pat = 0.0  # ignore per‑patient specific distribution
best_temp = 2.0  # stronger smoothing (reduces over‑confidence)

global_counts = df[TARGETS].sum().astype(np.float64) + best_smooth
overall_probs = global_counts / (global_counts.sum() + len(TARGETS) * best_smooth)

eeg_groups = df.groupby("eeg_id")[TARGETS].sum()
eeg_counts = eeg_groups + best_smooth
eeg_probs = (eeg_counts + epsilon).div(
    eeg_counts.sum(axis=1) + epsilon * len(TARGETS), axis=0
)
eeg_prob_dict = {eid: row.values for eid, row in eeg_probs.iterrows()}

patient_groups = df.groupby("patient_id")[TARGETS].sum()
patient_counts = patient_groups + best_smooth
patient_probs = (patient_counts + epsilon).div(
    patient_counts.sum(axis=1) + epsilon * len(TARGETS), axis=0
)
patient_prob_dict = {pid: row.values for pid, row in patient_probs.iterrows()}




## === cell 4
test_path = os.path.join(DATA_ROOT, "test.csv")
test = pd.read_csv(test_path)
print("\nTest shape:", test.shape)
print(test.head())

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

for idx, eid in enumerate(sub["eeg_id"]):
    pid = test.loc[test["eeg_id"] == eid, "patient_id"].values[0]

    per_eeg = eeg_prob_dict.get(eid, overall_probs.values)
    per_pat = patient_prob_dict.get(pid, overall_probs.values)

    blended = (
        best_w_eeg * per_eeg
        + best_w_pat * per_pat
        + (1 - best_w_eeg - best_w_pat) * overall_probs.values
    )
    blended = apply_temperature(blended, best_temp)

    blended = np.clip(blended, 1e-8, None)
    blended = blended / blended.sum()

    for col, prob in zip(TARGETS, blended):
        sub.at[idx, col] = prob

row_sums = sub[TARGETS].sum(axis=1)
if not np.allclose(row_sums, 1.0):
    sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print("\nSubmission saved to", submission_path)
print("\nSubmission preview:")
print(sub.head())
print("\nRow sum check (unique values, should be 1.0):")
print(sub[TARGETS].sum(axis=1).unique())
