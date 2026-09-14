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

0.3077253736303449

# 6. Current score

0.79199

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The notebook fails because `src` isn’t importable in the current runtime: `sys.path.append(...)` points to a dataset that doesn’t exist here, so `from src.settings import TARGET_COLS` errors and prevents ensembling/writing `submission.csv`. I fix this by using the already-available `sample_submission.csv` header to define `TARGET_COLS` and by making `merge_preds` robust (ensuring all required fold files exist and preserving `eeg_id` alignment). I also avoid `!pip install`/external-code calls that can’t work in this environment (given your installed packages list) and instead ensemble whatever per-fold submission files are present; if none exist, it safely fall back to a valid uniform-probability submission. This keeps the “core logic” of your script (averaging/normalizing model predictions) while ensuring an end-to-end run that always produces a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.40995, lower-is-better) strongly suggests the script is falling back to the uniform-probability submission because no per-fold prediction CSVs are found in `/kaggle/working`, which is typically much worse than even a simple baseline. To move toward the target, the smallest legitimate improvement is to generate a non-uniform, data-driven prior from `train.csv` vote proportions and use it as the fallback (and also to fill any missing rows when merging partial prediction files). This preserves your core logic (weighted averaging + normalization) and only changes the fallback behavior and missing-row fill behavior, which directly affects the final probabilities and KL score. The resulting submission remains valid (correct columns, alignment, row sums to 1) and should improve substantially versus uniform without changing any model/training code.'
- What this solution (achieved 1.68479) has done: 'Your current score indicates the script is still effectively producing a weak fallback (train prior only), likely because no per-fold prediction files are present; we can legitimately improve the fallback toward the target by making it conditional on `patient_id`, which is available in both train and test. This keeps your core logic intact (merge/average/normalize) and only improves the “no-fold-files / missing-rows” fill distribution, which directly impacts KL divergence. Concretely, we compute a per-patient empirical class prior from `train.csv` (vote proportions aggregated by patient), fall back to the global prior for unseen patients, and use these patient-conditioned priors both for full fallback and for filling missing `eeg_id` rows during merging. This is a minimal, score-relevant change and still guarantees a valid submission with row sums = 1.'
- What this solution (achieved 1.68479) has done: 'Your score (1.68479, lower-is-better) is far from the target (0.3077), and the debug cell strongly suggests you’re still not actually ensembling model outputs most of the time (i.e., `used==0` or nearly so), so the submission is dominated by the fallback distribution. The smallest score-relevant improvement without changing any modeling/training logic is to (1) make file discovery flexible (so it finds your real per-fold/per-version CSV names in `/kaggle/working`), and (2) make the fallback stronger by conditioning not just on `patient_id` but also on `spectrogram_id` (available in both train/test), with safe backoff to patient prior then global prior. This keeps the same core semantics (weighted averaging + normalization), but greatly increases the chance you use real predictions when present and makes missing-row fills closer to the true distribution when they aren’t. The submission format and row-sum-to-1 constraint are preserved.'
- What this solution (achieved 1.68479) has done: 'Your current score suggests you’re still mostly (or entirely) using the metadata-based fallback rather than real model predictions, so the smallest score-relevant improvement is to make the file discovery and merge more forgiving to common filename/format variations and to ensure we correctly average across folds/versions. I (1) accept any prediction CSVs in `/kaggle/working` that contain the required columns (not just ones with fold/version tokens), (2) ensemble all matched files with equal weights (preserving your “average then normalize” core logic) while keeping your explicit weights when exact fold/version matches exist, and (3) apply a light probability “floor + renormalize” after ensembling to avoid extreme zeros that can hurt KL. These changes keep the same semantics (probability ensembling + normalization) but greatly increase the chance of using your actual model outputs, which should move KL down toward the target.'
- What this solution (achieved 1.68851) has done: 'Your current KL (1.68479; lower is better) is far above the target (0.3077), and the debug behavior implies you’re still effectively not using any real per-fold model predictions, so you’re submitting mostly metadata priors. The minimal score-relevant change is to add a very lightweight “semi-supervised” fallback that uses the test EEG itself (available at inference) to pick a closer prior: compute simple amplitude/variance features from each test EEG parquet and map it to a nearest-neighbor prior computed from the same features on a small subset of train EEGs. This preserves your core logic (ensemble/average/normalize) and only changes the fill/fallback distribution when prediction CSVs are missing or incomplete, which directly impacts KL. To keep runtime within 600s, it samples a limited number of train EEGs and uses only a few cheap features; it also caches computed features to disk in `/kaggle/working`.'
- What this solution (achieved 0.81181) has done: 'Your current KL (1.68851, lower-is-better) is far worse than the target (0.3077), and the printouts strongly suggest you’re still mostly not using any real model prediction CSVs—so the submission is dominated by fallback priors. The smallest score-relevant improvement without changing any model/training logic is to make the fallback more “label-like” by conditioning on metadata that exists in both train and test: (1) build per-spectrogram and per-patient priors using *train vote probabilities* (not raw summed votes), (2) optionally blend those with your existing EEG kNN prior when available (so it doesn’t fully override), and (3) remove the relatively aggressive probability floor (1e-4) and replace it with a tiny epsilon to avoid hurting KL via over-smoothing. These changes keep your core semantics (ensemble → normalize; fallback only when needed) but should move KL materially downward toward the target when prediction files are missing/incomplete. The script still runs end-to-end and always writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.81181) has done: 'Your current KL (0.81181, lower-is-better) is still far above the target (0.3077), so we should cautiously improve performance, not degrade it. The biggest likely issue is that when you *do* find real model prediction CSVs, you currently **do not** fill NaNs within those CSVs unless the `eeg_id` order doesn’t match; that can quietly inject near-uniform/renormalized noise and worsen KL. I make filling missing/invalid prediction values **always** use your (stronger) `fill_mat` backoff, keep the same ensembling/normalization core logic, and add a tiny, metric-safe “sharpening” (temperature < 1) after ensembling to better match KL without changing semantics (still probabilities summing to 1). These changes are minimal, deterministic, and only affect post-processing/robustness, so they should move KL downward toward the target.'
- What this solution (achieved 0.81181) has done: 'You’re still far from the target (0.81181 vs 0.3077, lower-is-better), so we should improve cautiously with minimal semantic changes. The biggest likely gain without touching any training/modeling is to make ensembling robust to *bad but finite* values (negatives, >1, row-sums not 1) inside discovered prediction CSVs, because these can silently distort probabilities and inflate KL. I add a strict “sanitize to probabilities” step for every loaded prediction (clip to [eps,1], renormalize), then proceed with the same weighted averaging + normalization you already do. I keep your existing fallback priors/NN logic and temperature as-is, only ensuring all inputs to the ensemble are valid probabilities so the final distribution is more reliable and typically lowers KL.'
- What this solution (achieved 0.81181) has done: 'Your current KL (0.81181, lower-is-better) is still far above the target (0.3077), so we should improve (decrease KL) with minimal, post-processing-only changes. The biggest low-risk gain is to stop forcing predictions into a fixed [eps,1] range that can distort already-good model probabilities; instead, we only clip negatives, then renormalize, and only add epsilon *after* normalization to prevent exact zeros (KL-sensitive) without oversmoothing. We also make the temperature sharpening slightly milder (closer to 1.0) to avoid overconfident distributions that can worsen KL on the private set. Finally, we keep your fallback/NN/meta priors unchanged, but ensure any missing/invalid values are filled then normalized consistently.'
- What this solution (achieved 0.81181) has done: 'We keep your ensemble + normalize pipeline unchanged, and only make small post-processing adjustments that typically reduce KL without altering model outputs. Specifically: (1) make the temperature transform truly “mild” by moving it closer to neutral (toward 1.0) and (2) add a very small blend toward your metadata/NN fill prior after ensembling (a common KL-stabilizer that helps when any model file is slightly miscalibrated). We also ensure the same blend is applied in the fallback-only case for consistency and keep all probabilities strictly valid (non-negative, finite, row-sum=1). These changes are minimal, deterministic, and should move your KL down from 0.81181 toward the 0.3077 target without changing any training/model architecture.'
- What this solution (achieved 0.8083) has done: 'We keep your ensemble/normalize pipeline intact and only adjust post-processing knobs that affect calibration for KL: (1) make the prior-blend slightly stronger so predictions are pulled a bit toward a safer metadata/NN prior when model CSVs are imperfect, and (2) set temperature to neutral (1.0) to avoid over-sharpening that can inflate KL when the model is already somewhat miscalibrated. We also slightly increase the epsilon used after normalization to reduce KL blow-ups from near-zero probabilities without forcing heavy smoothing. These are minimal, deterministic changes that typically reduce KL from 0.81 toward your 0.307 target without changing any modeling/training logic or file I/O.'
- What this solution (achieved 0.81181) has done: 'We keep your ensemble/normalize pipeline intact and only make two calibration changes that typically reduce KL without touching any modeling: (1) set `final_prior_blend` back down to a very small value so we don’t over-pull good model predictions toward metadata priors, and (2) make the probability floor even smaller (near-metric-neutral) to avoid unnecessary smoothing that can worsen KL when predictions are already well-formed. We also ensure the fallback path uses exactly the same post-processing as the ensemble path (sanitize → optional temperature → tiny prior blend) so behavior is consistent. These are minimal, deterministic post-processing tweaks aimed at lowering your KL from 0.8083 toward 0.3077.'
- What this solution (achieved 0.78483) has done: 'We keep your ensemble/normalize pipeline intact and only tweak post-processing in ways that typically reduce KL without changing any model/training logic. Specifically, we (1) apply a tiny amount of *label-like smoothing* by blending predictions with the per-row `fill_mat` prior **before** the final normalization (this stabilizes KL when any model file is slightly miscalibrated), and (2) make the final prior blend conditional (weaker when real prediction files are used, stronger only in fallback mode). We also slightly reduce the NN prior influence (`nn_blend`) so the metadata priors don’t get overridden too aggressively, which can hurt KL if the NN prior is noisy. All changes preserve valid probabilities (finite, non-negative, row-sum=1) and still produce `submission.csv`.'
- What this solution (achieved 0.79199) has done: 'We make two minimal, score-relevant calibration tweaks to better match KL divergence while preserving your exact ensemble→normalize core logic. First, we reduce the fallback/metadata influence during ensembling by weakening the pre-normalization fill prior blend, because your current score (0.7848) suggests the blend is likely over-pulling predictions away from model outputs. Second, we apply a very mild “softening” temperature (>1) to reduce overconfident spikes (which KL penalizes heavily when wrong), while keeping all probabilities valid and normalized. These are tiny post-processing changes only, deterministic, and should move KL downward toward your 0.3077 target without touching any training/modeling.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import re
import numpy as np
import pandas as pd

np.random.seed(42)



## === cell 1
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)

SAMPLE_SUB_PATH = os.path.join(DATA_PATH, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_PATH, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_PATH, "train.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]

assert "eeg_id" in sample_sub.columns
assert len(TARGET_COLS) == 6

test_df = pd.read_csv(TEST_CSV_PATH, usecols=["eeg_id", "patient_id", "spectrogram_id"])
test_df = test_df.sort_values("eeg_id").reset_index(drop=True)




## === cell 2
def compute_train_priors(train_csv_path: str, target_cols):
    """
    Score-relevant fallback/fill improvement only:
    - global prior from all train votes
    - patient-conditioned prior (patient_id)
    - spectrogram-conditioned prior (spectrogram_id)

    Build group priors by averaging per-row vote-probabilities (not summing raw votes),
    which better matches the evaluation target distribution.
    Backoff order used later: spectrogram -> patient -> global.
    """
    usecols = ["patient_id", "spectrogram_id"] + list(target_cols)
    train = pd.read_csv(train_csv_path, usecols=usecols)

    votes = train[target_cols].to_numpy(dtype=np.float64)
    votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = votes.sum(axis=1, keepdims=True)
    good = (row_sum[:, 0] > 0) & np.isfinite(row_sum[:, 0])
    train = train.loc[good].copy()
    votes = votes[good]
    row_sum = row_sum[good]

    row_prob = votes / row_sum
    row_prob = np.clip(row_prob, 1e-15, 1.0)
    row_prob = row_prob / row_prob.sum(axis=1, keepdims=True)

    global_prior = row_prob.mean(axis=0)
    if (not np.isfinite(global_prior).all()) or global_prior.sum() <= 0:
        global_prior = np.full(
            len(target_cols), 1.0 / len(target_cols), dtype=np.float64
        )
    global_prior = np.clip(global_prior, 1e-15, 1.0)
    global_prior = global_prior / global_prior.sum()

    for j, c in enumerate(target_cols):
        train[c] = row_prob[:, j]

    def _make_group_priors(group_key: str):
        grp = train.groupby(group_key)[target_cols].mean()
        arr = grp.to_numpy(dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 1e-15, 1.0)
        arr = arr / arr.sum(axis=1, keepdims=True)

        priors = {}
        for key, d in zip(grp.index.tolist(), arr):
            try:
                key = int(key)
            except Exception:
                pass
            priors[key] = d.astype(np.float64, copy=True)
        return priors

    patient_priors = _make_group_priors("patient_id")
    spectrogram_priors = _make_group_priors("spectrogram_id")
    return global_prior, patient_priors, spectrogram_priors


TRAIN_PRIOR, PATIENT_PRIORS, SPECTRO_PRIORS = compute_train_priors(
    TRAIN_CSV_PATH, TARGET_COLS
)




## === cell 3
def _safe_normalize_rows(mat, eps=0.0):
    mat = np.asarray(mat, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    if eps > 0:
        mat = np.clip(mat, eps, None)
    else:
        mat = np.clip(mat, 0.0, None)
    rs = mat.sum(axis=1, keepdims=True)
    rs = np.where(rs <= 0, 1.0, rs)
    return mat / rs


def _discover_pred_files(work_dir, folds, versions):
    """
    Score-relevant: broaden discovery so we actually use model prediction files present in /kaggle/working.
    Priority:
      1) exact expected names: submission_fold{fold}_{version}.csv
      2) fold/version token match in filename
      3) any other CSV that looks like a submission (has eeg_id + all TARGET_COLS)
    """
    all_csv = glob.glob(os.path.join(work_dir, "*.csv"))
    found = {}  # (fold, version) -> path

    for fold in folds:
        for version in versions:
            p = os.path.join(work_dir, f"submission_fold{fold}_{version}.csv")
            if os.path.exists(p):
                found[(fold, version)] = p

    if len(found) < len(folds) * len(versions):
        for p in all_csv:
            bn = os.path.basename(p).lower()
            if bn in ("submission.csv", "sample_submission.csv"):
                continue

            for fold in folds:
                fold_ok = (f"fold{fold}" in bn) or (
                    re.search(rf"\bfold[_\- ]?{fold}\b", bn) is not None
                )
                if not fold_ok:
                    continue
                for version in versions:
                    vlow = version.lower()
                    ver_ok = (vlow in bn) or (
                        re.search(rf"\b{re.escape(vlow)}\b", bn) is not None
                    )
                    if not ver_ok:
                        continue
                    if (fold, version) not in found:
                        found[(fold, version)] = p

    plausible = []
    if len(found) == 0:
        for p in all_csv:
            bn = os.path.basename(p).lower()
            if bn in ("submission.csv", "sample_submission.csv"):
                continue
            try:
                df_head = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            if "eeg_id" in df_head.columns and all(
                c in df_head.columns for c in TARGET_COLS
            ):
                plausible.append(p)

    return found, plausible


EEG_TEST_DIR = os.path.join(DATA_PATH, "test_eegs")
EEG_TRAIN_DIR = os.path.join(DATA_PATH, "train_eegs")
EEG_FEATURE_CACHE = os.path.join(OUT_PATH, "eeg_feat_cache.npz")


def _eeg_feat_from_parquet(path: str):
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df is None or df.shape[0] == 0:
        return None

    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
    if X.size == 0:
        return None
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    absX = np.abs(X)
    feat = np.array(
        [
            float(np.mean(absX)),
            float(np.std(X)),
            float(np.quantile(absX, 0.90)),
            float(np.quantile(absX, 0.99)),
            float(np.mean(X * X)),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(feat).all():
        return None
    return feat


def build_eeg_nn_prior(
    train_csv_path: str,
    train_eeg_dir: str,
    test_eeg_dir: str,
    target_cols,
    out_cache_path: str,
    n_train_eegs: int = 1200,
    n_test_eegs: int = 9850,
    k: int = 25,
):
    """
    Produces per-test-eeg prior matrix (n_test, n_classes) via kNN in a small feature space.
    Runtime bounded by sampling n_train_eegs and computing 5 features per file.
    """
    if os.path.exists(out_cache_path):
        try:
            z = np.load(out_cache_path, allow_pickle=False)
            if (
                "test_eeg_id" in z
                and "test_feat" in z
                and "train_feat" in z
                and "train_prior" in z
                and "target_cols" in z
            ):
                cached_cols = list(z["target_cols"])
                if cached_cols == list(target_cols):
                    return (
                        z["test_eeg_id"].astype(np.int64),
                        z["test_feat"].astype(np.float64),
                        z["train_feat"].astype(np.float64),
                        z["train_prior"].astype(np.float64),
                    )
        except Exception:
            pass

    train_meta = pd.read_csv(train_csv_path, usecols=["eeg_id"] + list(target_cols))
    g = train_meta.groupby("eeg_id")[target_cols].sum()
    g_arr = g.to_numpy(dtype=np.float64)
    g_rs = g_arr.sum(axis=1, keepdims=True)
    g_rs = np.where(g_rs <= 0, 1.0, g_rs)
    g_prob = g_arr / g_rs
    g_prob = np.clip(g_prob, 1e-15, 1.0)
    g_prob = g_prob / g_prob.sum(axis=1, keepdims=True)
    train_eeg_ids_all = g.index.to_numpy(dtype=np.int64)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(train_eeg_ids_all))
    train_feats = []
    train_priors = []
    kept_ids = []
    for idx in perm:
        if len(train_feats) >= n_train_eegs:
            break
        eid = int(train_eeg_ids_all[idx])
        p = os.path.join(train_eeg_dir, f"{eid}.parquet")
        if not os.path.exists(p):
            continue
        feat = _eeg_feat_from_parquet(p)
        if feat is None:
            continue
        train_feats.append(feat)
        train_priors.append(g_prob[idx])
        kept_ids.append(eid)

    if len(train_feats) < 50:
        return None

    train_feat = np.vstack(train_feats).astype(np.float64)
    train_prior = np.vstack(train_priors).astype(np.float64)

    test_ids = sorted(
        [
            int(os.path.splitext(os.path.basename(p))[0])
            for p in glob.glob(os.path.join(test_eeg_dir, "*.parquet"))
        ]
    )
    if n_test_eegs is not None:
        test_ids = test_ids[:n_test_eegs]

    test_feats = []
    test_ids_kept = []
    for eid in test_ids:
        p = os.path.join(test_eeg_dir, f"{eid}.parquet")
        feat = _eeg_feat_from_parquet(p)
        if feat is None:
            continue
        test_feats.append(feat)
        test_ids_kept.append(int(eid))

    if len(test_feats) == 0:
        return None

    test_feat = np.vstack(test_feats).astype(np.float64)
    test_eeg_id = np.array(test_ids_kept, dtype=np.int64)

    try:
        np.savez_compressed(
            out_cache_path,
            test_eeg_id=test_eeg_id,
            test_feat=test_feat,
            train_feat=train_feat,
            train_prior=train_prior,
            target_cols=np.array(list(target_cols), dtype=object),
        )
    except Exception:
        pass

    return test_eeg_id, test_feat, train_feat, train_prior


NN_CACHE = build_eeg_nn_prior(
    TRAIN_CSV_PATH,
    EEG_TRAIN_DIR,
    EEG_TEST_DIR,
    TARGET_COLS,
    EEG_FEATURE_CACHE,
    n_train_eegs=1200,
    n_test_eegs=None,
    k=25,
)


def _nn_prior_for_test_ids(test_eeg_ids_sorted, global_prior, nn_cache, k=25):
    """
    Returns (n_test, n_classes) priors aligned to test_eeg_ids_sorted.
    If nn_cache missing/unusable, returns None.
    """
    if nn_cache is None:
        return None
    test_eeg_id, test_feat, train_feat, train_prior = nn_cache

    mu = train_feat.mean(axis=0, keepdims=True)
    sd = train_feat.std(axis=0, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)

    tr = (train_feat - mu) / sd
    te = (test_feat - mu) / sd

    idx_map = {int(eid): i for i, eid in enumerate(test_eeg_id.tolist())}

    out = np.empty((len(test_eeg_ids_sorted), train_prior.shape[1]), dtype=np.float64)
    out[:] = global_prior[None, :]

    train_prior = train_prior.astype(np.float64, copy=False)
    for i, eid in enumerate(test_eeg_ids_sorted):
        j = idx_map.get(int(eid), None)
        if j is None:
            continue
        v = te[j]
        d2 = np.sum((tr - v[None, :]) ** 2, axis=1)
        if k >= len(d2):
            nn_idx = np.argsort(d2)
        else:
            nn_idx = np.argpartition(d2, kth=k - 1)[:k]
            nn_idx = nn_idx[np.argsort(d2[nn_idx])]
        prior = train_prior[nn_idx].mean(axis=0)
        prior = np.clip(prior, 1e-15, 1.0)
        prior = prior / prior.sum()
        out[i] = prior
    return out


def _fill_missing_with_fillmat(arr: np.ndarray, fill_mat: np.ndarray):
    """
    Always fill NaN/inf entries inside prediction arrays using fill_mat backoff.
    """
    arr = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr = arr.copy()
        arr[bad] = fill_mat[bad]
    return arr


def _sanitize_pred_probs(arr: np.ndarray, eps_after_norm: float = 1e-15) -> np.ndarray:
    """
    Score-relevant:
    - set non-finite -> 0, clip negatives -> 0, normalize rows
    - then apply a tiny epsilon floor AFTER normalization (KL-safe), and renormalize
    """
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)

    rs = arr.sum(axis=1, keepdims=True)
    rs = np.where(rs <= 0, 1.0, rs)
    arr = arr / rs

    eps = float(eps_after_norm)
    if eps > 0:
        arr = np.clip(arr, eps, None)
        arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def _apply_temperature(pred: np.ndarray, temp: float, eps_after: float = 1e-15):
    """
    Mild sharpening/softening with safety:
    - Apply power transform, renormalize.
    - Add tiny eps AFTER transform to avoid exact zeros, renormalize.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if temp is None or float(temp) == 1.0:
        return pred
    t = float(temp)
    t = max(0.5, min(2.0, t))

    pred = np.clip(pred, 0.0, None)
    pred = _safe_normalize_rows(pred)

    pred = np.clip(pred, 1e-300, None)
    pred = pred ** (1.0 / t)
    pred = pred / pred.sum(axis=1, keepdims=True)

    eps = float(eps_after)
    if eps > 0:
        pred = np.clip(pred, eps, None)
        pred = pred / pred.sum(axis=1, keepdims=True)
    return pred


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.3, 0.3, 0.4),
    base_submission_path=SAMPLE_SUB_PATH,
    work_dir=OUT_PATH,
    test_meta=None,
    global_prior=TRAIN_PRIOR,
    patient_priors=PATIENT_PRIORS,
    spectrogram_priors=SPECTRO_PRIORS,
    prob_floor=1e-15,
    nn_cache=NN_CACHE,
    nn_k=25,
    nn_blend=0.28,
    temp=1.0,
    final_prior_blend=0.02,
    pre_norm_fill_blend=0.015,
):
    """
    Core logic preserved: (weighted) averaging of prediction files then row-wise normalization.

    Score-relevant knobs:
      - pre_norm_fill_blend: tiny stabilizer toward fill_mat (too large can hurt by washing out model signal)
      - temp: mild softening (>1) can reduce KL blow-ups from overconfident wrong probabilities
    """
    base = pd.read_csv(base_submission_path).copy()
    base = base.sort_values("eeg_id").reset_index(drop=True)

    if test_meta is None:
        test_meta = pd.DataFrame({"eeg_id": base["eeg_id"].values})
    else:
        test_meta = test_meta.copy()
        test_meta = test_meta.sort_values("eeg_id").reset_index(drop=True)
        if not np.array_equal(test_meta["eeg_id"].values, base["eeg_id"].values):
            test_meta = base[["eeg_id"]].merge(test_meta, on="eeg_id", how="left")

    if "patient_id" not in test_meta.columns:
        test_meta["patient_id"] = -1
    if "spectrogram_id" not in test_meta.columns:
        test_meta["spectrogram_id"] = -1
    test_meta["patient_id"] = test_meta["patient_id"].fillna(-1)
    test_meta["spectrogram_id"] = test_meta["spectrogram_id"].fillna(-1)

    global_prior = np.asarray(global_prior, dtype=np.float64)
    if (
        global_prior.shape != (len(TARGET_COLS),)
        or not np.isfinite(global_prior).all()
        or global_prior.sum() <= 0
    ):
        global_prior = np.full(
            len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
    global_prior = np.clip(global_prior, 1e-15, 1.0)
    global_prior = global_prior / global_prior.sum()

    nn_prior = _nn_prior_for_test_ids(
        test_eeg_ids_sorted=base["eeg_id"].to_numpy(dtype=np.int64),
        global_prior=global_prior,
        nn_cache=nn_cache,
        k=int(nn_k),
    )

    fill_mat = np.empty((len(base), len(TARGET_COLS)), dtype=np.float64)
    pids = test_meta["patient_id"].to_numpy()
    sids = test_meta["spectrogram_id"].to_numpy()
    for i, (pid, sid) in enumerate(zip(pids, sids)):
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = -1
        try:
            pid_int = int(pid)
        except Exception:
            pid_int = -1

        prior_meta = spectrogram_priors.get(sid_int, None)
        if prior_meta is None:
            prior_meta = patient_priors.get(pid_int, global_prior)

        if nn_prior is not None:
            a = float(nn_blend)
            prior = a * nn_prior[i, :] + (1.0 - a) * prior_meta
            prior = np.clip(prior, 1e-15, 1.0)
            prior = prior / prior.sum()
            fill_mat[i, :] = prior
        else:
            fill_mat[i, :] = prior_meta

    pred_sum = np.zeros((len(base), len(TARGET_COLS)), dtype=np.float64)
    weight_sum = 0.0
    used_files = []

    file_map, plausible_files = _discover_pred_files(work_dir, folds, versions)

    eps_floor = max(1e-15, float(prob_floor))

    for fold in folds:
        for version, w in zip(versions, weights):
            path = file_map.get((fold, version), None)
            if path is None or (not os.path.exists(path)):
                continue

            df = pd.read_csv(path)
            if "eeg_id" not in df.columns:
                continue
            if any(c not in df.columns for c in TARGET_COLS):
                continue

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = df.sort_values("eeg_id").reset_index(drop=True)

            if not np.array_equal(df["eeg_id"].values, base["eeg_id"].values):
                df = base[["eeg_id"]].merge(df, on="eeg_id", how="left")

            arr = df[TARGET_COLS].to_numpy(dtype=np.float64)
            arr = _fill_missing_with_fillmat(arr, fill_mat)
            arr = _sanitize_pred_probs(arr, eps_after_norm=eps_floor)

            pred_sum += arr * float(w)
            weight_sum += float(w)
            used_files.append(path)

    if weight_sum == 0.0 and plausible_files:
        w = 1.0
        for path in plausible_files:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "eeg_id" not in df.columns:
                continue
            if any(c not in df.columns for c in TARGET_COLS):
                continue

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = df.sort_values("eeg_id").reset_index(drop=True)
            if not np.array_equal(df["eeg_id"].values, base["eeg_id"].values):
                df = base[["eeg_id"]].merge(df, on="eeg_id", how="left")

            arr = df[TARGET_COLS].to_numpy(dtype=np.float64)
            arr = _fill_missing_with_fillmat(arr, fill_mat)
            arr = _sanitize_pred_probs(arr, eps_after_norm=eps_floor)

            pred_sum += arr * w
            weight_sum += w
            used_files.append(path)

    if weight_sum == 0.0:
        pred = _sanitize_pred_probs(fill_mat, eps_after_norm=eps_floor)
        pred = _apply_temperature(pred, temp=float(temp), eps_after=eps_floor)

        b = float(final_prior_blend)
        b = max(0.0, min(0.2, b))
        if b > 0:
            pred = (1.0 - b) * pred + b * _safe_normalize_rows(fill_mat)
            pred = _safe_normalize_rows(pred, eps=eps_floor)

        base[TARGET_COLS] = pred
        base.attrs["used_files"] = []
        base.attrs["used_mode"] = (
            "fallback_priors_blend_nn" if (nn_prior is not None) else "fallback_priors"
        )
        return base

    pred = pred_sum / weight_sum
    pred = _safe_normalize_rows(pred)

    a = float(pre_norm_fill_blend)
    a = max(0.0, min(0.05, a))
    if a > 0:
        pred = (1.0 - a) * pred + a * _safe_normalize_rows(fill_mat)
        pred = _safe_normalize_rows(pred)

    pred = np.clip(pred, eps_floor, None)
    pred = pred / pred.sum(axis=1, keepdims=True)

    pred = _apply_temperature(pred, temp=float(temp), eps_after=eps_floor)

    b = float(final_prior_blend) * 0.5
    b = max(0.0, min(0.2, b))
    if b > 0:
        pred = (1.0 - b) * pred + b * _safe_normalize_rows(fill_mat)
        pred = _safe_normalize_rows(pred, eps=eps_floor)

    base[TARGET_COLS] = pred
    base.attrs["used_files"] = used_files
    base.attrs["used_mode"] = "ensemble"
    return base




## === cell 4
sol = merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.3, 0.3, 0.4),
    test_meta=test_df,
    global_prior=TRAIN_PRIOR,
    patient_priors=PATIENT_PRIORS,
    spectrogram_priors=SPECTRO_PRIORS,
    prob_floor=1e-15,
    nn_cache=NN_CACHE,
    nn_k=25,
    nn_blend=0.28,
    temp=1.08,  # was 1.0
    final_prior_blend=0.02,
    pre_norm_fill_blend=0.006,  # was 0.015
)

assert list(sol.columns) == ["eeg_id"] + TARGET_COLS
assert sol[TARGET_COLS].isna().sum().sum() == 0
row_sum = sol[TARGET_COLS].sum(axis=1).values
assert np.allclose(row_sum, 1.0, rtol=1e-6, atol=1e-6)

SUB_PATH = os.path.join(OUT_PATH, "submission.csv")
sol.to_csv(SUB_PATH, index=False)

print(f"Wrote: {SUB_PATH}")
print("Used mode:", sol.attrs.get("used_mode", None))
used_files = sol.attrs.get("used_files", [])
print(f"Used {len(used_files)} prediction files.")
print("First few used files:", used_files[:10])
print("Global train prior (backoff):", dict(zip(TARGET_COLS, TRAIN_PRIOR.round(6))))
print(f"Patient priors available for {len(PATIENT_PRIORS)} train patients.")
print(f"Spectrogram priors available for {len(SPECTRO_PRIORS)} train spectrograms.")
print("NN cache available:", NN_CACHE is not None)
print(sol.head())



## === cell 5
expected = []
found_exact = []
for fold in (0, 1, 2, 3, 4):
    for version in ("v0", "v2", "v3"):
        p = os.path.join(OUT_PATH, f"submission_fold{fold}_{version}.csv")
        expected.append(p)
        if os.path.exists(p):
            found_exact.append(p)

file_map, plausible_files = _discover_pred_files(
    OUT_PATH, folds=(0, 1, 2, 3, 4), versions=("v0", "v2", "v3")
)

print(
    f"Found {len(found_exact)}/{len(expected)} exact expected per-fold submission files."
)
print(
    f"Found {len(file_map)}/{len(expected)} fold/version files via flexible discovery."
)
print(
    f"Found {len(plausible_files)} plausible prediction CSVs (eeg_id + all target cols)."
)

if file_map:
    ex_key = sorted(file_map.keys())[0]
    print("Example flexible match:", ex_key, "->", file_map[ex_key])
elif plausible_files:
    print("Example plausible file:", plausible_files[0])
else:
    print(
        "No usable prediction CSVs found. Working dir CSVs:",
        glob.glob(os.path.join(OUT_PATH, "*.csv"))[:50],
    )
