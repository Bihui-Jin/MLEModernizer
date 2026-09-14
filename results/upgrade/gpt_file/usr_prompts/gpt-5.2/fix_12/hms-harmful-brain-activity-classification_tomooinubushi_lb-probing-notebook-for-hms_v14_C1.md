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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.989084

# 6. Current score

1.48867

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'Your code already follows a “constant prior” submission strategy, but it can fail to yield a valid score because `mean_vote_ratio` may become all-NaN if any of the brittle hypotheses checks differ across environments/files, producing an invalid submission. I make the submission robust by always computing the class prior from `train.csv` (vote counts normalized to probabilities), then (optionally) using your hypothesis-based priors only if they are valid and sum to 1. I also enforce exact row-wise normalization and clip to a tiny epsilon to avoid any KL issues with zeros, while preserving the same core “constant probabilities for all rows” logic. This should reliably produce a valid `submission.csv` and generally improve (lower) KL compared with NaNs/invalid rows.'
- What this solution (achieved 1.41937) has done: 'We keep your “constant prior for all test rows” core logic, but make it slightly better aligned to the evaluation by estimating a more appropriate prior from the *same label definition used in the competition*: first aggregate votes per `eeg_id` (since test is one row per `eeg_id`) and then compute the global class distribution from those aggregated vote totals. This typically lowers KL versus averaging over overlapping train rows because it reduces overweighting EEGs that have many overlapping labeled segments. We also keep your hypothesis-based priors only if valid, and we preserve the strict row-wise normalization + epsilon clipping to avoid invalid KL due to zeros. No model/training is introduced; runtime improves because we stop scanning all parquet files (that scan doesn’t help the submission quality and can risk timeouts).'
- What this solution (achieved 1.40726) has done: 'We keep your core “constant prior for all test rows” logic, but make the prior better aligned to the competition’s KL metric by computing it from the *Dirichlet-smoothed mean of per‑eeg normalized vote distributions* (so each eeg contributes equally, matching the test granularity). This is a minimal change from your current “aggregate then normalize” prior and typically reduces KL when some eegs have many overlapping labeled segments. We also keep your hypothesis-based priors as an override only if valid, and we retain strict epsilon-clipping plus row-wise normalization to guarantee a valid submission (no zeros, sums to 1). No model/training is introduced, and we keep the parquet scan skipped to stay within time.'
- What this solution (achieved 1.40586) has done: 'I keep your constant-prior submission strategy (same semantics) but make the prior slightly better matched to the test distribution by estimating it from per‑patient normalized vote distributions (patient-level averaging often reduces KL when some patients contribute many EEGs/segments). To avoid over-shifting and accidentally worsening score, I blend this patient prior with your current per‑eeg prior using a small weight, and keep the same Dirichlet smoothing, epsilon clipping, and strict row-wise normalization to guarantee a valid submission. This is a minimal, fast change (no parquet reads) and should move the score down toward your target without changing the overall approach. If the patient prior cannot be computed safely (edge case), it fall back to your existing prior.'
- What this solution (achieved 1.40878) has done: 'We keep the exact same “constant prior for all test rows” strategy, but make the prior closer to the competition’s per‑EEG evaluation by (1) using a patient-weighted average of per‑EEG normalized vote distributions (instead of averaging per‑patient totals), and (2) blending that with your current per‑EEG prior. This is a minimal semantic change that often reduces KL because it avoids overweighting patients with many overlapping segments while still matching the test granularity (one row per eeg_id). We keep your existing Dirichlet smoothing, hypothesis override logic (still off), epsilon clipping, and strict row-wise normalization so the submission stays valid. No parquet reads are introduced, so runtime stays fast and within constraints.'
- What this solution (achieved 1.40878) has done: 'I keep your constant-prior submission strategy intact, but adjust the *single* lever that controls performance: the blend weight between the per‑EEG prior and the patient-weighted prior. Since your current score (1.40878, lower-is-better) is still far from the target (0.989084), we should push a bit more toward the patient-weighted prior in a controlled, minimal way. I also add an internal (train-only) KL proxy evaluation to pick between just two nearby blend weights (0.35 vs 0.55) without changing any modeling logic, and then write the submission with the chosen weight. This keeps runtime low (no parquet reads) and preserves the exact “same probability for all test rows” semantics.'
- What this solution (achieved 1.40726) has done: 'Your current approach is a constant-prior submission, so the only meaningful lever is estimating that prior in a way that better matches the test/evaluation granularity (one row per `eeg_id`) without changing the overall semantics. I keep your blending idea but make the proxy selection less brittle by evaluating a small grid of nearby blend weights (still fast, no parquet reads) so we can move the prior more decisively toward the better proxy KL direction. I also compute an additional “plain per-eeg mean prior” candidate and include it in the proxy competition, then pick the best among these constant priors. Finally, I keep the strict epsilon-clip + row normalization to guarantee a valid submission (no zeros, sums to 1).'
- What this solution (achieved 8.07733) has done: 'Your current constant-prior strategy is already stable and valid, but the proxy selection is using KL(P_true || Q) whereas the competition uses KL(Q_pred || P_true); for a fixed constant prior, those lead to different “best” priors and can meaningfully shift the public score. I keep the exact same core logic (single constant probability vector for all test rows), but change only the proxy computation to match the competition direction, using the same smoothing/normalization safeguards. To reduce overfitting to one heuristic prior, I also include two additional constant-prior candidates that are still within your current semantics (mean per-eeg distribution and geometric-mean per-eeg distribution) and pick the best by the corrected proxy. The output submission format and strict row-wise normalization/epsilon clipping remain unchanged to guarantee a valid CSV.'
- What this solution (achieved 1.48867) has done: 'Your current score (8.07733, lower-is-better) is far worse than the target (0.989084), so we should make a minimal correction that moves the constant-prior strategy back toward your previously good behavior. The big issue is that the proxy selection currently optimizes KL(pred||true) on train, but the competition scores KL(true||pred); this mismatch can choose a much worse constant prior. I change only the proxy computation to match the competition direction, keep your same candidate priors/blends and the same constant-probability submission semantics, and retain epsilon-clipping + exact row normalization to guarantee a valid CSV. This is a tiny change that should substantially reduce the KL and move the score closer to the target band without altering the overall approach.'
- What this solution (achieved 8.07733) has done: 'We keep your constant-prior submission strategy (same probabilities for every test row) and only fix the proxy-selection bug that is currently choosing a worse prior: your proxy KL is computed as KL(true||pred) but the competition uses KL(pred||true). We switch the proxy to KL(pred||true) while keeping the same candidate priors/blends, Dirichlet smoothing, and strict epsilon+row normalization so the submission stays valid. This is a minimal change that should move your score back down (lower is better) toward your target, since the selected prior be optimized in the correct direction. No parquet reads or model/training is introduced, and runtime remains fast.'
- What this solution (achieved 1.48867) has done: 'I keep your constant-prior submission strategy unchanged, but fix the proxy-selection objective mismatch so you select the prior that minimizes the same KL direction used on Kaggle (KL(true || pred), not KL(pred || true)). This is a minimal change confined to the train-only proxy scoring function, and it should move your score back toward your previously good ~1.4 behavior (and closer to the 0.989 target) instead of the current ~8.08. I also keep your existing candidate priors/blends, Dirichlet smoothing, epsilon clipping, and strict row-wise normalization so the submission stays valid and stable. No parquet reads, no modeling/training, and the output path/format remain the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

sns.set(style="whitegrid")

train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
all_df = pd.concat([train, test]).reset_index(drop=True)

display(train.head())
display(test.head())
display(all_df.head())



## === cell 1
print(len(train.eeg_id.unique()))
print(len(train.spectrogram_id.unique()))
print(len(train.patient_id.unique()))
print(len(train.eeg_id.unique()) / len(train.patient_id.unique()))
print(len(train.spectrogram_id.unique()) / len(train.patient_id.unique()))



## === cell 2
train_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
train_spectrogram_files = os.listdir(train_spectrogram_dir)
print(f"There are {len(train_spectrogram_files)} train spectrogram parquets")
test_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
test_spectrogram_files = os.listdir(test_spectrogram_dir)
print(f"There are {len(test_spectrogram_files)} test spectrogram parquets")
train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
train_eeg_files = os.listdir(train_eeg_dir)
print(f"There are {len(train_eeg_files)} train eeg parquets")
test_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
test_eeg_files = os.listdir(test_eeg_dir)
print(f"There are {len(test_eeg_files)} test eeg parquets")




## === cell 3
def get_files_info(files, file_dir):
    nan_ratio = []
    shapes = []
    for file in tqdm(files):
        data = np.array(pd.read_parquet(f"{file_dir}{file}"))
        nan_ratio.append(np.isnan(data).sum() / len(data.flatten()))
        shapes.append(data.shape)
    nan_ratio = np.array(nan_ratio)
    shapes = np.array(shapes)
    return nan_ratio, shapes




## === cell 4
test_spectrogram_nan_ratio = np.array([np.nan], dtype=float)
test_spectrogram_shapes = np.array([[np.nan, np.nan]], dtype=float)
test_eeg_nan_ratio = np.array([np.nan], dtype=float)
test_eeg_shapes = np.array([[np.nan, np.nan]], dtype=float)

print("Skipping parquet scan; using constant-prior submission strategy only.")



## === cell 5
hypothesis0 = False
hypothesis1 = False

print(f"hypothesis0: {hypothesis0}")
print(f"hypothesis1: {hypothesis1}")



## === cell 6
hypotheses = []
hypotheses = False
print(f"hypotheses: {hypotheses}")



## === cell 7
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

targets = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

alpha = 1.0  # keep your existing fixed Dirichlet smoothing for stability

blend_candidates = [0.0, 0.2, 0.35, 0.55, 0.75, 1.0]


def _safe_row_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, eps, None)
    s = x.sum(axis=1, keepdims=True)
    return x / s


def _safe_vec_normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, eps, None)
    return v / v.sum()


def _kl_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


train_eeg_votes = train.groupby("eeg_id", as_index=False)[targets].sum()
V_eeg = train_eeg_votes[targets].to_numpy(dtype=np.float64)

P_eeg = _safe_row_normalize(V_eeg, eps=1e-12)

V_eeg_sm = V_eeg + alpha
V_eeg_sm = V_eeg_sm / V_eeg_sm.sum(axis=1, keepdims=True)
prior_eeg = _safe_vec_normalize(V_eeg_sm.mean(axis=0), eps=1e-12)

prior_eeg_mean = _safe_vec_normalize(P_eeg.mean(axis=0), eps=1e-12)

eps_geo = 1e-12
prior_eeg_geo = np.exp(np.log(np.clip(P_eeg, eps_geo, None)).mean(axis=0))
prior_eeg_geo = _safe_vec_normalize(prior_eeg_geo, eps=1e-12)

eeg_to_patient = train.drop_duplicates("eeg_id")[["eeg_id", "patient_id"]]
train_eeg_votes = train_eeg_votes.merge(
    eeg_to_patient, on="eeg_id", how="left", validate="one_to_one"
)
V_pat_from_eeg = train_eeg_votes.groupby("patient_id", as_index=False)[targets].mean()
prior_pat_from_eeg = V_pat_from_eeg[targets].to_numpy(dtype=np.float64).mean(axis=0)
prior_pat_from_eeg = _safe_vec_normalize(prior_pat_from_eeg, eps=1e-12)

prior_agg = V_eeg.sum(axis=0) + alpha
prior_agg = _safe_vec_normalize(prior_agg, eps=1e-12)

P_true = P_eeg  # per-eeg observed distribution proxy (sums to 1)

best_name = None
best_params = None
best_proxy = np.inf


def _proxy_score(prior_vec):
    prior_vec = _safe_vec_normalize(prior_vec, eps=1e-12)
    Q_pred = np.tile(prior_vec[None, :], (P_true.shape[0], 1))
    return float(_kl_divergence(P_true, Q_pred, eps=1e-12).mean())


base_priors = {
    "prior_eeg_sm_mean": prior_eeg,
    "prior_eeg_mean": prior_eeg_mean,
    "prior_eeg_geo": prior_eeg_geo,
}
for base_name, base_prior in base_priors.items():
    for w in blend_candidates:
        prior_blend = (1.0 - w) * base_prior + w * prior_pat_from_eeg
        prior_blend = _safe_vec_normalize(prior_blend, eps=1e-12)
        proxy = _proxy_score(prior_blend)
        if proxy < best_proxy:
            best_proxy = proxy
            best_name = f"blend_{base_name}_with_patient"
            best_params = {"blend_w_patient": float(w), "prior": prior_blend}

proxy_agg = _proxy_score(prior_agg)
if proxy_agg < best_proxy:
    best_proxy = proxy_agg
    best_name = "aggregate_votes"
    best_params = {"blend_w_patient": None, "prior": prior_agg}

chosen_prior_vec = best_params["prior"]
blend_w_patient = best_params["blend_w_patient"]

print("Chosen constant-prior strategy via train-only proxy KL:", best_name)
print("proxy_KL(true||pred):", best_proxy)
print("blend_w_patient:", blend_w_patient)

train_prior = dict(zip(targets, chosen_prior_vec.tolist()))

if (hypothesis0 == True) & (hypothesis1 == True) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.196002,
        "gpd_vote": 0.156386,
        "lrda_vote": 0.155805,
        "other_vote": 0.17610,
        "grda_vote": 0.17660,
        "lpd_vote": 0.139101,
    }
elif (hypothesis0 == True) & (hypothesis1 == False) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.174031,
        "lpd_vote": 0.112700,
        "gpd_vote": 0.090854,
        "lrda_vote": 0.071484,
        "grda_vote": 0.136408,
        "other_vote": 0.414523,
    }
elif (hypothesis0 == False) & (hypothesis1 == True) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.152810,
        "lpd_vote": 0.142456,
        "gpd_vote": 0.104062,
        "lrda_vote": 0.065407,
        "grda_vote": 0.114851,
        "other_vote": 0.420414,
    }
elif (hypothesis0 == False) & (hypothesis1 == False) & (hypotheses == True):
    mean_vote_ratio = {
        "seizure_vote": 0.310718,
        "lpd_vote": 0.046279,
        "gpd_vote": 0.051885,
        "lrda_vote": 0.081796,
        "grda_vote": 0.231471,
        "other_vote": 0.277851,
    }
else:
    mean_vote_ratio = None


def _is_valid_prior(d, cols, tol=1e-6):
    if d is None:
        return False
    for c in cols:
        if c not in d:
            return False
        v = d[c]
        if not np.isfinite(v):
            return False
        if v < 0:
            return False
    s = sum(float(d[c]) for c in cols)
    return np.isfinite(s) and abs(s - 1.0) <= tol


prior = (
    mean_vote_ratio
    if _is_valid_prior(mean_vote_ratio, targets, tol=1e-3)
    else train_prior
)

for t in targets:
    sub[t] = float(prior[t])

eps = 1e-12
probs = sub[targets].to_numpy(dtype=np.float64)
probs = np.clip(probs, eps, None)
probs = probs / probs.sum(axis=1, keepdims=True)
sub[targets] = probs

display(sub.head())
print("Row-wise sum stats:", sub[targets].sum(axis=1).describe())
print("Using prior:", prior)
print("Chosen strategy:", best_name)
print("blend_w_patient:", blend_w_patient)



## === cell 8
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", sub.shape)
