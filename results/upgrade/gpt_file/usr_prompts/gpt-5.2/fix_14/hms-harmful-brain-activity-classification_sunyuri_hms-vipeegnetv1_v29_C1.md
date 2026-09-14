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

0.4706474959928369

# 6. Current score

0.94603

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I remove the hard dependency on external pre-trained weight files (both EfficientNet ImageNet weights and fold `.h5` weights) by providing a safe fallback that still produces a valid submission when those assets aren’t present. Finally, since no trained fold weights are available and training is disabled, I add a deterministic “prior” prediction based on the normalized mean vote distribution from `train.csv`, which yields a sane KL-divergence baseline and guarantees a valid `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash by switching to a robust fallback: try importing TensorFlow normally first, and only if it fails, set the protobuf environment variables and re-import in a clean way. This keeps your original logic intact but prevents the `MessageFactory.GetPrototype` error from stopping execution at cell 1. I also make the fallback submission generation always run even if TensorFlow can’t be imported (so you always get a valid `submission.csv`). Finally, I keep your current “prior” baseline (since it’s score-relevant and currently far from target) but ensure it is computed and written without relying on TensorFlow at all.'
- What this solution (achieved 1.17939) has done: 'I fix the TensorFlow/protobuf crash at the root by setting the protobuf env vars before any TensorFlow import attempt, and I make the TensorFlow import fully optional so the notebook always reaches submission writing. To move the score down toward the target (lower is better) without changing the core model/training logic, I improve the “no-weights/no-TF” fallback from a single global prior to a patient-conditional prior (computed from train by `patient_id`, with a safe fallback to the global prior for unseen patients), which is still a legitimate, label-free baseline. I also ensure the submission probabilities are strictly normalized and clipped, matching the KL metric requirements. Paths, model architecture, and training code remain unchanged; only robustness and fallback calibration are adjusted.'
- What this solution (achieved 0.82523) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* ensuring TensorFlow is imported with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` set before any TF/protobuf modules load; the current “retry import” path can still leave incompatible protobuf modules resident and triggers the `MessageFactory.GetPrototype` error. I also make the fallback path completely independent of TensorFlow so a valid `submission.csv` is always produced even if TF fails to import. To move your KL score down toward the target (lower is better) with minimal semantic change, I keep your patient-prior idea but improve it to a smoothed blend of patient prior + global prior (reduces overconfident patient-specific biases) and add a safe “unknown patient” handling. Finally, I keep strict normalization/clipping to satisfy Kaggle submission constraints.'
- What this solution (achieved 0.83424) has done: 'I fix the TensorFlow/protobuf crash by setting both protobuf environment variables *before* any TensorFlow import and by making the TensorFlow import strictly optional (so the notebook always reaches submission writing). This unblocks execution end-to-end and preserves your existing training/inference logic when TensorFlow works. To move KL down toward the target without changing the model/training approach, I keep your patient+global prior fallback but make it safer and slightly better calibrated by smoothing patient priors using vote-count–weighted aggregation (still label-free for test, and consistent with the metric). Finally, I ensure the written `submission.csv` always has the correct columns/order and each row sums to 1 (with clipping), preventing submission format failures.'
- What this solution (achieved 0.9076) has done: 'I fix the runtime crash by preventing any TensorFlow/protobuf import from happening at all when `NEEDTRAIN=False` and no fold weights are present, because your current notebook still imports TensorFlow in cell 1 and hits the `MessageFactory.GetPrototype` error. This keeps your model/training code intact, but makes TF truly optional by deferring its import until it’s actually needed for inference with existing `.h5` weights. Since your current score (0.83424) is far worse than the target (0.47065, lower is better), I also minimally improve the prior-only fallback (still label-free on test) by blending patient prior + global prior + a tiny uniform “floor” and by using a more conservative patient weight to reduce KL from overconfident priors. The script always write a valid `submission.csv` with correct columns/order and strictly normalized probabilities.'
- What this solution (achieved 1.01761) has done: 'Your current score (0.9076, lower is better) is still far from the target (0.47065), so we should improve the fallback predictions (since you are not using trained fold weights). With minimal changes and without touching the model/training logic, I make the prior fallback more informative by conditioning not only on `patient_id` but also on `spectrogram_id` when it exists in train, and I smooth these priors with a small Dirichlet-style additive prior to avoid overconfident zeros (important for KL). I also tune the blend weights slightly toward the more specific priors while keeping a uniform floor and strict normalization/clipping so the submission remains valid. TensorFlow import behavior and all model code remain unchanged; only the prior-only fallback used when no weights are present is adjusted.'
- What this solution (achieved 0.95313) has done: 'Your current gap is large (1.01761 vs target 0.47065, lower is better), and since you’re not using fold weights, the only score-relevant lever is the prior-only fallback. I keep your existing “structured prior” core idea but make it less noisy/more robust by (1) using vote-count–weighted + additive-Dirichlet smoothing when building the patient/spec priors, and (2) blending patient/spec priors with weights that depend on how much support (total votes) each prior has in train. This typically reduces KL by avoiding overconfident/undersupported conditional priors while preserving identical evaluation semantics (valid probabilities summing to 1). All TensorFlow/model code paths are left intact; only the prior computation and the fallback predictor are adjusted.'
- What this solution (achieved 0.94603) has done: 'We keep your current “structured prior-only” fallback (since no training/weights are used) but make it more informative in a strictly label-safe way by conditioning on `expert_consensus` *when available for that patient/spec in train* (it’s metadata derived from train labels only, and we only use it to build priors from train, then apply those priors using test’s patient/spec IDs). Concretely, we add an `expert_consensus`-conditional prior per patient and per spectrogram (with strong Dirichlet smoothing toward the global prior), and then evidence-weighted blend these new priors together with your existing patient/spec/global priors. This is a minimal extension of your existing prior logic (no model/training changes) and should reduce KL versus the current 0.953 by capturing strong per-patient/per-spec label tendencies without overconfidence. Submission writing/normalization stays identical to ensure a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402022"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 128
LENGTH = 256

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
    sample_sub = pd.read_csv(
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    sample_sub = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )

TARGETS = sample_sub.columns[1:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)

vote_counts = (
    df.groupby("eeg_id")[TARGETS].sum().sum(axis=1).reindex(train["eeg_id"]).values
)
vote_counts = np.asarray(vote_counts, dtype=np.float32)
vote_counts = np.clip(vote_counts, 1.0, None)

PRIOR_GLOBAL = (train[TARGETS].values * vote_counts[:, None]).sum(
    axis=0
) / vote_counts.sum()
PRIOR_GLOBAL = PRIOR_GLOBAL.astype(np.float32)
PRIOR_GLOBAL = np.clip(PRIOR_GLOBAL, 1e-8, 1.0)
PRIOR_GLOBAL = PRIOR_GLOBAL / PRIOR_GLOBAL.sum()
print("Global prior distribution:", dict(zip(TARGETS, PRIOR_GLOBAL.round(6))))

_train_tmp = train[["patient_id"] + list(TARGETS)].copy()
_train_tmp["_w"] = vote_counts.astype(np.float32)
for t in TARGETS:
    _train_tmp[t] = _train_tmp[t].astype(np.float32) * _train_tmp["_w"]

patient_num = _train_tmp.groupby("patient_id")[list(TARGETS)].sum().astype(np.float32)
patient_den = _train_tmp.groupby("patient_id")["_w"].sum().astype(np.float32)

ALPHA_PAT = 40.0  # moderate smoothing to reduce KL from undersupported patient priors
patient_smoothed = (patient_num + (ALPHA_PAT * PRIOR_GLOBAL[None, :])) / (
    patient_den.values[:, None] + ALPHA_PAT
)
patient_prior_vals = patient_smoothed.values.astype(np.float32)
patient_prior_vals = np.clip(patient_prior_vals, 1e-8, 1.0)
patient_prior_vals = patient_prior_vals / patient_prior_vals.sum(axis=1, keepdims=True)
patient_prior_df = pd.DataFrame(
    patient_prior_vals, index=patient_smoothed.index, columns=TARGETS
)
patient_den_map = patient_den.to_dict()
print("Patient priors computed for #patients:", patient_prior_df.shape[0])

_spec_tmp = train[["spec_id"] + list(TARGETS)].copy()
_spec_tmp["_w"] = vote_counts.astype(np.float32)
for t in TARGETS:
    _spec_tmp[t] = _spec_tmp[t].astype(np.float32) * _spec_tmp["_w"]

spec_num = _spec_tmp.groupby("spec_id")[list(TARGETS)].sum().astype(np.float32)
spec_den = _spec_tmp.groupby("spec_id")["_w"].sum().astype(np.float32)

ALPHA_SPEC = 80.0  # stronger smoothing; spec_id can be noisier than patient_id
spec_smoothed = (spec_num + (ALPHA_SPEC * PRIOR_GLOBAL[None, :])) / (
    spec_den.values[:, None] + ALPHA_SPEC
)
spec_prior_vals = spec_smoothed.values.astype(np.float32)
spec_prior_vals = np.clip(spec_prior_vals, 1e-8, 1.0)
spec_prior_vals = spec_prior_vals / spec_prior_vals.sum(axis=1, keepdims=True)
spec_prior_df = pd.DataFrame(
    spec_prior_vals, index=spec_smoothed.index, columns=TARGETS
)
spec_den_map = spec_den.to_dict()
print("Spectrogram priors computed for #specs:", spec_prior_df.shape[0])

_cons = train[["patient_id", "spec_id", "target"] + list(TARGETS)].copy()
_cons["_w"] = vote_counts.astype(np.float32)
for t in TARGETS:
    _cons[t] = _cons[t].astype(np.float32) * _cons["_w"]

ALPHA_CONS_PAT = 120.0
pat_cons_num = (
    _cons.groupby(["patient_id", "target"])[list(TARGETS)].sum().astype(np.float32)
)
pat_cons_den = _cons.groupby(["patient_id", "target"])["_w"].sum().astype(np.float32)
pat_cons_smoothed = (pat_cons_num + (ALPHA_CONS_PAT * PRIOR_GLOBAL[None, :])) / (
    pat_cons_den.values[:, None] + ALPHA_CONS_PAT
)
pat_cons_vals = pat_cons_smoothed.values.astype(np.float32)
pat_cons_vals = np.clip(pat_cons_vals, 1e-8, 1.0)
pat_cons_vals = pat_cons_vals / pat_cons_vals.sum(axis=1, keepdims=True)
patient_cons_prior_df = pd.DataFrame(
    pat_cons_vals, index=pat_cons_smoothed.index, columns=TARGETS
)
patient_cons_den_map = pat_cons_den.to_dict()
print("Patient+consensus priors computed for #keys:", patient_cons_prior_df.shape[0])

ALPHA_CONS_SPEC = 160.0
spec_cons_num = (
    _cons.groupby(["spec_id", "target"])[list(TARGETS)].sum().astype(np.float32)
)
spec_cons_den = _cons.groupby(["spec_id", "target"])["_w"].sum().astype(np.float32)
spec_cons_smoothed = (spec_cons_num + (ALPHA_CONS_SPEC * PRIOR_GLOBAL[None, :])) / (
    spec_cons_den.values[:, None] + ALPHA_CONS_SPEC
)
spec_cons_vals = spec_cons_smoothed.values.astype(np.float32)
spec_cons_vals = np.clip(spec_cons_vals, 1e-8, 1.0)
spec_cons_vals = spec_cons_vals / spec_cons_vals.sum(axis=1, keepdims=True)
spec_cons_prior_df = pd.DataFrame(
    spec_cons_vals, index=spec_cons_smoothed.index, columns=TARGETS
)
spec_cons_den_map = spec_cons_den.to_dict()
print("Spec+consensus priors computed for #keys:", spec_cons_prior_df.shape[0])


def _find_weight_files_quick(ver: int):
    import glob
    import re

    pattern = f"EB2_v{ver}_f*.h5"
    roots = []
    if os.path.isdir(LOAD_MODELS_FROM):
        roots.append(LOAD_MODELS_FROM)
    roots.append("/kaggle/input")

    found = []
    for root in roots:
        found.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    found = sorted(list(dict.fromkeys(found)))

    fold_to_path = {}
    for p in found:
        m = re.search(rf"EB2_v{ver}_f(\d+)\.h5$", os.path.basename(p))
        if m:
            fold_to_path[int(m.group(1))] = p
    return [fold_to_path[k] for k in sorted(fold_to_path.keys())]


VER = 1
WEIGHT_FILES = _find_weight_files_quick(VER)
print(f"Pre-scan: found {len(WEIGHT_FILES)} fold weight files for VER={VER}.")

NEED_TF = bool(NEEDTRAIN) or (len(WEIGHT_FILES) > 0)

tf = None
K = None
strategy = None


def _safe_import_tensorflow():
    """
    Import TensorFlow robustly in Kaggle's environment where protobuf binary/TF combos
    can crash with: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    """
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e1:
        try:
            import importlib
            import sys

            for m in list(sys.modules.keys()):
                if m.startswith(("tensorflow", "google.protobuf")):
                    sys.modules.pop(m, None)

            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

            tf = importlib.import_module("tensorflow")
            return tf
        except Exception as e2:
            print("TensorFlow import failed (will fall back to prior-only submission).")
            print("First error:", repr(e1))
            print("Retry error:", repr(e2))
            return None


if NEED_TF:
    tf = _safe_import_tensorflow()
    if tf is not None:
        import tensorflow.keras.backend as K  # noqa: F401

        print("TensorFlow version =", tf.__version__)
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) <= 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print(f"Using {len(gpus)} GPU")
        else:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {len(gpus)} GPUs")

        MIX = True
        if MIX:
            try:
                tf.config.optimizer.set_experimental_options(
                    {"auto_mixed_precision": True}
                )
                print("Mixed precision enabled")
            except Exception as e:
                print("Mixed precision enable failed (continuing):", repr(e))
else:
    print("Skipping TensorFlow import (no training and no weights detected).")




## === cell 1
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")

    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()

if NEEDTRAIN:
    from scipy import signal

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")

    if READ_EEG_FILES:
        eegs = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
                time_temp = train[train.eeg_id == name].eeg_median.iloc[-1]
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

                list_eeg = list()
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, chan.split("-")[0]]
                            - eeg_default.loc[:, chan.split("-")[1]]
                        ).values

                    eeg[np.isnan(eeg)] = 0

                    if 200 != SFREQ:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs[name] = list_eeg

        if not os.path.exists("./input/brain-eegs"):
            os.makedirs("./input/brain-eegs")
        np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()




## === cell 2
try:
    import albumentations as albu
except Exception:
    albu = None

import matplotlib

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}

if tf is not None:

    class DataGenerator(tf.keras.utils.Sequence):
        "Generates data for Keras"

        def __init__(
            self,
            data,
            batch_size=32,
            shuffle=False,
            augment=False,
            mode="train",
            specs=None,
            eegs=None,
        ):

            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.data) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            X, X_eeg, y = self.__data_generation(indexes)
            return [X, X_eeg], y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            X_eeg = np.zeros(
                (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
            y = np.zeros((len(indexes), 6), dtype="float32")
            img_map = np.zeros((100 * 300, 3))

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                if self.mode == "test":
                    r = 0
                elif self.mode == "valid":
                    r = int((row["min"] + row["max"]) // 4)
                else:
                    r = np.random.randint(row["min"], row["max"] + 1) // 2

                for k in range(4):
                    img = self.specs[row.spec_id][
                        r : r + 300, k * 100 : (k + 1) * 100
                    ].T
                    img_eeg = self.eegs[row.eeg_id][:, :, k]

                    img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                    img = np.log(img)
                    img = np.nan_to_num(img, nan=0.0)

                    img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                    img = np.reshape(img, (img.shape[0] * img.shape[1]))
                    img = np.array(img, dtype=np.int16)
                    img_map = self.cmaps[img - 1]
                    img_map = np.reshape(img_map, (100, 300, 3))

                    img_map = img_map[
                        :,
                        max(round((600 / 2 - LENGTH) / 2), 0) : min(
                            (round((600 / 2 - LENGTH) / 2) + LENGTH), img_map.shape[1]
                        ),
                        :,
                    ]
                    if HIGH != 100:
                        img_map = np.array(
                            tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                            dtype=np.float32,
                        )
                        X[
                            j,
                            round((HIGH - img_map.shape[0]) / 2) : round(
                                (HIGH + img_map.shape[0]) / 2
                            ),
                            :,
                            :,
                            k,
                        ] = img_map
                    else:
                        X[j, :, :, :, k] = img

                    X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / (0.229**2)
                    X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / (0.224**2)
                    X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / (0.225**2)

                    X_eeg[j, 1, :, k] = img_eeg[0, :]
                    X_eeg[j, 2, :, k] = img_eeg[1, :]
                    X_eeg[j, 3, :, k] = img_eeg[2, :]
                    X_eeg[j, 4, :, k] = img_eeg[3, :]

                    X_eeg[j, :, :, k] = (
                        X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                    ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if self.mode != "test":
                    y[j] = row[TARGETS].values

            return X, X_eeg, y

        def __random_transform(self, img):
            if albu is None:
                return img
            composition = albu.Compose(
                [
                    albu.HorizontalFlip(p=0.5),
                    albu.CoarseDropout(
                        max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                    ),
                ]
            )
            return composition(image=img)["image"]

        def __augment_batch(self, img_batch):
            for i in range(img_batch.shape[0]):
                img_batch[i,] = self.__random_transform(img_batch[i,])
            return img_batch




## === cell 3
if NEEDTRAIN:
    import math

    LR_START = 1e-6
    LR_MAX = 1e-3
    LR_MIN = 1e-6
    LR_RAMPUP_EPOCHS = 0
    LR_SUSTAIN_EPOCHS = 0
    EPOCHS2 = 10

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            decay_total_epochs = EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
            decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
            phase = math.pi * decay_epoch_index / decay_total_epochs
            cosine_decay = 0.5 * (1 + math.cos(phase))
            lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
        return lr

    LR2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

    LR_START = 1e-4
    LR_MAX = 1e-3
    LR_RAMPUP_EPOCHS = 0
    LR_SUSTAIN_EPOCHS = 0
    LR_STEP_DECAY = 0.1
    EVERY = 2
    EPOCHS = 6

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MAX * LR_STEP_DECAY ** (
                (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) // EVERY
            )
        return max(lr, 1e-4)

    LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)




## === cell 4
if tf is not None:

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="tanh",
                dilation_rate=dilation_rate,
            )(x)
            sigm_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                dilation_rate=dilation_rate,
            )(x)
            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(
                x
            )
            res_x = tf.keras.layers.Add()([res_x, x])
        return res_x

    def _supports_include_preprocessing(app_fn):
        import inspect

        try:
            return "include_preprocessing" in inspect.signature(app_fn).parameters
        except Exception:
            return False

    def build_model():
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

        v2_kwargs = dict(include_top=False, weights=None)
        if _supports_include_preprocessing(tf.keras.applications.EfficientNetV2B0):
            v2_kwargs["include_preprocessing"] = False
        base_model = tf.keras.applications.EfficientNetV2B0(**v2_kwargs)

        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnetv2-b0_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnetv2-b0_notop.h5"
                )

        x0 = inp[:, :, :, :, 0]
        x1 = inp[:, :, :, :, 1]
        x2 = inp[:, :, :, :, 2]
        x3 = inp[:, :, :, :, 3]
        x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

        b0_kwargs = dict(include_top=False, weights=None)
        if _supports_include_preprocessing(tf.keras.applications.EfficientNetB0):
            b0_kwargs["include_preprocessing"] = False
        base_model_eeg = tf.keras.applications.EfficientNetB0(**b0_kwargs)

        base_model_eeg._name = "eeg_extractor"
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_notop.h5"
                )

        x0_eeg = inp_eeg[:, :, :, :1]
        x1_eeg = inp_eeg[:, :, :, 1:2]
        x2_eeg = inp_eeg[:, :, :, 2:3]
        x3_eeg = inp_eeg[:, :, :, 3:4]
        x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)

        x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
        x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.KLDivergence()
        model.compile(loss=loss, optimizer=opt)
        return model




## === cell 5
if NEEDTRAIN and tf is not None:
    from sklearn.model_selection import GroupKFold
    import gc

    all_oof = []
    all_true = []

    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i + 1}")

        train_gen = DataGenerator(
            train.iloc[train_index],
            shuffle=True,
            batch_size=16,
            specs=spectrograms,
            eegs=eegs,
        )
        valid_gen = DataGenerator(
            train.iloc[valid_index],
            shuffle=False,
            batch_size=32,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
        )

        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        K.clear_session()
        callbacks_list = [
            LR,
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"EB2_v{VER}_f{i}.h5",
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5, monitor="val_loss", mode="min"
            ),
        ]

        with strategy.scope():
            model = build_model()
        model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )

        model.load_weights(f"EB2_v{VER}_f{i}.h5")
        oof = model.predict(valid_gen, verbose=1)
        all_oof.append(oof)
        all_true.append(train.iloc[valid_index][TARGETS].values)

        del model, oof
        K.clear_session()
        gc.collect()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)




## === cell 6
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    def _predict_structured_prior(test_df: pd.DataFrame) -> np.ndarray:
        """
        Change (score-relevant, minimal): extend the existing evidence-weighted blending
        with consensus-conditional priors (patient_id+expert_consensus, spec_id+expert_consensus)
        when they exist in train. This usually lowers KL by capturing strong tendencies
        while using strong smoothing + evidence weighting to avoid overconfident priors.
        """
        n = len(test_df)
        k = len(TARGETS)
        pred = np.zeros((n, k), dtype=np.float32)

        TAU_SPEC = 250.0
        TAU_PAT = 400.0

        TAU_SPEC_CONS = 600.0
        TAU_PAT_CONS = 800.0

        eps_uniform = 0.01
        uniform = np.full((k,), 1.0 / k, dtype=np.float32)

        pats = test_df["patient_id"].values
        specs = test_df["spectrogram_id"].values

        for i, (pid, sid) in enumerate(zip(pats, specs)):
            spec_w = float(spec_den_map.get(sid, 0.0))
            pat_w = float(patient_den_map.get(pid, 0.0))

            w_spec = spec_w / (spec_w + TAU_SPEC) if spec_w > 0 else 0.0
            w_pat = pat_w / (pat_w + TAU_PAT) if pat_w > 0 else 0.0

            p_spec = (
                spec_prior_df.loc[sid, TARGETS].values.astype(np.float32)
                if w_spec > 0
                else PRIOR_GLOBAL
            )
            p_pat = (
                patient_prior_df.loc[pid, TARGETS].values.astype(np.float32)
                if w_pat > 0
                else PRIOR_GLOBAL
            )

            w_global = max(0.0, 1.0 - (w_spec + w_pat))
            base0 = (w_spec * p_spec) + (w_pat * p_pat) + (w_global * PRIOR_GLOBAL)
            base0 = np.clip(base0, 1e-8, 1.0)
            base0 = base0 / base0.sum()

            cons_guess = str(
                train.loc[train[TARGETS].idxmax(axis=1).index[0], "target"]
            )
            cons_guess = str(
                train.loc[train[TARGETS].idxmax(axis=1).index[0], "target"]
            )
            break

        _train_class = train[TARGETS].values.argmax(axis=1)
        _ec = train["target"].astype(str).values
        class_to_ec = {}
        for ci in range(k):
            vals = _ec[_train_class == ci]
            if len(vals) == 0:
                class_to_ec[ci] = _ec[0]
            else:
                vc = pd.Series(vals).value_counts()
                class_to_ec[ci] = str(vc.index[0])

        for i, (pid, sid) in enumerate(zip(pats, specs)):
            spec_w = float(spec_den_map.get(sid, 0.0))
            pat_w = float(patient_den_map.get(pid, 0.0))

            w_spec = spec_w / (spec_w + TAU_SPEC) if spec_w > 0 else 0.0
            w_pat = pat_w / (pat_w + TAU_PAT) if pat_w > 0 else 0.0

            p_spec = (
                spec_prior_df.loc[sid, TARGETS].values.astype(np.float32)
                if w_spec > 0
                else PRIOR_GLOBAL
            )
            p_pat = (
                patient_prior_df.loc[pid, TARGETS].values.astype(np.float32)
                if w_pat > 0
                else PRIOR_GLOBAL
            )
            w_global = max(0.0, 1.0 - (w_spec + w_pat))
            base0 = (w_spec * p_spec) + (w_pat * p_pat) + (w_global * PRIOR_GLOBAL)
            base0 = np.clip(base0, 1e-8, 1.0)
            base0 = base0 / base0.sum()

            cons_guess = class_to_ec[int(base0.argmax())]

            key_pat = (pid, cons_guess)
            key_spec = (sid, cons_guess)

            patc_w_raw = float(patient_cons_den_map.get(key_pat, 0.0))
            specc_w_raw = float(spec_cons_den_map.get(key_spec, 0.0))

            w_patc = patc_w_raw / (patc_w_raw + TAU_PAT_CONS) if patc_w_raw > 0 else 0.0
            w_specc = (
                specc_w_raw / (specc_w_raw + TAU_SPEC_CONS) if specc_w_raw > 0 else 0.0
            )

            p_patc = (
                patient_cons_prior_df.loc[key_pat, TARGETS].values.astype(np.float32)
                if w_patc > 0
                else PRIOR_GLOBAL
            )
            p_specc = (
                spec_cons_prior_df.loc[key_spec, TARGETS].values.astype(np.float32)
                if w_specc > 0
                else PRIOR_GLOBAL
            )

            w_extra = min(0.6, w_patc + w_specc)  # cap to keep conservative
            if w_extra > 0:
                extra = (w_specc * p_specc) + (w_patc * p_patc)
                extra = np.clip(extra, 1e-8, 1.0)
                extra = extra / extra.sum()
                base = (1.0 - w_extra) * base0 + w_extra * extra
            else:
                base = base0

            base = (1.0 - eps_uniform) * base + eps_uniform * uniform
            pred[i] = base

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        return pred

    def _write_submission(
        eeg_ids: np.ndarray, pred: np.ndarray, out_path: str = "submission.csv"
    ):
        pred = np.asarray(pred, dtype=np.float32)
        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"eeg_id": eeg_ids})
        sub = pd.concat([sub, pd.DataFrame(pred, columns=TARGETS)], axis=1)
        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv(out_path, index=False)
        print(f"Wrote {out_path}. Shape:", sub.shape)
        rs = sub[TARGETS].sum(axis=1)
        print("Row prob sum min/max:", float(rs.min()), float(rs.max()))
        return sub

    if tf is None or len(WEIGHT_FILES) == 0:
        if tf is None:
            print("TensorFlow unavailable -> using structured prior-only predictions.")
        else:
            print("No fold weights found -> using structured prior-only predictions.")
        pred = _predict_structured_prior(test)
        _write_submission(test.eeg_id.values, pred, "submission.csv")
    else:
        import glob
        import re
        from scipy import signal

        weight_files = WEIGHT_FILES
        print(f"Using {len(weight_files)} fold weight files.")

        if PLATFORM == "local":
            PATH2 = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

        files2 = sorted(os.listdir(PATH2))
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()

        test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

        if PLATFORM == "local":
            PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = sorted(os.listdir(PATH2))
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])

            if len(test[test.eeg_id == name]) > 0:
                time_temp = 0
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

                list_eeg = list()
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, chan.split("-")[0]]
                            - eeg_default.loc[:, chan.split("-")[1]]
                        ).values

                    eeg[np.isnan(eeg)] = 0

                    if 200 != SFREQ:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs2[name] = list_eeg
        print()

        preds = []
        with strategy.scope():
            model = build_model()

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
        )

        for wf in weight_files:
            print("Loading", wf)
            model.load_weights(wf)
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)

        pred = np.mean(preds, axis=0).astype(np.float32)
        print("\nTest preds shape", pred.shape)

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        _write_submission(test.eeg_id.values, pred, "submission.csv")
