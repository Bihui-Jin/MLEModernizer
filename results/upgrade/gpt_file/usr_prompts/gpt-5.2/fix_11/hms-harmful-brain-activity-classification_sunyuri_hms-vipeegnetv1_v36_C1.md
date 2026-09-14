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

0.4681750280233946

# 6. Current score

0.8829

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime error caused by a TensorFlow/protobuf incompatibility by forcing the Python protobuf implementation before importing TensorFlow. Then I make the inference path robust to missing external weight files: if the expected `/kaggle/input/models20240204/*.h5` files are not present, the script still run end-to-end by outputting a valid, normalized fallback submission using the provided `sample_submission.csv` priors (so you get a “yielded” score rather than failing). I also guard the GPU strategy selection so it doesn’t crash when no GPU is available, and ensure the submission columns exactly match the competition’s required order and sum-to-1 constraint.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely in this run (since you’re doing inference-only and the external `.h5` weights aren’t available, TF import is both unnecessary and currently failing). Then I improve the fallback submission (previously scoring ~1.41) by computing label priors from `train.csv` at the `eeg_id` level (same aggregation you already do) and using those normalized priors for every test row, which is a legitimate, score-improving calibration for KL-divergence. Finally, I keep the required submission format/column order and enforce per-row sum-to-1 with clipping for numerical safety. These changes are minimal and directly address the crash + move score toward the target.'
- What this solution (achieved 1.17939) has done: 'I fix the crash in the fallback path by ensuring the `test` metadata you map from has a unique `eeg_id` index (the error indicates duplicated `eeg_id` values). This is a minimal, score-neutral correctness fix: we deduplicate `test` by `eeg_id` (keeping the first row) before building the `eeg_id -> patient_id` mapping. I also make the unseen-patient debug computation correct (it currently mixes patient/eeg counts) without changing predictions. The script then reliably write a valid `submission.csv` with the required columns and per-row probabilities summing to 1.'
- What this solution (achieved 0.78004) has done: 'Your current fallback already uses sensible global + patient-level priors, but it treats each patient’s EEGs as equally reliable; KL on this task benefits from using vote-count–weighted class distributions instead. I keep the same overall fallback logic (global prior + patient prior, then map test eeg_id→patient_id, then normalize and write submission), but compute priors from raw vote totals (sums) rather than means of already-normalized rows. I also add a small convex blend of patient prior with the global prior (very light smoothing) to reduce overconfident patient priors, which typically improves KL. These are minimal changes confined to the fallback path and still produce a valid `submission.csv` with per-row probabilities summing to 1.'
- What this solution (achieved 0.77767) has done: 'Your current fallback is already legitimate but likely under-calibrated for KL because it only uses patient/global priors; we can usually reduce KL a bit by adding an eeg-level prior (more specific) while still smoothing toward patient/global to avoid overconfidence. I keep the same core fallback structure (priors from train votes, map test eeg_id→patient_id, normalize, write submission) and only change the blending: use eeg_id prior when available, otherwise patient prior, all smoothed with the global prior. I also ensure the mapping uses the test.csv eeg_id index directly and that every row sums to 1 with safe clipping.'
- What this solution (achieved 0.79906) has done: 'You’re currently far above the target (0.77767 vs 0.46818, lower is better), so we should make a minimal calibration change that typically reduces KL without changing your fallback “priors + blending” core logic. The smallest safe lever is to tune the convex weights (eeg/patient/global) and add a tiny Dirichlet-style smoothing to the eeg/patient vote totals before normalization to avoid overconfident near-zero probabilities that KL penalizes. This keeps the same data sources, same aggregation level, same prediction structure, and still guarantees valid per-row sum-to-1 output. The patch only touches the fallback prior computation and blending weights, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.8829) has done: 'Your current score (0.79906, lower-is-better) is still far above the target (0.46818), so we should make the smallest legitimate change that tends to reduce KL without changing the overall fallback logic (vote-based priors + blending). I keep the same global/patient/eeg prior structure, but (1) compute the EEG prior *conditioned on patient_id* (more relevant than global eeg_id, since test eeg_ids aren’t in train), and (2) apply a tiny temperature smoothing on the final probabilities to reduce overconfidence (KL heavily penalizes near-zero probs). These are calibration-level changes only: no new models, no new features, same inputs, and the submission remains properly normalized and valid. The rest of the pipeline, paths, and output schema stay unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240204"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 64
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

import pandas as pd, numpy as np

VER = 1

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
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



## === cell 1
try:
    import albumentations as albu  # noqa: F401
except Exception:
    albu = None

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}



## === cell 2
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    print("Test shape", test.shape)

    expected_weights = [
        os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5") for i in range(5)
    ]
    have_all_weights = all(os.path.exists(p) for p in expected_weights)

    if not have_all_weights:
        missing = [p for p in expected_weights if not os.path.exists(p)]
        print(
            "WARNING: Missing model weights; will create improved fallback submission."
        )
        print("Missing (first 3):", missing[:3])

        alpha = 2.0  # slightly stronger smoothing than before to reduce overconfidence
        votes_by_eeg = df.groupby("eeg_id")[TARGETS].sum().astype(np.float64)
        votes_by_eeg_sm = votes_by_eeg + alpha

        global_votes = votes_by_eeg_sm.sum(axis=0).values
        global_prior = np.clip(global_votes, 1e-8, np.inf)
        global_prior = global_prior / global_prior.sum()

        eeg_to_pid = df.groupby("eeg_id")["patient_id"].first()
        votes_by_eeg_with_pid = votes_by_eeg_sm.join(eeg_to_pid, how="left")

        patient_votes = (
            votes_by_eeg_with_pid.groupby("patient_id")[TARGETS]
            .sum()
            .astype(np.float64)
        )
        patient_priors_df = patient_votes.div(patient_votes.sum(axis=1), axis=0)
        patient_priors_df = patient_priors_df.clip(1e-8, 1.0)
        patient_priors_df = patient_priors_df.div(patient_priors_df.sum(axis=1), axis=0)

        eeg_mass_within_patient = votes_by_eeg_with_pid[TARGETS].sum(axis=1)
        patient_total_mass = (
            votes_by_eeg_with_pid.groupby("patient_id")[TARGETS].sum().sum(axis=1)
        )
        eeg_mass_share = eeg_mass_within_patient / votes_by_eeg_with_pid[
            "patient_id"
        ].map(patient_total_mass)
        eeg_mass_share = (
            eeg_mass_share.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        )

        eeg_prior_df = votes_by_eeg_sm.div(votes_by_eeg_sm.sum(axis=1), axis=0)
        eeg_prior_df = eeg_prior_df.clip(1e-8, 1.0)
        eeg_prior_df = eeg_prior_df.div(eeg_prior_df.sum(axis=1), axis=0)

        sub = pd.DataFrame({"eeg_id": sample_sub["eeg_id"].values})

        test_unique = test.drop_duplicates(subset=["eeg_id"], keep="first").copy()
        test_pid = test_unique.set_index("eeg_id")["patient_id"]
        sub["patient_id"] = sub["eeg_id"].map(test_pid)

        preds = np.empty((len(sub), len(TARGETS)), dtype=np.float64)

        w_patient = 0.80
        w_global = 0.20

        temperature = 1.15  # >1 flattens slightly

        for i, (eid, pid) in enumerate(
            zip(sub["eeg_id"].values, sub["patient_id"].values)
        ):
            have_patient = pid in patient_priors_df.index

            if have_patient:
                ppat = patient_priors_df.loc[pid, TARGETS].values

                patient_eegs = votes_by_eeg_with_pid.index[
                    votes_by_eeg_with_pid["patient_id"] == pid
                ]
                if len(patient_eegs) > 0:
                    shares = eeg_mass_share.loc[patient_eegs].values.astype(np.float64)
                    if shares.sum() > 0:
                        shares = shares / shares.sum()
                        peegs = eeg_prior_df.loc[patient_eegs, TARGETS].values
                        peeg_mix = (shares[:, None] * peegs).sum(axis=0)
                        peeg_mix = np.clip(peeg_mix, 1e-8, 1.0)
                        peeg_mix = peeg_mix / peeg_mix.sum()

                        p = 0.90 * ppat + 0.10 * peeg_mix
                    else:
                        p = ppat
                else:
                    p = ppat

                preds[i, :] = w_patient * p + w_global * global_prior
            else:
                preds[i, :] = global_prior

        preds = np.clip(preds, 1e-8, 1.0)
        preds = preds / preds.sum(axis=1, keepdims=True)

        logp = np.log(preds)
        logp = logp / temperature
        logp = logp - logp.max(axis=1, keepdims=True)
        preds = np.exp(logp)
        preds = np.clip(preds, 1e-8, 1.0)
        preds = preds / preds.sum(axis=1, keepdims=True)

        for j, c in enumerate(TARGETS):
            sub[c] = preds[:, j].astype(np.float32)

        sub = sub.drop(columns=["patient_id"])
        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv("submission.csv", index=False)

        print("Submission shape", sub.shape)
        row_sums = sub[TARGETS].sum(axis=1)
        print("Row prob sum stats:", float(row_sums.min()), float(row_sums.max()))

        test_patients = pd.Series(test_unique["patient_id"].unique())
        unseen_patients = int((~test_patients.isin(patient_priors_df.index)).sum())
        print("Using global prior:", dict(zip(TARGETS, global_prior.round(6).tolist())))
        print(
            "Patient priors available for #patients:", int(patient_priors_df.shape[0])
        )
        print("Unseen test patients (vs train):", unseen_patients)
        print(sub.head())
    else:
        raise RuntimeError(
            "Model weights found but TensorFlow import/inference is disabled due to protobuf incompatibility. "
            "Remove this guard only if your environment supports TensorFlow."
        )
