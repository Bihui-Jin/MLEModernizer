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

0.3256166983473387

# 6. Current score

0.81232

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I make the pipeline robust to missing external datasets/modules by (1) auto-detecting whether the MK codebase and checkpoints actually exist and only running them if they do, and (2) providing a fast, deterministic fallback that produces a valid submission when inference can’t run. The main bug is the fold inference `os.system` call failing (exit code 512) because the referenced `/kaggle/input/hms-mk-data/...` and/or `/kaggle/input/hms-mk-codes` aren’t available in your environment; this currently prevents any CSV from being generated. I also fix the cell numbering to start at 1 and ensure we always write `/kaggle/working/submission.csv` with the exact required columns and per-row normalization. The fallback uses the training label distribution as a prior (score-neutral baseline), guaranteeing a valid CSV so you can submit and then iterate when the missing inputs are restored.'
- What this solution (achieved 0.73577) has done: 'Your current score indicates the external MK inference isn’t being used (or isn’t aligned), so the simplest way to move toward the target is to make the fallback stronger without changing the overall approach: keep it as a “no-model” baseline but condition the class-prior by `patient_id` (using train metadata) and apply light additive smoothing so probabilities are never too sharp or too flat. This stays within your existing logic (still just priors from train votes), but usually reduces KL versus a single global prior because label distributions vary by patient. I also harden ID alignment (unique `eeg_id` in submission order) and ensure probabilities are clipped and renormalized exactly once at the end to avoid numerical drift. Paths and submission schema remain unchanged and it still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.77275) has done: 'Your current score (0.73577, lower-is-better) is far from the target (0.3256), and since the external MK inference isn’t available, the only legitimate way to move toward the target without changing core modeling is to make the metadata-prior baseline more informative. I keep the same “patient-conditioned prior from train votes” core idea, but add a second conditioning signal using `spectrogram_id` (also available in test/train metadata) and then blend patient- and spectrogram-priors with a small weight toward the more specific one when it has enough support. I also compute priors using total vote counts aggregated per group (more statistically stable than averaging per-row normalized distributions when vote totals vary), while keeping the same smoothing and final normalization so the submission stays valid. Paths, output schema, and fallback/inference branching remain unchanged.'
- What this solution (achieved 0.81232) has done: 'Your current score (0.77275, lower-is-better) is far from the target (0.3256), and since external inference isn’t available, the only safe lever is improving the metadata-prior baseline without changing its core “priors from train vote counts” logic. I keep the same patient+spectrogram Dirichlet-smoothed priors, but make the blend weight depend on actual vote mass (total annotator votes) rather than row counts, which better matches the competition target distribution and usually reduces KL. I also add a small “pooling” step that mixes each group prior slightly back toward the global prior when group vote mass is low (hierarchical shrinkage), which stabilizes rare patients/spectrograms. Finally, I keep the same strict normalization/clipping and submission alignment to avoid format-induced score regressions.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import subprocess
import pandas as pd
import numpy as np

MK_CODES_DIR = "/kaggle/input/hms-mk-codes"
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

FALLBACK_TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

TARGET_COLS = FALLBACK_TARGET_COLS
if os.path.isdir(MK_CODES_DIR):
    if MK_CODES_DIR not in sys.path:
        sys.path.append(MK_CODES_DIR)
    try:
        from src.settings import TARGET_COLS as _TARGET_COLS  # type: ignore

        TARGET_COLS = list(_TARGET_COLS)
    except Exception:
        TARGET_COLS = FALLBACK_TARGET_COLS

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
TEST_CSV_PATH = f"{DATA_PATH}/test.csv"
TRAIN_CSV_PATH = f"{DATA_PATH}/train.csv"

print("TARGET_COLS =", TARGET_COLS)
print("MK_CODES_DIR exists:", os.path.isdir(MK_CODES_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Train exists:", os.path.exists(TRAIN_CSV_PATH))
print("Test exists:", os.path.exists(TEST_CSV_PATH))




## === cell 1
def run_parquet_to_npy_conversion():
    if not os.path.isdir(MK_CODES_DIR):
        print(f"MK_CODES_DIR not found ({MK_CODES_DIR}); skipping conversion.")
        return False

    expected_any = glob.glob(os.path.join(OUT_PATH, "*.npy"))
    if len(expected_any) > 0:
        print(
            f"Found {len(expected_any)} .npy files in {OUT_PATH}, skipping conversion."
        )
        return True

    cmd = [
        sys.executable,
        "-m",
        "src.convert_parquet_to_npy",
        f"--data_dir={DATA_PATH}",
        f"--out_dir={OUT_PATH}",
    ]
    print("Running conversion:", " ".join(cmd), " (cwd=", MK_CODES_DIR, ")")
    try:
        subprocess.run(cmd, cwd=MK_CODES_DIR, check=True)
        return True
    except Exception as e:
        print("Conversion failed; will proceed without .npy files. Error:", repr(e))
        return False


_ = run_parquet_to_npy_conversion()




## === cell 2
def try_run_external_inference():
    if not os.path.isdir(MK_CODES_DIR):
        print(
            f"MK_CODES_DIR not found ({MK_CODES_DIR}); cannot run external inference."
        )
        return False

    ckpts = [
        "/kaggle/input/hms-mk-data/fold0_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold1_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold2_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold3_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold4_pseudo_resv2.ckpt",
    ]
    missing_ckpts = [p for p in ckpts if not os.path.exists(p)]
    if missing_ckpts:
        print(
            "Missing checkpoints; cannot run external inference. Missing (showing up to 3):",
            missing_ckpts[:3],
        )
        return False

    for fold, ckpt in enumerate(ckpts):
        out_csv = f"/kaggle/working/submission_fold{fold}_v2.csv"
        if os.path.exists(out_csv):
            print(f"Exists, skipping fold {fold}:", out_csv)
            continue

        cmd = [
            sys.executable,
            "-m",
            "test",
            f"paths.data_dir={DATA_PATH}",
            f"data.test_eegs_dir={OUT_PATH}",
            f"ckpt_path={ckpt}",
            "hydra=test",
            f"+model.test_output_dir={OUT_PATH}",
            "experiment=conv1d_resv2",
            "+model.net.pretrained=False",
        ]
        print("Running inference:", " ".join(cmd), " (cwd=", MK_CODES_DIR, ")")
        try:
            subprocess.run(cmd, cwd=MK_CODES_DIR, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"Inference failed for fold={fold} with returncode={e.returncode}. Falling back."
            )
            return False

        src_csv = "/kaggle/working/submission.csv"
        if not os.path.exists(src_csv):
            print(f"Expected {src_csv} was not created for fold={fold}. Falling back.")
            return False

        os.replace(src_csv, out_csv)
        print("Wrote:", out_csv)

    return True


external_ok = try_run_external_inference()
print("external_ok =", external_ok)




## === cell 3
def _finalize_probs(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float64")
    mat = np.clip(mat, 1e-12, 1.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def merge_preds(folds=(0, 1, 2, 3, 4), version="v2"):
    base = pd.read_csv(SAMPLE_SUB_PATH)
    if "eeg_id" not in base.columns:
        raise ValueError("sample_submission.csv must contain eeg_id")

    base_ids = base["eeg_id"].values
    all_fold_preds = []

    for fold in folds:
        path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fold submission: {path}")

        df = pd.read_csv(path)
        missing = [c for c in (["eeg_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df = df.set_index("eeg_id").loc[base_ids, TARGET_COLS].astype("float64")
        all_fold_preds.append(df.values)

    preds = np.mean(np.stack(all_fold_preds, axis=0), axis=0)
    preds = _finalize_probs(preds)

    out = base.copy()
    out[TARGET_COLS] = preds
    return out


def _dirichlet_smooth_from_counts(
    counts: pd.DataFrame, alpha: float, global_prior: np.ndarray
) -> pd.DataFrame:
    """
    Vote-count aggregation + Dirichlet smoothing.
    """
    cnt = counts.sum(axis=1).astype("float64").values.reshape(-1, 1)
    raw = counts[TARGET_COLS].astype("float64").values
    smooth = raw + alpha * global_prior.reshape(1, -1)
    denom = cnt + alpha
    denom = np.where(denom == 0, 1.0, denom)
    out = smooth / denom
    out = _finalize_probs(out)
    return pd.DataFrame(out, index=counts.index, columns=TARGET_COLS)


def _shrink_group_prior_to_global(
    group_prior_df: pd.DataFrame,
    group_vote_mass: pd.Series,
    global_prior: np.ndarray,
    tau: float,
) -> pd.DataFrame:
    """
    Change (score-improving, still same 'prior' logic): hierarchical shrinkage by vote mass.
    For low vote mass groups, mix more toward global prior; for high mass, keep group-specific.
    This typically reduces KL by preventing overconfident priors for rare groups.
    """
    m = (
        group_vote_mass.astype("float64")
        .reindex(group_prior_df.index)
        .fillna(0.0)
        .values
    )
    w = (m / (m + float(tau))).reshape(-1, 1)  # in [0,1)
    out = w * group_prior_df.values + (1.0 - w) * global_prior.reshape(1, -1)
    out = _finalize_probs(out)
    return pd.DataFrame(out, index=group_prior_df.index, columns=TARGET_COLS)


def make_prior_baseline():
    base = pd.read_csv(SAMPLE_SUB_PATH)
    test = pd.read_csv(
        TEST_CSV_PATH, usecols=["eeg_id", "patient_id", "spectrogram_id"]
    )
    train = pd.read_csv(
        TRAIN_CSV_PATH, usecols=["patient_id", "spectrogram_id"] + TARGET_COLS
    )

    global_counts = train[TARGET_COLS].astype("float64").sum(axis=0).values
    global_prior = np.clip(global_counts, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    alpha = 0.5  # keep same small, deterministic smoothing strength

    patient_counts = train.groupby("patient_id")[TARGET_COLS].sum()
    spectro_counts = train.groupby("spectrogram_id")[TARGET_COLS].sum()

    patient_prior = _dirichlet_smooth_from_counts(
        patient_counts, alpha=alpha, global_prior=global_prior
    )
    spectro_prior = _dirichlet_smooth_from_counts(
        spectro_counts, alpha=alpha, global_prior=global_prior
    )

    train_vote_mass = train[TARGET_COLS].sum(axis=1).astype("float64")
    patient_vote_mass = train_vote_mass.groupby(train["patient_id"]).sum()
    spectro_vote_mass = train_vote_mass.groupby(train["spectrogram_id"]).sum()

    patient_prior = _shrink_group_prior_to_global(
        patient_prior, patient_vote_mass, global_prior=global_prior, tau=30.0
    )
    spectro_prior = _shrink_group_prior_to_global(
        spectro_prior, spectro_vote_mass, global_prior=global_prior, tau=30.0
    )

    test = test.drop_duplicates(subset=["eeg_id"]).set_index("eeg_id")
    base_ids = base["eeg_id"].values
    test = test.loc[base_ids]  # align and preserve submission order

    pid = test["patient_id"].values
    sid = test["spectrogram_id"].values

    patient_map = {k: patient_prior.loc[k].values for k in patient_prior.index.values}
    spectro_map = {k: spectro_prior.loc[k].values for k in spectro_prior.index.values}
    patient_m_map = patient_vote_mass.to_dict()
    spectro_m_map = spectro_vote_mass.to_dict()

    pred = np.zeros((len(base_ids), len(TARGET_COLS)), dtype="float64")
    for i, (p, s) in enumerate(zip(pid, sid)):
        pp = patient_map.get(p, global_prior)
        sp = spectro_map.get(s, global_prior)

        mp = float(patient_m_map.get(p, 0.0))
        ms = float(spectro_m_map.get(s, 0.0))

        w = (ms + 1.0) / (ms + mp + 2.0)
        w = float(np.clip(w, 0.15, 0.85))
        pred[i] = (1.0 - w) * pp + w * sp

    pred = _finalize_probs(pred)

    out = base.copy()
    out[TARGET_COLS] = pred
    return out


if external_ok:
    sol = merge_preds(folds=(0, 1, 2, 3, 4), version="v2")
else:
    sol = make_prior_baseline()

print(sol.head())
print("sol shape:", sol.shape)



## === cell 4
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)

chk = pd.read_csv(sub_path)
assert list(chk.columns) == ["eeg_id"] + TARGET_COLS, "Submission columns mismatch"
row_sum = chk[TARGET_COLS].sum(axis=1).values
assert np.all(np.isfinite(row_sum)), "Non-finite probabilities"
assert np.max(np.abs(row_sum - 1.0)) < 1e-6, "Row probabilities do not sum to 1"
print("Saved valid submission:", sub_path, "shape=", chk.shape)
print(chk.head())
