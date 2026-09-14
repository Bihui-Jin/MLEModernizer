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

0.3407056092205677

# 6. Current score

0.78445

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I make the script robust to missing private inputs by (1) removing the failing offline `pip install` steps and instead using a safe, optional import path, and (2) making the parquet→npy conversion and multi-fold inference conditional on the required code/checkpoints actually existing. If the pretrained pipeline can’t run, I still produce a valid `submission.csv` by falling back to a calibrated label-prior baseline derived from `train.csv` (score won’t be great, but it be valid and should run end-to-end). I also fix the cell numbering to start at 1 (your current “cell 0” breaks the required format) and ensure the final submission has the exact required columns, row count, and per-row probabilities summing to 1.'
- What this solution (achieved 0.76634) has done: 'I fix the crash in the fallback path by removing the incorrect `validate="one_to_one"` merge: `sample_submission.csv` can contain duplicate `eeg_id`, so we must map `patient_id` from `test.csv` via a de-duplicated lookup to keep row counts intact. I also renumber cells to start at 1 (your current “cell 0” violates the required format) and keep the rest of the logic identical. Finally, I make the fallback prior computation deterministic and ensure the submission is always written even when inference artifacts are missing, with strict probability normalization so the file passes Kaggle checks.'
- What this solution (achieved 0.78445) has done: 'Your current fallback is a patient-level label-prior, which is a reasonable baseline but can be improved toward the target by (1) computing priors at the more specific `eeg_id` level when possible (train has many rows per `eeg_id`, and test has unique `eeg_id`), and (2) applying a small, deterministic shrinkage mix between the more specific prior and the global prior to reduce KL penalties from overconfident priors. This keeps the same “prior-from-train-votes” core logic (no model/feature changes), but makes the fallback predictions better calibrated for the KL metric. I also fix the merge/alignment in the inference path to preserve row order exactly as `sample_submission.csv` (avoid accidental row drops/NaNs that can inflate KL). The changes are minimal and still guarantee a valid `submission.csv` with per-row probabilities summing to 1.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

CODES_PATH = "/kaggle/input/hms-mk-codes"
if os.path.isdir(CODES_PATH) and CODES_PATH not in sys.path:
    sys.path.append(CODES_PATH)

INNER_PATH = os.path.join(CODES_PATH, "src")
if os.path.isdir(INNER_PATH) and INNER_PATH not in sys.path:
    sys.path.append(INNER_PATH)



## === cell 1
import subprocess


def _run(cmd: str, check: bool = True):
    print(cmd)
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=False)




## === cell 2
REQ_PATH = "/kaggle/input/requirements-mk"
wheels = [
    "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    "omegaconf-2.3.0-py3-none-any.whl",
    "hydra_core-1.3.2-py3-none-any.whl",
    "lightning-2.2.1-py3-none-any.whl",
]

if os.path.isdir(REQ_PATH):
    for whl in wheels:
        fp = os.path.join(REQ_PATH, whl)
        if os.path.exists(fp):
            _run(f"pip install {fp} --no-index --no-deps", check=False)
        else:
            print(f"Wheel not found, skipping: {fp}")
else:
    print(f"Requirements directory not found, skipping pip installs: {REQ_PATH}")



## === cell 3
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
os.makedirs(OUT_PATH, exist_ok=True)



## === cell 4
convert_ok = False
if os.path.isdir(CODES_PATH):
    cmd = f"cd {CODES_PATH} && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
    r = _run(cmd, check=False)
    convert_ok = r.returncode == 0
    print(f"convert_parquet_to_npy returncode={r.returncode}")
else:
    print(f"CODES_PATH not available, skipping conversion: {CODES_PATH}")



## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 6
_run("ls /kaggle/input/hms-mk-data || true", check=False)



## === cell 7
mk_data_path = "/kaggle/input/hms-mk-data"
fold_sub_paths = []
inference_ok = False

if os.path.isdir(CODES_PATH) and os.path.isdir(mk_data_path):
    for fold in range(5):
        ckpt = os.path.join(mk_data_path, f"fold{fold}_pseudo_log.ckpt")
        if not os.path.exists(ckpt):
            print(f"Missing checkpoint, skipping fold {fold}: {ckpt}")
            continue
        cmd = (
            f"cd {CODES_PATH} && python -m test "
            f"paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
            f"ckpt_path={ckpt} hydra=test +model.test_output_dir={OUT_PATH} "
            f"experiment=conv1d_pseudo +model.net.pretrained=False"
        )
        r = _run(cmd, check=False)
        if r.returncode != 0:
            print(
                f"Fold {fold} inference failed (returncode={r.returncode}); will fall back if needed."
            )
            continue

        src_fp = "/kaggle/working/submission.csv"
        dst_fp = f"/kaggle/working/submission_fold{fold}.csv"
        if os.path.exists(src_fp):
            _run(f"mv {src_fp} {dst_fp}", check=False)
            if os.path.exists(dst_fp):
                fold_sub_paths.append(dst_fp)

    inference_ok = len(fold_sub_paths) > 0
else:
    print(
        f"Private code or checkpoints not available; skipping model inference. CODES_PATH={CODES_PATH}, mk_data_path={mk_data_path}"
    )

print(f"inference_ok={inference_ok}, found_fold_submissions={len(fold_sub_paths)}")



## === cell 8
try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception:
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def merge_preds(fold_paths):
    preds = []
    sol0 = None
    for fp in fold_paths:
        sol = pd.read_csv(fp)
        if sol0 is None:
            sol0 = sol.copy()
        missing = [c for c in TARGET_COLS if c not in sol.columns]
        if missing:
            raise ValueError(f"Fold submission {fp} missing columns: {missing}")
        preds.append(sol[TARGET_COLS].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(preds, axis=0), axis=0)

    eps = 1e-12
    preds = np.clip(preds, eps, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    sol0[TARGET_COLS] = preds
    return sol0




## === cell 9
sample_fp = f"{DATA_PATH}/sample_submission.csv"
sample_sub = pd.read_csv(sample_fp)

if inference_ok:
    sol_pred = merge_preds(fold_sub_paths)
    if "eeg_id" not in sol_pred.columns:
        raise ValueError("Merged predictions missing `eeg_id` column.")

    sol = sample_sub[["eeg_id"]].copy()
    pred_lookup = sol_pred.drop_duplicates(subset=["eeg_id"], keep="first").set_index(
        "eeg_id"
    )[TARGET_COLS]
    for c in TARGET_COLS:
        sol[c] = sol["eeg_id"].map(pred_lookup[c])

else:
    train_fp = f"{DATA_PATH}/train.csv"
    test_fp = f"{DATA_PATH}/test.csv"

    train = pd.read_csv(train_fp, usecols=["eeg_id", "patient_id"] + TARGET_COLS)
    test = pd.read_csv(test_fp, usecols=["eeg_id", "patient_id"])

    eps = 1e-12
    K = len(TARGET_COLS)

    vote_sums_global = train[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    alpha_global = 1.0
    prior_global = (vote_sums_global + alpha_global) / (
        vote_sums_global.sum() + alpha_global * K
    )
    prior_global = np.clip(prior_global, eps, 1.0)
    prior_global = prior_global / prior_global.sum()

    grp_eeg = train.groupby("eeg_id", sort=False)[TARGET_COLS].sum()
    eeg_ids = grp_eeg.index.to_numpy()
    counts_eeg = grp_eeg.to_numpy(dtype=np.float64)
    alpha_eeg = 1.0
    denom_eeg = counts_eeg.sum(axis=1, keepdims=True) + alpha_eeg * K
    priors_eeg = (counts_eeg + alpha_eeg) / denom_eeg
    priors_eeg = np.clip(priors_eeg, eps, 1.0)
    priors_eeg = priors_eeg / priors_eeg.sum(axis=1, keepdims=True)
    eeg_to_prior = {int(eid): priors_eeg[i] for i, eid in enumerate(eeg_ids)}

    grp_pat = train.groupby("patient_id", sort=False)[TARGET_COLS].sum()
    pat_ids = grp_pat.index.to_numpy()
    counts_pat = grp_pat.to_numpy(dtype=np.float64)
    alpha_patient = 2.0
    denom_pat = counts_pat.sum(axis=1, keepdims=True) + alpha_patient * K
    priors_patient = (counts_pat + alpha_patient) / denom_pat
    priors_patient = np.clip(priors_patient, eps, 1.0)
    priors_patient = priors_patient / priors_patient.sum(axis=1, keepdims=True)
    patient_to_prior = {int(pid): priors_patient[i] for i, pid in enumerate(pat_ids)}

    test_lookup = test.drop_duplicates(subset=["eeg_id"], keep="first").set_index(
        "eeg_id"
    )["patient_id"]

    sol = sample_sub.copy()
    sol["patient_id"] = sol["eeg_id"].map(test_lookup)

    preds = np.zeros((len(sol), K), dtype=np.float64)
    eegs = sol["eeg_id"].to_numpy()
    pids = sol["patient_id"].to_numpy()

    w_global = (
        0.15  # small; preserves core "prior baseline" logic while improving calibration
    )
    for i, (eid, pid) in enumerate(zip(eegs, pids)):
        base = None
        if not pd.isna(eid):
            base = eeg_to_prior.get(int(eid), None)
        if base is None:
            if pd.isna(pid):
                base = prior_global
            else:
                base = patient_to_prior.get(int(pid), prior_global)
        preds[i] = (1.0 - w_global) * base + w_global * prior_global

    for j, c in enumerate(TARGET_COLS):
        sol[c] = preds[:, j]

    sol = sol[["eeg_id"] + TARGET_COLS]

vals = sol[TARGET_COLS].to_numpy(dtype=np.float64)
nan_mask = ~np.isfinite(vals)
if nan_mask.any():
    fallback = np.full((len(TARGET_COLS),), 1.0 / len(TARGET_COLS), dtype=np.float64)
    vals[nan_mask] = np.take(fallback, np.where(nan_mask)[1])

vals = np.clip(vals, 1e-12, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = vals

out_fp = "/kaggle/working/submission.csv"
sol.to_csv(out_fp, index=False)
print(f"Wrote: {out_fp}  shape={sol.shape}")
print(sol.head())



## === cell 10
assert os.path.exists("/kaggle/working/submission.csv")
sub = pd.read_csv("/kaggle/working/submission.csv")
assert list(sub.columns) == ["eeg_id"] + TARGET_COLS
assert len(sub) == len(sample_sub)
row_sums = sub[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert np.max(np.abs(row_sums - 1.0)) < 1e-6
print("Submission sanity checks passed.")
