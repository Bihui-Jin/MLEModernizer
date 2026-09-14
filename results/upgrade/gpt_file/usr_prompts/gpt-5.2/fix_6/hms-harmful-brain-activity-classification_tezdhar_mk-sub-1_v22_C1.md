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

0.3298398298703465

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Your notebook fails because it depends on two missing Kaggle Datasets (`/kaggle/input/hms-mk-codes` and `/kaggle/input/hms-mk-data`) that contain the inference code and checkpoints, so no per-fold prediction CSVs are ever produced. I remove the hard-fail on those missing inputs and add a safe fallback that generates a valid submission directly from `sample_submission.csv` (uniform class probabilities), ensuring correct columns and row-wise normalization so Kaggle accepts the file. This is score-worse than a real model but gets you a “yielded” score and an end-to-end runnable pipeline in the given environment. If you later attach the missing datasets, the original inference/ensemble path run as intended without changing core logic.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.40995, lower-is-better) comes from the uniform fallback because the real inference assets aren’t available, so the smallest legitimate way to move toward the target is to make the fallback less “wrong” while still being a proper probability distribution. I replace the uniform fallback with a class-prior fallback computed from `train.csv` by converting vote counts into per-row probabilities and averaging them (a calibrated prior under the same label definition), then apply the same row-normalization and clipping you already do. This keeps the pipeline end-to-end, preserves the existing ensemble path if the missing fold CSVs ever appear, and should reduce KL divergence versus uniform without changing any model logic. I also ensure the fallback aligns exactly to the `sample_submission.csv` `eeg_id` order.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)



## === cell 1
MK_CODES_DIR = "/kaggle/input/hms-mk-codes"
if os.path.isdir(MK_CODES_DIR):
    sys.path.append(MK_CODES_DIR)


def run_cmd(cmd, cwd=None):
    """Run a shell command reliably in Kaggle Python scripts (replaces notebook !magic)."""
    print(f"[run] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    if res.stdout:
        print(res.stdout[-2000:])
    if res.stderr:
        print(res.stderr[-2000:])
    return res




## === cell 2
REQ_DIR = "/kaggle/input/requirements-mk"
wheels = [
    "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    "omegaconf-2.3.0-py3-none-any.whl",
    "hydra_core-1.3.2-py3-none-any.whl",
    "lightning-2.2.1-py3-none-any.whl",
]
if os.path.isdir(REQ_DIR):
    for w in wheels:
        whl_path = os.path.join(REQ_DIR, w)
        if os.path.exists(whl_path):
            extra = "--no-deps --no-index"
            if "antlr4" in w:
                extra += " --force-reinstall"
            run_cmd(f"pip install {whl_path} {extra}")
        else:
            print(f"[warn] wheel not found: {whl_path} (skipping)")
else:
    print(f"[warn] requirements dir not found: {REQ_DIR} (skipping wheel installs)")



## === cell 3
HAS_MK_CODES = os.path.isdir(MK_CODES_DIR)
if not HAS_MK_CODES:
    print(
        f"[warn] Optional code repo not found at {MK_CODES_DIR}. Will use fallback submission."
    )
else:
    run_cmd(
        f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
        cwd=MK_CODES_DIR,
    )



## === cell 4
MK_DATA_DIR = "/kaggle/input/hms-mk-data"
print("MK_DATA_DIR exists:", os.path.isdir(MK_DATA_DIR))
if os.path.isdir(MK_DATA_DIR):
    print("MK_DATA_DIR files (sample):", sorted(os.listdir(MK_DATA_DIR))[:20])




## === cell 5
def run_fold(ckpt_path, experiment, out_csv_path, data_test_eegs_dir=OUT_PATH):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cmd = (
        "python -m test "
        f"paths.data_dir={DATA_PATH} "
        f"data.test_eegs_dir={data_test_eegs_dir} "
        f"ckpt_path={ckpt_path} "
        "hydra=test "
        f"+model.test_output_dir={OUT_PATH} "
        f"experiment={experiment} "
        "+model.net.pretrained=False"
    )
    run_cmd(cmd, cwd=MK_CODES_DIR)

    produced = os.path.join(OUT_PATH, "submission.csv")
    if not os.path.exists(produced):
        raise FileNotFoundError(f"Expected output submission not found at {produced}")

    shutil.move(produced, out_csv_path)
    print(f"[ok] wrote {out_csv_path}")


HAS_MK_DATA = os.path.isdir(MK_DATA_DIR)
if HAS_MK_CODES and HAS_MK_DATA:
    for fold in range(5):
        run_fold(
            ckpt_path=f"{MK_DATA_DIR}/fold{fold}_pseudo_log.ckpt",
            experiment="conv1d_pseudo",
            out_csv_path=f"{OUT_PATH}/submission_fold{fold}_v0.csv",
        )

    for fold in range(5):
        run_fold(
            ckpt_path=f"{MK_DATA_DIR}/fold{fold}_pseudo_resv2.ckpt",
            experiment="conv1d_resv2",
            out_csv_path=f"{OUT_PATH}/submission_fold{fold}_v2.csv",
        )
else:
    if not HAS_MK_CODES:
        print("[warn] Skipping inference: MK_CODES_DIR missing.")
    if not HAS_MK_DATA:
        print("[warn] Skipping inference: MK_DATA_DIR missing.")



## === cell 6
sample_path = os.path.join(DATA_PATH, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
if not os.path.exists(sample_path):
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

sample_sub = pd.read_csv(sample_path)

if sample_sub["eeg_id"].duplicated().any():
    print(
        "[warn] sample_submission has duplicated eeg_id; deduplicating by keeping first."
    )
    sample_sub = sample_sub.loc[~sample_sub["eeg_id"].duplicated(keep="first")].copy()

TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]
expected = {
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
}
if set(TARGET_COLS) != expected:
    raise ValueError(
        f"Unexpected target columns: {TARGET_COLS} vs expected {sorted(expected)}"
    )


def merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v2")):
    paths = []
    for fold in folds:
        for version in versions:
            p = f"{OUT_PATH}/submission_fold{fold}_{version}.csv"
            if os.path.exists(p):
                paths.append(p)
            else:
                raise FileNotFoundError(f"Missing prediction file: {p}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        missing = {"eeg_id", *TARGET_COLS} - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")
        dfs.append(df[["eeg_id"] + TARGET_COLS].copy())

    base = dfs[0][["eeg_id"]].copy()
    for i, df in enumerate(dfs):
        if not base["eeg_id"].equals(df["eeg_id"]):
            df = df.set_index("eeg_id").reindex(base["eeg_id"].values).reset_index()
            if df[TARGET_COLS].isna().any().any():
                raise ValueError(
                    f"Alignment introduced NaNs for {paths[i]} (eeg_id mismatch)."
                )
            dfs[i] = df

    preds = np.mean([df[TARGET_COLS].to_numpy(dtype=np.float64) for df in dfs], axis=0)

    row_sums = preds.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    preds = preds / row_sums

    out = base.copy()
    out[TARGET_COLS] = preds.astype(np.float32)
    return out


def fallback_uniform_submission():
    sub = sample_sub[["eeg_id"]].copy()
    k = len(TARGET_COLS)
    sub[TARGET_COLS] = 1.0 / k
    return sub


def fallback_train_prior_submission():
    """
    Global train prior from vote distributions.
    """
    train_path = os.path.join(DATA_PATH, "train.csv")
    if not os.path.exists(train_path):
        train_path = "/kaggle/input/train.csv"
    if not os.path.exists(train_path):
        print("[warn] train.csv not found; falling back to uniform.")
        return fallback_uniform_submission()

    train = pd.read_csv(train_path, usecols=TARGET_COLS)
    votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
    row_sums = votes.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs = votes / row_sums
    prior = probs.mean(axis=0)

    prior = np.clip(prior, 0.0, None)
    prior = prior / prior.sum()

    sub = sample_sub[["eeg_id"]].copy()
    for i, c in enumerate(TARGET_COLS):
        sub[c] = float(prior[i])
    return sub


def fallback_patient_prior_submission():
    """
    Change (score-directed): use patient-conditioned priors (avg vote distribution per patient_id).

    Bugfix: test.csv and/or sample_submission.csv can contain duplicate eeg_id rows in some environments,
    so validate='one_to_one' can crash. We instead deduplicate test.csv by eeg_id (keeping first),
    then do a safe left merge to preserve sample_submission's eeg_id order.
    """
    train_path = os.path.join(DATA_PATH, "train.csv")
    if not os.path.exists(train_path):
        train_path = "/kaggle/input/train.csv"
    test_path = os.path.join(DATA_PATH, "test.csv")
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/test.csv"

    if (not os.path.exists(train_path)) or (not os.path.exists(test_path)):
        print(
            "[warn] train.csv or test.csv not found; falling back to global train prior."
        )
        return fallback_train_prior_submission()

    usecols_train = ["patient_id"] + TARGET_COLS
    train = pd.read_csv(train_path, usecols=usecols_train)
    votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
    row_sums = votes.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs = votes / row_sums
    probs_df = pd.DataFrame(probs, columns=TARGET_COLS)
    probs_df["patient_id"] = train["patient_id"].values

    patient_prior = probs_df.groupby("patient_id")[TARGET_COLS].mean()

    global_prior = probs_df[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)
    global_prior = np.clip(global_prior, 0.0, None)
    global_prior = global_prior / global_prior.sum()

    test = pd.read_csv(test_path, usecols=["eeg_id", "patient_id"])
    if test["eeg_id"].duplicated().any():
        test = test.loc[~test["eeg_id"].duplicated(keep="first")].copy()

    base = sample_sub[["eeg_id"]].merge(test, on="eeg_id", how="left")
    if base["patient_id"].isna().any():
        base["patient_id"] = -1

    pred = np.empty((len(base), len(TARGET_COLS)), dtype=np.float64)
    pids = base["patient_id"].to_numpy()
    pred[:] = global_prior[None, :]

    seen_mask = np.isin(pids, patient_prior.index.values)
    if seen_mask.any():
        pred_seen = patient_prior.reindex(pids[seen_mask]).to_numpy(dtype=np.float64)
        ok = ~np.isnan(pred_seen).any(axis=1)
        if ok.any():
            idx = np.flatnonzero(seen_mask)[ok]
            pred[idx] = pred_seen[ok]

    pred = np.clip(pred, 0.0, None)
    rs = pred.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    pred = pred / rs

    sub = base[["eeg_id"]].copy()
    sub[TARGET_COLS] = pred.astype(np.float32)
    return sub




## === cell 7
can_ensemble = True
needed = [
    f"{OUT_PATH}/submission_fold{f}_{v}.csv" for f in range(5) for v in ["v0", "v2"]
]
for p in needed:
    if not os.path.exists(p):
        can_ensemble = False
        break

if can_ensemble:
    sol = merge_preds(folds=[0, 1, 2, 3, 4], versions=["v0", "v2"])
else:
    print(
        "[warn] Missing fold prediction files; writing patient-prior fallback submission."
    )
    sol = fallback_patient_prior_submission()

p = sol[TARGET_COLS].to_numpy(dtype=np.float64)
p = np.clip(p, 1e-12, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = p.astype(np.float32)



## === cell 8
sub_path = os.path.join(OUT_PATH, "submission.csv")
sol.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("Shape:", sol.shape)
print(
    "Row-sum min/max:",
    float(sol[TARGET_COLS].sum(axis=1).min()),
    float(sol[TARGET_COLS].sum(axis=1).max()),
)
print(sol.head())
