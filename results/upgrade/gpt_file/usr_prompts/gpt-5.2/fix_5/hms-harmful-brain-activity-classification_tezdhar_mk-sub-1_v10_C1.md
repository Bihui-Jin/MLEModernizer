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

0.3556530763563889

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The crashes come from assuming an external code repo (`/kaggle/input/hms-mk-codes`) and checkpoints exist; in your environment they don’t, so no fold submissions are ever created and the merge step fails. To make the notebook run end-to-end and still produce a valid `.csv`, I add a robust fallback path: if the repo (or any fold file) is missing, generate a submission directly from `sample_submission.csv` using safe, normalized probabilities. This keeps the original multi-fold merge logic intact when fold files exist, but guarantees a correct-format submission otherwise (rows sum to 1, correct columns, correct row count/order). This is score-neutral vs “no submission” and should at least yield a valid leaderboard score.'
- What this solution (achieved 1.41937) has done: 'Your current 1.40995 score suggests you’re already producing a valid submission, but the fallback path is still extremely weak because it predicts uniform probabilities for every class, which is far from the empirical label distribution and hurts KL divergence. To move toward the target 0.3556 with minimal change and without altering the core “use fold predictions if available” logic, I only improve the fallback by using the class prior estimated from `train.csv` vote totals (normalized) and use that as a constant probability vector for all test rows. This keeps evaluation semantics (probabilities summing to 1), avoids any leakage (train labels are allowed for priors), and typically yields a much better baseline than uniform when no model outputs exist. If fold submissions are present, the original merge behavior remains unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/input/hms-mk-codes")
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR))

print("Python:", sys.version)
print("Repo exists:", REPO_DIR.exists())
print("Repo in sys.path:", str(REPO_DIR) in sys.path)



## === cell 1
import subprocess


def _pip_install(wheel_path, extra_args=None):
    extra_args = extra_args or []
    if Path(wheel_path).exists():
        cmd = ["pip", "install", wheel_path] + extra_args
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"Wheel not found, skipping: {wheel_path}")


_pip_install(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    ["--no-index", "--no-deps", "--force-reinstall"],
)
_pip_install(
    "/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl",
    ["--no-index", "--no-deps"],
)
_pip_install(
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl",
    ["--no-index", "--no-deps"],
)
_pip_install(
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
    ["--no-deps", "--no-index"],
)



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)



## === cell 3
convert_ok = False
if REPO_DIR.exists():
    try:
        subprocess.check_call(
            [
                "python",
                "-m",
                "src.convert_parquet_to_npy",
                "--data_dir",
                DATA_PATH,
                "--out_dir",
                OUT_PATH,
            ],
            cwd=str(REPO_DIR),
        )
        convert_ok = True
    except Exception as e:
        print(
            "Warning: convert_parquet_to_npy failed; continuing with fallback path.",
            repr(e),
        )
else:
    print(
        f"Warning: repo directory missing at {REPO_DIR}; skipping parquet->npy conversion."
    )
print("convert_ok:", convert_ok)




## === cell 4
def _run_test_with_ckpt(ckpt_path: str) -> bool:
    ckpt = Path(ckpt_path)
    if not REPO_DIR.exists():
        print("Skipping test run: missing repo:", REPO_DIR)
        return False
    if not ckpt.exists():
        print("Skipping test run: missing checkpoint:", ckpt)
        return False

    try:
        subprocess.check_call(
            [
                "python",
                "-m",
                "test",
                f"paths.data_dir={DATA_PATH}",
                f"data.test_eegs_dir={OUT_PATH}",
                f"ckpt_path={ckpt_path}",
                "hydra=test",
                f"+model.test_output_dir={OUT_PATH}",
                "+model.net.pretrained=False",
            ],
            cwd=str(REPO_DIR),
        )
        return True
    except Exception as e:
        print("Warning: test run failed:", ckpt_path, repr(e))
        return False


fold2_ok = _run_test_with_ckpt(
    "/kaggle/input/hms-mk-data/epoch_014_val_loss_0.4944.ckpt"
)
print("fold2_ok:", fold2_ok)



## === cell 5
from pathlib import Path

src_sub = Path("/kaggle/working/submission.csv")
dst_sub = Path("/kaggle/working/submission_fold2.csv")
if src_sub.exists():
    src_sub.replace(dst_sub)
    print("Wrote:", dst_sub)
else:
    print("No submission.csv produced for fold2; will rely on other folds/fallback.")



## === cell 6
fold1_ok = _run_test_with_ckpt(
    "/kaggle/input/hms-mk-data/epoch_014_val_loss_0.4728.ckpt"
)
print("fold1_ok:", fold1_ok)



## === cell 7
src_sub = Path("/kaggle/working/submission.csv")
dst_sub = Path("/kaggle/working/submission_fold1.csv")
if src_sub.exists():
    src_sub.replace(dst_sub)
    print("Wrote:", dst_sub)
else:
    print("No submission.csv produced for fold1; will rely on other folds/fallback.")



## === cell 8
fold0_ok = _run_test_with_ckpt(
    "/kaggle/input/hms-mk-data/epoch_012_val_loss_0.5037.ckpt"
)
print("fold0_ok:", fold0_ok)



## === cell 9
src_sub = Path("/kaggle/working/submission.csv")
dst_sub = Path("/kaggle/working/submission_fold0.csv")
if src_sub.exists():
    src_sub.replace(dst_sub)
    print("Wrote:", dst_sub)
else:
    print("No submission.csv produced for fold0; will rely on other folds/fallback.")



## === cell 10
import pandas as pd
import numpy as np

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print("Warning: could not import TARGET_COLS from src.settings:", repr(e))
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def _train_priors_from_votes(data_path: str = DATA_PATH) -> np.ndarray:
    """
    Use the empirical class prior from train.csv vote totals.
    """
    train_path = Path(data_path) / "train.csv"
    if not train_path.exists():
        train_path = Path("/kaggle/input") / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train.csv at {train_path}")

    usecols = TARGET_COLS
    train = pd.read_csv(train_path, usecols=usecols)

    vote_sums = train[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    total = float(vote_sums.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid vote totals computed from train.csv")

    priors = vote_sums / total
    priors = np.clip(priors, 1e-15, None)
    priors = priors / priors.sum()
    return priors


def merge_preds(folds=(0, 1, 2), data_path: str = DATA_PATH):
    preds_frames = []
    eeg_union = None

    for fold in folds:
        path = f"/kaggle/working/submission_fold{fold}.csv"
        if not Path(path).exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        missing = [c for c in (["eeg_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        df = df[["eeg_id"] + TARGET_COLS].copy()
        if df["eeg_id"].duplicated().any():
            df = df.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

        preds_frames.append(df.set_index("eeg_id"))

        if eeg_union is None:
            eeg_union = set(df["eeg_id"].tolist())
        else:
            eeg_union |= set(df["eeg_id"].tolist())

    eeg_union = sorted(eeg_union)

    mats = []
    for fdf in preds_frames:
        aligned = fdf.reindex(eeg_union)
        if aligned.isna().any(axis=None):
            priors = _train_priors_from_votes(data_path=data_path)
            fill = np.tile(priors.reshape(1, -1), (aligned.shape[0], 1))
            aligned = aligned.to_numpy(dtype=np.float64)
            mask = ~np.isfinite(aligned)
            if mask.any():
                aligned[mask] = fill[mask]
            aligned = pd.DataFrame(aligned, index=eeg_union, columns=TARGET_COLS)
        mats.append(aligned[TARGET_COLS].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(mats, axis=0), axis=0)

    eps = 0.02
    priors = _train_priors_from_votes(data_path=data_path).reshape(1, -1)
    preds = (1.0 - eps) * preds + eps * priors

    preds = np.clip(preds, 1e-15, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    sol = pd.DataFrame(preds, columns=TARGET_COLS)
    sol.insert(0, "eeg_id", eeg_union)

    row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
    if not np.all(np.isfinite(row_sums)):
        raise ValueError("Non-finite probabilities found in merged submission.")
    if np.max(np.abs(row_sums - 1.0)) > 1e-6:
        raise ValueError(
            f"Row probabilities do not sum to 1 (max deviation {np.max(np.abs(row_sums - 1.0))})."
        )

    return sol


def fallback_submission(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    If no fold predictions exist, produce a valid submission using train-vote priors.
    """
    sample_path = Path(data_path) / "sample_submission.csv"
    if not sample_path.exists():
        sample_path = Path("/kaggle/input") / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Could not find sample_submission.csv at {sample_path}"
        )

    sub = pd.read_csv(sample_path)
    missing = [c for c in (["eeg_id"] + TARGET_COLS) if c not in sub.columns]
    if missing:
        raise ValueError(f"sample_submission missing columns: {missing}")

    priors = _train_priors_from_votes(data_path=data_path)
    probs = np.tile(priors.reshape(1, -1), (len(sub), 1)).astype(np.float64, copy=False)
    probs = np.clip(probs, 1e-15, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    sub.loc[:, TARGET_COLS] = probs

    row_sums = sub[TARGET_COLS].sum(axis=1).to_numpy()
    if np.max(np.abs(row_sums - 1.0)) > 1e-9:
        raise ValueError("Fallback submission rows do not sum to 1.")
    return sub[["eeg_id"] + TARGET_COLS].copy()




## === cell 11
available_folds = []
for f in (0, 1, 2):
    if Path(f"/kaggle/working/submission_fold{f}.csv").exists():
        available_folds.append(f)

if available_folds:
    print("Merging available folds:", available_folds)
    sol = merge_preds(folds=tuple(available_folds), data_path=DATA_PATH)
else:
    print("No fold submissions found; using fallback submission.")
    sol = fallback_submission(data_path=DATA_PATH)

sample_path = Path(DATA_PATH) / "sample_submission.csv"
if not sample_path.exists():
    sample_path = Path("/kaggle/input") / "sample_submission.csv"
sample = pd.read_csv(sample_path, usecols=["eeg_id"] + TARGET_COLS)

sol = sample[["eeg_id"]].merge(sol, on="eeg_id", how="left")
if sol[TARGET_COLS].isna().any(axis=None):
    priors = _train_priors_from_votes(data_path=DATA_PATH)
    sol.loc[:, TARGET_COLS] = sol[TARGET_COLS].to_numpy(dtype=np.float64)
    arr = sol[TARGET_COLS].to_numpy(dtype=np.float64)
    mask = ~np.isfinite(arr)
    if mask.any():
        fill = np.tile(priors.reshape(1, -1), (arr.shape[0], 1))
        arr[mask] = fill[mask]
        sol.loc[:, TARGET_COLS] = arr

arr = sol[TARGET_COLS].to_numpy(dtype=np.float64)
arr = np.clip(arr, 1e-15, None)
arr = arr / arr.sum(axis=1, keepdims=True)
sol.loc[:, TARGET_COLS] = arr

sol.head()



## === cell 12
out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)
print("Final submission written to:", out_file)
print("Shape:", sol.shape)
print("Columns:", list(sol.columns))

row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
print("Row sum stats:", float(row_sums.min()), float(row_sums.max()))



## === cell 13
sol
