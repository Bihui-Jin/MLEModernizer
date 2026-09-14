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

0.3127713983230008

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the `ModuleNotFoundError: No module named 'src'` by ensuring `/kaggle/input/hms-mk-codes` is on `sys.path` (your current path points to a different dataset), and I make the ensembling code independent of that import by defining `TARGET_COLS` directly from `sample_submission.csv`. I also fix the cell numbering (must start at 1) and make the merge robust by starting from the sample submission (guarantees correct `eeg_id` ordering/row count), then averaging fold predictions and re-normalizing so each row sums to 1 (required by the competition). These changes are score-neutral except for preventing invalid/misaligned submissions, and they ensure a valid `/kaggle/working/submission.csv` is always produced.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.3128), so we should improve performance with minimal risk while preserving the same inference core. The biggest likely issue is that `data.test_eegs_dir` is being pointed at `OUT_PATH` (where converted files may or may not exist), which can silently make the model read the wrong/empty inputs and collapse predictions; we instead point it at the real `test_eegs/` directory and only use `OUT_PATH` as a fallback if conversion actually produced `.npy` files. We also add a tiny, metric-consistent safety: clip probabilities away from exactly 0/1 and re-normalize, which prevents pathological KL blow-ups if any fold outputs hard zeros due to missing rows. Finally, we keep the sample-submission alignment but enforce float columns and handle duplicate `eeg_id` defensively (averaging duplicates) to avoid subtle merge artifacts.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (KL is lower-is-better), so the most likely “minimal-change” improvement is to ensure the inference code is actually reading the correct test EEG inputs rather than silently falling back to an empty/wrong directory. I make the test EEG directory selection stricter by requiring that converted `.npy` files match *test eeg_ids* (not just “any .npy exists”), otherwise we always use the official `test_eegs/` parquet directory. I also make the fold-output directory unique per fold so folds can’t overwrite each other’s outputs in `OUT_PATH`, and I keep your same averaging/normalization but add a tiny floor before renorm (still consistent with KL and avoids pathological zeros). These are execution- and alignment-safety changes intended to move your score down toward the target without changing the model or training/inference semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import numpy as np
import pandas as pd

MK_CODES_DIR = "/kaggle/input/hms-mk-codes"
if os.path.isdir(MK_CODES_DIR) and MK_CODES_DIR not in sys.path:
    sys.path.append(MK_CODES_DIR)

print("Python:", sys.version)
print("MK_CODES_DIR exists:", os.path.isdir(MK_CODES_DIR))
print("sys.path contains MK_CODES_DIR:", MK_CODES_DIR in sys.path)



## === cell 1
import subprocess


def _maybe_pip_install(wheel_path, extra_args=None):
    extra_args = extra_args or []
    if os.path.exists(wheel_path):
        cmd = ["pip", "install", wheel_path] + extra_args
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"Skipping install (not found): {wheel_path}")


_maybe_pip_install(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    ["--no-index", "--no-deps", "--force-reinstall"],
)
_maybe_pip_install(
    "/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl",
    ["--no-index", "--no-deps"],
)
_maybe_pip_install(
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl",
    ["--no-index", "--no-deps"],
)
_maybe_pip_install(
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
    ["--no-deps", "--no-index"],
)



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)

print("DATA_PATH exists:", os.path.isdir(DATA_PATH))
print("OUT_PATH exists:", os.path.isdir(OUT_PATH))



## === cell 3
convert_mod = os.path.join(MK_CODES_DIR, "src", "convert_parquet_to_npy.py")
if os.path.exists(convert_mod):
    cmd = [
        sys.executable,
        "-m",
        "src.convert_parquet_to_npy",
        f"--data_dir={DATA_PATH}",
        f"--out_dir={OUT_PATH}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=MK_CODES_DIR)
else:
    print("Conversion module not found, skipping:", convert_mod)



## === cell 4
path_to_list = "/kaggle/input/hms-mk-data"
if os.path.isdir(path_to_list):
    files = sorted(os.listdir(path_to_list))
    print("Found", len(files), "files in", path_to_list)
    print("First 20:", files[:20])
else:
    print(
        "Directory not found (this may be OK if the dataset isn't attached):",
        path_to_list,
    )



## === cell 5
SAMPLE_SUB_PATH = os.path.join(DATA_PATH, "sample_submission.csv")
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]
assert "eeg_id" in sample_sub.columns
assert len(TARGET_COLS) == 6, f"Unexpected target columns: {TARGET_COLS}"
print("TARGET_COLS:", TARGET_COLS)

test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
test_eeg_ids = set(test_df["eeg_id"].astype(str).tolist())

TEST_EEGS_PARQUET_DIR = os.path.join(DATA_PATH, "test_eegs")
TEST_EEGS_NPY_DIR = OUT_PATH  # used only if conversion created matching .npy files here


def _select_test_eegs_dir():
    npy_files = glob.glob(os.path.join(TEST_EEGS_NPY_DIR, "*.npy"))
    npy_basenames = set(os.path.splitext(os.path.basename(p))[0] for p in npy_files)
    matched = len(npy_basenames.intersection(test_eeg_ids))

    if len(npy_files) > 0:
        print(
            f"Found .npy in OUT_PATH: total={len(npy_files)} matched_test_ids={matched} (need >0 match to use)."
        )

    if matched > 0:
        print("Using converted .npy test EEGs in OUT_PATH.")
        return TEST_EEGS_NPY_DIR

    if os.path.isdir(TEST_EEGS_PARQUET_DIR):
        pq_files = glob.glob(os.path.join(TEST_EEGS_PARQUET_DIR, "*.parquet"))
        print(f"Using parquet test EEGs in DATA_PATH (count={len(pq_files)}).")
        return TEST_EEGS_PARQUET_DIR

    print("WARNING: No test EEG directory found; falling back to OUT_PATH.")
    return TEST_EEGS_NPY_DIR


SELECTED_TEST_EEGS_DIR = _select_test_eegs_dir()
print("SELECTED_TEST_EEGS_DIR:", SELECTED_TEST_EEGS_DIR)

ckpt_dir = "/kaggle/input/hms-mk-data"
ckpts = {
    0: os.path.join(ckpt_dir, "fold0_effb3_sim_pseudo.ckpt"),
    1: os.path.join(ckpt_dir, "fold1_effb3_sim_pseudo.ckpt"),
    2: os.path.join(ckpt_dir, "fold2_effb3_sim_pseudo.ckpt"),
    3: os.path.join(ckpt_dir, "fold3_effb3_sim_pseudo.ckpt"),
    4: os.path.join(ckpt_dir, "fold4_effb3_sim_pseudo.ckpt"),
}


def run_fold(fold, ckpt_path, version="v3"):
    if not os.path.exists(ckpt_path):
        print(f"[fold {fold}] Checkpoint missing, skipping: {ckpt_path}")
        return None

    fold_out_dir = os.path.join(OUT_PATH, f"fold{fold}_out")
    os.makedirs(fold_out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "test",
        f"paths.data_dir={DATA_PATH}",
        f"data.test_eegs_dir={SELECTED_TEST_EEGS_DIR}",
        f"ckpt_path={ckpt_path}",
        "hydra=test",
        f"+model.test_output_dir={fold_out_dir}",
        "experiment=conv1d_effv2_pseudo",
        "+model.net.pretrained=False",
    ]
    print(f"[fold {fold}] Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=MK_CODES_DIR)

    src_sub = os.path.join(fold_out_dir, "submission.csv")
    dst_sub = os.path.join(OUT_PATH, f"submission_fold{fold}_{version}.csv")
    if os.path.exists(src_sub):
        os.replace(src_sub, dst_sub)
        print(f"[fold {fold}] Wrote:", dst_sub)
        return dst_sub
    else:
        alt_src = os.path.join(OUT_PATH, "submission.csv")
        if os.path.exists(alt_src):
            os.replace(alt_src, dst_sub)
            print(f"[fold {fold}] Wrote (fallback):", dst_sub)
            return dst_sub
        print(f"[fold {fold}] Expected submission not found at:", src_sub)
        return None


produced = []
for f in [0, 1, 2, 3, 4]:
    out = run_fold(f, ckpts[f], version="v3")
    if out is not None:
        produced.append(out)

print("Produced fold submissions:", produced)




## === cell 6
def merge_preds(folds=(0, 1, 2, 3, 4), version="v3", weights=None, eps=1e-6):
    """
    Merge fold predictions by weighted average, aligned on eeg_id using sample_submission as template.
    Ensures probabilities sum to 1 for each row.

    eps is score-relevant for KL: avoids exact 0/1 probabilities that can produce extreme KL.
    """
    weights = weights if weights is not None else [1.0] * len(folds)
    if len(weights) != len(folds):
        raise ValueError("weights must have same length as folds")

    base = sample_sub.copy()
    base[TARGET_COLS] = 0.0

    total_w = 0.0
    used = 0
    for fold, w in zip(folds, weights):
        path = os.path.join(OUT_PATH, f"submission_fold{fold}_{version}.csv")
        if not os.path.exists(path):
            print(f"Missing fold file, skipping: {path}")
            continue
        df = pd.read_csv(path)

        if df["eeg_id"].duplicated().any():
            df = df.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

        df = base[["eeg_id"]].merge(df, on="eeg_id", how="left", validate="1:1")

        arr = df[TARGET_COLS].to_numpy(dtype=np.float64)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)

        base[TARGET_COLS] += arr * float(w)
        total_w += float(w)
        used += 1

    if used == 0 or total_w == 0:
        print("No fold predictions found; writing uniform-probability submission.")
        base[TARGET_COLS] = 1.0 / len(TARGET_COLS)
        return base

    base[TARGET_COLS] /= total_w

    probs = base[TARGET_COLS].to_numpy(dtype=np.float64)

    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    base[TARGET_COLS] = probs
    base[TARGET_COLS] = base[TARGET_COLS].astype(np.float64)
    return base




## === cell 7
sol = merge_preds(
    folds=[0, 1, 2, 3, 4], version="v3", weights=[1.0, 1.0, 1.0, 1.0, 1.0], eps=1e-6
)
print(sol.shape)
print(sol.head())



## === cell 8
SUB_PATH = "/kaggle/working/submission.csv"
sol.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH)
print("Column check:", list(sol.columns))
print(
    "Row-sum min/max:",
    sol[TARGET_COLS].sum(axis=1).min(),
    sol[TARGET_COLS].sum(axis=1).max(),
)

assert sol.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"
assert list(sol.columns) == list(
    sample_sub.columns
), "Column order mismatch vs sample_submission"
assert np.allclose(
    sol[TARGET_COLS].sum(axis=1).values, 1.0, atol=1e-6
), "Probabilities do not sum to 1"
print(sol.head())
