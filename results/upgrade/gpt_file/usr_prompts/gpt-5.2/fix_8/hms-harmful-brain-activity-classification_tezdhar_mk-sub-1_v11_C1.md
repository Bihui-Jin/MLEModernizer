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

0.347951519440718

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the submission length mismatch by ensuring we always generate predictions for exactly the `eeg_id`s present in `sample_submission.csv` (and in that exact order), instead of relying on `test.csv` (which can differ). I also make the fold-merge robust by reindexing each fold’s predictions onto the sample’s `eeg_id` list, preventing accidental row drops/duplication and eliminating the need for a potentially lossy merge. These changes are score-neutral (they only fix alignment/format), and they guarantee the output CSV is valid (correct row count, column order, and per-row probabilities summing to 1). Finally, I keep the existing inference/ensemble logic intact and only adjust the data alignment and sanity checks.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far worse than the target (0.34795), so we should improve performance without changing the external model/inference core logic. The smallest likely win is to align the *ensemble averaging* with the KL metric by averaging in log/probability space (geometric mean) instead of plain arithmetic mean, which typically reduces KL for probability ensembles and is a minimal post-processing change. I also add an optional, very light Dirichlet-style smoothing toward uniform **only when folds disagree heavily**, to reduce overconfident wrong predictions (which KL penalizes), while keeping outputs valid and normalized. All I/O paths, fold generation, and submission alignment remain identical.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.34795), so we should improve predictive calibration with minimal, safe post-processing rather than changing the model. I keep your existing fold alignment and geometric-mean ensemble, but replace the heuristic disagreement-based uniform smoothing with a small, global Dirichlet prior (additive smoothing) which is a standard KL-friendly calibration step and usually reduces overconfident errors. I also add a very light temperature scaling on the ensembled probabilities (in log space) to further reduce KL spikes without altering per-fold inference. All changes preserve submission ordering/shape and keep probabilities normalized to sum to 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far above the target (0.34795), so we should reduce KL with minimal, low-risk calibration changes while keeping your inference and ensembling logic intact. I keep the fold alignment + geometric-mean ensemble, but make the two post-processing hyperparameters (Dirichlet additive smoothing `alpha` and temperature `T`) slightly stronger to reduce overconfident spikes, which KL heavily penalizes. I also add a tiny probability floor before normalization at the very end to avoid near-zero probabilities that can blow up KL on some samples, without changing the row order/format. These are small, metric-aligned post-processing tweaks only.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far above the target (0.34795), so we should improve calibration with minimal post-processing rather than touching the model/inference core. I keep your fold alignment and geometric-mean ensemble intact, but (1) make the Dirichlet smoothing slightly stronger and (2) apply a slightly higher temperature so the final probabilities are less overconfident—this typically reduces KL spikes. I also add an ultra-small final probability floor (and renormalize) to prevent near-zero probabilities, which can heavily penalize KL when the true class has nonzero mass. All paths, ordering, and submission schema remain unchanged.'

# 9. Code solution

## === cell 0
import sys
from pathlib import Path

MK_CODES_PATH = Path("/kaggle/input/hms-mk-codes")
if MK_CODES_PATH.exists():
    sys.path.insert(0, str(MK_CODES_PATH))



## === cell 1
import os
from pathlib import Path

REQ_DIR = Path("/kaggle/input/requirements-mk")
if REQ_DIR.exists():
    os.system(
        "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
    )
    os.system(
        "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
    )
    os.system(
        "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
    )
    os.system(
        "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
    )



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 3
import subprocess
from pathlib import Path

MK_REPO_CWD = Path("/kaggle/input/hms-mk-codes")
CAN_RUN_MK = MK_REPO_CWD.exists()

if CAN_RUN_MK:
    cmd = [
        "python",
        "-m",
        "src.convert_parquet_to_npy",
        f"--data_dir={DATA_PATH}",
        f"--out_dir={OUT_PATH}",
    ]
    subprocess.run(cmd, cwd=str(MK_REPO_CWD), check=True)
else:
    print(f"WARNING: {MK_REPO_CWD} not found; skipping parquet->npy conversion.")



## === cell 4
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 5
import shutil
from pathlib import Path
import subprocess

ckpt_base = Path("/kaggle/input/hms-mk-data")
fold_files = []

if CAN_RUN_MK and ckpt_base.exists():
    for i, ckpt_name in enumerate(checkpoints):
        ckpt_path = ckpt_base / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        cmd = [
            "python",
            "-m",
            "test",
            f"paths.data_dir={DATA_PATH}",
            f"data.test_eegs_dir={OUT_PATH}",
            f"ckpt_path={str(ckpt_path)}",
            "hydra=test",
            f"+model.test_output_dir={OUT_PATH}",
            "+model.net.pretrained=False",
        ]
        subprocess.run(cmd, cwd=str(MK_REPO_CWD), check=True)

        src_sub = Path("/kaggle/working/submission.csv")
        dst_sub = Path(f"/kaggle/working/submission_fold{i}.csv")
        if not src_sub.exists():
            raise FileNotFoundError(
                f"Expected {src_sub} to be created by inference, but it does not exist."
            )
        shutil.move(str(src_sub), str(dst_sub))
        fold_files.append(dst_sub)
else:
    print(
        "WARNING: External repo/checkpoints not available; skipping model inference and will fall back to a safe baseline submission."
    )



## === cell 6
import pandas as pd
import numpy as np
from pathlib import Path

sample_path = Path(DATA_PATH) / "sample_submission.csv"
sample = pd.read_csv(sample_path)
sample_eeg_ids = sample["eeg_id"].to_numpy()

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception:
    TARGET_COLS = [c for c in sample.columns if c != "eeg_id"]

FINAL_EPS = 1e-6

DIRICHLET_ALPHA = 0.18
TEMPERATURE = 1.70

POST_FLOOR = 3e-4


def _normalize_probs(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    s = arr.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return arr / s


def _geometric_mean_ensemble(preds_stack: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    preds_stack = np.asarray(preds_stack, dtype=np.float64)
    preds_stack = np.clip(preds_stack, eps, 1.0)
    logp = np.log(preds_stack)
    mean_logp = np.mean(logp, axis=0)
    p = np.exp(mean_logp)
    return _normalize_probs(p, eps=eps)


def _dirichlet_additive_smooth(probs: np.ndarray, alpha: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.shape[1]
    probs = probs + (alpha / k)
    return _normalize_probs(probs)


def _temperature_scale_probs(
    probs: np.ndarray, temperature: float, eps: float = 1e-12
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    logp = np.log(probs)
    logp = logp / float(temperature)
    p = np.exp(logp)
    return _normalize_probs(p, eps=eps)


def merge_preds(sample_eeg_ids_in_order, folds=(0, 1, 2, 3, 4)):
    preds_list = []

    for fold in folds:
        fold_path = Path(f"/kaggle/working/submission_fold{fold}.csv")
        if not fold_path.exists():
            raise FileNotFoundError(f"Missing fold submission: {fold_path}")

        sol = pd.read_csv(fold_path)
        missing = [c for c in (["eeg_id"] + list(TARGET_COLS)) if c not in sol.columns]
        if missing:
            raise ValueError(f"{fold_path} is missing required columns: {missing}")

        sol = sol[["eeg_id"] + list(TARGET_COLS)].copy()
        sol = sol.drop_duplicates(subset=["eeg_id"], keep="last").set_index("eeg_id")
        sol = sol.reindex(sample_eeg_ids_in_order)

        if sol.isna().any().any():
            k = len(TARGET_COLS)
            sol = sol.fillna(1.0 / k)

        preds_list.append(sol[list(TARGET_COLS)].to_numpy(dtype=np.float64))

    preds_stack = np.stack(preds_list, axis=0)  # (n_folds, n_rows, k)

    preds = _geometric_mean_ensemble(preds_stack)

    preds = _dirichlet_additive_smooth(preds, alpha=DIRICHLET_ALPHA)

    preds = _temperature_scale_probs(preds, temperature=TEMPERATURE)

    out = pd.DataFrame({"eeg_id": sample_eeg_ids_in_order})
    out.loc[:, TARGET_COLS] = preds
    return out


have_all_folds = all(
    Path(f"/kaggle/working/submission_fold{i}.csv").exists() for i in range(5)
)
if have_all_folds:
    sol = merge_preds(sample_eeg_ids, folds=[0, 1, 2, 3, 4])
else:
    sol = pd.DataFrame({"eeg_id": sample_eeg_ids})
    k = len(TARGET_COLS)
    for c in TARGET_COLS:
        sol[c] = 1.0 / k



## === cell 7
sol_out = sol.set_index("eeg_id").reindex(sample_eeg_ids).reset_index()
sol_out.rename(columns={"index": "eeg_id"}, inplace=True)

if sol_out[list(TARGET_COLS)].isna().any().any():
    k = len(TARGET_COLS)
    sol_out.loc[:, list(TARGET_COLS)] = sol_out[list(TARGET_COLS)].fillna(1.0 / k)

vals = sol_out[list(TARGET_COLS)].to_numpy(dtype=np.float64)
vals = _normalize_probs(vals, eps=FINAL_EPS)

k = len(TARGET_COLS)
vals = np.maximum(vals, POST_FLOOR / k)
vals = _normalize_probs(vals, eps=FINAL_EPS)

sol_out.loc[:, TARGET_COLS] = vals
sol_out = sol_out[["eeg_id"] + list(TARGET_COLS)]
sol_out.to_csv("/kaggle/working/submission.csv", index=False)



## === cell 8
sol_out.head()



## === cell 9
assert sol_out.shape[0] == sample.shape[0], "Row count mismatch vs sample_submission."
assert list(sol_out.columns) == list(
    sample.columns
), "Column order/names mismatch vs sample_submission."
row_sums = sol_out[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums)), "Non-finite probabilities found."
assert (
    np.max(np.abs(row_sums - 1.0)) < 1e-6
), "Probabilities do not sum to 1 within tolerance."
"/kaggle/working/submission.csv"
