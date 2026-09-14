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

0.3405953164188462

# 6. Current score

0.85134

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48697) has done: 'The failures come from relying on external `/kaggle/input/hms-mk-codes` and `/kaggle/input/hms-mk-data` assets that aren’t present in this environment, so the conversion and fold inference commands exit immediately and no per-fold submissions are produced. To make the notebook run end-to-end and still generate a valid submission, I replace those external calls with an in-notebook baseline that uses the training vote distribution aggregated by `eeg_id` as class priors, then predicts the same calibrated prior for every test row (this is score-stable and usually reasonable for KL). I also keep your ensembling/validation logic, but make it robust to the new single-file prediction generation. The output is a correct `submission.csv` with required columns and row-wise probabilities summing to 1.'
- What this solution (achieved 0.85134) has done: 'The timeout is coming from the external pipeline: installing extra wheels, converting all parquet files to NPY, and running 5 separate `python -m test` inference subprocesses; that work is far beyond a 600s budget. To preserve correctness and core logic of the submission generation, the optimized script skips those heavyweight steps by default and directly produces the same in-notebook probabilistic baselines (Dirichlet-smoothed vote priors), which are already present in your code. Additionally, the invalid-submission length issue is fixed by always aligning predictions to `sample_submission`’s `eeg_id` set/order via an index-based reindex (no accidental row loss/duplication). Finally, the slow per-row Python loop in the blended baseline is replaced with a fully vectorized join-based computation that is mathematically identical but much faster.'

# 9. Code solution

## === cell 0
import sys
from pathlib import Path

sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 1
import os

wheel_paths = [
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    "/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl",
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl",
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
]
SKIP_WHEEL_INSTALL = True  # must be True for 600s budget
if not SKIP_WHEEL_INSTALL:
    for wp in wheel_paths:
        if os.path.exists(wp):
            if "antlr4" in wp:
                os.system(f"pip install {wp} --no-index --no-deps --force-reinstall")
            elif "lightning" in wp:
                os.system(f"pip install {wp} --no-deps --no-index")
            else:
                os.system(f"pip install {wp} --no-index --no-deps")
        else:
            print(f"[info] wheel not found, skipping install: {wp}")
else:
    print("[info] Skipping wheel installs for runtime.")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)



## === cell 3
import os
from pathlib import Path

SKIP_PARQUET_TO_NPY = True
cmd = f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
if not SKIP_PARQUET_TO_NPY and Path("/kaggle/input/hms-mk-codes").exists():
    ret = os.system(cmd)
    if ret != 0:
        print(
            f"[warn] convert_parquet_to_npy failed with exit code {ret}; continuing with fallback baseline."
        )
else:
    print("[info] Skipping parquet->npy conversion for runtime.")



## === cell 4
import os
from pathlib import Path

RUN_EXTERNAL_INFERENCE = False

have_codes = Path("/kaggle/input/hms-mk-codes").exists()
have_data = Path("/kaggle/input/hms-mk-data").exists()

if RUN_EXTERNAL_INFERENCE and have_codes and have_data:
    for fold in range(5):
        ckpt = f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_log.ckpt"
        cmd = (
            "cd /kaggle/input/hms-mk-codes && "
            f"python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
            f"ckpt_path={ckpt} hydra=test +model.test_output_dir={OUT_PATH} "
            "experiment=conv1d_pseudo +model.net.pretrained=False"
        )
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"Fold {fold} inference failed with exit code {ret}")
        src = "/kaggle/working/submission.csv"
        dst = f"/kaggle/working/submission_fold{fold}_v0.csv"
        if not Path(src).exists():
            raise FileNotFoundError(f"Expected inference output not found: {src}")
        os.system(f"mv {src} {dst}")
else:
    print(
        "[info] External inference disabled/unavailable; will generate an in-notebook baseline submission."
    )



## === cell 5
import pandas as pd
import numpy as np
from pathlib import Path

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
TRAIN_PATH = f"{DATA_PATH}/train.csv"
TEST_PATH = f"{DATA_PATH}/test.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]


def merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0",), working_dir="/kaggle/working"):
    preds_list = []
    eeg_id_ref = None
    sol_ref = None

    for fold in folds:
        for version in versions:
            fp = Path(working_dir) / f"submission_fold{fold}_{version}.csv"
            if not fp.exists():
                raise FileNotFoundError(f"Missing prediction file: {fp}")

            sol = pd.read_csv(fp)

            if "eeg_id" not in sol.columns:
                raise ValueError(f"'eeg_id' column missing in {fp}")
            missing = [c for c in TARGET_COLS if c not in sol.columns]
            if missing:
                raise ValueError(f"Missing target columns {missing} in {fp}")

            sol = sol.sort_values("eeg_id").reset_index(drop=True)
            if eeg_id_ref is None:
                eeg_id_ref = sol["eeg_id"].to_numpy()
                sol_ref = sol[["eeg_id"] + TARGET_COLS].copy()
            else:
                if not np.array_equal(eeg_id_ref, sol["eeg_id"].to_numpy()):
                    raise ValueError(f"eeg_id order/content mismatch in {fp}")

            preds_list.append(sol[TARGET_COLS].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)
    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = sol_ref.copy()
    out[TARGET_COLS] = preds
    return out


def _dirichlet_mean_from_votes(votes_mat: np.ndarray, alpha: float) -> np.ndarray:
    K = votes_mat.shape[1]
    denom = votes_mat.sum(axis=1, keepdims=True) + alpha * K
    return (votes_mat + alpha) / denom


def _mix_with_uniform(prob_mat: np.ndarray, mix: float) -> np.ndarray:
    K = prob_mat.shape[1]
    uniform = np.full((1, K), 1.0 / K, dtype=np.float64)
    out = (1.0 - mix) * prob_mat + mix * uniform
    out = np.clip(out, 1e-12, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def make_spectro_patient_global_blend_submission(
    train_path=TRAIN_PATH, test_path=TEST_PATH, sample_sub_path=SAMPLE_SUB_PATH
):
    train = pd.read_csv(
        train_path,
        usecols=["eeg_id", "patient_id", "spectrogram_id"] + TARGET_COLS,
    )
    test = pd.read_csv(test_path, usecols=["eeg_id", "patient_id", "spectrogram_id"])
    sample = pd.read_csv(sample_sub_path, usecols=["eeg_id"] + TARGET_COLS)

    K = len(TARGET_COLS)
    alpha = 2.0
    mix = 0.02

    eeg_votes = (
        train.groupby(["eeg_id", "patient_id", "spectrogram_id"], sort=False)[
            TARGET_COLS
        ]
        .sum()
        .reset_index()
    )
    spectro_votes = eeg_votes.groupby("spectrogram_id", sort=False)[TARGET_COLS].sum()
    patient_votes = eeg_votes.groupby("patient_id", sort=False)[TARGET_COLS].sum()

    global_votes = (
        eeg_votes[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)[None, :]
    )
    global_prob = _mix_with_uniform(
        _dirichlet_mean_from_votes(global_votes, alpha=alpha), mix=mix
    )[0]

    sp_prob = _mix_with_uniform(
        _dirichlet_mean_from_votes(
            spectro_votes.to_numpy(dtype=np.float64), alpha=alpha
        ),
        mix=mix,
    )
    sp_prob_df = pd.DataFrame(sp_prob, index=spectro_votes.index, columns=TARGET_COLS)

    pv_prob = _mix_with_uniform(
        _dirichlet_mean_from_votes(
            patient_votes.to_numpy(dtype=np.float64), alpha=alpha
        ),
        mix=mix,
    )
    pv_prob_df = pd.DataFrame(pv_prob, index=patient_votes.index, columns=TARGET_COLS)

    base = sample[["eeg_id"]].merge(test, on="eeg_id", how="left")
    base = base.sort_values("eeg_id").reset_index(drop=True)

    w_global = 0.10
    w_spec = 0.70
    w_pat = 0.20

    base2 = base.merge(
        sp_prob_df.reset_index().rename(columns={"index": "spectrogram_id"}),
        on="spectrogram_id",
        how="left",
        suffixes=("", "_sp"),
    )
    base2 = base2.merge(
        pv_prob_df.reset_index().rename(columns={"index": "patient_id"}),
        on="patient_id",
        how="left",
        suffixes=("", "_pv"),
    )

    sp_mat = base2[TARGET_COLS].to_numpy(dtype=np.float64)  # from spectrogram join
    pv_mat = base2[[c + "_pv" for c in TARGET_COLS]].to_numpy(dtype=np.float64)

    sp_present = ~np.isnan(sp_mat).any(axis=1)
    pv_present = ~np.isnan(pv_mat).any(axis=1)

    sp_mat = np.nan_to_num(sp_mat, nan=0.0)
    pv_mat = np.nan_to_num(pv_mat, nan=0.0)

    pat_scale = np.where(sp_present, 0.5, 1.0) * pv_present.astype(np.float64)

    preds = (
        w_global * global_prob[None, :]
        + w_spec * sp_mat
        + (w_pat * pat_scale)[:, None] * pv_mat
    )

    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = base[["eeg_id"]].copy()
    out[TARGET_COLS] = preds
    return out


def make_patient_prior_baseline_submission(
    train_path=TRAIN_PATH, test_path=TEST_PATH, sample_sub_path=SAMPLE_SUB_PATH
):
    train = pd.read_csv(train_path, usecols=["eeg_id", "patient_id"] + TARGET_COLS)
    test = pd.read_csv(test_path, usecols=["eeg_id", "patient_id"])
    sample = pd.read_csv(sample_sub_path, usecols=["eeg_id"] + TARGET_COLS)

    K = len(TARGET_COLS)

    eeg_votes = (
        train.groupby(["eeg_id", "patient_id"], sort=False)[TARGET_COLS]
        .sum()
        .reset_index()
    )
    patient_votes = eeg_votes.groupby("patient_id", sort=False)[TARGET_COLS].sum()

    alpha = 1.0

    global_votes = patient_votes.sum(axis=0).to_numpy(dtype=np.float64)
    global_total = float(global_votes.sum())
    global_prob = (global_votes + alpha) / (global_total + alpha * K)

    mix = 0.01
    uniform = np.full(K, 1.0 / K, dtype=np.float64)
    global_prob = (1.0 - mix) * global_prob + mix * uniform
    global_prob = np.clip(global_prob, 1e-12, None)
    global_prob = global_prob / global_prob.sum()

    pv = patient_votes.to_numpy(dtype=np.float64)
    pv_total = pv.sum(axis=1, keepdims=True)
    pv = (pv + alpha) / (pv_total + alpha * K)
    pv = (1.0 - mix) * pv + mix * uniform
    pv = np.clip(pv, 1e-12, None)
    pv = pv / pv.sum(axis=1, keepdims=True)

    patient_prob = pd.DataFrame(pv, index=patient_votes.index, columns=TARGET_COLS)

    out = sample[["eeg_id"]].merge(test, on="eeg_id", how="left")
    out = out.sort_values("eeg_id").reset_index(drop=True)

    joined = out.merge(
        patient_prob.reset_index().rename(columns={"index": "patient_id"}),
        on="patient_id",
        how="left",
    )
    preds = joined[TARGET_COLS].to_numpy(dtype=np.float64)
    missing = np.isnan(preds).any(axis=1)
    if missing.any():
        preds[missing] = global_prob

    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out2 = out[["eeg_id"]].copy()
    out2[TARGET_COLS] = preds
    return out2


def make_prior_baseline_submission(
    train_path=TRAIN_PATH, test_path=TEST_PATH, sample_sub_path=SAMPLE_SUB_PATH
):
    train = pd.read_csv(train_path, usecols=["eeg_id"] + TARGET_COLS)
    test = pd.read_csv(test_path, usecols=["eeg_id"])
    sample = pd.read_csv(sample_sub_path, usecols=["eeg_id"] + TARGET_COLS)

    K = len(TARGET_COLS)

    eeg_votes = train.groupby("eeg_id", sort=False)[TARGET_COLS].sum()
    global_votes = eeg_votes.sum(axis=0).to_numpy(dtype=np.float64)

    alpha = 1.0
    global_total = float(global_votes.sum())
    global_prob = (global_votes + alpha) / (global_total + alpha * K)

    mix = 0.01
    uniform = np.full(K, 1.0 / K, dtype=np.float64)
    global_prob = (1.0 - mix) * global_prob + mix * uniform
    global_prob = np.clip(global_prob, 1e-12, None)
    global_prob = global_prob / global_prob.sum()

    out = sample[["eeg_id"]].merge(test, on="eeg_id", how="left")
    out = out.sort_values("eeg_id").reset_index(drop=True)

    preds = np.tile(global_prob, (len(out), 1))
    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out2 = out[["eeg_id"]].copy()
    out2[TARGET_COLS] = preds
    return out2




## === cell 6
try:
    sol = merge_preds(
        folds=(0, 1, 2, 3, 4), versions=("v0",), working_dir="/kaggle/working"
    )
    print("[info] Using ensembled fold predictions.")
except FileNotFoundError as e:
    print(f"[warn] {e}")
    try:
        sol = make_spectro_patient_global_blend_submission()
        print(
            "[info] Using in-notebook blended spectrogram+patient+global priors (Dirichlet posterior mean + mild uniform mixing)."
        )
    except Exception as e2:
        print(
            f"[warn] blended spectrogram/patient/global baseline failed ({type(e2).__name__}: {e2}); trying patient baseline."
        )
        try:
            sol = make_patient_prior_baseline_submission()
            print(
                "[info] Using in-notebook patient-prior baseline predictions (Dirichlet posterior mean + mild uniform mixing)."
            )
        except Exception as e3:
            print(
                f"[warn] patient-prior baseline failed ({type(e3).__name__}: {e3}); falling back to global prior."
            )
            sol = make_prior_baseline_submission()
            print(
                "[info] Using in-notebook global-prior baseline predictions (Dirichlet posterior mean + mild uniform mixing)."
            )



## === cell 7
import pandas as pd
import numpy as np

sample = (
    pd.read_csv(SAMPLE_SUB_PATH, usecols=["eeg_id"] + TARGET_COLS)
    .sort_values("eeg_id")
    .reset_index(drop=True)
)

sol = sol.copy()
sol = sol.drop_duplicates(subset=["eeg_id"], keep="first").set_index("eeg_id")

aligned = sol.reindex(sample["eeg_id"].to_numpy())
K = len(TARGET_COLS)
aligned[TARGET_COLS] = aligned[TARGET_COLS].fillna(1.0 / K)

preds = aligned[TARGET_COLS].to_numpy(dtype=np.float64)
preds = np.clip(preds, 1e-12, None)
preds = preds / preds.sum(axis=1, keepdims=True)

out = pd.DataFrame({"eeg_id": sample["eeg_id"].to_numpy()})
out[TARGET_COLS] = preds

out_fp = "/kaggle/working/submission.csv"
out.to_csv(out_fp, index=False)

chk = pd.read_csv(out_fp)
assert (
    chk.columns.tolist() == ["eeg_id"] + TARGET_COLS
), "Submission columns/order mismatch."

chk_sorted = chk.sort_values("eeg_id").reset_index(drop=True)
if not np.array_equal(sample["eeg_id"].to_numpy(), chk_sorted["eeg_id"].to_numpy()):
    raise ValueError("eeg_id mismatch vs sample_submission (set or order differs).")

row_sums = chk[TARGET_COLS].sum(axis=1).to_numpy()
if not np.allclose(row_sums, 1.0, atol=1e-6):
    raise ValueError(
        f"Probabilities do not sum to 1 (min={row_sums.min()}, max={row_sums.max()})"
    )

if chk[TARGET_COLS].min().min() <= 0.0:
    raise ValueError("Probabilities must be strictly positive to avoid KL issues.")

if len(chk) != len(sample):
    raise ValueError(
        f"Row count mismatch vs sample_submission: {len(chk)} vs {len(sample)}"
    )

print(f"[info] Wrote submission: {out_fp}  shape={chk.shape}")



## === cell 8
out.head()
