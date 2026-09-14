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

0.3070588873339696

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.39779) has done: 'I remove the hard dependency on missing external codebases (`/kaggle/input/hms-mk-codes*`) and failing `pip install` steps, since they prevent any submission from being produced. Instead, I generate a simple, valid baseline submission directly from `sample_submission.csv` by using the normalized class vote priors from `train.csv`, which is score-better than uniform and fully deterministic. I keep your existing ensembling/merge logic in place (but gated) so if those fold CSVs ever exist, they still be used unchanged. Finally, I ensure the written `/kaggle/working/submission.csv` has the exact required columns, sums to 1 per row, and is always produced.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd



## === cell 1
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
OUT_PATH2 = "/kaggle/working/v2"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
Path(OUT_PATH2).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None):
    """Run a shell command and raise if it fails (so we fail fast)."""
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)




## === cell 2
p = Path("/kaggle/input/hms-mk-codes/")
if p.exists():
    sys.path.append(str(p))
else:
    print("WARNING: /kaggle/input/hms-mk-codes/ not found; not adding to sys.path")



## === cell 3
req_dir = Path("/kaggle/input/requirements-mk")
if req_dir.exists():
    run_cmd(
        "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
    )
    run_cmd(
        "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
    )
    run_cmd(
        "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
    )
    run_cmd(
        "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
    )
else:
    print("WARNING: /kaggle/input/requirements-mk not found; skipping pip installs.")



## === cell 4
codes_v1 = Path("/kaggle/input/hms-mk-codes")
if codes_v1.exists():
    run_cmd(
        f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
        cwd=str(codes_v1),
    )
else:
    print(
        "WARNING: /kaggle/input/hms-mk-codes not found; skipping parquet->npy conversion (v1)."
    )



## === cell 5
mk_data_dir = Path("/kaggle/input/hms-mk-data")
print("Exists /kaggle/input/hms-mk-data:", mk_data_dir.exists())
if mk_data_dir.exists():
    print("Sample files:", sorted([p.name for p in mk_data_dir.glob("*")])[:10])



## === cell 6
codes_v1 = Path("/kaggle/input/hms-mk-codes")
if codes_v1.exists() and mk_data_dir.exists():
    for fold in [0, 1, 2, 3, 4]:
        try:
            run_cmd(
                "python -m test "
                f"paths.data_dir={DATA_PATH} "
                f"data.test_eegs_dir={OUT_PATH} "
                f"ckpt_path=/kaggle/input/hms-mk-data/fold{fold}_levit_pseudo.ckpt "
                "hydra=test "
                f"+model.test_output_dir={OUT_PATH} "
                "experiment=conv1d_tfm2d_pseudo "
                "+model.net.pretrained=False",
                cwd=str(codes_v1),
            )
            src = Path("/kaggle/working/submission.csv")
            dst = Path(f"/kaggle/working/submission_fold{fold}_v0.csv")
            if src.exists():
                src.replace(dst)
            else:
                print(
                    f"WARNING: Expected {src} to be created by inference; skipping fold {fold} v0."
                )
        except Exception as e:
            print(
                f"WARNING: v1 inference failed for fold {fold} v0; skipping. Reason: {repr(e)}"
            )

    for fold in [0, 1, 2, 3, 4]:
        try:
            run_cmd(
                "python -m test "
                f"paths.data_dir={DATA_PATH} "
                f"data.test_eegs_dir={OUT_PATH} "
                f"ckpt_path=/kaggle/input/hms-mk-data/fold{fold}_effb3_sim_pseudo.ckpt "
                "hydra=test "
                f"+model.test_output_dir={OUT_PATH} "
                "experiment=conv1d_effv2_pseudo "
                "+model.net.pretrained=False",
                cwd=str(codes_v1),
            )
            src = Path("/kaggle/working/submission.csv")
            dst = Path(f"/kaggle/working/submission_fold{fold}_v3.csv")
            if src.exists():
                src.replace(dst)
            else:
                print(
                    f"WARNING: Expected {src} to be created by inference; skipping fold {fold} v3."
                )
        except Exception as e:
            print(
                f"WARNING: v1 inference failed for fold {fold} v3; skipping. Reason: {repr(e)}"
            )
else:
    print(
        "WARNING: v1 inference skipped (missing /kaggle/input/hms-mk-codes and/or /kaggle/input/hms-mk-data)."
    )



## === cell 7
try:
    sys.path.remove("/kaggle/input/hms-mk-codes/")
except ValueError:
    pass

p2 = Path("/kaggle/input/hms-mk-codesv2")
if p2.exists():
    sys.path.append(str(p2))
else:
    print("WARNING: /kaggle/input/hms-mk-codesv2 not found; not adding to sys.path")



## === cell 8
codes_v2 = Path("/kaggle/input/hms-mk-codesv2")
if codes_v2.exists():
    run_cmd(
        f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH2}",
        cwd=str(codes_v2),
    )
else:
    print(
        "WARNING: /kaggle/input/hms-mk-codesv2 not found; skipping parquet->npy conversion (v2)."
    )



## === cell 9
codes_v2 = Path("/kaggle/input/hms-mk-codesv2")
if codes_v2.exists() and mk_data_dir.exists():
    for fold in [0, 1, 2, 3, 4]:
        try:
            run_cmd(
                "python -m test "
                f"paths.data_dir={DATA_PATH} "
                "data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG "
                f"data.test_dataset.eeg_dir={OUT_PATH2}/test_eegs "
                f"ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold{fold}.ckpt "
                "hydra=test "
                f"+model.test_output_dir={OUT_PATH} "
                "experiment=clean_tfm_corr_pseudo "
                "data.num_workers=2 "
                "+model.net.pretrained=False",
                cwd=str(codes_v2),
            )
            src = Path("/kaggle/working/submission.csv")
            dst = Path(f"/kaggle/working/submission_fold{fold}_v4.csv")
            if src.exists():
                src.replace(dst)
            else:
                print(
                    f"WARNING: Expected {src} to be created by inference; skipping fold {fold} v4."
                )
        except Exception as e:
            print(
                f"WARNING: v2 inference failed for fold {fold} v4; skipping. Reason: {repr(e)}"
            )

    for fold in [0, 1, 2, 3, 4]:
        try:
            run_cmd(
                "python -m test "
                f"paths.data_dir={DATA_PATH} "
                "data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG "
                f"data.test_dataset.eeg_dir={OUT_PATH2}/test_eegs "
                f"ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold{fold}.ckpt "
                "hydra=test "
                f"+model.test_output_dir={OUT_PATH} "
                "experiment=clean_effb1_corr_pseudo "
                "data.num_workers=2 "
                "+model.net.pretrained=False",
                cwd=str(codes_v2),
            )
            src = Path("/kaggle/working/submission.csv")
            dst = Path(f"/kaggle/working/submission_fold{fold}_v5.csv")
            if src.exists():
                src.replace(dst)
            else:
                print(
                    f"WARNING: Expected {src} to be created by inference; skipping fold {fold} v5."
                )
        except Exception as e:
            print(
                f"WARNING: v2 inference failed for fold {fold} v5; skipping. Reason: {repr(e)}"
            )
else:
    print(
        "WARNING: v2 inference skipped (missing /kaggle/input/hms-mk-codesv2 and/or /kaggle/input/hms-mk-data)."
    )



## === cell 10
p1 = Path("/kaggle/working/submission_fold1_v4.csv")
p2 = Path("/kaggle/working/submission_fold1_v5.csv")
if p1.exists():
    print("Head:", p1)
    print(pd.read_csv(p1, nrows=5).head())
if p2.exists():
    print("Head:", p2)
    print(pd.read_csv(p2, nrows=5).head())



## === cell 11
DEFAULT_TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _get_target_cols():
    ss_path = f"{DATA_PATH}/sample_submission.csv"
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path, nrows=1)
        cols = [c for c in ss.columns if c != "eeg_id"]
        if len(cols) == 6:
            return cols
    return DEFAULT_TARGET_COLS


TARGET_COLS = _get_target_cols()


def _get_unique_test_base():
    """
    test.csv is one row per eeg_id; build the base index from test.csv to guarantee
    correct row count and alignment.
    """
    test = pd.read_csv(f"{DATA_PATH}/test.csv", usecols=["eeg_id"])
    base = (
        test.drop_duplicates(subset=["eeg_id"])
        .sort_values("eeg_id")
        .reset_index(drop=True)
    )
    return base


def merge_preds(folds, versions, weights):
    """
    Keep your ensemble semantics, but ensure weights renormalize over files that exist
    and every file is aligned by eeg_id; then enforce valid probabilities for KL.
    """
    base = _get_unique_test_base()

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    mats = []
    ws = []

    for fold in folds:
        for w, version in zip(weights, versions):
            path = Path(f"/kaggle/working/submission_fold{fold}_{version}.csv")
            if not path.exists():
                print("WARNING: missing prediction file, skipping:", str(path))
                continue

            sol = pd.read_csv(path)
            need_cols = ["eeg_id"] + TARGET_COLS
            missing_cols = [c for c in need_cols if c not in sol.columns]
            if missing_cols:
                raise ValueError(f"{path} is missing columns: {missing_cols}")

            sol = sol[need_cols].copy()
            if sol["eeg_id"].duplicated().any():
                sol = sol.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

            aligned = base.merge(sol, on="eeg_id", how="left", validate="one_to_one")
            vals = aligned[TARGET_COLS].to_numpy(dtype=np.float64)
            if np.isnan(vals).any():
                vals = np.nan_to_num(vals, nan=1.0 / len(TARGET_COLS))

            vals = np.clip(vals, 1e-12, None)
            vals = vals / vals.sum(axis=1, keepdims=True)

            mats.append(vals)
            ws.append(float(w))

    if len(mats) == 0:
        raise RuntimeError("No prediction CSVs were found; cannot ensemble.")

    ws = np.asarray(ws, dtype=np.float64)
    ws = ws / ws.sum()

    pred = np.zeros((len(base), len(TARGET_COLS)), dtype=np.float64)
    for w, vals in zip(ws, mats):
        pred += w * vals

    pred = np.clip(pred, 1e-12, None)
    pred = pred / pred.sum(axis=1, keepdims=True)

    out = base.copy()
    out[TARGET_COLS] = pred
    return out


def make_prior_submission(
    alpha_global=0.15,
    alpha_patient=0.35,
    alpha_spect=0.60,
    alpha_pair=0.80,
    min_n_patient=40,
    min_n_spect=40,
    min_n_pair=15,
    shrink_patient=0.15,
    shrink_spect=0.15,
    shrink_pair=0.10,
):
    """
    Same prior-based fallback, but adjusted to better match KL evaluation:
      - CHANGE (score-relevant): convert each training row's vote counts to a probability
        distribution, then aggregate those probabilities with weights = total votes.
        This better approximates the "observed target" distribution than summing raw votes.
      - Keep deterministic backoff:
          (patient_id, spectrogram_id) -> spectrogram_id -> patient_id -> global
      - Keep smoothing + small shrinkage to global to avoid extreme/zero probs (KL blow-ups).
    """
    use_cols = ["eeg_id", "patient_id", "spectrogram_id"] + TARGET_COLS
    train = pd.read_csv(f"{DATA_PATH}/train.csv", usecols=use_cols)

    votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
    vote_sums = votes.sum(axis=1, keepdims=True)
    vote_sums = np.clip(vote_sums, 1.0, None)  # safety
    row_probs = votes / vote_sums
    train_probs = train[["eeg_id", "patient_id", "spectrogram_id"]].copy()
    train_probs[TARGET_COLS] = row_probs
    train_probs["n_votes"] = vote_sums.ravel()

    def _wavg(g: pd.DataFrame) -> pd.Series:
        w = g["n_votes"].to_numpy(dtype=np.float64)
        x = g[TARGET_COLS].to_numpy(dtype=np.float64)
        wsum = float(w.sum())
        if wsum <= 0:
            return pd.Series({c: 1.0 / len(TARGET_COLS) for c in TARGET_COLS})
        return pd.Series((x * w[:, None]).sum(axis=0) / wsum, index=TARGET_COLS)

    train_eeg = (
        train_probs.groupby(["eeg_id", "patient_id", "spectrogram_id"], as_index=False)
        .apply(_wavg, include_groups=False)
        .reset_index()
    )

    n_patient = train_eeg.groupby("patient_id").size()
    n_spect = train_eeg.groupby("spectrogram_id").size()
    n_pair = train_eeg.groupby(["patient_id", "spectrogram_id"]).size()

    global_probs = train_eeg[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)
    global_probs = np.clip(global_probs, 1e-12, None)
    global_probs = global_probs / global_probs.sum()
    global_counts = global_probs + float(alpha_global)
    global_prior = global_counts / global_counts.sum()

    def _smoothed_group_prior(group_cols, alpha, shrink):
        grp = train_eeg.groupby(group_cols)[TARGET_COLS].mean().astype(np.float64)
        grp = grp + float(alpha)
        grp = grp.div(grp.sum(axis=1), axis=0)
        grp = (1.0 - float(shrink)) * grp + float(shrink) * pd.Series(
            global_prior, index=TARGET_COLS
        )
        return grp

    patient_priors = _smoothed_group_prior("patient_id", alpha_patient, shrink_patient)
    spect_priors = _smoothed_group_prior("spectrogram_id", alpha_spect, shrink_spect)
    pair_priors = _smoothed_group_prior(
        ["patient_id", "spectrogram_id"], alpha_pair, shrink_pair
    )

    test = pd.read_csv(
        f"{DATA_PATH}/test.csv", usecols=["eeg_id", "patient_id", "spectrogram_id"]
    )
    test = (
        test.drop_duplicates(subset=["eeg_id"])
        .sort_values("eeg_id")
        .reset_index(drop=True)
    )

    mat = np.tile(global_prior, (len(test), 1)).astype(np.float64)

    pid = test["patient_id"].values
    sid = test["spectrogram_id"].values
    pair_idx = pd.MultiIndex.from_arrays([pid, sid])

    pair_support = n_pair.reindex(pair_idx).fillna(0).to_numpy(dtype=np.int64)
    spect_support = n_spect.reindex(sid).fillna(0).to_numpy(dtype=np.int64)
    patient_support = n_patient.reindex(pid).fillna(0).to_numpy(dtype=np.int64)

    use_pair = pair_support >= int(min_n_pair)
    use_spect = (~use_pair) & (spect_support >= int(min_n_spect))
    use_patient = (~use_pair) & (~use_spect) & (patient_support >= int(min_n_patient))

    if use_pair.any():
        pair_mapped = pair_priors.reindex(pair_idx[use_pair])
        pair_mat = pair_mapped.to_numpy(dtype=np.float64)
        ok = np.isfinite(pair_mat).all(axis=1)
        idx = np.flatnonzero(use_pair)
        mat[idx[ok]] = pair_mat[ok]

    if use_spect.any():
        spect_mapped = spect_priors.reindex(sid[use_spect])
        spect_mat = spect_mapped.to_numpy(dtype=np.float64)
        ok = np.isfinite(spect_mat).all(axis=1)
        idx = np.flatnonzero(use_spect)
        mat[idx[ok]] = spect_mat[ok]

    if use_patient.any():
        patient_mapped = patient_priors.reindex(pid[use_patient])
        patient_mat = patient_mapped.to_numpy(dtype=np.float64)
        ok = np.isfinite(patient_mat).all(axis=1)
        idx = np.flatnonzero(use_patient)
        mat[idx[ok]] = patient_mat[ok]

    mat = np.clip(mat, 1e-12, None)
    mat = mat / mat.sum(axis=1, keepdims=True)

    out = test[["eeg_id"]].copy()
    out[TARGET_COLS] = mat
    return out




## === cell 12
try:
    sol = merge_preds(
        folds=[0, 1, 2, 3, 4],
        versions=["v0", "v3", "v4", "v5"],
        weights=[0.25, 0.25, 0.25, 0.25],
    )
    print("Ensemble created from available fold prediction CSVs.")
except Exception as e:
    print(
        "WARNING: ensembling unavailable, using (adaptive pair/spectrogram/patient/global, shrunk & smoothed) prior baseline instead."
    )
    print("Reason:", repr(e))
    sol = make_prior_submission(
        alpha_global=0.15,
        alpha_patient=0.35,
        alpha_spect=0.60,
        alpha_pair=0.80,
        min_n_patient=40,
        min_n_spect=40,
        min_n_pair=15,
        shrink_patient=0.15,
        shrink_spect=0.15,
        shrink_pair=0.10,
    )



## === cell 13
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape=", sol.shape)

assert list(sol.columns) == ["eeg_id"] + TARGET_COLS, "Submission columns mismatch."
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))
assert np.all(np.isfinite(row_sums)), "Non-finite row sums."
assert np.max(np.abs(row_sums - 1.0)) < 1e-6, "Row probabilities do not sum to 1."
assert sol["eeg_id"].nunique() == len(sol), "Duplicate eeg_id in submission."
print(sol.head())
