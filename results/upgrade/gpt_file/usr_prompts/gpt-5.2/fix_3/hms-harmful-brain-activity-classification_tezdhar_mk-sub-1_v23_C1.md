# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3265433581152409

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'The main failure is that the fold prediction CSVs were never created, but the notebook continues without checking for errors; this is because `CODE_PATH` and the extra wheel paths likely don’t exist in your environment, so the `test` commands never run successfully. I (1) make the script robust to missing external code/ckpts by conditionally running those steps only if the paths exist, and (2) add a guaranteed fallback that always writes a valid `submission.csv` by using the competition’s `sample_submission.csv` and filling with a safe prior (uniform probabilities) when model outputs are unavailable. I also make `merge_preds` resilient by skipping missing files (instead of raising) and verifying row alignment by `eeg_id` before averaging, ensuring correct submission format and probability normalization. This run end-to-end and produce a valid `.csv`; if the external models are present, it ensemble them as originally intended.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

CODE_PATH = "/kaggle/input/hms-mk-codes"
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)


def _run(cmd, check=False):
    if isinstance(cmd, str):
        printable = cmd
    else:
        printable = " ".join(cmd)
    print("RUN:", printable)
    return subprocess.run(cmd, check=check)


if os.path.isdir(CODE_PATH) and CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)

print("Python:", sys.version)
print("CODE_PATH exists:", os.path.isdir(CODE_PATH))
print("DATA_PATH exists:", os.path.isdir(DATA_PATH))
print("OUT_PATH:", OUT_PATH)



## === cell 1
wheel_dir = "/kaggle/input/requirements-mk"
wheels = [
    "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    "omegaconf-2.3.0-py3-none-any.whl",
    "hydra_core-1.3.2-py3-none-any.whl",
    "lightning-2.2.1-py3-none-any.whl",
]

if os.path.isdir(wheel_dir):
    for w in wheels:
        p = os.path.join(wheel_dir, w)
        if os.path.exists(p):
            _run(
                ["pip", "install", p, "--no-index", "--no-deps", "--force-reinstall"],
                check=False,
            )
        else:
            print("Wheel missing (skipping):", p)
else:
    print("Wheel directory missing (skipping installs):", wheel_dir)



## === cell 2
if os.path.isdir(CODE_PATH):
    res = _run(
        [
            "bash",
            "-lc",
            f"cd {CODE_PATH} && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
        ],
        check=False,
    )
    print("convert_parquet_to_npy returncode:", res.returncode)
else:
    print("CODE_PATH not available; skipping parquet->npy conversion step.")



## === cell 3
_run(
    ["bash", "-lc", "ls -la /kaggle/input/hms-mk-data 2>/dev/null || true"], check=False
)



## === cell 4
ckpt_root = "/kaggle/input/hms-mk-data"
can_run_infer = os.path.isdir(CODE_PATH) and os.path.isdir(ckpt_root)

cmds = [
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold0_pseudo_log.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_pseudo +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v0.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold1_pseudo_log.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_pseudo +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v0.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold2_pseudo_log.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_pseudo +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v0.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold3_pseudo_log.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_pseudo +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v0.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold4_pseudo_log.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_pseudo +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v0.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold0_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v2.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold1_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v2.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold2_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v2.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold3_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v2.csv",
    f"cd {CODE_PATH} && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path={ckpt_root}/fold4_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v2.csv",
]

if can_run_infer:
    for c in cmds:
        res = _run(["bash", "-lc", c], check=False)
        print("infer returncode:", res.returncode)
else:
    print("Missing CODE_PATH or ckpt_root; skipping external inference commands.")
    print(
        "CODE_PATH ok:",
        os.path.isdir(CODE_PATH),
        "ckpt_root ok:",
        os.path.isdir(ckpt_root),
    )



## === cell 5
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


def _load_sample_submission():
    p1 = os.path.join(DATA_PATH, "sample_submission.csv")
    if os.path.exists(p1):
        return pd.read_csv(p1)
    p2 = "/kaggle/input/sample_submission.csv"
    if os.path.exists(p2):
        return pd.read_csv(p2)
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )


def merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v2"), weights=(0.2, 0.8)):
    """
    Weighted average across multiple fold submissions and model versions.
    Core logic preserved: sum weighted probabilities then row-normalize.
    Robustness fix: skip missing files and align by eeg_id to avoid silent mis-ordering.
    """
    if len(weights) != len(versions):
        raise ValueError(
            f"weights and versions must have same length. Got {len(weights)} vs {len(versions)}"
        )

    base = _load_sample_submission()
    base = base[["eeg_id"] + TARGET_COLS].copy()

    preds_sum = np.zeros((len(base), len(TARGET_COLS)), dtype=np.float64)
    used = 0

    for fold in folds:
        for weight, version in zip(weights, versions):
            path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
            if not os.path.exists(path):
                print("Missing prediction file (skipping):", path)
                continue

            df = pd.read_csv(path)
            missing = [c for c in (["eeg_id"] + TARGET_COLS) if c not in df.columns]
            if missing:
                print(
                    "Bad prediction file (missing columns; skipping):",
                    path,
                    "missing:",
                    missing,
                )
                continue

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = base[["eeg_id"]].merge(df, on="eeg_id", how="left")
            if df[TARGET_COLS].isna().any().any():
                print("Prediction file has missing eeg_id rows (skipping):", path)
                continue

            arr = df[TARGET_COLS].to_numpy(dtype=np.float64) * float(weight)
            preds_sum += arr
            used += 1

    if used == 0:
        print("No prediction files were usable; falling back to uniform probabilities.")
        preds = np.full(
            (len(base), len(TARGET_COLS)), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
    else:
        preds = preds_sum
        preds = np.clip(preds, 1e-12, None)
        preds = preds / preds.sum(axis=1, keepdims=True)
        print(f"Ensembled from {used} prediction files.")

    sol = base.copy()
    sol.loc[:, TARGET_COLS] = preds
    return sol




## === cell 6
sol = merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v2"), weights=(0.2, 0.8))
print(sol.head())
print(
    "Row sums (min/max):",
    sol[TARGET_COLS].sum(axis=1).min(),
    sol[TARGET_COLS].sum(axis=1).max(),
)



## === cell 7
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("Shape:", sol.shape)
print("Columns:", list(sol.columns))



## === cell 8
assert sol.columns.tolist() == ["eeg_id"] + TARGET_COLS
assert sol["eeg_id"].is_unique, "eeg_id must be unique in submission."
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
if not (np.all(np.isfinite(row_sums)) and np.allclose(row_sums, 1.0, atol=1e-6)):
    raise ValueError(
        f"Row sums are not ~1.0. min={row_sums.min()} max={row_sums.max()}"
    )
print("Submission looks valid.")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/4125777550.py in <cell line: 0>()
      1 # Final quick validation: check ordering/uniqueness and probability constraints.
      2 assert sol.columns.tolist() == ["eeg_id"] + TARGET_COLS
----> 3 assert sol["eeg_id"].is_unique, "eeg_id must be unique in submission."
      4 row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
      5 if not (np.all(np.isfinite(row_sums)) and np.allclose(row_sums, 1.0, atol=1e-6)):

AssertionError: eeg_id must be unique in submission.
