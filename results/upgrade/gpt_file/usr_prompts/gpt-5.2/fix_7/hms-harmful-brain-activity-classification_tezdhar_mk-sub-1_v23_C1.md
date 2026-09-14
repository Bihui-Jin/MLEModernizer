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

0.3265433581152409

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'The main failure is that the fold prediction CSVs were never created, but the notebook continues without checking for errors; this is because `CODE_PATH` and the extra wheel paths likely don’t exist in your environment, so the `test` commands never run successfully. I (1) make the script robust to missing external code/ckpts by conditionally running those steps only if the paths exist, and (2) add a guaranteed fallback that always writes a valid `submission.csv` by using the competition’s `sample_submission.csv` and filling with a safe prior (uniform probabilities) when model outputs are unavailable. I also make `merge_preds` resilient by skipping missing files (instead of raising) and verifying row alignment by `eeg_id` before averaging, ensuring correct submission format and probability normalization. This run end-to-end and produce a valid `.csv`; if the external models are present, it ensemble them as originally intended.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

CODE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
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
ckpt_root = "/kaggle/input/hms-harmful-brain-activity-classification"
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

res = _run(
    [
        "bash",
        "-lc",
        "ls -la /kaggle/working/submission_fold*_v*.csv 2>/dev/null || true",
    ],
    check=False,
)
print("fold submission list returncode:", res.returncode)



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


def _load_test_ids_full():
    p = os.path.join(DATA_PATH, "test.csv")
    if not os.path.exists(p):
        p2 = "/kaggle/input/test.csv"
        if os.path.exists(p2):
            p = p2
        else:
            raise FileNotFoundError("Could not find test.csv in expected locations.")
    test = pd.read_csv(p, usecols=["eeg_id", "patient_id", "spectrogram_id"])
    test = (
        test.drop_duplicates("eeg_id", keep="first")
        .sort_values("eeg_id")
        .reset_index(drop=True)
    )
    return test


def _load_test_ids():
    return _load_test_ids_full()[["eeg_id"]].copy()


def _load_train_prior_probs() -> np.ndarray:
    """
    Score-improvement fallback:
    Empirical class distribution from train vote fractions is a much better KL prior than uniform.
    """
    p = os.path.join(DATA_PATH, "train.csv")
    if not os.path.exists(p):
        p2 = "/kaggle/input/train.csv"
        if os.path.exists(p2):
            p = p2
        else:
            return np.full(
                (len(TARGET_COLS),), 1.0 / len(TARGET_COLS), dtype=np.float64
            )

    tr = pd.read_csv(p, usecols=TARGET_COLS)
    votes = tr[TARGET_COLS].to_numpy(dtype=np.float64)
    row_sum = votes.sum(axis=1, keepdims=True)
    row_sum = np.clip(row_sum, 1e-12, None)
    probs = votes / row_sum
    prior = probs.mean(axis=0)
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()
    return prior.astype(np.float64)


def _compute_meta_conditional_priors():
    """
    Score-improvement (still a fallback, no model changes):
    Build smoothed conditional priors P(class | patient_id) and P(class | spectrogram_id)
    from train vote fractions. This typically improves KL vs global prior when external
    inference outputs are missing.
    """
    p = os.path.join(DATA_PATH, "train.csv")
    if not os.path.exists(p):
        p2 = "/kaggle/input/train.csv"
        if os.path.exists(p2):
            p = p2
        else:
            return None

    usecols = ["patient_id", "spectrogram_id"] + TARGET_COLS
    tr = pd.read_csv(p, usecols=usecols)

    votes = tr[TARGET_COLS].to_numpy(dtype=np.float64)
    row_sum = votes.sum(axis=1, keepdims=True)
    row_sum = np.clip(row_sum, 1e-12, None)
    probs = votes / row_sum
    probs_df = pd.DataFrame(probs, columns=TARGET_COLS)
    probs_df["patient_id"] = tr["patient_id"].values
    probs_df["spectrogram_id"] = tr["spectrogram_id"].values

    patient_mean = probs_df.groupby("patient_id", as_index=True)[TARGET_COLS].mean()
    spect_mean = probs_df.groupby("spectrogram_id", as_index=True)[TARGET_COLS].mean()
    return patient_mean, spect_mean


_META_PRIORS = _compute_meta_conditional_priors()


def _meta_baseline_probs(test_full: pd.DataFrame) -> np.ndarray:
    """
    Score-improvement fallback probabilities for each test eeg_id:
    Prefer patient prior; blend with spectrogram prior; back off to global prior.
    """
    global_prior = _load_train_prior_probs()
    n = len(test_full)
    out = np.tile(global_prior.reshape(1, -1), (n, 1)).astype(np.float64)

    if _META_PRIORS is None:
        return out

    patient_mean, spect_mean = _META_PRIORS

    smooth = 0.15

    pids = test_full["patient_id"].values
    has_patient = np.isin(pids, patient_mean.index.values)
    if has_patient.any():
        pm = patient_mean.reindex(pids[has_patient]).to_numpy(dtype=np.float64)
        pm = (1.0 - smooth) * pm + smooth * global_prior.reshape(1, -1)
        out[has_patient] = pm

    sids = test_full["spectrogram_id"].values
    has_spec = np.isin(sids, spect_mean.index.values)
    if has_spec.any():
        smv = spect_mean.reindex(sids[has_spec]).to_numpy(dtype=np.float64)
        smv = (1.0 - smooth) * smv + smooth * global_prior.reshape(1, -1)
        out[has_spec] = 0.7 * out[has_spec] + 0.3 * smv

    out = np.clip(out, 1e-12, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _safe_groupby_mean_preds(df: pd.DataFrame) -> pd.DataFrame:
    if not df["eeg_id"].is_unique:
        df = df.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()
    return df


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2"),
    weights=(0.2, 0.8),
    baseline_weight=0.10,
):
    """
    Weighted average across multiple fold submissions and model versions.
    Core logic preserved: sum weighted probabilities then row-normalize.

    Score-relevant robustness:
    - Fill missing eeg_id rows inside any prediction file with a stronger meta-conditional prior.
    - Always blend in a small amount of the same baseline to reduce KL blowups on bad rows
      (keeps semantics: still probabilities, just more calibrated ensemble).
    """
    if len(weights) != len(versions):
        raise ValueError(
            f"weights and versions must have same length. Got {len(weights)} vs {len(versions)}"
        )

    test_full = _load_test_ids_full()
    base = test_full[["eeg_id"]].copy()

    baseline = _meta_baseline_probs(test_full)

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
            df = _safe_groupby_mean_preds(df)

            aligned = base.merge(df, on="eeg_id", how="left")

            if aligned[TARGET_COLS].isna().any().any():
                na_mask = aligned[TARGET_COLS].isna().any(axis=1).to_numpy()
                na_rows = int(na_mask.sum())
                print(
                    f"Prediction file missing {na_rows} / {len(aligned)} eeg_id rows; filling with meta baseline:",
                    path,
                )
                fill_vals = baseline[na_mask]
                for j, c in enumerate(TARGET_COLS):
                    aligned.loc[na_mask, c] = fill_vals[:, j]

            arr = aligned[TARGET_COLS].to_numpy(dtype=np.float64) * float(weight)
            preds_sum += arr
            used += 1

    if used == 0:
        print("No prediction files were usable; using meta baseline probabilities.")
        preds = baseline
    else:
        preds = preds_sum
        preds = np.clip(preds, 1e-12, None)
        preds = preds / preds.sum(axis=1, keepdims=True)
        bw = float(baseline_weight)
        if bw > 0:
            preds = (1.0 - bw) * preds + bw * baseline
            preds = np.clip(preds, 1e-12, None)
            preds = preds / preds.sum(axis=1, keepdims=True)
        print(f"Ensembled from {used} prediction files; baseline_weight={bw:g}.")

    sol = base.copy()
    sol.loc[:, TARGET_COLS] = preds
    return sol




## === cell 6
sol = merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2"),
    weights=(0.2, 0.8),
    baseline_weight=0.10,
)
print(sol.head())
print("n_rows:", len(sol), "unique eeg_id:", sol["eeg_id"].nunique())
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
