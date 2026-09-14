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

0.3045701602754038

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I make the pipeline robust to the missing external `/kaggle/input/hms-mk-codes` repo by conditionally skipping the parquet→npy conversion and model inference steps when those assets aren’t present, instead of crashing. To ensure you still get a valid `.csv` submission end-to-end, I add a safe fallback that generates calibrated probabilities from the train label distribution (a legitimate prior) aligned to `sample_submission.csv`. I also keep the existing ensembling/merge logic intact when fold prediction CSVs are actually available, including strict row alignment and row-wise probability normalization (required by the competition). This produce `/kaggle/working/submission.csv` every time and avoids invalid rows that don’t sum to 1.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far worse than the target (0.30457), and this is because the notebook is almost certainly falling back to a constant “global prior” prediction when the external MK repo/checkpoints aren’t available. To move the score materially toward the target with minimal changes and without changing any modeling/training logic, I keep your inference/merge logic intact but replace the fallback with a stronger, still-legitimate baseline: per-patient class priors computed from `train.csv` (using only `patient_id` and vote columns), with a safe global-prior fallback for unseen patients. I also add light Dirichlet-style smoothing and keep the strict row-wise normalization required by the metric to avoid submission failure. This should improve calibration versus a single constant distribution and generally reduces KL on this competition.'
- What this solution (achieved 1.39779) has done: 'Your score is far above (worse than) the target, and the main reason is that your pipeline is almost certainly skipping MK inference and then using a weak fallback that doesn’t use the most predictive metadata available. I keep your existing inference/merge path unchanged, but strengthen only the fallback by conditioning on both `patient_id` and `spectrogram_id` (test-time metadata) using priors learned from `train.csv`, with a safe backoff chain (patient+spectrogram → patient → spectrogram → global) and the same row-wise probability normalization required by the metric. This is a minimal, legitimate improvement that typically reduces KL substantially compared with a single global prior or patient-only prior, while still producing a valid `submission.csv` end-to-end. I also make the patient-prior mapping deterministic and faster by aligning columns explicitly rather than relying on dict value order.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far worse than the target (0.30457) because the pipeline is almost certainly using the metadata-prior fallback rather than real model predictions, and the fallback is still too weak. To move the score toward the target with minimal change, I keep your inference/merge path untouched and only strengthen the fallback by learning priors at the more-specific `patient_id + spectrogram_id` level using *sum of votes* (not mean of per-row-normalized probabilities), then backing off to patient-only → spectrogram-only → global, which is better calibrated for the KL metric. I also compute a Dirichlet prior from the global distribution and apply it as pseudo-counts in every group (still legitimate and deterministic), and keep strict row-wise normalization to avoid submission failures. This should materially improve the fallback submission without changing your core logic or requiring the external MK repo/assets.'

# 9. Code solution

## === cell 0
import os
import sys

MK_CODES_PATH = "/kaggle/input/hms-mk-codes"
if os.path.isdir(MK_CODES_PATH) and MK_CODES_PATH not in sys.path:
    sys.path.append(MK_CODES_PATH)



## === cell 1
import subprocess


def _pip_install_if_exists(whl_path, extra_args=None):
    if extra_args is None:
        extra_args = []
    if os.path.exists(whl_path):
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            whl_path,
            "--no-index",
            "--no-deps",
        ] + extra_args
        subprocess.check_call(cmd)
    else:
        print(f"[WARN] Wheel not found, skipping install: {whl_path}")


_pip_install_if_exists(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    extra_args=["--force-reinstall"],
)
_pip_install_if_exists("/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl")
_pip_install_if_exists(
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl"
)
_pip_install_if_exists(
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
    extra_args=["--no-deps"],
)



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)



## === cell 3
if os.path.isdir(MK_CODES_PATH):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "src.convert_parquet_to_npy",
            "--data_dir",
            DATA_PATH,
            "--out_dir",
            OUT_PATH,
        ],
        cwd=MK_CODES_PATH,
    )
else:
    print(
        f"[WARN] External repo not found at {MK_CODES_PATH}; skipping parquet->npy conversion."
    )



## === cell 4
import glob

print("Listing /kaggle/input (first 50):")
for p in sorted(glob.glob("/kaggle/input/*"))[:50]:
    print(" ", p)




## === cell 5
def run_test_and_move(ckpt_path, experiment, fold, version):
    if not os.path.isdir(MK_CODES_PATH):
        raise FileNotFoundError(f"Missing external repo dir: {MK_CODES_PATH}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "test",
            f"paths.data_dir={DATA_PATH}",
            f"data.test_eegs_dir={OUT_PATH}",
            f"ckpt_path={ckpt_path}",
            "hydra=test",
            f"+model.test_output_dir={OUT_PATH}",
            f"experiment={experiment}",
            "+model.net.pretrained=False",
        ],
        cwd=MK_CODES_PATH,
    )
    src_csv = os.path.join(OUT_PATH, "submission.csv")
    dst_csv = os.path.join(OUT_PATH, f"submission_fold{fold}_{version}.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Expected submission not found at: {src_csv}")
    os.replace(src_csv, dst_csv)
    return dst_csv


if os.path.isdir(MK_CODES_PATH):
    for fold in range(5):
        run_test_and_move(
            ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_levit_pseudo.ckpt",
            experiment="conv1d_tfm2d_pseudo",
            fold=fold,
            version="v0",
        )

    for fold in range(5):
        run_test_and_move(
            ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_resv2.ckpt",
            experiment="conv1d_resv2",
            fold=fold,
            version="v2",
        )

    for fold in range(5):
        run_test_and_move(
            ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_effb3_sim_pseudo.ckpt",
            experiment="conv1d_effv2_pseudo",
            fold=fold,
            version="v3",
        )
else:
    print(f"[WARN] Skipping model inference because {MK_CODES_PATH} is unavailable.")



## === cell 6
import pandas as pd
import numpy as np

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print(
        f"[WARN] Could not import TARGET_COLS from src.settings ({e}). Falling back to default target cols."
    )
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.3, 0.3, 0.4),
    base_submission_path=f"{DATA_PATH}/sample_submission.csv",
    work_dir=OUT_PATH,
    eps=1e-12,
):
    sol = pd.read_csv(base_submission_path)
    if "eeg_id" not in sol.columns:
        raise ValueError("sample_submission.csv missing 'eeg_id'")

    eeg_ids = sol["eeg_id"].values
    pred_sum = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)

    for fold in folds:
        for w, ver in zip(weights, versions):
            fn = os.path.join(work_dir, f"submission_fold{fold}_{ver}.csv")
            if not os.path.exists(fn):
                raise FileNotFoundError(f"Missing prediction file: {fn}")

            df = pd.read_csv(fn)

            if "eeg_id" not in df.columns:
                raise ValueError(f"{fn} missing 'eeg_id'")
            missing_cols = [c for c in TARGET_COLS if c not in df.columns]
            if missing_cols:
                raise ValueError(f"{fn} missing target cols: {missing_cols}")

            df = df.set_index("eeg_id").reindex(eeg_ids)
            if df.isna().any().any():
                nan_rows = int(df.isna().any(axis=1).sum())
                raise ValueError(
                    f"{fn} has missing predictions after align: {nan_rows} rows"
                )

            pred_sum += df[TARGET_COLS].to_numpy(dtype=np.float64) * float(w)

    pred_sum = np.clip(pred_sum, eps, None)
    pred_sum = pred_sum / pred_sum.sum(axis=1, keepdims=True)

    sol[TARGET_COLS] = pred_sum
    return sol


def fallback_prior_submission(
    train_csv_path=f"{DATA_PATH}/train.csv",
    sample_submission_path=f"{DATA_PATH}/sample_submission.csv",
    eps=1e-12,
):
    train = pd.read_csv(train_csv_path, usecols=TARGET_COLS)
    row_sum = train[TARGET_COLS].sum(axis=1).replace(0, np.nan)
    probs = train[TARGET_COLS].div(row_sum, axis=0)
    prior = probs.mean(axis=0).to_numpy(dtype=np.float64)
    prior = np.clip(prior, eps, None)
    prior = prior / prior.sum()

    sol = pd.read_csv(sample_submission_path)
    for i, c in enumerate(TARGET_COLS):
        sol[c] = prior[i]
    arr = sol[TARGET_COLS].to_numpy(dtype=np.float64)
    arr = np.clip(arr, eps, None)
    arr = arr / arr.sum(axis=1, keepdims=True)
    sol[TARGET_COLS] = arr
    return sol


def fallback_patient_prior_submission(
    train_csv_path=f"{DATA_PATH}/train.csv",
    test_csv_path=f"{DATA_PATH}/test.csv",
    sample_submission_path=f"{DATA_PATH}/sample_submission.csv",
    alpha=1.0,
    eps=1e-12,
):
    usecols = ["patient_id"] + TARGET_COLS
    train = pd.read_csv(train_csv_path, usecols=usecols)

    row_sum = train[TARGET_COLS].sum(axis=1).replace(0, np.nan)
    probs = train[TARGET_COLS].div(row_sum, axis=0)

    K = len(TARGET_COLS)
    smooth = alpha / K
    probs = (probs + smooth).div((1.0 + alpha), axis=0)

    probs["patient_id"] = train["patient_id"].values
    patient_prior = probs.groupby("patient_id")[TARGET_COLS].mean()

    global_prior = probs[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)
    global_prior = np.clip(global_prior, eps, None)
    global_prior = global_prior / global_prior.sum()

    sol = pd.read_csv(sample_submission_path)
    test = pd.read_csv(test_csv_path, usecols=["eeg_id", "patient_id"])
    sol = sol.merge(test, on="eeg_id", how="left", validate="one_to_one")

    patient_arr = patient_prior.reindex(sol["patient_id"].values).to_numpy(
        dtype=np.float64
    )
    mask = np.isnan(patient_arr).any(axis=1)
    patient_arr[mask] = global_prior

    patient_arr = np.clip(patient_arr, eps, None)
    patient_arr = patient_arr / patient_arr.sum(axis=1, keepdims=True)

    sol = sol.drop(columns=["patient_id"])
    sol[TARGET_COLS] = patient_arr
    return sol


def fallback_patient_spectrogram_prior_submission(
    train_csv_path=f"{DATA_PATH}/train.csv",
    test_csv_path=f"{DATA_PATH}/test.csv",
    sample_submission_path=f"{DATA_PATH}/sample_submission.csv",
    alpha=1.0,
    eps=1e-12,
):
    usecols = ["patient_id", "spectrogram_id"] + TARGET_COLS
    train = pd.read_csv(train_csv_path, usecols=usecols)
    test = pd.read_csv(
        test_csv_path, usecols=["eeg_id", "patient_id", "spectrogram_id"]
    )
    sol = pd.read_csv(sample_submission_path).merge(
        test, on="eeg_id", how="left", validate="one_to_one"
    )

    K = len(TARGET_COLS)

    global_counts = train[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    global_counts = np.clip(global_counts, eps, None)
    global_prior = global_counts / global_counts.sum()

    prior_vec = global_prior * float(alpha)

    def _group_prob_sum(df, keys):
        gsum = df.groupby(keys, sort=False)[TARGET_COLS].sum()
        arr = gsum.to_numpy(dtype=np.float64) + prior_vec[None, :]
        arr = np.clip(arr, eps, None)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return gsum.index, arr

    idx_ps, arr_ps = _group_prob_sum(train, ["patient_id", "spectrogram_id"])
    ps_prior = pd.DataFrame(arr_ps, index=idx_ps, columns=TARGET_COLS)

    idx_p, arr_p = _group_prob_sum(train, ["patient_id"])
    patient_prior = pd.DataFrame(arr_p, index=idx_p, columns=TARGET_COLS)

    idx_s, arr_s = _group_prob_sum(train, ["spectrogram_id"])
    spect_prior = pd.DataFrame(arr_s, index=idx_s, columns=TARGET_COLS)

    idx_ps_test = pd.MultiIndex.from_arrays(
        [sol["patient_id"].values, sol["spectrogram_id"].values],
        names=["patient_id", "spectrogram_id"],
    )
    out = ps_prior.reindex(idx_ps_test).to_numpy(dtype=np.float64)

    miss = np.isnan(out).any(axis=1)
    if miss.any():
        out[miss] = patient_prior.reindex(sol.loc[miss, "patient_id"].values).to_numpy(
            dtype=np.float64
        )
    miss = np.isnan(out).any(axis=1)
    if miss.any():
        out[miss] = spect_prior.reindex(
            sol.loc[miss, "spectrogram_id"].values
        ).to_numpy(dtype=np.float64)
    miss = np.isnan(out).any(axis=1)
    if miss.any():
        out[miss] = global_prior

    out = np.clip(out, eps, None)
    out = out / out.sum(axis=1, keepdims=True)

    sol = sol.drop(columns=["patient_id", "spectrogram_id"])
    sol[TARGET_COLS] = out
    return sol




## === cell 7
try:
    sol = merge_preds(
        folds=(0, 1, 2, 3, 4),
        versions=("v0", "v2", "v3"),
        weights=(0.3, 0.3, 0.4),
    )
    print("[INFO] Successfully merged fold prediction CSVs.")
except Exception as e:
    print(
        f"[WARN] merge_preds failed ({type(e).__name__}: {e}). Using patient+spectrogram conditional prior submission."
    )
    try:
        sol = fallback_patient_spectrogram_prior_submission(alpha=6.0)
        print(
            "[INFO] Successfully built patient+spectrogram conditional prior submission."
        )
    except Exception as e2:
        print(
            f"[WARN] patient+spectrogram prior fallback failed ({type(e2).__name__}: {e2}). Trying patient-only prior."
        )
        try:
            sol = fallback_patient_prior_submission(alpha=1.0)
            print("[INFO] Successfully built patient-conditional prior submission.")
        except Exception as e3:
            print(
                f"[WARN] patient prior fallback failed ({type(e3).__name__}: {e3}). Using global prior submission."
            )
            sol = fallback_prior_submission()



## === cell 8
out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)
print("Wrote:", out_file)
print("Shape:", sol.shape)
print(
    "Row sums (min/mean/max):",
    float(sol[TARGET_COLS].sum(axis=1).min()),
    float(sol[TARGET_COLS].sum(axis=1).mean()),
    float(sol[TARGET_COLS].sum(axis=1).max()),
)



## === cell 9
sol.head()
