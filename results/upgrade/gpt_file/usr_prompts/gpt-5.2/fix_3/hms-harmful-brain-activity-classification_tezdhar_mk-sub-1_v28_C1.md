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

0.3071621873722135

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.41937) has done: 'Your failures come from relying on external `/kaggle/input/hms-mk-codes` and `/kaggle/input/hms-mk-data` assets that are not present, so the fold prediction CSVs are never created and ensembling crashes. To make the notebook run end-to-end in this environment, I keep your ensemble/normalization logic but add a safe fallback that generates a valid baseline submission (class-prior probabilities from `train.csv`, strictly normalized) when the fold files are missing. This produces a correct `.csv` submission with the required columns and row-sum-to-1 constraint, and should yield a reasonable (though not SOTA) KL score rather than “Not yielded”. I also update the pip/convert/test cells to no-op cleanly when the external assets aren’t available, avoiding silent downstream missing-file errors.'

# 9. Code solution

## === cell 0
import os, sys, glob
import numpy as np
import pandas as pd

MK_CODES_PATH = "/kaggle/input/hms-mk-codes/"
if os.path.isdir(MK_CODES_PATH) and MK_CODES_PATH not in sys.path:
    sys.path.append(MK_CODES_PATH)




## === cell 1
def _maybe_system(cmd: str):
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is not None:
            ip.system(cmd)
    except Exception:
        pass


REQ_DIR = "/kaggle/input/requirements-mk"
if os.path.isdir(REQ_DIR):
    _maybe_system(
        f"pip install {REQ_DIR}/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
    )
    _maybe_system(
        f"pip install {REQ_DIR}/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
    )
    _maybe_system(
        f"pip install {REQ_DIR}/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
    )
    _maybe_system(
        f"pip install {REQ_DIR}/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
    )



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 3
if os.path.isdir(MK_CODES_PATH):
    _maybe_system(
        f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
    )



## === cell 4
if os.path.isdir("/kaggle/input/hms-mk-data"):
    _maybe_system("ls /kaggle/input/hms-mk-data")



## === cell 5
MK_DATA_PATH = "/kaggle/input/hms-mk-data"
if os.path.isdir(MK_CODES_PATH) and os.path.isdir(MK_DATA_PATH):
    cmds = [
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v0.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold1_levit_pseudo.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v0.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold2_levit_pseudo.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v0.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold3_levit_pseudo.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v0.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold4_levit_pseudo.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v0.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold0_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v2.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold1_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v2.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold2_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v2.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold3_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v2.csv",
        f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} ckpt_path=/kaggle/input/hms-mk-data/fold4_pseudo_resv2.ckpt hydra=test +model.test_output_dir={OUT_PATH} experiment=conv1d_resv2 +model.net.pretrained=False",
        "mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v2.csv",
    ]
    for c in cmds:
        _maybe_system(c)



## === cell 6
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def merge_preds(folds=(0, 1, 2), versions=("v0",), weights=(1.0,)):
    """
    Minimal robust ensembling:
    - reads each fold/version csv created above
    - aligns by eeg_id (prevents mis-ordering bugs)
    - weighted average then strict renormalization so each row sums to 1
    """
    weights = np.asarray(list(weights), dtype=np.float64)
    if len(weights) != len(versions):
        raise ValueError(
            f"weights (len={len(weights)}) must match versions (len={len(versions)})"
        )

    all_pred = None
    base_eeg = None

    for fold in folds:
        for w, ver in zip(weights, versions):
            path = f"/kaggle/working/submission_fold{fold}_{ver}.csv"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing prediction file: {path}")

            df = pd.read_csv(path)
            if "eeg_id" not in df.columns:
                raise ValueError(f"{path} missing eeg_id column")
            missing = [c for c in TARGET_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{path} missing target columns: {missing}")

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = df.sort_values("eeg_id").reset_index(drop=True)

            pred = df[TARGET_COLS].to_numpy(dtype=np.float64) * float(w)

            if all_pred is None:
                all_pred = pred
                base_eeg = df["eeg_id"].to_numpy()
            else:
                if not np.array_equal(base_eeg, df["eeg_id"].to_numpy()):
                    raise ValueError(
                        f"eeg_id order mismatch in {path}; cannot ensemble safely."
                    )
                all_pred += pred

    all_pred = np.clip(all_pred, 0.0, np.inf)
    row_sums = all_pred.sum(axis=1, keepdims=True)
    zero_mask = row_sums.squeeze() == 0
    if np.any(zero_mask):
        all_pred[zero_mask] = 1.0
        row_sums = all_pred.sum(axis=1, keepdims=True)

    all_pred = all_pred / row_sums

    sol = pd.DataFrame({"eeg_id": base_eeg})
    sol[TARGET_COLS] = all_pred
    return sol


def make_class_prior_submission(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Fallback to ensure a valid submission is always produced:
    - Uses global class priors from train vote counts (converted to probabilities).
    - Applies the same distribution to every test eeg_id.
    This is score-reasonable for KL and far better than failing to submit.
    """
    train_csv = os.path.join(data_path, "train.csv")
    test_csv = os.path.join(data_path, "test.csv")

    train = pd.read_csv(train_csv, usecols=TARGET_COLS)
    sums = train[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    sums = np.clip(sums, 0.0, np.inf)
    if sums.sum() == 0:
        prior = np.ones(len(TARGET_COLS), dtype=np.float64) / len(TARGET_COLS)
    else:
        prior = sums / sums.sum()

    test = pd.read_csv(test_csv, usecols=["eeg_id"])
    sol = pd.DataFrame({"eeg_id": test["eeg_id"].to_numpy()})
    for i, c in enumerate(TARGET_COLS):
        sol[c] = prior[i]

    arr = sol[TARGET_COLS].to_numpy(dtype=np.float64)
    arr = np.clip(arr, 0.0, np.inf)
    arr = arr / arr.sum(axis=1, keepdims=True)
    sol[TARGET_COLS] = arr
    return sol




## === cell 7
try:
    sol = merge_preds(folds=[0, 1, 2, 3, 4], versions=["v0", "v2"], weights=[0.5, 0.5])
except FileNotFoundError:
    sol = make_class_prior_submission(DATA_PATH)

sol = sol[["eeg_id"] + TARGET_COLS].copy()



## === cell 8
pred = sol[TARGET_COLS].to_numpy(dtype=np.float64)
pred = np.clip(pred, 0.0, np.inf)
pred_sum = pred.sum(axis=1, keepdims=True)
pred_sum[pred_sum == 0] = 1.0
pred = pred / pred_sum
sol[TARGET_COLS] = pred

out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)

print("Wrote:", out_file)
print("Shape:", sol.shape)
print(
    "Row-sum min/max:",
    sol[TARGET_COLS].sum(axis=1).min(),
    sol[TARGET_COLS].sum(axis=1).max(),
)
print(sol.head())
