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

0.3256166983473387

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import glob
import pandas as pd
import numpy as np

MK_CODES_DIR = "/kaggle/input/hms-mk-codes"
if MK_CODES_DIR not in sys.path:
    sys.path.append(MK_CODES_DIR)

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

FALLBACK_TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
try:
    from src.settings import TARGET_COLS as _TARGET_COLS  # type: ignore

    TARGET_COLS = list(_TARGET_COLS)
except Exception:
    TARGET_COLS = FALLBACK_TARGET_COLS

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
TEST_CSV_PATH = f"{DATA_PATH}/test.csv"

print("TARGET_COLS =", TARGET_COLS)
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))



## === cell 1
expected_any = glob.glob(os.path.join(OUT_PATH, "*.npy"))
if len(expected_any) == 0:
    cmd = f"cd {MK_CODES_DIR} && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
    print("Running:", cmd)
    os.system(cmd)
else:
    print(f"Found {len(expected_any)} .npy files in {OUT_PATH}, skipping conversion.")



## === cell 2
ckpts = [
    "/kaggle/input/hms-mk-data/fold0_pseudo_resv2.ckpt",
    "/kaggle/input/hms-mk-data/fold1_pseudo_resv2.ckpt",
    "/kaggle/input/hms-mk-data/fold2_pseudo_resv2.ckpt",
    "/kaggle/input/hms-mk-data/fold3_pseudo_resv2.ckpt",
    "/kaggle/input/hms-mk-data/fold4_pseudo_resv2.ckpt",
]

for fold, ckpt in enumerate(ckpts):
    out_csv = f"/kaggle/working/submission_fold{fold}_v2.csv"
    if os.path.exists(out_csv):
        print(f"Exists, skipping fold {fold}:", out_csv)
        continue

    cmd = (
        f"cd {MK_CODES_DIR} && python -m test "
        f"paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path={ckpt} hydra=test +model.test_output_dir={OUT_PATH} "
        f"experiment=conv1d_resv2 +model.net.pretrained=False"
    )
    print("Running:", cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(
            f"Inference command failed for fold={fold} with exit code {ret}"
        )

    src_csv = "/kaggle/working/submission.csv"
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Expected {src_csv} was not created for fold={fold}")
    os.replace(src_csv, out_csv)
    print("Wrote:", out_csv)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1092079147.py in <cell line: 0>()
     23     ret = os.system(cmd)
     24     if ret != 0:
---> 25         raise RuntimeError(
     26             f"Inference command failed for fold={fold} with exit code {ret}"
     27         )

RuntimeError: Inference command failed for fold=0 with exit code 512

## === cell 3
def merge_preds(folds=(0, 1, 2, 3, 4), version="v2"):
    base = pd.read_csv(SAMPLE_SUB_PATH)
    if "eeg_id" not in base.columns:
        raise ValueError("sample_submission.csv must contain eeg_id")

    base_ids = base["eeg_id"].values
    all_fold_preds = []

    for fold in folds:
        path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fold submission: {path}")

        df = pd.read_csv(path)
        missing = [c for c in (["eeg_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df = df.set_index("eeg_id").loc[base_ids, TARGET_COLS].astype("float64")
        all_fold_preds.append(df.values)

    preds = np.mean(np.stack(all_fold_preds, axis=0), axis=0)

    row_sums = preds.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    preds = preds / row_sums

    preds = np.clip(preds, 1e-12, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = base.copy()
    out[TARGET_COLS] = preds
    return out


sol = merge_preds(folds=(0, 1, 2, 3, 4), version="v2")
sol.head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2004309073.py in <cell line: 0>()
     38 
     39 
---> 40 sol = merge_preds(folds=(0, 1, 2, 3, 4), version="v2")
     41 sol.head()
     42 

/tmp/ipykernel_11/2004309073.py in merge_preds(folds, version)
     11         path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
     12         if not os.path.exists(path):
---> 13             raise FileNotFoundError(f"Missing fold submission: {path}")
     14 
     15         df = pd.read_csv(path)

FileNotFoundError: Missing fold submission: /kaggle/working/submission_fold0_v2.csv

## === cell 4
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)

chk = pd.read_csv(sub_path)
assert list(chk.columns) == ["eeg_id"] + TARGET_COLS, "Submission columns mismatch"
row_sum = chk[TARGET_COLS].sum(axis=1).values
assert np.all(np.isfinite(row_sum)), "Non-finite probabilities"
assert np.max(np.abs(row_sum - 1.0)) < 1e-6, "Row probabilities do not sum to 1"
print("Saved valid submission:", sub_path, "shape=", chk.shape)
print(chk.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/399943202.py in <cell line: 0>()
      1 # Final submission
      2 sub_path = "/kaggle/working/submission.csv"
----> 3 sol.to_csv(sub_path, index=False)
      4 
      5 # Quick validation

NameError: name 'sol' is not defined
