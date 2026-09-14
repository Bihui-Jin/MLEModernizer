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

0.3070588873339696

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
You’re failing at the ensembling step because you removed the codebase from `sys.path`, so `from src.settings import TARGET_COLS` can’t resolve and the pipeline never writes a final `submission.csv`. I’ll make the ensemble independent of that import by defining `TARGET_COLS` directly from the competition’s required column names (or from `sample_submission.csv` as a fallback), and I’ll also make `merge_preds()` robust to ordering mismatches by merging on `eeg_id` and re-normalizing probabilities. Finally, I’ll ensure the script always produces `/kaggle/working/submission.csv` with correct columns, correct row count, and row-wise sums of 1.0.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_11/4162154251.py", line 1
    You’re failing at the ensembling step because you removed the codebase from `sys.path`, so `from src.settings import TARGET_COLS` can’t resolve and the pipeline never writes a final `submission.csv`. I’ll make the ensemble independent of that import by defining `TARGET_COLS` directly from the competition’s required column names (or from `sample_submission.csv` as a fallback), and I’ll also make `merge_preds()` robust to ordering mismatches by merging on `eeg_id` and re-normalizing probabilities. Finally, I’ll ensure the script always produces `/kaggle/working/submission.csv` with correct columns, correct row count, and row-wise sums of 1.0.
       ^
SyntaxError: invalid character '’' (U+2019)


## === cell 1
import sys
import os

sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 2
!pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall
!pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps
!pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps
!pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index



## === cell 3
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
OUT_PATH2 = "/kaggle/working/v2"



## === cell 4
!cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir=$DATA_PATH --out_dir=$OUT_PATH



## === cell 5
!ls /kaggle/input/hms-mk-data



## === cell 6
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v0.csv 
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold1_levit_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v0.csv 
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold2_levit_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v0.csv 
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold3_levit_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v0.csv 
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold4_levit_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v0.csv 

!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold0_effb3_sim_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_effv2_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v3.csv 
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold1_effb3_sim_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_effv2_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v3.csv 
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold2_effb3_sim_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_effv2_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v3.csv 
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold3_effb3_sim_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_effv2_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v3.csv
!cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=$DATA_PATH data.test_eegs_dir=$OUT_PATH ckpt_path=/kaggle/input/hms-mk-data/fold4_effb3_sim_pseudo.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=conv1d_effv2_pseudo +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v3.csv



## === cell 7
sys.path.remove('/kaggle/input/hms-mk-codes/')



## === cell 8
sys.path.append('/kaggle/input/hms-mk-codesv2')



## === cell 9
!cd /kaggle/input/hms-mk-codesv2 && python -m src.convert_parquet_to_npy --data_dir=$DATA_PATH --out_dir=$OUT_PATH2



## === cell 10
!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold0.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_tfm_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v4.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold1.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_tfm_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v4.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold2.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_tfm_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v4.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold3.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_tfm_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v4.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold4.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_tfm_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v4.csv 


!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold0.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_effb1_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v5.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold1.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_effb1_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v5.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold2.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_effb1_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v5.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold3.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_effb1_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v5.csv 

!cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=$DATA_PATH data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=$OUT_PATH2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold4.ckpt hydra=test +model.test_output_dir=$OUT_PATH experiment=clean_effb1_corr_pseudo data.num_workers=2 +model.net.pretrained=False 
!mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v5.csv



## === cell 11
!head /kaggle/working/submission_fold1_v4.csv 



## === cell 12
!head /kaggle/working/submission_fold1_v5.csv 



## === cell 13
import pandas as pd
import numpy as np

DEFAULT_TARGET_COLS = [
    "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"
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

def merge_preds(folds=[0, 1, 2, 3, 4], versions=["v4", "v5"], weights=[0.5, 0.5]):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    base = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")[["eeg_id"]].copy()
    pred_sum = np.zeros((len(base), len(TARGET_COLS)), dtype=np.float64)

    for fold in folds:
        for w, version in zip(weights, versions):
            path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
            sol = pd.read_csv(path)

            sol = base.merge(sol[["eeg_id"] + TARGET_COLS], on="eeg_id", how="left", validate="one_to_one")
            vals = sol[TARGET_COLS].to_numpy(dtype=np.float64)

            if np.isnan(vals).any():
                vals = np.nan_to_num(vals, nan=1.0 / len(TARGET_COLS))

            pred_sum += w * vals

    pred_sum = np.clip(pred_sum, 1e-12, None)
    pred_sum = pred_sum / pred_sum.sum(axis=1, keepdims=True)

    out = base.copy()
    out[TARGET_COLS] = pred_sum
    return out



## === cell 14
sol = merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v3", "v4", "v5"],
    weights=[0.25, 0.25, 0.25, 0.25],
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3843250423.py in <cell line: 0>()
----> 1 sol = merge_preds(
      2     folds=[0, 1, 2, 3, 4],
      3     versions=["v0", "v3", "v4", "v5"],
      4     weights=[0.25, 0.25, 0.25, 0.25],
      5 )

/tmp/ipykernel_11/2262472828.py in merge_preds(folds, versions, weights)
     30         for w, version in zip(weights, versions):
     31             path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
---> 32             sol = pd.read_csv(path)
     33 
     34             # Align predictions to base eeg_id order

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/submission_fold0_v0.csv'

## === cell 15
sol.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote:", "/kaggle/working/submission.csv", "shape=", sol.shape)

row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2697932586.py in <cell line: 0>()
      1 # Write final submission with required filename and columns.
----> 2 sol.to_csv("/kaggle/working/submission.csv", index=False)
      3 print("Wrote:", "/kaggle/working/submission.csv", "shape=", sol.shape)
      4 
      5 # Quick validity checks (score-neutral; prevents accidental invalid submissions)

NameError: name 'sol' is not defined

## === cell 16
sol.head()
```

## --- ERROR in cell 16, traceback:
  File "/tmp/ipykernel_11/4275505007.py", line 2
    ```
    ^
SyntaxError: invalid syntax
