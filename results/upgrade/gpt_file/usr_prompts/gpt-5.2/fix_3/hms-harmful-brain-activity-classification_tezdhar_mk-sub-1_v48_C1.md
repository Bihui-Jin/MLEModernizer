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

0.3346129051398209

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys

sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 1
import subprocess, textwrap, os, shlex, pathlib, time


def _run(cmd):
    print(f"\n[RUN] {cmd}")
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if p.stdout:
        print(p.stdout)
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr)
        raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
    return p


_run(
    "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
)
_run(
    "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
)
_run(
    "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
)
_run(
    "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3003749200.py in <cell line: 0>()
     17 
     18 
---> 19 _run(
     20     "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
     21 )

/tmp/ipykernel_11/3003749200.py in _run(cmd)
     13         if p.stderr:
     14             print(p.stderr)
---> 15         raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
     16     return p
     17 

RuntimeError: Command failed with code 1: pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall

## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
OUT_PATH2 = "/kaggle/working/v2"



## === cell 3
_run(
    f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH)}"
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3775472328.py in <cell line: 0>()
----> 1 _run(
      2     f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH)}"
      3 )
      4 

/tmp/ipykernel_11/3003749200.py in _run(cmd)
     13         if p.stderr:
     14             print(p.stderr)
---> 15         raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
     16     return p
     17 

RuntimeError: Command failed with code 2: cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir=/kaggle/input/hms-harmful-brain-activity-classification --out_dir=/kaggle/working

## === cell 4
try:
    _run("ls /kaggle/input/hms-mk-data")
except Exception as e:
    print(f"Warning: could not list /kaggle/input/hms-mk-data: {e}")



## === cell 5
_run(
    f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_eegs_dir={shlex.quote(OUT_PATH)} ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v0.csv")

_run(
    f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_eegs_dir={shlex.quote(OUT_PATH)} ckpt_path=/kaggle/input/hms-mk-data/fold1_levit_pseudo.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v0.csv")

_run(
    f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_eegs_dir={shlex.quote(OUT_PATH)} ckpt_path=/kaggle/input/hms-mk-data/fold2_levit_pseudo.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v0.csv")

_run(
    f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_eegs_dir={shlex.quote(OUT_PATH)} ckpt_path=/kaggle/input/hms-mk-data/fold3_levit_pseudo.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v0.csv")

_run(
    f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_eegs_dir={shlex.quote(OUT_PATH)} ckpt_path=/kaggle/input/hms-mk-data/fold4_levit_pseudo.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v0.csv")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1344112883.py in <cell line: 0>()
      2 # we raise (to keep core logic strict). If it succeeds but doesn't produce submission.csv,
      3 # later ensembling will handle missing files.
----> 4 _run(
      5     f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_eegs_dir={shlex.quote(OUT_PATH)} ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False"
      6 )

/tmp/ipykernel_11/3003749200.py in _run(cmd)
     13         if p.stderr:
     14             print(p.stderr)
---> 15         raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
     16     return p
     17 

RuntimeError: Command failed with code 2: cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=/kaggle/input/hms-harmful-brain-activity-classification data.test_eegs_dir=/kaggle/working ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt hydra=test +model.test_output_dir=/kaggle/working experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False

## === cell 6
try:
    sys.path.remove("/kaggle/input/hms-mk-codes/")
except ValueError:
    pass



## === cell 7
sys.path.append("/kaggle/input/hms-mk-codesv2")



## === cell 8
_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH2)}"
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3806881584.py in <cell line: 0>()
----> 1 _run(
      2     f"cd /kaggle/input/hms-mk-codesv2 && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH2)}"
      3 )
      4 

/tmp/ipykernel_11/3003749200.py in _run(cmd)
     13         if p.stderr:
     14             print(p.stderr)
---> 15         raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
     16     return p
     17 

RuntimeError: Command failed with code 2: cd /kaggle/input/hms-mk-codesv2 && python -m src.convert_parquet_to_npy --data_dir=/kaggle/input/hms-harmful-brain-activity-classification --out_dir=/kaggle/working/v2

## === cell 9
_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold0.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_tfm_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v4.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold1.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_tfm_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v4.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold2.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_tfm_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v4.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold3.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_tfm_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v4.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold4.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_tfm_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v4.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold0.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_effb1_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v5.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold1.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_effb1_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v5.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold2.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_effb1_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v5.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold3.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_effb1_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v5.csv")

_run(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold4.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_effb1_pseudo data.num_workers=2 +model.net.pretrained=False"
)
_run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v5.csv")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/607619385.py in <cell line: 0>()
      1 # v4 (xcit) folds
----> 2 _run(
      3     f"cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold0.ckpt hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} experiment=clean_tfm_pseudo data.num_workers=2 +model.net.pretrained=False"
      4 )
      5 _run("mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v4.csv")

/tmp/ipykernel_11/3003749200.py in _run(cmd)
     13         if p.stderr:
     14             print(p.stderr)
---> 15         raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
     16     return p
     17 

RuntimeError: Command failed with code 2: cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=/kaggle/input/hms-harmful-brain-activity-classification data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=/kaggle/working/v2/test_eegs ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold0.ckpt hydra=test +model.test_output_dir=/kaggle/working experiment=clean_tfm_pseudo data.num_workers=2 +model.net.pretrained=False

## === cell 10
try:
    _run("head /kaggle/working/submission_fold1_v4.csv")
except Exception as e:
    print(f"Warning: could not preview v4 file: {e}")



## === cell 11
try:
    _run("head /kaggle/working/submission_fold1_v5.csv")
except Exception as e:
    print(f"Warning: could not preview v5 file: {e}")



## === cell 12
import os
import pandas as pd
import numpy as np

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v4", "v5"),
    weights=(0.5, 0.5),
    base_dir="/kaggle/working",
    allow_missing=True,
):
    """
    Bugfix: previously raised FileNotFoundError and prevented producing any submission.
    This keeps the same averaging logic when files exist, but can proceed using the
    available prediction files if some are missing (score-neutral vs failing).
    """
    if len(versions) != len(weights):
        raise ValueError("versions and weights must have the same length")

    sol = sample_sub.copy()
    eeg_order = sol["eeg_id"].values
    pred_sum = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)
    weight_sum = 0.0

    missing = []
    for fold in folds:
        for version, w in zip(versions, weights):
            fp = os.path.join(base_dir, f"submission_fold{fold}_{version}.csv")
            if not os.path.exists(fp):
                missing.append(fp)
                if allow_missing:
                    continue
                raise FileNotFoundError(f"Missing prediction file: {fp}")

            df = pd.read_csv(fp)
            if "eeg_id" not in df.columns:
                raise ValueError(f"{fp} missing eeg_id column")
            for c in TARGET_COLS:
                if c not in df.columns:
                    raise ValueError(f"{fp} missing required column: {c}")

            df = df.set_index("eeg_id").reindex(eeg_order)
            if df.isna().any().any():
                df[TARGET_COLS] = df[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

            pred_sum += df[TARGET_COLS].to_numpy(dtype=np.float64) * float(w)
            weight_sum += float(w)

    if weight_sum == 0.0:
        raise RuntimeError(
            "No prediction files were merged (all missing). "
            f"First 3 missing examples: {missing[:3]}"
        )

    pred_sum /= weight_sum

    pred_sum = np.nan_to_num(
        pred_sum,
        nan=1.0 / len(TARGET_COLS),
        posinf=1.0 / len(TARGET_COLS),
        neginf=1.0 / len(TARGET_COLS),
    )
    pred_sum = np.clip(pred_sum, 1e-15, 1.0)
    pred_sum /= pred_sum.sum(axis=1, keepdims=True)

    sol[TARGET_COLS] = pred_sum
    if missing:
        print(
            f"merge_preds: missing {len(missing)} files (allowed). Example: {missing[0]}"
        )
    return sol




## === cell 13
sol = merge_preds(
    folds=(0, 1, 2, 3, 4), versions=("v4", "v5"), weights=(0.5, 0.5), allow_missing=True
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/538339626.py in <cell line: 0>()
----> 1 sol = merge_preds(
      2     folds=(0, 1, 2, 3, 4), versions=("v4", "v5"), weights=(0.5, 0.5), allow_missing=True
      3 )
      4 

/tmp/ipykernel_11/711967331.py in merge_preds(folds, versions, weights, base_dir, allow_missing)
     55 
     56     if weight_sum == 0.0:
---> 57         raise RuntimeError(
     58             "No prediction files were merged (all missing). "
     59             f"First 3 missing examples: {missing[:3]}"

RuntimeError: No prediction files were merged (all missing). First 3 missing examples: ['/kaggle/working/submission_fold0_v4.csv', '/kaggle/working/submission_fold0_v5.csv', '/kaggle/working/submission_fold1_v4.csv']

## === cell 14
for c in ["eeg_id"] + TARGET_COLS:
    if c not in sol.columns:
        raise ValueError(f"Submission missing required column: {c}")

probs = sol[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.nan_to_num(
    probs,
    nan=1.0 / len(TARGET_COLS),
    posinf=1.0 / len(TARGET_COLS),
    neginf=1.0 / len(TARGET_COLS),
)
probs = np.clip(probs, 1e-15, 1.0)  # avoid exact zeros for KL stability
probs = probs / probs.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = probs

out_fp = "/kaggle/working/submission.csv"
sol.to_csv(out_fp, index=False)
print(f"Wrote submission to: {out_fp}")
print(sol.head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3571175463.py in <cell line: 0>()
      1 # Ensure valid submission formatting and probability constraints
      2 for c in ["eeg_id"] + TARGET_COLS:
----> 3     if c not in sol.columns:
      4         raise ValueError(f"Submission missing required column: {c}")
      5 

NameError: name 'sol' is not defined

## === cell 15
assert sol.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"
assert list(sol.columns) == list(
    sample_sub.columns
), "Column order mismatch vs sample_submission"
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums)), "Non-finite probabilities detected"
assert (
    np.max(np.abs(row_sums - 1.0)) < 1e-6
), "Probabilities do not sum to 1 within tolerance"
sol.head()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2297572464.py in <cell line: 0>()
      1 # Final quick sanity checks
----> 2 assert sol.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"
      3 assert list(sol.columns) == list(
      4     sample_sub.columns
      5 ), "Column order mismatch vs sample_submission"

NameError: name 'sol' is not defined
