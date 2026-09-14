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

0.3092453848929784

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'The runtime failure is caused by `from src.settings import TARGET_COLS` after you removed the `hms-mk-codes` path, so `src` is no longer importable at merge time. I make the merge step self-contained by deriving `TARGET_COLS` directly from `sample_submission.csv`, and I also make the merge robust to missing fold/version files and ensure the final probabilities are valid (non-negative and row-sum=1). This keeps the core inference logic unchanged (you still generate the same per-fold CSVs), but fixes the crash and guarantees a valid `submission.csv` is written. Finally, I keep your intended ensemble weights and simply normalize safely to avoid submission format failures.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 1
import os, subprocess, textwrap, shlex


def run_cmd(cmd: str):
    print(cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
    return r.stdout


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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3349408745.py in <cell line: 0>()
     13 
     14 
---> 15 run_cmd(
     16     "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
     17 )

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 1: pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall

## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
OUT_PATH2 = "/kaggle/working/v2"



## === cell 3
run_cmd(
    f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH)}"
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1557857328.py in <cell line: 0>()
----> 1 run_cmd(
      2     f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH)}"
      3 )
      4 

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 2: cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir=/kaggle/input/hms-harmful-brain-activity-classification --out_dir=/kaggle/working

## === cell 4
run_cmd("ls /kaggle/input/hms-mk-data")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2762870559.py in <cell line: 0>()
----> 1 run_cmd("ls /kaggle/input/hms-mk-data")
      2 

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 2: ls /kaggle/input/hms-mk-data

## === cell 5
base = f"cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir={shlex.quote(DATA_PATH)} data.test_eegs_dir={shlex.quote(OUT_PATH)} hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} +model.net.pretrained=False"
cmds = [
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v0.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold1_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v0.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold2_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v0.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold3_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v0.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold4_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v0.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold0_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v3.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold1_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v3.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold2_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v3.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold3_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v3.csv",
    f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold4_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v3.csv",
]
for c in cmds:
    run_cmd(c)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1269964413.py in <cell line: 0>()
     16 ]
     17 for c in cmds:
---> 18     run_cmd(c)
     19 

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 2: cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=/kaggle/input/hms-harmful-brain-activity-classification data.test_eegs_dir=/kaggle/working hydra=test +model.test_output_dir=/kaggle/working +model.net.pretrained=False ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v0.csv

## === cell 6
try:
    sys.path.remove("/kaggle/input/hms-mk-codes/")
except ValueError:
    pass



## === cell 7
sys.path.append("/kaggle/input/hms-mk-codesv2")



## === cell 8
run_cmd(
    f"cd /kaggle/input/hms-mk-codesv2 && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH2)}"
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3570445830.py in <cell line: 0>()
----> 1 run_cmd(
      2     f"cd /kaggle/input/hms-mk-codesv2 && python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH2)}"
      3 )
      4 

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 2: cd /kaggle/input/hms-mk-codesv2 && python -m src.convert_parquet_to_npy --data_dir=/kaggle/input/hms-harmful-brain-activity-classification --out_dir=/kaggle/working/v2

## === cell 9
base2 = (
    f"cd /kaggle/input/hms-mk-codesv2 && python -m test "
    f"paths.data_dir={shlex.quote(DATA_PATH)} "
    f"data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG "
    f"data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs "
    f"hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} "
    f"data.num_workers=2 +model.net.pretrained=False"
)
cmds2 = [
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold0.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v4.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold1.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v4.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold2.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v4.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold3.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v4.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold4.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v4.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold0.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v5.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold1.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v5.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold2.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v5.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold3.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v5.csv",
    f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold4.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v5.csv",
]
for c in cmds2:
    run_cmd(c)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2306847724.py in <cell line: 0>()
     23 ]
     24 for c in cmds2:
---> 25     run_cmd(c)
     26 

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 2: cd /kaggle/input/hms-mk-codesv2 && python -m test paths.data_dir=/kaggle/input/hms-harmful-brain-activity-classification data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG data.test_dataset.eeg_dir=/kaggle/working/v2/test_eegs hydra=test +model.test_output_dir=/kaggle/working data.num_workers=2 +model.net.pretrained=False ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold0.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v4.csv

## === cell 10
run_cmd("head /kaggle/working/submission_fold1_v4.csv")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1671033692.py in <cell line: 0>()
----> 1 run_cmd("head /kaggle/working/submission_fold1_v4.csv")
      2 

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 1: head /kaggle/working/submission_fold1_v4.csv

## === cell 11
run_cmd("head /kaggle/working/submission_fold1_v5.csv")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3270088605.py in <cell line: 0>()
----> 1 run_cmd("head /kaggle/working/submission_fold1_v5.csv")
      2 

/tmp/ipykernel_11/3349408745.py in run_cmd(cmd)
      9         print(r.stdout)
     10         print(r.stderr)
---> 11         raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
     12     return r.stdout
     13 

RuntimeError: Command failed with exit code 1: head /kaggle/working/submission_fold1_v5.csv

## === cell 12
import pandas as pd
import numpy as np
import os

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v4", "v5"),
    weights=(0.5, 0.5),
    workdir="/kaggle/working",
    eps=1e-12,
):
    weights = list(weights)
    versions = list(versions)
    if len(weights) != len(versions):
        raise ValueError("weights and versions must have the same length")

    sol = sample_sub.copy()

    acc = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)
    used = 0

    for fold in folds:
        for w, v in zip(weights, versions):
            fp = f"{workdir}/submission_fold{fold}_{v}.csv"
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = sol[["eeg_id"]].merge(df, on="eeg_id", how="left")

            vals = df[TARGET_COLS].to_numpy(np.float64)
            if np.isnan(vals).any():
                nan_rows = np.isnan(vals).any(axis=1)
                vals[nan_rows] = 1.0 / len(TARGET_COLS)

            acc += vals * float(w)
            used += 1

    if used == 0:
        preds = np.full_like(acc, 1.0 / len(TARGET_COLS))
    else:
        preds = acc / (np.sum(weights) * len(folds))

    preds = np.clip(preds, eps, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    sol[TARGET_COLS] = preds
    return sol




## === cell 13
sol = merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v3", "v4", "v5"),
    weights=(0.3, 0.3, 0.2, 0.2),
)



## === cell 14
sol.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv")
print(sol.shape)
print(sol.head())



## === cell 15
assert list(sol.columns) == ["eeg_id"] + TARGET_COLS
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert np.max(np.abs(row_sums - 1.0)) < 1e-6
assert (sol[TARGET_COLS].to_numpy() >= 0).all()
print("Submission looks valid: probs non-negative and sum to 1 per row.")
