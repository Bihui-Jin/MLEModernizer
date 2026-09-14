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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.4962510187102379

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=Warning)

INPUT_DIR = Path("/kaggle/input/hms-harmful-brain-activity-classification")
TRAIN_CSV = INPUT_DIR / "train.csv"
TEST_CSV = INPUT_DIR / "test.csv"
SAMPLE_SUB = INPUT_DIR / "sample_submission.csv"
TEST_SPEC_DIR = INPUT_DIR / "test_spectrograms"

gc.collect()



## === cell 1
import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
from IPython.display import display

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=Warning)
gc.collect()



## === cell 2
LABELS = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTE_COLS = [f"{l}_vote" for l in LABELS]




## === cell 3
class Config:
    seed = 3131
    image_transform = transforms.Resize((512, 512))
    num_folds = 5
    dataset_wide_mean = -0.2972692229201065  # From Train notebook
    dataset_wide_std = 2.5997336315611026  # From Train notebook


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(Config.seed)



## === cell 4
train_df_for_prior = pd.read_csv(TRAIN_CSV, usecols=VOTE_COLS)
prior = train_df_for_prior[VOTE_COLS].sum(axis=0).values.astype(np.float64)
prior = prior / (prior.sum() + 1e-12)
prior = np.clip(prior, 1e-8, 1.0)
prior = prior / prior.sum()
del train_df_for_prior
gc.collect()



## === cell 5
test_df = pd.read_csv(TEST_CSV, usecols=["eeg_id", "spectrogram_id", "patient_id"])
sample_sub = pd.read_csv(SAMPLE_SUB)

submission = sample_sub[["eeg_id"]].merge(
    test_df[["eeg_id", "spectrogram_id"]],
    on="eeg_id",
    how="left",
    validate="one_to_one",
)

submission["path"] = submission["spectrogram_id"].apply(
    lambda x: str(TEST_SPEC_DIR / f"{x}.parquet")
)

display(submission.head())
print("n_test_rows:", len(submission), "unique_eeg_id:", submission["eeg_id"].nunique())
gc.collect()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
MergeError                                Traceback (most recent call last)
/tmp/ipykernel_54/1866458784.py in <cell line: 0>()
      5 
      6 # enforce exact ordering/rows as required by submission template
----> 7 submission = sample_sub[["eeg_id"]].merge(
      8     test_df[["eeg_id", "spectrogram_id"]],
      9     on="eeg_id",

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    811         # are in fact unique.
    812         if validate is not None:
--> 813             self._validate_validate_kwd(validate)
    814 
    815     def _maybe_require_matching_dtypes(

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _validate_validate_kwd(self, validate)
   1646         if validate in ["one_to_one", "1:1"]:
   1647             if not left_unique and not right_unique:
-> 1648                 raise MergeError(
   1649                     "Merge keys are not unique in either left "
   1650                     "or right dataset; not a one-to-one merge"

MergeError: Merge keys are not unique in either left or right dataset; not a one-to-one merge

## === cell 6
def _find_first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return str(p)
    return None


def try_load_state_dict(model, ckpt_path):
    if ckpt_path is None:
        return False
    try:
        state = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(state)
        return True
    except Exception:
        return False




## === cell 7
models = []

loaded_any = False
for i in range(Config.num_folds):
    model_effnet_b0 = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=6, in_chans=1
    )
    ckpt = _find_first_existing(
        [
            f"/kaggle/input/hms-train-efficientnetb0/efficientnet_b0_fold{i}.pth",
            f"/kaggle/input/efficientnet-b0/efficientnet_b0_fold{i}.pth",
            f"/kaggle/input/efficientnetb0/efficientnet_b0_fold{i}.pth",
        ]
    )
    ok = try_load_state_dict(model_effnet_b0, ckpt)
    if ok:
        loaded_any = True
        models.append(model_effnet_b0)

models_datawide1 = []

for i in range(Config.num_folds):
    model_effnet_b1 = timm.create_model(
        "efficientnet_b1", pretrained=False, num_classes=6, in_chans=1
    )
    ckpt = _find_first_existing(
        [
            f"/kaggle/input/train/efficientnet_b1_fold{i}.pth",
            f"/kaggle/input/efficientnet-b1/efficientnet_b1_fold{i}.pth",
            f"/kaggle/input/efficientnetb1/efficientnet_b1_fold{i}.pth",
        ]
    )
    ok = try_load_state_dict(model_effnet_b1, ckpt)
    if ok:
        loaded_any = True
        models_datawide1.append(model_effnet_b1)

print("Loaded models:", len(models), "Loaded models_datawide1:", len(models_datawide1))
gc.collect()



## === cell 8
norm_csv = _find_first_existing(
    [
        "/kaggle/input/efficientnet-b0-naomi/normalization.csv",
        "/kaggle/input/normalization/normalization.csv",
    ]
)
if norm_csv is not None:
    df_normalization = pd.read_csv(norm_csv)
    data_mean2 = float(df_normalization["mean"].values[0])
    data_std2 = float(df_normalization["std"].values[0])
else:
    data_mean2 = float(Config.dataset_wide_mean)
    data_std2 = float(Config.dataset_wide_std)

gc.collect()



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for m in models:
    m.to(device)
for m in models_datawide1:
    m.to(device)
gc.collect()



## === cell 10
paths_to_parquets = submission["path"].values
n_test = len(paths_to_parquets)


def preprocess(path_to_parquet):
    data = pd.read_parquet(path_to_parquet)
    data = data.fillna(-1).values[:, 1:].T
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)
    return data


def normalize_datawide1(data_point):
    eps = 1e-6
    data_point = (data_point - Config.dataset_wide_mean) / (
        Config.dataset_wide_std + eps
    )
    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    data_point = Config.image_transform(data_tensor)
    return data_point


def normalize_datawide2(data, data_mean, data_std):
    eps = 1e-6
    data = (data - data_mean) / (data_std + eps)
    data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
    data = Config.image_transform(data_tensor)
    return data


def normalize_instance_wise(data_point):
    eps = 1e-6
    data_mean = data_point.mean(axis=(0, 1))
    data_std = data_point.std(axis=(0, 1))
    data_point = (data_point - data_mean) / (data_std + eps)
    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    data_point = Config.image_transform(data_tensor)
    return data_point


test_predictions = np.zeros((n_test, 6), dtype=np.float64)

if (len(models) + len(models_datawide1)) == 0:
    test_predictions[:] = prior[None, :]
else:
    for idx, path in enumerate(paths_to_parquets):
        test_predictions_per_model = []
        try:
            preprocessed_data = preprocess(path)
        except Exception:
            test_predictions[idx] = prior
            continue

        for i in range(len(models)):
            models[i].eval()
            current_parquet_data = (
                normalize_instance_wise(preprocessed_data).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                model_output = models[i](current_parquet_data)
                current_model_prediction = (
                    F.softmax(model_output, dim=1)[0].detach().cpu().numpy()
                )
            test_predictions_per_model.append(current_model_prediction)

        for i in range(len(models_datawide1)):
            models_datawide1[i].eval()
            current_parquet_data = (
                normalize_datawide1(preprocessed_data).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                model_output = models_datawide1[i](current_parquet_data)
                current_model_prediction = (
                    F.softmax(model_output, dim=1)[0].detach().cpu().numpy()
                )
            test_predictions_per_model.append(current_model_prediction)

        ensemble_prediction = np.mean(test_predictions_per_model, axis=0)
        ensemble_prediction = np.clip(ensemble_prediction, 1e-8, 1.0)
        ensemble_prediction = ensemble_prediction / ensemble_prediction.sum()
        test_predictions[idx] = ensemble_prediction

gc.collect()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2549799626.py in <cell line: 0>()
----> 1 paths_to_parquets = submission["path"].values
      2 n_test = len(paths_to_parquets)
      3 
      4 
      5 def preprocess(path_to_parquet):

NameError: name 'submission' is not defined

## === cell 11
test_predictions = np.asarray(test_predictions, dtype=np.float64)
test_predictions = np.clip(test_predictions, 1e-8, 1.0)
test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)

assert test_predictions.shape == (len(submission), 6)
row_sums = test_predictions.sum(axis=1)
print("Row sum min/max:", row_sums.min(), row_sums.max())



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/781022657.py in <cell line: 0>()
      1 # Ensure valid probabilities per row (required by Kaggle; also helps KL metric avoid inf/nan)
----> 2 test_predictions = np.asarray(test_predictions, dtype=np.float64)
      3 test_predictions = np.clip(test_predictions, 1e-8, 1.0)
      4 test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)
      5 

NameError: name 'test_predictions' is not defined

## === cell 12
final_sub = pd.read_csv(SAMPLE_SUB)
for i, lab in enumerate(LABELS):
    final_sub[f"{lab}_vote"] = test_predictions[:, i]

final_sub.to_csv("submission.csv", index=False)
display(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)
gc.collect()

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3666443473.py in <cell line: 0>()
      2 final_sub = pd.read_csv(SAMPLE_SUB)
      3 for i, lab in enumerate(LABELS):
----> 4     final_sub[f"{lab}_vote"] = test_predictions[:, i]
      5 
      6 final_sub.to_csv("submission.csv", index=False)

NameError: name 'test_predictions' is not defined
