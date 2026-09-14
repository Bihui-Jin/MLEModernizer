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

0.4814297300688451

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from IPython.display import display  # added for safe head display

warnings.filterwarnings("ignore", category=Warning)
gc.collect()




## === cell 1
class Config:
    seed = 3131
    image_transform = transforms.Resize((512, 512))
    num_folds = 5
    dataset_wide_mean = -0.2972692229201065  # From original notebook
    dataset_wide_std = 2.5997336315611026  # From original notebook


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(Config.seed)




## === cell 2
possible_paths = [
    os.path.join(".", "data", "hms-harmful-brain-activity-classification"),
    "/kaggle/input/hms-harmful-brain-activity-classification",
]
BASE_DIR = next((p for p in possible_paths if os.path.isdir(p)), None)
if BASE_DIR is None:
    raise FileNotFoundError("Could not locate the competition data directory.")

test_csv_path = os.path.join(BASE_DIR, "test.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
else:
    test_df = pd.DataFrame(columns=["eeg_id", "spectrogram_id", "patient_id"])
    warnings.warn(f"{test_csv_path} not found; using empty test DataFrame.")

if os.path.exists(sample_sub_path):
    submission = pd.read_csv(sample_sub_path)
else:
    raise FileNotFoundError(
        f"{sample_sub_path} not found; cannot create submission template."
    )

submission = submission.merge(test_df, on="eeg_id", how="left")
submission["path"] = submission["spectrogram_id"].apply(
    lambda x: os.path.join(BASE_DIR, "test_spectrograms", f"{x}.parquet")
)




## === cell 3
train_csv_path = os.path.join(BASE_DIR, "train.csv")
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
else:
    raise FileNotFoundError(
        f"{train_csv_path} not found; training data required for baseline probabilities."
    )

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
class_totals = train_df[label_cols].sum().astype(float)
baseline_probs = (class_totals / class_totals.sum()).values  # shape (6,)




## === cell 4
models = []
models_datawide1 = []
models_datawide2 = []

for i in range(Config.num_folds):
    try:
        m = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=6, in_chans=1
        )
        m.load_state_dict(
            torch.load(
                f"/kaggle/input/hms-train-efficientnetb0/efficientnet_b0_fold{i}.pth",
                map_location=torch.device("cpu"),
            )
        )
        models.append(m)
    except FileNotFoundError:
        pass

for i in range(Config.num_folds):
    try:
        m = timm.create_model(
            "efficientnet_b1", pretrained=False, num_classes=6, in_chans=1
        )
        m.load_state_dict(
            torch.load(
                f"/kaggle/input/train/efficientnet_b1_fold{i}.pth",
                map_location=torch.device("cpu"),
            )
        )
        models_datawide1.append(m)
    except FileNotFoundError:
        pass

for i in range(Config.num_folds):
    try:
        m = timm.create_model(
            "efficientnet_b1", pretrained=False, num_classes=6, in_chans=1
        )
        m.load_state_dict(
            torch.load(
                f"/kaggle/input/efficientnet-b0-naomi/efficientnet_b1_fold{i}_datawide_ReduceLROnPlateau_0.001_False.pth",
                map_location=torch.device("cpu"),
            )
        )
        models_datawide2.append(m)
    except FileNotFoundError:
        pass
    try:
        m = timm.create_model(
            "efficientnet_b1", pretrained=False, num_classes=6, in_chans=1
        )
        m.load_state_dict(
            torch.load(
                f"/kaggle/input/efficientnet-b0-naomi/efficientnet_b1_fold{i}_datawide_CosineAnnealingLR_0.001_False.pth",
                map_location=torch.device("cpu"),
            )
        )
        models_datawide2.append(m)
    except FileNotFoundError:
        pass




## === cell 5
try:
    df_norm = pd.read_csv(os.path.join(BASE_DIR, "normalization.csv"))
    data_mean2 = df_norm["mean"].values[0]
    data_std2 = df_norm["std"].values[0]
except FileNotFoundError:
    data_mean2 = None
    data_std2 = None




## === cell 6
def preprocess(path_to_parquet):
    data = pd.read_parquet(path_to_parquet)
    data = data.fillna(-1).values[:, 1:].T
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)
    return data


def normalize_instance_wise(data_point):
    eps = 1e-6
    data_mean = data_point.mean(axis=(0, 1))
    data_std = data_point.std(axis=(0, 1))
    data_point = (data_point - data_mean) / (data_std + eps)
    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    return Config.image_transform(data_tensor)


def normalize_datawide1(data_point):
    eps = 1e-6
    data_point = (data_point - Config.dataset_wide_mean) / (
        Config.dataset_wide_std + eps
    )
    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    return Config.image_transform(data_tensor)


def normalize_datawide2(data, data_mean, data_std):
    eps = 1e-6
    data = (data - data_mean) / (data_std + eps)
    data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
    return Config.image_transform(data_tensor)




## === cell 7
paths = submission["path"].values
num_test = len(paths)
test_predictions = np.empty((num_test, 6), dtype=np.float32)

use_models = bool(models) or bool(models_datawide1) or bool(models_datawide2)

for idx, path in enumerate(paths):
    if use_models:
        per_model_preds = []

        if not os.path.exists(path):
            per_model_preds = []
        else:
            preproc = preprocess(path)

            for m in models:
                m.eval()
                inp = normalize_instance_wise(preproc)
                with torch.no_grad():
                    out = m(inp)
                    pred = F.softmax(out, dim=1).cpu().numpy()[0]
                per_model_preds.append(pred)

            for m in models_datawide1:
                m.eval()
                inp = normalize_datawide1(preproc)
                with torch.no_grad():
                    out = m(inp)
                    pred = F.softmax(out, dim=1).cpu().numpy()[0]
                per_model_preds.append(pred)

            if data_mean2 is not None and data_std2 is not None:
                for m in models_datawide2:
                    m.eval()
                    inp = normalize_datawide2(preproc, data_mean2, data_std2)
                    with torch.no_grad():
                        out = m(inp)
                        pred = F.softmax(out, dim=1).cpu().numpy()[0]
                    per_model_preds.append(pred)

        if per_model_preds:
            test_predictions[idx] = np.mean(per_model_preds, axis=0)
        else:
            test_predictions[idx] = baseline_probs
    else:
        test_predictions[idx] = baseline_probs

uniform_probs = np.full_like(test_predictions, 1.0 / 6)
alpha = 0.1  # blend factor; kept tiny to preserve original signal
test_predictions = (1 - alpha) * test_predictions + alpha * uniform_probs

test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)




## === cell 8
labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
for i, lab in enumerate(labels):
    submission[f"{lab}_vote"] = test_predictions[:, i]

submission_cols = ["eeg_id"] + [f"{lab}_vote" for lab in labels]
final_submission = submission[submission_cols]

submission_path = "submission.csv"
final_submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(final_submission.head())
gc.collect()
