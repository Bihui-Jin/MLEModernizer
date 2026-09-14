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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
tqdm==4.67.1

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

0.5399889763869918

# 6. Current score

1.405

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The merge in cell 2 is exploding the row count because both `sample_submission.csv` and `test.csv` contain duplicate `eeg_id` values, creating a many-to-many join. The minimal fix is to avoid merging entirely and instead build the inference dataframe directly from `test.csv`, adding a `path` column for each `spectrogram_id`. To keep the rest of the pipeline unchanged, the dataset return a dummy `y` for test rows (since labels don’t exist there), and we ensure predictions are aligned 1:1 with `test` order before writing `submission.csv`. This should run end-to-end and produce a valid submission with exactly 9850 rows and row-wise probabilities summing to 1.'
- What this solution (achieved 1.405) has done: 'Your current pipeline is producing a valid submission, but the score (KL divergence) indicates the predictions are badly miscalibrated for this competition’s target distribution. The smallest legitimate improvement (without changing architecture/training) is to apply a light, deterministic post-processing calibration: mix the model probabilities with a reasonable prior estimated from `train.csv` vote distributions, then renormalize. This typically reduces extreme/confident wrong predictions and improves KL. I keep everything else identical and only add: loading `train.csv`, computing the global mean label distribution, and blending with a small weight chosen conservatively to move the score toward your target.'
- What this solution (achieved 1.405) has done: 'I fix the merge error by not forcing a one-to-one merge against `sample_submission`, since `sample_submission` can contain duplicate `eeg_id` rows; instead, I map predictions onto `sample_submission` by `eeg_id` while preserving its row order. I also ensure the output CSV always contains exactly the required columns in the required order and that each row sums to 1, which addresses the “missing target columns” invalid-submission issue that can happen when the merge fails mid-cell. These changes are minimal and keep the model/inference logic identical, only stabilizing the final alignment step and guaranteeing a valid `submission.csv` is written end-to-end.'

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from torch.utils.data import Dataset, DataLoader




## === cell 1
class CFG:
    base_dir = pathlib.Path("/kaggle/input/hms-harmful-brain-activity-classification")
    path_test = base_dir / "test.csv"
    path_submission = base_dir / "sample_submission.csv"
    path_train = base_dir / "train.csv"

    spec_dir = base_dir / "test_spectrograms"
    model_name = "tf_efficientnet_b0_ns"
    model_weights = sorted(
        list(pathlib.Path("/kaggle/input/hms-pytorch-baseline-training").glob("*.pt"))
    )
    transform = transforms.Resize((512, 512), antialias=False)
    batch_size = 16
    num_workers = min(2, os.cpu_count() or 1)
    label_columns = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    prior_blend = 0.15  # 0=no change; higher pushes predictions toward global prior


print("Found model weights:", len(CFG.model_weights))
CFG.model_weights[:5]



## === cell 2
test = pd.read_csv(CFG.path_test)
sample_submission = pd.read_csv(CFG.path_submission)

submission = test.copy()
submission["path"] = submission["spectrogram_id"].map(
    lambda x: CFG.spec_dir / f"{x}.parquet"
)

assert len(submission) == len(
    test
), f"Unexpected test length change: {len(submission)} vs {len(test)}"
assert (
    submission["path"].map(lambda p: pathlib.Path(p).exists()).all()
), "Some spectrogram parquet files are missing."
submission.head()




## === cell 3
def preprocess(x):
    x = np.clip(x, np.exp(-6), np.exp(10))
    x = np.log(x)
    m, s = x.mean(), x.std()
    x = (x - m) / (s + 1e-6)
    return x


class SpecDataset(Dataset):
    def __init__(self, df, transform=CFG.transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = pd.read_parquet(row.path)
        x = x.fillna(-1).values[:, 1:].T
        x = preprocess(x)
        x = torch.Tensor(x[None, :])  # [1, H, W]
        if self.transform:
            x = self.transform(x)

        y = torch.zeros(len(CFG.label_columns), dtype=torch.float32)
        return x, y




## === cell 4
data_ds = SpecDataset(df=submission)

data_loader = DataLoader(
    dataset=data_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

len(data_loader)



## === cell 5
x, y = next(iter(data_loader))
x.shape, y.shape



## === cell 6
plt.imshow(x[0, 0])
plt.show()



## === cell 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")



## === cell 8
model = timm.create_model(
    model_name=CFG.model_name, pretrained=False, num_classes=6, in_chans=1
)
model.to(DEVICE)
num_parameter = sum(p.numel() for p in model.parameters())
print(f"Model has {num_parameter} parameters.")



## === cell 9
if len(CFG.model_weights) == 0:
    prediction = np.full(
        (len(submission), len(CFG.label_columns)),
        1.0 / len(CFG.label_columns),
        dtype=np.float32,
    )
    prediction = pd.DataFrame(prediction, columns=CFG.label_columns)
else:
    prediction = pd.DataFrame(
        0.0, columns=CFG.label_columns, index=np.arange(len(submission))
    )
    for i, path_weight in enumerate(CFG.model_weights):
        print(f"Model {i}: {path_weight}")
        state = torch.load(path_weight, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            res = []
            for xb, _ in tqdm(data_loader, total=len(data_loader), leave=False):
                xb = xb.to(DEVICE, non_blocking=True)
                pred = model(xb)
                pred = F.softmax(pred, dim=1)
                res.append(pred.detach().cpu().numpy())
            res = np.concatenate(res, axis=0)
        res = pd.DataFrame(
            res, columns=CFG.label_columns, index=np.arange(len(submission))
        )
        prediction = prediction + res
        print()
    prediction = prediction / len(CFG.model_weights)

prediction.head()



## === cell 10
pred_arr = prediction[CFG.label_columns].to_numpy(dtype=np.float64)
pred_arr = np.clip(pred_arr, 1e-12, 1.0)
pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)
prediction = pd.DataFrame(pred_arr, columns=CFG.label_columns)

prediction.sum(axis=1).describe()



## === cell 11
pred_with_ids = submission[["eeg_id"]].copy()
pred_with_ids[CFG.label_columns] = prediction[CFG.label_columns].values

pred_by_eeg = pred_with_ids.groupby("eeg_id", sort=False)[CFG.label_columns].mean()
pred_by_eeg = pred_by_eeg.reset_index()

submission_eeg = (
    test[["eeg_id"]]
    .drop_duplicates(subset=["eeg_id"], keep="first")
    .reset_index(drop=True)
)
submission_eeg = submission_eeg.merge(
    pred_by_eeg, on="eeg_id", how="left", validate="one_to_one"
)

miss = submission_eeg[CFG.label_columns].isna().any(axis=1)
if miss.any():
    submission_eeg.loc[miss, CFG.label_columns] = 1.0 / len(CFG.label_columns)

pred_arr = submission_eeg[CFG.label_columns].to_numpy(dtype=np.float64)
pred_arr = np.clip(pred_arr, 1e-12, 1.0)
pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)
submission_eeg[CFG.label_columns] = pred_arr

print(
    "Unique eeg_id rows for submission:",
    len(submission_eeg),
    "out of test rows:",
    len(test),
)



## === cell 12
train = pd.read_csv(CFG.path_train, usecols=CFG.label_columns)
train_votes = train[CFG.label_columns].to_numpy(dtype=np.float64)
train_votes = np.clip(train_votes, 0.0, None)
row_sums = train_votes.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
train_probs = train_votes / row_sums
prior = train_probs.mean(axis=0)
prior = np.clip(prior, 1e-12, 1.0)
prior = prior / prior.sum()

a = float(CFG.prior_blend)
if not (0.0 <= a <= 1.0):
    raise ValueError("CFG.prior_blend must be in [0,1].")

pred_arr = submission_eeg[CFG.label_columns].to_numpy(dtype=np.float64)
pred_arr = (1.0 - a) * pred_arr + a * prior[None, :]
pred_arr = np.clip(pred_arr, 1e-12, 1.0)
pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)
submission_eeg[CFG.label_columns] = pred_arr

print("Using prior:", dict(zip(CFG.label_columns, prior.round(6))))
submission_eeg[CFG.label_columns].sum(axis=1).describe()



## === cell 13
sub_map = submission_eeg.set_index("eeg_id")[CFG.label_columns]

submission_out = sample_submission[["eeg_id"]].copy()
for c in CFG.label_columns:
    submission_out[c] = submission_out["eeg_id"].map(sub_map[c])

miss = submission_out[CFG.label_columns].isna().any(axis=1)
if miss.any():
    submission_out.loc[miss, CFG.label_columns] = 1.0 / len(CFG.label_columns)

pred_arr = submission_out[CFG.label_columns].to_numpy(dtype=np.float64)
pred_arr = np.clip(pred_arr, 1e-12, 1.0)
pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)
submission_out[CFG.label_columns] = pred_arr

assert submission_out.shape[0] == sample_submission.shape[0]
assert list(submission_out.columns) == ["eeg_id"] + CFG.label_columns
row_sums = submission_out[CFG.label_columns].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums)) and np.allclose(row_sums, 1.0, atol=1e-6)

submission_out.head()



## === cell 14
submission_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_out.shape)



## === cell 15
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
