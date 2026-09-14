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

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the stray explanatory text that caused a SyntaxError, replace the merge‑based creation of the submission dataframe with a direct construction from the test metadata (guaranteeing the correct number of rows), and rewrite the final cell to use ordinary Python printing instead of a shell command. These minimal fixes prevent runtime errors, ensure a valid CSV of the proper length, and keep the original model‑based pipeline intact.'
- What this solution (achieved 1.41937) has done: 'I modify the dataset class so it no longer expects label columns in the test‑only dataframe; it return a dummy zero‑vector for y. This fixes the KeyError during DataLoader iteration and lets the pipeline run to produce a valid submission CSV. The rest of the logic (model loading, averaging, softmax, baseline fallback) remains unchanged, preserving the original workflow while moving the score toward the lower‑is‑better target.'
- What this solution (achieved 1.41937) has done: 'We keep the original pipeline unchanged but add a light regularisation step after the model (or baseline) predictions: blend the obtained probabilities with the overall training‑set baseline (70 % model, 30 % baseline) and renormalise each row so they sum to 1. This small smoothing usually reduces over‑confident mistakes and moves the KL‑divergence closer to the target lower score without altering the core architecture or training logic.'
- What this solution (achieved 1.41937) has done: 'I add a temperature scaling hyper‑parameter to soften the model logits during inference and adjust the blending ratio between model predictions and the class‑frequency baseline (moving it closer to 50 % each). This small change keeps the original pipeline intact while making the predictions less over‑confident, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I adjust the inference‑time calibration to bring the predictions closer to the target KL score: increase the temperature (softening the softmax) and give the class‑frequency baseline a larger weight, which typically reduces over‑confidence and lowers the divergence. These changes are confined to the configuration constants and keep the original modeling pipeline untouched.'
- What this solution (achieved 1.41937) has done: 'Implemented a higher temperature and stronger baseline blending to soften model predictions, which reduces over‑confidence and typically lowers KL‑divergence. Added a tiny epsilon clipping step after blending to avoid zero probabilities and renormalised rows to keep each prediction sum equal to 1.'
- What this solution (achieved 1.41937) has done: 'I adjust the inference‑time calibration to bring the predictions closer to the true label distribution: lower the temperature from 5.0 to 2.0 so the softmax is less overly smoothed, and shift the blending ratio to give 70 % weight to the model and 30 % to the class‑frequency baseline (instead of the previous 20/80). These small config tweaks keep the core model and data pipeline unchanged while reducing over‑confidence and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I adjust the inference‑time calibration to make the predictions less confident and rely more on the class‑frequency baseline, which should lower the KL‑divergence and move the score toward the target. Specifically, I increase the temperature back to 5.0 and set both the model and baseline blending weights to 0.5, keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'The update reduces the KL‑divergence by making the predictions less confident: the softmax temperature is increased and the baseline distribution is given a larger share of the final blend (80 % baseline / 20 % model). These tiny calibration tweaks keep the original pipeline intact while moving the score toward the lower‑is‑better target.'

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
    label_columns = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    temperature = 8.0  # soften model logits more
    weight_model = 0.2  # give less weight to the model
    weight_baseline = 0.8  # rely more on the class‑frequency baseline


train_df = pd.read_csv(CFG.path_train)
baseline_probs = train_df[CFG.label_columns].mean()
baseline_probs = baseline_probs / baseline_probs.sum()
CFG.baseline = baseline_probs

CFG.model_weights




## === cell 2
test_df = pd.read_csv(CFG.path_test)
submission = pd.DataFrame({"eeg_id": test_df["eeg_id"]})
submission["path"] = test_df["spectrogram_id"].map(
    lambda x: CFG.spec_dir / f"{x}.parquet"
)
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
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = pd.read_parquet(row.path)
        x = x.fillna(-1).values[:, 1:].T  # shape (H, W)
        x = preprocess(x)
        x = torch.Tensor(x[None, :])  # shape: (1, H, W)
        if self.transform:
            x = self.transform(x)
        y = torch.zeros(len(CFG.label_columns), dtype=torch.float32)
        return x, y




## === cell 4
data_ds = SpecDataset(df=submission)
data_loader = DataLoader(
    dataset=data_ds,
    batch_size=CFG.batch_size,
    num_workers=os.cpu_count(),
    pin_memory=True,
    shuffle=False,
)
data_loader




## === cell 5
x, y = next(iter(data_loader))
x.shape, y.shape




## === cell 6
plt.imshow(x[0, 0].cpu())
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
prediction = pd.DataFrame(0.0, columns=CFG.label_columns, index=submission.index)

if CFG.model_weights:
    for i, path_weight in enumerate(CFG.model_weights):
        print(f"Model {i}: {path_weight}")
        model.load_state_dict(torch.load(path_weight, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            res = []
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(DEVICE)
                logits = model(x_batch)
                pred = F.softmax(logits / CFG.temperature, dim=1)
                res.append(pred.cpu().numpy())
            res = np.concatenate(res)
            res_df = pd.DataFrame(
                res, columns=CFG.label_columns, index=submission.index
            )
            display(res_df.head())
            prediction += res_df
        print("\n")
    prediction = prediction / len(CFG.model_weights)
else:
    print("No model weights found – using baseline probabilities.")
    baseline_array = np.tile(CFG.baseline.values, (len(submission), 1))
    prediction = pd.DataFrame(
        baseline_array, columns=CFG.label_columns, index=submission.index
    )

baseline_df = pd.DataFrame(
    np.tile(CFG.baseline.values, (len(prediction), 1)),
    columns=CFG.label_columns,
    index=prediction.index,
)
prediction = (prediction * CFG.weight_model) + (baseline_df * CFG.weight_baseline)
prediction = prediction.clip(lower=1e-6)
prediction = prediction.div(prediction.sum(axis=1), axis=0)




## === cell 10
prediction




## === cell 11
submission[CFG.label_columns] = prediction
submission = submission[["eeg_id"] + CFG.label_columns]
submission




## === cell 12
output_path = pathlib.Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## === cell 13
print(submission.head())
