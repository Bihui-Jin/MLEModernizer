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
librosa==0.11.0
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
PyWavelets==1.8.0
sklearn-pandas==2.2.0
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

0.4036638285480033

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fixed the undefined `DATA_TYPE` by giving `DataGenerator` a sensible default, and simplified test inference to use only the spectrogram data (avoiding the heavy EEG loading). This eliminates the NameError and ensures a valid `submission.csv` is written, while keeping the original modeling logic unchanged.'
- What this solution (achieved 1.41937) has done: 'I fixed the group‑by sum error, corrected the column rename, and ensured the fallback prediction logic runs when no model files are found. These changes let the script load data, generate predictions, normalize them, and write a valid `submission.csv` without runtime errors, moving the pipeline toward the target score.'

# 9. Code solution

## === cell 0
import glob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

TRAIN_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(TRAIN_PATH)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

baseline_counts = train_df[TARGETS].mean()
baseline_probs = baseline_counts / baseline_counts.sum()

eeg_sum = train_df.groupby("eeg_id")[TARGETS].sum()
row_totals = eeg_sum.sum(axis=1)
eeg_probs = eeg_sum.div(row_totals, axis=0).fillna(baseline_probs)

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)
test.head()

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquet files")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(os.path.join(PATH2, f))
    spectrograms2[int(f.split(".")[0])] = tmp.iloc[:, 1:].values

test = test.rename(columns={"spectrogram_id": "spec_id"})
print("Finished loading data.")




## === cell 1
class DataGenerator:
    "Generates data for Keras"

    def __init__(
        self,
        data,
        specs=None,
        eeg_specs=None,
        raw_eegs=None,
        augment=False,
        mode="train",
        data_type="both",  # default to both when not specified
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.on_epoch_end()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X, y = self.data_generation(index)
        if self.augment:
            X = self.augmentation(X)
        return X, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def data_generation(self, index):
        if self.data_type == "both":
            X, y = self.generate_all_specs(index)
        elif self.data_type in ("eeg", "kaggle"):
            X, y = self.generate_specs(index)
        elif self.data_type == "raw":
            X, y = self.generate_raw(index)
        else:
            raise ValueError(f"Unsupported data_type {self.data_type}")
        return X, y

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        row = self.data.iloc[index]
        offset = 0 if self.mode == "test" else int(row.offset / 2)
        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]
        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)
        mn, mx = img.min(), img.max()
        img = 255 * (img - mn) / (mx - mn + 1e-5)

        X[56:156, :256, 0] = img[:, 22:-22, 0]
        X[156:256, :256, 0] = img[:, 22:-22, 2]
        X[56:156, :256, 1] = img[:, 22:-22, 1]
        X[156:256, :256, 1] = img[:, 22:-22, 3]
        X[56:156, :256, 2] = img[:, 22:-22, 2]
        X[156:256, :256, 2] = img[:, 22:-22, 1]

        X[56:156, 256:, 0] = img[:, 22:-22, 0]
        X[156:256, 256:, 0] = img[:, 22:-22, 2]
        X[56:156, 256:, 1] = img[:, 22:-22, 1]
        X[156:256, 256:, 1] = img[:, 22:-22, 3]

        img = eeg
        mn, mx = img.min(), img.max()
        img = 255 * (img - mn) / (mx - mn + 1e-5)

        X[356:456, :256, 0] = img[:, 22:-22, 0]
        X[456:556, :256, 0] = img[:, 22:-22, 2]
        X[356:456, :256, 1] = img[:, 22:-22, 1]
        X[456:556, :256, 1] = img[:, 22:-22, 3]
        X[356:456, :256, 2] = img[:, 22:-22, 2]
        X[456:556, :256, 2] = img[:, 22:-22, 1]

        X[356:456, 256:, 0] = img[:, 22:-22, 0]
        X[456:556, 256:, 0] = img[:, 22:-22, 2]
        X[356:456, 256:, 1] = img[:, 22:-22, 1]
        X[456:556, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]
        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")
        row = self.data.iloc[index]
        offset = 0 if self.mode == "test" else int(row.offset / 2)
        if self.data_type == "eeg":
            img = self.eeg_specs[row.eeg_id]
        else:  # kaggle
            spec = self.specs[row.spec_id]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]
            img = np.stack(imgs, axis=-1)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
        mn, mx = img.min(), img.max()
        img = 255 * (img - mn) / (mx - mn + 1e-5)

        X[56:156, :256, 0] = img[:, 22:-22, 0]
        X[156:256, :256, 0] = img[:, 22:-22, 2]
        X[56:156, :256, 1] = img[:, 22:-22, 1]
        X[156:256, :256, 1] = img[:, 22:-22, 3]
        X[56:156, :256, 2] = img[:, 22:-22, 2]
        X[156:256, :256, 2] = img[:, 22:-22, 1]

        X[56:156, 256:, 0] = img[:, 22:-22, 0]
        X[156:256, 256:, 0] = img[:, 22:-22, 1]
        X[56:156, 256:, 1] = img[:, 22:-22, 2]
        X[156:256, 256:, 1] = img[:, 22:-22, 3]

        X[356:456, :256, 0] = img[:, 22:-22, 0]
        X[456:556, :256, 0] = img[:, 22:-22, 1]
        X[356:456, :256, 1] = img[:, 22:-22, 2]
        X[456:556, :256, 1] = img[:, 22:-22, 3]
        X[356:456, :256, 2] = img[:, 22:-22, 3]
        X[456:556, :256, 2] = img[:, 22:-56, 2]

        X[356:456, 256:, 0] = img[:, 22:-22, 0]
        X[456:556, 256:, 0] = img[:, 22:-22, 2]
        X[356:456, 256:, 1] = img[:, 22:-22, 1]
        X[456:556, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]
        return X, y




## === cell 2
def run_inference_loop(model, test_gen, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm.tqdm(test_gen):
            batch_data = torch.from_numpy(batch_data).float()
            batch_data = batch_data.permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(batch_data)
            pred_list.append(out.softmax(dim=1).cpu().numpy())
    pred_arr = np.concatenate(pred_list, axis=0)
    return pred_arr


test_gen = DataGenerator(
    test,
    mode="test",
    data_type="kaggle",
    specs=spectrograms2,
    eeg_specs=None,  # not needed for kaggle mode
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = glob.glob("/kaggle/input/model90-both-5model/model_90/*/*.pt")
preds = []
if model_paths:
    for model_path in model_paths:
        print("Loading model:", model_path)
        model = torch.load(model_path, map_location=device)
        pred = run_inference_loop(model, test_gen, device)
        preds.append(pred)

if preds:
    test_pred = np.mean(preds, axis=0)
else:
    print("No model files found – using per‑eeg_id or baseline probabilities.")
    prob_rows = []
    for eid in test["eeg_id"]:
        if eid in eeg_probs.index:
            prob_rows.append(eeg_probs.loc[eid].values)
        else:
            prob_rows.append(baseline_probs.values)
    test_pred = np.vstack(prob_rows)

test_pred = 0.9 * test_pred + 0.1 * baseline_probs.values

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

row_sums = test_pred_df[CLASSES].sum(axis=1)
test_pred_df[CLASSES] = test_pred_df[CLASSES].div(row_sums, axis=0)

test_pred_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
test_pred_df.head()
