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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

1.033637338537991

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the run-stopping issues by (1) removing the hard GPU-only exit so the notebook can run on CPU too, and (2) handling the missing pretrained weights file by falling back to a deterministic, label-distribution baseline (computed from train vote proportions) when the `.pth` isn’t available. I also fix the submission length/alignment bug by generating predictions strictly from `test.csv` order (not from `os.listdir` order) and by ensuring every `eeg_id` in `test.csv` gets exactly one row. Finally, I guarantee probabilities are valid (non-negative and row-sum exactly 1) and write `submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'I fix the merge error by making `eeg_id` dtypes consistent (use `int64` everywhere) so the left-join succeeds and the notebook writes a valid `submission.csv`. I also ensure the `eeg_id` collected from the DataLoader is converted to an integer and that the prediction dataframe is aligned 1:1 with `test.csv` order without relying on an extra merge (still keeping the current logic intact). This is score-neutral (it only fixes I/O/alignment), and preserves your baseline/model behavior exactly. Finally, I keep the probability normalization and add a uniqueness check to prevent accidental duplicate IDs.'
- What this solution (achieved 1.39779) has done: 'We fix the duplicate-`eeg_id` failure by ensuring the inference loop produces exactly one prediction per `eeg_id` in `test.csv`. Since `test.csv` can contain repeated `eeg_id`s, we deduplicate `test_df` (keeping first occurrence) before building the Dataset/DataLoader, and then expand back to the original `sample_submission.csv`/`test.csv` order via a left-merge (so submission length stays 9850). This is a correctness/alignment fix (not a model change) and should also slightly improve the KL score because we avoid accidental repeated/overwritten IDs and ensure every row is filled deterministically. The rest of the model/baseline logic and probability normalization is preserved.'
- What this solution (achieved 1.48384) has done: 'Your current score (1.39779, lower-is-better) is worse than the target (1.0336), so we should cautiously improve without changing your model or training (there is none) and while keeping your inference semantics. The biggest legitimate gain with minimal change is to compute the fallback prior in a way that better matches the competition’s row-level target distribution: aggregate train votes by `eeg_id` first (so overlapped subsamples don’t overweight some recordings), then average normalized per-`eeg_id` distributions. Additionally, we blend that prior slightly with uniform (tiny smoothing) to reduce KL penalties from extreme small probabilities, while still producing valid probabilities. Everything else (checkpoint usage, dataset loading, dedup/merge expansion, normalization, and submission writing) stays the same.'
- What this solution (achieved 1.40995) has done: 'I keep your architecture and inference logic intact and only adjust the fallback “train-prior baseline” to better match the per-`eeg_id` distribution used by the competition: first aggregate votes per `eeg_id`, normalize, then apply Dirichlet-style additive smoothing in vote space before renormalizing. This reduces extreme near-zero probabilities that are heavily penalized by KL divergence, typically improving the score while preserving the semantics of a global prior. I also make the inference pipeline deterministic and slightly more stable by using a larger batch size on CPU/GPU (no semantic change) and enforcing float64 normalization consistently. The submission schema, row alignment to `test.csv`, and probability-sum-to-one constraints remain unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



## === cell 1
import os
from pathlib import Path

PATH_IN = Path("/kaggle/input/")
PATH_TMP = Path("/kaggle/temp/")
PATH_OUT = Path("/kaggle/working/")




## === cell 2
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


seed_everything(42)



## === cell 3
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device



## === cell 4
COMP_PATH = PATH_IN / "hms-harmful-brain-activity-classification"
TRAIN_CSV = COMP_PATH / "train.csv"
TEST_CSV = COMP_PATH / "test.csv"
SAMPLE_SUB = COMP_PATH / "sample_submission.csv"
TEST_EEG_DIR = COMP_PATH / "test_eegs"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

assert TRAIN_CSV.exists(), f"Missing: {TRAIN_CSV}"
assert TEST_CSV.exists(), f"Missing: {TEST_CSV}"
assert SAMPLE_SUB.exists(), f"Missing: {SAMPLE_SUB}"
assert TEST_EEG_DIR.exists(), f"Missing: {TEST_EEG_DIR}"




## === cell 5
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=5, stride=1)
        self.batch_norm1 = nn.BatchNorm1d(num_features=19)
        self.leaky_relu = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=5, stride=1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=19)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv3 = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=5, stride=1)
        self.batch_norm3 = nn.BatchNorm1d(num_features=19)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=19, hidden_size=19, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc = nn.Linear(in_features=19, out_features=6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.max_pool3(x)

        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]

        x = self.dropout1(x)

        x = self.fc(x)
        x = self.softmax(x)

        return x




## === cell 6
model = CustomModel().to(device)

ckpt_path = PATH_IN / "beca-2" / "custom_net_model.pth"
has_ckpt = ckpt_path.exists()

if has_ckpt:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
else:
    model.eval()  # keep consistent; model won't be used for baseline
    print(
        f"WARNING: checkpoint not found at {ckpt_path}. Will use a train-prior baseline prediction."
    )



## === cell 7
train_df = pd.read_csv(TRAIN_CSV, usecols=["eeg_id"] + TARGET_COLS)
train_df["eeg_id"] = train_df["eeg_id"].astype(np.int64)

agg_votes = train_df.groupby("eeg_id", sort=False)[
    TARGET_COLS
].sum()  # per recording total votes
V = agg_votes.to_numpy(dtype=np.float64)
row_sums = V.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
P = V / row_sums  # per-eeg distribution

k = len(TARGET_COLS)
pseudo_count = 1.0  # small, safe; typically improves KL vs purely empirical mean
P_smooth = (V + pseudo_count) / (row_sums + pseudo_count * k)

prior = P_smooth.mean(axis=0)
prior = np.clip(prior, 1e-12, None)
prior = prior / prior.sum()
prior




## === cell 8
class TestDataset(Dataset):
    def __init__(self, test_df: pd.DataFrame, data_dir: Path):
        self.test_df = test_df.reset_index(drop=True).copy()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        eeg_id = int(self.test_df.loc[idx, "eeg_id"])
        fp = self.data_dir / f"{eeg_id}.parquet"

        try:
            df = pd.read_parquet(fp)
            if "EKG" in df.columns:
                df = df.drop("EKG", axis=1)
            x = df.to_numpy(dtype=np.float32)  # shape (T, C)
            if x.ndim != 2 or x.shape[1] != 19:
                x = np.zeros((10000, 19), dtype=np.float32)
        except Exception:
            x = np.zeros((10000, 19), dtype=np.float32)

        x = torch.tensor(x, dtype=torch.float32)
        dummy_label = torch.tensor(1, dtype=torch.long)
        return x, dummy_label, eeg_id




## === cell 9
test_df_full = pd.read_csv(TEST_CSV, usecols=["eeg_id"])
test_df_full["eeg_id"] = test_df_full["eeg_id"].astype(np.int64)

test_df_unique = test_df_full.drop_duplicates(
    subset=["eeg_id"], keep="first"
).reset_index(drop=True)

test_dataset = TestDataset(test_df_unique, TEST_EEG_DIR)

bs = 8 if torch.cuda.is_available() else 4
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)

(len(test_df_full), len(test_df_unique), len(test_dataset), test_df_unique.shape)




## === cell 10
def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s




## === cell 11
predictions = np.zeros((len(test_dataset), 6), dtype=np.float64)
eeg_ids = []

use_model = has_ckpt and (model is not None)

with torch.no_grad():
    row = 0
    for inputs, labels, eeg_id in test_loader:
        if torch.is_tensor(eeg_id):
            eeg_id_batch = [int(x) for x in eeg_id.detach().cpu().numpy().tolist()]
        else:
            eeg_id_batch = [int(x) for x in eeg_id]

        bsz = len(eeg_id_batch)
        eeg_ids.extend(eeg_id_batch)

        if use_model:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.detach().cpu().numpy()
            probs = normalize_probs(probs)
            predictions[row : row + bsz] = probs
        else:
            predictions[row : row + bsz] = prior

        row += bsz

predictions = normalize_probs(predictions)
predictions.shape, len(eeg_ids), len(set(eeg_ids))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1685115486.py in <cell line: 0>()
      6 with torch.no_grad():
      7     row = 0
----> 8     for inputs, labels, eeg_id in test_loader:
      9         # Collect eeg_ids for the batch
     10         if torch.is_tensor(eeg_id):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     53         else:
     54             data = self.dataset[possibly_batched_index]
---> 55         return self.collate_fn(data)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
    396         >>> default_collate(batch)  # Handle `CustomType` automatically
    397     """
--> 398     return collate(batch, collate_fn_map=default_collate_fn_map)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    209 
    210         if isinstance(elem, tuple):
--> 211             return [
    212                 collate(samples, collate_fn_map=collate_fn_map)
    213                 for samples in transposed

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in <listcomp>(.0)
    210         if isinstance(elem, tuple):
    211             return [
--> 212                 collate(samples, collate_fn_map=collate_fn_map)
    213                 for samples in transposed
    214             ]  # Backwards compatibility.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    153     if collate_fn_map is not None:
    154         if elem_type in collate_fn_map:
--> 155             return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
    156 
    157         for collate_type in collate_fn_map:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate_tensor_fn(batch, collate_fn_map)
    270         storage = elem._typed_storage()._new_shared(numel, device=elem.device)
    271         out = elem.new(storage).resize_(len(batch), *list(elem.size()))
--> 272     return torch.stack(batch, 0, out=out)
    273 
    274 

RuntimeError: stack expects each tensor to be equal size, but got [16400, 19] at entry 0 and [15600, 19] at entry 1

## === cell 12
sub = pd.read_csv(SAMPLE_SUB, usecols=["eeg_id"] + TARGET_COLS)
sub["eeg_id"] = sub["eeg_id"].astype(np.int64)

pred_df = pd.DataFrame(predictions, columns=TARGET_COLS)
pred_df.insert(0, "eeg_id", np.asarray(eeg_ids, dtype=np.int64))

assert pred_df[
    "eeg_id"
].is_unique, "Duplicate eeg_id predictions detected (should be unique now)."
assert len(pred_df) == len(test_df_unique)
assert np.array_equal(pred_df["eeg_id"].to_numpy(), test_df_unique["eeg_id"].to_numpy())

sub = sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left", validate="many_to_one")

missing = sub[TARGET_COLS].isna().any(axis=1)
if missing.any():
    sub.loc[missing, TARGET_COLS] = prior

sub[TARGET_COLS] = normalize_probs(sub[TARGET_COLS].to_numpy(dtype=np.float64))
sub



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1672019849.py in <cell line: 0>()
      3 
      4 pred_df = pd.DataFrame(predictions, columns=TARGET_COLS)
----> 5 pred_df.insert(0, "eeg_id", np.asarray(eeg_ids, dtype=np.int64))
      6 
      7 assert pred_df[

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in insert(self, loc, column, value, allow_duplicates)
   5169             value = value.iloc[:, 0]
   5170 
-> 5171         value, refs = self._sanitize_column(value)
   5172         self._mgr.insert(loc, column, value, refs=refs)
   5173 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (1693)

## === cell 13
out_path = PATH_OUT / "submission.csv"
sub.to_csv(out_path, index=False)

assert out_path.exists() and out_path.suffix == ".csv"
assert (
    sub.shape[0] == pd.read_csv(TEST_CSV).shape[0]
), "Submission and test must have same length"
row_sums = sub[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert np.allclose(row_sums, 1.0, atol=1e-6), "Each row must sum to 1"

out_path, sub.head()
