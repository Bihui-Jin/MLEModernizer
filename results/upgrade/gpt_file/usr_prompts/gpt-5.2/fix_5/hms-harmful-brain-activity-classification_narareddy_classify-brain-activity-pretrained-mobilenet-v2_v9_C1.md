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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
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
transformers==4.53.3

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

1.132911383265961

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the pipeline so it runs end-to-end and writes a valid `submission.csv` with exactly the same `eeg_id` set/length as `test.csv`, with rows summing to 1. The main runtime blocker is the missing external MobileNet checkpoint/config path; to preserve the overall approach (predicting class probabilities) without adding new data sources, I replace that with a small CNN defined locally that can train on your constructed 3-channel PCA “images”. I also fix the PCA feature indexing bug (you were hardcoding 224 PCA columns even though the number of clusters varies) and add the missing `ast` import. Finally, for safety and to avoid Kaggle “length mismatch” errors, I build the submission by merging predictions onto `sample_submission.csv` / `test.csv` order and renormalizing probabilities per row.'

# 9. Code solution

## === cell 0
import os
import ast
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

pd.set_option("display.max_columns", None)

data_path = "/kaggle/input/hms-harmful-brain-activity-classification"
sg_path = f"{data_path}/train_spectrograms"
sg_test_path = f"{data_path}/test_spectrograms"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 52):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(52)
device



## === cell 1
train_df_M = pd.read_csv(f"{data_path}/train.csv")
print(train_df_M.shape)
train_df_M.head()



## === cell 2
print(
    "Train Data:\n",
    train_df_M["expert_consensus"]
    .value_counts()
    .rename("Count")
    .to_frame()
    .assign(Percentage=lambda x: round((x / x.sum()) * 100)),
)
plot = (
    train_df_M["expert_consensus"]
    .value_counts()
    .plot(kind="bar", title="Distribution of expert_consensus")
)



## === cell 3
N_TRAIN_SAMPLE = 4000  # was 300
train_df = train_df_M.sample(N_TRAIN_SAMPLE, random_state=52).reset_index(drop=True)
print(
    "Train Sample Data:\n",
    train_df["expert_consensus"]
    .value_counts()
    .rename("Count")
    .to_frame()
    .assign(Percentage=lambda x: round((x / x.sum()) * 100)),
)



## === cell 4
_sg_train_cache = {}


def _load_train_sg(sgid: int) -> pd.DataFrame:
    df = _sg_train_cache.get(sgid)
    if df is None:
        df = pd.read_parquet(f"{sg_path}/{sgid}.parquet")
        _sg_train_cache[sgid] = df
    return df


sg_null_flags = np.zeros(len(train_df), dtype=bool)
for sgid, idxs in train_df.groupby("spectrogram_id").indices.items():
    parquet_df = _load_train_sg(int(sgid))
    times = parquet_df["time"].to_numpy()
    feat_df = parquet_df.drop(columns=["time", "spectrogram_id"], errors="ignore")

    nan_row = feat_df.isnull().any(axis=1).to_numpy()

    for i in idxs:
        off = float(train_df.at[i, "spectrogram_label_offset_seconds"])
        m = (times >= off) & (times < off + 600.0)
        sg_null_flags[i] = bool(nan_row[m].any())

train_df["SG_NullData_Ind"] = sg_null_flags

print(train_df["SG_NullData_Ind"].value_counts())
print("\ncheck expert consensus where sg is not null...")
print(
    train_df[train_df["SG_NullData_Ind"] == False]["expert_consensus"]
    .value_counts()
    .rename("Count")
    .to_frame()
    .assign(Percentage=lambda x: round((x / x.sum()) * 100))
)



## === cell 5
train_df = train_df[train_df["SG_NullData_Ind"] != True]
train_df.shape



## === cell 6
train_df = train_df.sort_values(by="spectrogram_id")
sg_records_referred_list = []
spectrogram_id = None

for index, row in train_df.iterrows():
    if row["spectrogram_id"] != spectrogram_id:
        spectrogram_id = int(row["spectrogram_id"])
        parquet_df = _load_train_sg(spectrogram_id)

    off = float(row["spectrogram_label_offset_seconds"])
    filtered_df = parquet_df[
        (parquet_df["time"] >= off) & (parquet_df["time"] < off + 600)
    ].copy()
    filtered_df["spectrogram_id"] = spectrogram_id
    sg_records_referred_list.append(filtered_df)

sg_records_df = pd.concat(sg_records_referred_list, ignore_index=True)
sg_records_df = sg_records_df.drop_duplicates()

print(sg_records_df.shape)
sg_records_df.head()



## === cell 7
corr = sg_records_df.drop(columns=["time", "spectrogram_id"]).corr()
corr.style.background_gradient(cmap="coolwarm")



## === cell 8
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

linked = linkage(corr, "ward")
plt.figure(figsize=(50, 5))
dendrogram(linked, labels=corr.index.tolist(), leaf_rotation=90)

thresholds = [1, 2, 3, 4]
for thr in thresholds:
    plt.axhline(y=thr, color="r", linestyle="--")

plt.title("Hierarchical Clustering Dendrogram with Multiple Thresholds")
plt.xlabel("Feature")
plt.ylabel("Distance")
plt.show()

for thr in thresholds:
    clusters = fcluster(linked, thr, criterion="distance")
    num_unique_clusters = len(set(clusters))
    print(f"Number of unique features at threshold {thr}: {num_unique_clusters}")



## === cell 9
thr = 0.4286  # this 0.264 if you run on entire train SG records
clusters = fcluster(linked, thr, criterion="distance")

num_unique_clusters = len(set(clusters))
print(f"Number of unique features at threshold {thr}: {num_unique_clusters}")

cluster_assignments = pd.DataFrame({"Feature": corr.columns, "Cluster": clusters})
grouped_features = cluster_assignments.groupby("Cluster")["Feature"].apply(list)
grouped_features



## === cell 10
cluster_models = {}  # cluster_label -> (features, scaler, pca)

keys = sg_records_df[["spectrogram_id", "time"]].reset_index(drop=True)
pc_blocks = []

for cluster_label, features in grouped_features.items():
    cluster_data = sg_records_df[features]

    scaler = StandardScaler()
    cluster_data_standardized = scaler.fit_transform(cluster_data)

    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(cluster_data_standardized)

    cluster_models[int(cluster_label)] = (features, scaler, pca)

    pc_blocks.append(
        pd.DataFrame(principal_component, columns=[f"PCA_{cluster_label}"])
    )

principal_components_df = pd.concat(pc_blocks, axis=1)
final_pc_df = pd.concat([keys, principal_components_df], axis=1)

pca_cols = [c for c in final_pc_df.columns if c.startswith("PCA_")]
pca_cols_sorted = sorted(pca_cols, key=lambda x: int(x.split("_")[1]))
print(
    "PCA columns:",
    len(pca_cols_sorted),
    "min/max:",
    pca_cols_sorted[0],
    pca_cols_sorted[-1],
)

print(final_pc_df.shape)
final_pc_df.head()



## === cell 11
vote_columns = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
prob_columns = [
    "seizure_prob",
    "lpd_prob",
    "gpd_prob",
    "lrda_prob",
    "grda_prob",
    "other_prob",
]

train_df["total_votes"] = train_df[vote_columns].sum(axis=1)
for vote_col, prob_col in zip(vote_columns, prob_columns):
    train_df[prob_col] = train_df[vote_col] / train_df["total_votes"]
train_df.head()



## === cell 12
final_pc_df = final_pc_df.sort_values(["spectrogram_id", "time"]).reset_index(drop=True)

_pc_by_sgid = {}
for sgid, idxs in final_pc_df.groupby("spectrogram_id").indices.items():
    idxs = np.asarray(idxs, dtype=np.int64)
    times = final_pc_df.loc[idxs, "time"].to_numpy()
    X = final_pc_df.loc[idxs, pca_cols_sorted].to_numpy(dtype=np.float32, copy=False)
    _pc_by_sgid[int(sgid)] = (times, X)

processed_data = {}
EmptySGs = []

for _, row in train_df.iterrows():
    label_id = row["label_id"]
    spectrogram_id = int(row["spectrogram_id"])
    offset_seconds = (
        float(row["spectrogram_label_offset_seconds"]) + 76
    )  # original logic
    labels = row[prob_columns]

    if spectrogram_id not in _pc_by_sgid:
        EmptySGs.append(spectrogram_id)
        continue

    times, X = _pc_by_sgid[spectrogram_id]
    start = np.searchsorted(times, offset_seconds, side="left")
    end = np.searchsorted(times, offset_seconds + 448.0, side="left")
    if end - start != 224:
        EmptySGs.append(spectrogram_id)
        continue

    window = X[start:end]  # (224, C_pca)
    if np.isnan(window).any():
        EmptySGs.append(spectrogram_id)
        continue

    processed_data[label_id] = {
        "spectrogram_id": spectrogram_id,
        "patient_id": row["patient_id"],
    }
    for label in labels.index:
        processed_data[label_id][label] = float(labels[label])

    wT = window.T
    for j, c in enumerate(pca_cols_sorted):
        processed_data[label_id][c] = wT[j].tolist()

final_training_data_byPCA = pd.DataFrame.from_dict(processed_data, orient="index")
final_training_data_byPCA.reset_index(inplace=True)  # 'index' contains label_id

print(final_training_data_byPCA.shape)
print("Null or incorrect size #: ", len(EmptySGs))
final_training_data_byPCA.head()




## === cell 13
def transform_to_3channel_with_zeros(dataframe, pca_columns):
    N = len(dataframe)
    C = len(pca_columns)
    T = 224

    arr = np.empty((N, C, T), dtype=np.float32)
    for j, col in enumerate(pca_columns):
        col_vals = dataframe[col].values
        for i in range(N):
            v = col_vals[i]
            if isinstance(v, str):
                v = ast.literal_eval(v)
            arr[i, j, :] = np.asarray(v, dtype=np.float32)

    zeros = np.zeros_like(arr)
    out = np.stack([arr, zeros, zeros], axis=1)  # (N,3,C,T)
    return torch.from_numpy(out)


images_tensor = transform_to_3channel_with_zeros(
    final_training_data_byPCA, pca_cols_sorted
)
print(images_tensor.shape)  # (N, 3, C, 224)
images_tensor[0]



## === cell 14
labels = final_training_data_byPCA[prob_columns].values.astype(np.float32)
groups = final_training_data_byPCA["patient_id"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=52)
train_idx, val_idx = next(gss.split(images_tensor, labels, groups=groups))

train_images = images_tensor[train_idx]
val_images = images_tensor[val_idx]
train_labels = labels[train_idx]
val_labels = labels[val_idx]

print("** distribution of highest prob labels in train - check if it is skewed **")
print(
    pd.DataFrame(train_labels, columns=prob_columns)[prob_columns]
    .idxmax(axis=1)
    .value_counts()
)
print(
    "\n** distribution of highest prob labels in validation - check if it is skewed **"
)
print(
    pd.DataFrame(val_labels, columns=prob_columns)[prob_columns]
    .idxmax(axis=1)
    .value_counts()
)




## === cell 15
class SmallCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits


model = SmallCNN(num_classes=6).to(device)
model



## === cell 16
train_dataset = TensorDataset(
    train_images, torch.tensor(train_labels, dtype=torch.float32)
)
val_dataset = TensorDataset(val_images, torch.tensor(val_labels, dtype=torch.float32))

num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
)

criterion = torch.nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.00001)



## === cell 17
train_losses = []
val_losses = []

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, labels_b in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device, non_blocking=True)
        labels_b = labels_b.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(F.log_softmax(logits, dim=1), labels_b)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    avg_train_loss = total_loss / max(1, len(train_loader))
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels_b in val_loader:
            images = images.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(F.log_softmax(logits, dim=1), labels_b)
            total_val_loss += float(loss.item())

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
    )



## === cell 18
plt.figure(figsize=(8, 3))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("KLDivLoss")
plt.legend()
plt.show()



## === cell 19
import sys



## === cell 20
del (
    sg_records_df,
    principal_components_df,
    train_df_M,
)



## === cell 21
test_df = pd.read_csv(f"{data_path}/test.csv")
print(test_df.shape)
test_df.head()




## === cell 22
def build_test_pc_df_for_spectrogram(spectrogram_id: int, cache: dict):
    if spectrogram_id in cache:
        return cache[spectrogram_id]

    parquet_path = f"{sg_test_path}/{spectrogram_id}.parquet"
    parquet_df = pd.read_parquet(parquet_path)

    out = pd.DataFrame(
        {"spectrogram_id": parquet_df["spectrogram_id"], "time": parquet_df["time"]}
    )
    for cluster_label in sorted(cluster_models.keys()):
        features, scaler, pca = cluster_models[cluster_label]
        X = parquet_df[features].values
        Xs = scaler.transform(X)
        pc = pca.transform(Xs).astype(np.float32).reshape(-1)
        out[f"PCA_{cluster_label}"] = pc

    cache[spectrogram_id] = out
    return out


def make_image_tensor_from_pc_window(
    pc_df: pd.DataFrame, pca_cols_sorted, offset_seconds: float
):
    mask = (pc_df["time"] >= offset_seconds) & (pc_df["time"] < offset_seconds + 448)
    temp_df = pc_df.loc[mask, ["time"] + pca_cols_sorted].reset_index(drop=True)

    if temp_df.shape[0] != 224:
        return None
    if temp_df[pca_cols_sorted].isnull().sum().sum() != 0:
        return None

    single_channel_img = (
        temp_df[pca_cols_sorted].to_numpy(dtype=np.float32).T
    )  # (C,224)
    zeros_channel = np.zeros_like(single_channel_img)
    three_channel_img = np.stack(
        [single_channel_img, zeros_channel, zeros_channel], axis=0
    )  # (3,C,224)
    return torch.from_numpy(three_channel_img)


model.eval()
candidate_offsets = [0.0, 38.0, 76.0, 114.0, 152.0]
best_offset = 76.0
best_val_kl = float("inf")

with torch.no_grad():
    for off in candidate_offsets:
        total = 0.0
        count = 0
        if off != 76.0:
            continue

        Xv = val_images.to(device)
        yv = torch.tensor(val_labels, dtype=torch.float32, device=device)
        logits = model(Xv)
        loss = criterion(F.log_softmax(logits, dim=1), yv)
        total = float(loss.item()) * 1.0
        count = 1

        avg = total / count
        if avg < best_val_kl:
            best_val_kl = avg
            best_offset = off

print(
    "Selected offset_seconds for test inference:",
    best_offset,
    " (val KL approx:",
    best_val_kl,
    ")",
)

sample_sub = pd.read_csv(f"{data_path}/sample_submission.csv")
uniform = np.array([1 / 6] * 6, dtype=np.float32)

pc_cache = {}
preds = np.zeros((len(test_df), 6), dtype=np.float32)

test_sgid = test_df["spectrogram_id"].astype(int).to_numpy()
unique_sgids, inv = np.unique(test_sgid, return_inverse=True)

with torch.no_grad():
    for sgid in tqdm(unique_sgids, desc="Infer test (by spectrogram)"):
        rows_idx = np.where(test_sgid == sgid)[0]
        try:
            pc_df = build_test_pc_df_for_spectrogram(int(sgid), pc_cache)
            img = make_image_tensor_from_pc_window(
                pc_df, pca_cols_sorted, float(best_offset)
            )
            if img is None:
                preds[rows_idx] = uniform
                continue

            x = img.unsqueeze(0).to(device)
            logits = model(x)
            prob = (
                F.softmax(logits, dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                .reshape(-1)
            )
            s = float(prob.sum())
            p = prob / s if s > 0 else uniform
            preds[rows_idx] = p
        except Exception:
            preds[rows_idx] = uniform

test_preds = pd.DataFrame(
    {
        "eeg_id": test_df["eeg_id"].values,
        "seizure_vote": preds[:, 0],
        "lpd_vote": preds[:, 1],
        "gpd_vote": preds[:, 2],
        "lrda_vote": preds[:, 3],
        "grda_vote": preds[:, 4],
        "other_vote": preds[:, 5],
    }
)

test_preds = sample_sub[["eeg_id"]].merge(test_preds, on="eeg_id", how="left")
test_preds[vote_columns] = test_preds[vote_columns].fillna(1 / 6)

prob_sum = test_preds[vote_columns].sum(axis=1).values.reshape(-1, 1)
test_preds[vote_columns] = test_preds[vote_columns] / prob_sum
test_preds.head()



## === cell 23
out_path = "submission.csv"
test_preds.to_csv(out_path, index=False)
print(f"submission file generated: {out_path} with shape {test_preds.shape}")
print("Column check:", list(test_preds.columns))
print(
    "Row-sum min/max:",
    test_preds[vote_columns].sum(axis=1).min(),
    test_preds[vote_columns].sum(axis=1).max(),
)
print("Any NaNs:", test_preds.isna().any().any())
print("Unique eeg_ids:", test_preds["eeg_id"].nunique(), " expected:", len(test_preds))
