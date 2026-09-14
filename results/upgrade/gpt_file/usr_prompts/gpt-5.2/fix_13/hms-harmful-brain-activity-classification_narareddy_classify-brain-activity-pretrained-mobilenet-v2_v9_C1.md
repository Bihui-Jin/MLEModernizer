# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


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
N_TRAIN_SAMPLE = None  # was 4000
if N_TRAIN_SAMPLE is None:
    train_df = train_df_M.reset_index(drop=True)
else:
    train_df = train_df_M.sample(N_TRAIN_SAMPLE, random_state=52).reset_index(drop=True)

print("train_df shape:", train_df.shape)
print(
    "Train Sample Data:\n",
    train_df["expert_consensus"]
    .value_counts()
    .rename("Count")
    .to_frame()
    .assign(Percentage=lambda x: round((x / x.sum()) * 100)),
)



## === cell 4
train_df["SG_NullData_Ind"] = False
print(train_df["SG_NullData_Ind"].value_counts())



## === cell 5
train_df = train_df[train_df["SG_NullData_Ind"] != True]
train_df.shape



## === cell 6
from collections import OrderedDict

_SG_CACHE_MAX = 256  # safe bound; correctness unchanged
_sg_train_cache = OrderedDict()


def _load_train_sg(sgid: int) -> pd.DataFrame:
    df = _sg_train_cache.get(sgid)
    if df is not None:
        _sg_train_cache.move_to_end(sgid)
        return df
    df = pd.read_parquet(f"{sg_path}/{sgid}.parquet")
    _sg_train_cache[sgid] = df
    if len(_sg_train_cache) > _SG_CACHE_MAX:
        _sg_train_cache.popitem(last=False)
    return df


MAX_SGIDS_FOR_FIT = 200  # keep as provided
ROWS_PER_SGID_FOR_FIT = 120  # keep as provided

unique_train_sgids = train_df["spectrogram_id"].astype(int).unique()
rng = np.random.default_rng(52)
if len(unique_train_sgids) > MAX_SGIDS_FOR_FIT:
    fit_sgids = rng.choice(unique_train_sgids, size=MAX_SGIDS_FOR_FIT, replace=False)
else:
    fit_sgids = unique_train_sgids

sg_fit_blocks = []
for sgid in tqdm(fit_sgids, desc="Load SGs for corr/PCA fit"):
    df = _load_train_sg(int(sgid))
    feat_df = df.drop(columns=["time", "spectrogram_id"], errors="ignore")
    if len(feat_df) == 0:
        continue
    if len(feat_df) > ROWS_PER_SGID_FOR_FIT:
        idx = rng.choice(len(feat_df), size=ROWS_PER_SGID_FOR_FIT, replace=False)
        feat_df = feat_df.iloc[idx]
    sg_fit_blocks.append(feat_df)

sg_fit_df = pd.concat(sg_fit_blocks, ignore_index=True)
sg_fit_df = sg_fit_df.replace([np.inf, -np.inf], np.nan)
sg_fit_col_means = sg_fit_df.mean(axis=0, skipna=True)
sg_fit_df = sg_fit_df.fillna(sg_fit_col_means)

print("sg_fit_df shape (for corr/PCA fit):", sg_fit_df.shape)
sg_fit_df.head()



## === cell 7
corr = sg_fit_df.corr()
print("corr shape:", corr.shape)



## === cell 8
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

linked = linkage(corr, "ward")

thresholds = [1, 2, 3, 4]
for thr in thresholds:
    clusters = fcluster(linked, thr, criterion="distance")
    num_unique_clusters = len(set(clusters))
    print(f"Number of unique features at threshold {thr}: {num_unique_clusters}")



## === cell 9
thr = 0.4286  # keep as-is
clusters = fcluster(linked, thr, criterion="distance")

num_unique_clusters = len(set(clusters))
print(f"Number of unique features at threshold {thr}: {num_unique_clusters}")

cluster_assignments = pd.DataFrame({"Feature": corr.columns, "Cluster": clusters})
grouped_features = cluster_assignments.groupby("Cluster")["Feature"].apply(list)
grouped_features



## === cell 10
cluster_models = {}  # cluster_label -> (features, scaler, pca)

for cluster_label, features in grouped_features.items():
    cluster_data = sg_fit_df[features].copy()
    cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).fillna(
        cluster_data.mean(axis=0, skipna=True)
    )

    scaler = StandardScaler()
    cluster_data_standardized = scaler.fit_transform(cluster_data)

    pca = PCA(n_components=1, random_state=52)
    _ = pca.fit_transform(cluster_data_standardized)

    cluster_models[int(cluster_label)] = (features, scaler, pca)

pca_cols_sorted = [f"PCA_{int(k)}" for k in sorted(cluster_models.keys(), key=int)]
print(
    "Num PCA columns:",
    len(pca_cols_sorted),
    "first/last:",
    pca_cols_sorted[0],
    pca_cols_sorted[-1],
)



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
_all_feat_cols = list(sg_fit_df.columns)
_feat2idx = {c: i for i, c in enumerate(_all_feat_cols)}

_global_feat_means = sg_fit_col_means.reindex(_all_feat_cols).to_numpy(
    dtype=np.float32, copy=True
)

_cluster_runtime = {}  # cluster_label -> dict with pre-extracted params
for cluster_label in sorted(cluster_models.keys()):
    features, scaler, pca = cluster_models[cluster_label]
    col_idx = np.fromiter(
        (_feat2idx[f] for f in features), dtype=np.int64, count=len(features)
    )

    scaler_mean = np.asarray(scaler.mean_, dtype=np.float32)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float32)

    pca_mean = np.asarray(pca.mean_, dtype=np.float32)  # mean in standardized space
    pca_comp = np.asarray(pca.components_, dtype=np.float32)  # shape (1, k)

    _cluster_runtime[int(cluster_label)] = {
        "col_idx": col_idx,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "pca_mean": pca_mean,
        "pca_comp": pca_comp,
    }

_pca_cluster_labels_sorted = [int(c.split("_")[1]) for c in pca_cols_sorted]


def _build_pc_matrix_for_parquet_df_fast(parquet_df: pd.DataFrame):
    parquet_df = parquet_df.sort_values("time").reset_index(drop=True)
    times = parquet_df["time"].to_numpy()

    feat_df = parquet_df.drop(columns=["time", "spectrogram_id"], errors="ignore")
    feat_df = feat_df.reindex(columns=_all_feat_cols)
    Xall = feat_df.to_numpy(dtype=np.float32, copy=False)

    bad = ~np.isfinite(Xall)
    if bad.any():
        Xall = Xall.copy()
        rows, cols = np.nonzero(bad)
        Xall[rows, cols] = _global_feat_means[cols]

    T = Xall.shape[0]
    C = len(_pca_cluster_labels_sorted)
    pc_mat = np.empty((T, C), dtype=np.float32)

    for j, cl in enumerate(_pca_cluster_labels_sorted):
        rt = _cluster_runtime[cl]
        col_idx = rt["col_idx"]
        X = Xall[:, col_idx]  # (T,k) view/copy depending on column selection

        Xs = (X - rt["scaler_mean"]) / rt["scaler_scale"]

        pc = (Xs - rt["pca_mean"]) @ rt["pca_comp"].T  # (T,1)
        pc_mat[:, j] = pc.reshape(-1).astype(np.float32, copy=False)

    return times, pc_mat


sgid_arr = train_df["spectrogram_id"].astype(int).to_numpy()
offset_arr = train_df["spectrogram_label_offset_seconds"].to_numpy(dtype=np.float64)
patient_arr = train_df["patient_id"].to_numpy()
label_arr = train_df[prob_columns].to_numpy(dtype=np.float32, copy=True)

_train_pc_cache = {}  # sgid -> (times, pc_mat)

Nmax = len(train_df)
C = len(_pca_cluster_labels_sorted)
images_np = np.empty((Nmax, 3, C, 224), dtype=np.float32)
labels_np = np.empty((Nmax, 6), dtype=np.float32)
groups_np = np.empty((Nmax,), dtype=patient_arr.dtype)

kept = 0
skipped = 0

for sgid, idxs in tqdm(
    train_df.groupby("spectrogram_id").indices.items(),
    desc="Build train tensors (prealloc)",
):
    sgid = int(sgid)
    parquet_df = _load_train_sg(sgid)

    out = _train_pc_cache.get(sgid)
    if out is None:
        times, pc_mat = _build_pc_matrix_for_parquet_df_fast(parquet_df)
        _train_pc_cache[sgid] = (times, pc_mat)
    else:
        times, pc_mat = out

    idxs = np.asarray(idxs, dtype=np.int64)
    offs = offset_arr[idxs] + 76.0

    start = np.searchsorted(times, offs, side="left")
    end = np.searchsorted(times, offs + 448.0, side="left")
    good = (end - start) == 224
    if not np.any(good):
        skipped += int(len(idxs))
        continue

    idxs_good = idxs[good]
    start_good = start[good]

    n_good = len(idxs_good)

    windows = np.empty((n_good, 224, C), dtype=np.float32)
    for k, s in enumerate(start_good):
        windows[k] = pc_mat[s : s + 224, :]

    ok = ~np.isnan(windows).any(axis=(1, 2))
    if not np.any(ok):
        skipped += int(len(idxs_good)) + int((~good).sum())
        continue

    windows_ok = windows[ok]  # (m,224,C)
    single = np.transpose(windows_ok, (0, 2, 1))  # (m,C,224)
    m = single.shape[0]
    images_np[kept : kept + m, 0, :, :] = single
    images_np[kept : kept + m, 1, :, :] = 0.0
    images_np[kept : kept + m, 2, :, :] = 0.0

    kept_rows_idx = idxs_good[ok]
    labels_np[kept : kept + m] = label_arr[kept_rows_idx]
    groups_np[kept : kept + m] = patient_arr[kept_rows_idx]
    kept += m

    skipped += int((~ok).sum()) + int((~good).sum())

if kept == 0:
    raise RuntimeError("No training windows were built (kept_rows=0).")

images_np = images_np[:kept]
labels = labels_np[:kept]
groups = groups_np[:kept]
images_tensor = torch.from_numpy(images_np)

print("Built images_tensor:", images_tensor.shape, "kept:", kept, "skipped:", skipped)




## === cell 13
def _stratified_group_split(groups, y_probs, test_size=0.2, seed=52, n_tries=50):
    rng = np.random.default_rng(seed)
    y_cls = np.argmax(y_probs, axis=1)
    unique_groups = np.unique(groups)

    best = None
    best_score = float("inf")

    overall = np.bincount(y_cls, minlength=6).astype(np.float64)
    overall = overall / overall.sum()

    for _ in range(n_tries):
        perm = rng.permutation(unique_groups)
        val_groups = []
        val_n = 0
        target_n = int(round(len(y_cls) * test_size))

        for g in perm:
            idx = np.where(groups == g)[0]
            if val_n + len(idx) > target_n and val_n > 0:
                continue
            val_groups.append(g)
            val_n += len(idx)
            if val_n >= target_n:
                break

        val_mask = np.isin(groups, np.array(val_groups, dtype=groups.dtype))
        tr_mask = ~val_mask
        if tr_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        val_dist = np.bincount(y_cls[val_mask], minlength=6).astype(np.float64)
        val_dist = val_dist / max(1.0, val_dist.sum())

        drift = float(np.abs(val_dist - overall).sum())
        if drift < best_score:
            best_score = drift
            best = (np.where(tr_mask)[0], np.where(val_mask)[0])

    if best is None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(gss.split(images_tensor, labels, groups=groups))
    return best


train_idx, val_idx = _stratified_group_split(groups, labels, test_size=0.2, seed=52)

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




## === cell 14
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



## === cell 15
train_dataset = TensorDataset(
    train_images, torch.tensor(train_labels, dtype=torch.float32)
)
val_dataset = TensorDataset(val_images, torch.tensor(val_labels, dtype=torch.float32))

cpu_cnt = os.cpu_count() or 0
num_workers = min(4, max(0, (cpu_cnt // 2) - 1))
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



## === cell 16
train_losses = []
val_losses = []

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images_b, labels_b in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
    ):
        images_b = images_b.to(device, non_blocking=True)
        labels_b = labels_b.to(device, non_blocking=True)

        logits = model(images_b)
        loss = criterion(F.log_softmax(logits, dim=1), labels_b)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    avg_train_loss = total_loss / max(1, len(train_loader))
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.inference_mode():
        for images_b, labels_b in val_loader:
            images_b = images_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            logits = model(images_b)
            loss = criterion(F.log_softmax(logits, dim=1), labels_b)
            total_val_loss += float(loss.item())

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
    )



## === cell 17
plt.figure(figsize=(8, 3))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("KLDivLoss")
plt.legend()
plt.show()



## === cell 18
import sys



## === cell 19
del sg_fit_df, train_df_M



## === cell 20
test_df = pd.read_csv(f"{data_path}/test.csv")
print(test_df.shape)
test_df.head()



## === cell 21
_sg_test_cache = OrderedDict()


def _load_test_sg(sgid: int) -> pd.DataFrame:
    df = _sg_test_cache.get(sgid)
    if df is not None:
        _sg_test_cache.move_to_end(sgid)
        return df
    df = pd.read_parquet(f"{sg_test_path}/{sgid}.parquet")
    _sg_test_cache[sgid] = df
    if len(_sg_test_cache) > _SG_CACHE_MAX:
        _sg_test_cache.popitem(last=False)
    return df


def build_test_pc_arrays_for_spectrogram(spectrogram_id: int, cache: dict):
    out = cache.get(spectrogram_id)
    if out is not None:
        return out

    parquet_df = _load_test_sg(int(spectrogram_id))
    times, pc_mat = _build_pc_matrix_for_parquet_df_fast(parquet_df)

    cache[spectrogram_id] = (times, pc_mat)
    return times, pc_mat


def make_image_tensor_from_pc_window_arrays(
    times: np.ndarray, pc_mat: np.ndarray, offset_seconds: float
):
    start = np.searchsorted(times, offset_seconds, side="left")
    end = np.searchsorted(times, offset_seconds + 448.0, side="left")
    if end - start != 224:
        return None

    window = pc_mat[start:end, :]  # (224,C)
    if np.isnan(window).any():
        return None

    single_channel_img = window.T.astype(np.float32, copy=False)  # (C,224)
    zeros_channel = np.zeros_like(single_channel_img)
    three_channel_img = np.stack(
        [single_channel_img, zeros_channel, zeros_channel], axis=0
    )  # (3,C,224)
    return torch.from_numpy(three_channel_img)


model.eval()
candidate_offsets = [0.0, 38.0, 76.0, 114.0, 152.0]
best_offset = 76.0
best_val_kl = float("inf")

with torch.inference_mode():
    tot_loss = 0.0
    tot_n = 0
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(F.log_softmax(logits, dim=1), yb)  # batchmean
        bs = int(xb.shape[0])
        tot_loss += float(loss.item()) * bs
        tot_n += bs
    best_val_kl = tot_loss / max(1, tot_n)
    best_offset = 76.0

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
unique_sgids = np.unique(test_sgid)

PROB_FLOOR = 1e-6  # keep as-is

BATCH_INFER = 256

pending_imgs = []
pending_row_idxs = []


def _flush_pending():
    if not pending_imgs:
        return
    xb = torch.stack(pending_imgs, dim=0).to(device, non_blocking=True)
    logits = model(xb)
    prob = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
    prob = np.clip(prob, PROB_FLOOR, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    for rix, p in zip(pending_row_idxs, prob):
        preds[rix] = p
    pending_imgs.clear()
    pending_row_idxs.clear()


with torch.inference_mode():
    for sgid in tqdm(unique_sgids, desc="Infer test (batched by spectrogram)"):
        rows_idx = np.where(test_sgid == sgid)[0]
        try:
            times, pc_mat = build_test_pc_arrays_for_spectrogram(int(sgid), pc_cache)
            img = make_image_tensor_from_pc_window_arrays(
                times, pc_mat, float(best_offset)
            )
            if img is None:
                preds[rows_idx] = uniform
                continue

            pending_imgs.append(img)
            pending_row_idxs.append(rows_idx)

            if len(pending_imgs) >= BATCH_INFER:
                _flush_pending()
        except Exception:
            preds[rows_idx] = uniform

    _flush_pending()

test_pred_map = (
    pd.DataFrame(
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
    .groupby("eeg_id", as_index=False)
    .mean()
)

test_preds = sample_sub[["eeg_id"]].merge(test_pred_map, on="eeg_id", how="left")
test_preds[vote_columns] = test_preds[vote_columns].fillna(1 / 6)

test_preds[vote_columns] = test_preds[vote_columns].clip(PROB_FLOOR, 1.0)
prob_sum = test_preds[vote_columns].sum(axis=1).values.reshape(-1, 1)
test_preds[vote_columns] = test_preds[vote_columns] / prob_sum
test_preds.head()



## === cell 22
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
