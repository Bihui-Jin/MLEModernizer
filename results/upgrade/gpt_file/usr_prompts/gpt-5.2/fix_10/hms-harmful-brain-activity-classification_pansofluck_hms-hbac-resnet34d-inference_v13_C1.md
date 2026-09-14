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

albumentations==2.0.8
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
PyYAML==6.0.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.5631250263626647

# 6. Current score

0.92524

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.06358) has done: 'I fix the root cause of the missing `best_model.pth` by ensuring we always save a model checkpoint (even if validation loss is NaN/empty) and by falling back to the current model weights if the checkpoint isn’t found. I also remove the incorrect test-file existence filtering (it can silently drop rows and misalign predictions) and instead generate predictions for all test spectrogram_ids, defaulting safely to uniform probabilities if any read fails. Finally, I make the submission creation robust so `test_pred_df` is always defined and the output CSV has the exact required columns with row-wise probabilities summing to 1.'
- What this solution (achieved 0.91849) has done: 'Your current score is much worse than the target (lower-is-better), and the biggest likely cause is that train labels are aggregated by `spectrogram_id` but test-time predictions are averaged by `eeg_id`, creating a train/test target mismatch that hurts KL a lot. I keep the same model, loss, and training loop, but change the label aggregation and split to be by `eeg_id` (matching the submission unit), and train on `eeg_id`-level targets while still using one representative `spectrogram_id` per `eeg_id` for the image input. I also make the validation split group-aware by `patient_id` to reduce leakage and improve generalization without changing the core training approach. These are minimal, semantics-preserving fixes aligned with the metric and submission format and should move the score substantially toward your target.'
- What this solution (achieved 0.7885) has done: 'Your current score (0.91849, lower-is-better) is still far from the target (0.5631), so we should make small, metric-aligned improvements without changing the overall model/training setup. The biggest low-risk gain here is to better match the competition’s “eeg_id-level” targets by using **vote-count-weighted targets per eeg_id** (instead of simple-summing then normalizing, which implicitly weights overlapping segments equally). Additionally, we should make the validation split more stable and less leaky by using a **group split on patient_id via GroupShuffleSplit** (same idea as you already do, but deterministic and exact proportion), and slightly stabilize KL training by ensuring targets are strictly normalized (sum=1) after clipping. These changes keep the same architecture, same loss, same loop, and should move KL down toward your target without “over-optimizing”.'
- What this solution (achieved 0.88611) has done: 'Your current KL (0.7885, lower-is-better) is still well above the target (0.5631), so we should make a small, metric-aligned improvement without changing the model or training loop. The biggest low-risk gain here is to reduce train/test input mismatch: you train using a “first spectrogram_id per eeg_id” representative, but at test time you average multiple spectrograms per eeg_id; we instead train on the same distribution by sampling one random spectrogram per eeg_id each epoch (and use the same per-eeg vote-weighted targets). Additionally, we remove HorizontalFlip (it can break left/right EEG spatial semantics and often hurts generalization for this task) while keeping the rest identical. These changes preserve architecture, loss, and loop structure, but should reduce KL toward the target.'
- What this solution (achieved 0.88611) has done: 'Your current KL (0.88611, lower-is-better) is still far above the target (0.5631), so we should make a small, metric-aligned improvement without changing the model, loss, or training loop. The biggest low-risk issue is that you train on per‑`eeg_id` targets but validate against only the **representative** `spectrogram_id`, while at inference you effectively average multiple windows per `eeg_id`; we can make validation and inference consistent by using **the same aggregation rule** (mean over multiple spectrograms per `eeg_id`) without changing the network. Concretely: (1) build a validation dataset that contains *multiple spectrogram samples per eeg_id* (sample a small fixed number per eeg, deterministic), (2) compute validation KL after aggregating predictions back to eeg_id, and (3) at test time, predict on a small fixed number of spectrograms per eeg_id and average (instead of whatever test.csv provides), which usually reduces KL by reducing window-selection noise. These are minimal changes that preserve architecture/training semantics and should move the score down toward your target band.'
- What this solution (achieved 0.92524) has done: 'Your current KL (0.88611, lower-is-better) is still far above the target (0.5631), so we should make the smallest metric-aligned changes that improve generalization without changing your model, loss, or training loop. The biggest low-risk issue is a train/infer preprocessing mismatch: the dataset currently splits the spectrogram into `channels` chunks, but `channels=1` means you never actually use multi-region information; we keep `channels=1` (to preserve core logic) but fix an important normalization bug by standardizing per-channel correctly (currently it normalizes over wrong axes after ToTensorV2). Next, we make inference/validation aggregation more stable by using **log-prob averaging** (geometric mean) across multiple spectrogram windows per `eeg_id`, which is better aligned with KL than arithmetic mean of probabilities and is a minimal post-processing change. These changes are directly score-relevant, keep runtime similar, and preserve architecture/training semantics.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import GroupShuffleSplit



## === cell 1
ROOT = Path("/kaggle")
INPUT = ROOT / "input"
WORKING = ROOT / "working"

DATA = INPUT / "hms-harmful-brain-activity-classification"
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TMP = WORKING / "tmp_hms"
TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"
TMP.mkdir(exist_ok=True, parents=True)
TRAIN_SPEC_SPLIT.mkdir(exist_ok=True, parents=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True, parents=True)

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

RANDAM_SEED = 1086  # keep original name to minimize changes


def seed_everything(seed: int = 1086):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(RANDAM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)




## === cell 2
class CFG:
    model_name = "tf_efficientnetv2_l_in21ft1k"
    img_size_h = 400
    img_size_w = 300
    channels = 1
    max_epoch = 1  # keep runtime < 600s
    batch_size = 16
    lr = 1.0e-03
    weight_decay = 1.0e-02
    seed = 1086
    deterministic = True
    enable_amp = True

    n_val_specs_per_eeg = 4
    n_test_specs_per_eeg = 4


if CFG.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



## === cell 3
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

print(train.shape, test.shape, smpl_sub.shape)
print("train cols ok:", all(c in train.columns for c in CLASSES))



## === cell 4
train_votes = train[["eeg_id"] + CLASSES].copy()
train_votes[CLASSES] = train_votes[CLASSES].astype("float32")
train_votes["row_total_votes"] = train_votes[CLASSES].sum(axis=1).astype("float32")

eps = 1e-6
row_probs = train_votes[CLASSES].div(
    train_votes["row_total_votes"].values.reshape(-1, 1) + eps
)
row_probs = row_probs.clip(eps, 1.0)
row_probs = row_probs.div(row_probs.sum(axis=1).values.reshape(-1, 1) + eps)
train_votes[CLASSES] = row_probs

for c in CLASSES:
    train_votes[c] = train_votes[c] * train_votes["row_total_votes"]

train_g = train_votes.groupby("eeg_id", as_index=False)[
    CLASSES + ["row_total_votes"]
].sum()
train_g[CLASSES] = train_g[CLASSES].div(
    train_g["row_total_votes"].values.reshape(-1, 1) + eps
)
train_g[CLASSES] = train_g[CLASSES].clip(eps, 1.0)
train_g[CLASSES] = train_g[CLASSES].div(
    train_g[CLASSES].sum(axis=1).values.reshape(-1, 1) + eps
)
train_g.drop(columns=["row_total_votes"], inplace=True)

eeg_to_specs = (
    train.groupby("eeg_id")["spectrogram_id"]
    .apply(lambda s: s.dropna().astype("int64").unique().tolist())
    .to_dict()
)

rep_pid = (
    train.sort_values(["eeg_id", "patient_id"])
    .groupby("eeg_id", as_index=False)["patient_id"]
    .first()
)
train_g = train_g.merge(rep_pid, on="eeg_id", how="left")

rep_spec = (
    train.sort_values(["eeg_id", "spectrogram_id"])
    .groupby("eeg_id", as_index=False)["spectrogram_id"]
    .first()
)
train_g = train_g.merge(rep_spec, on="eeg_id", how="left")

print("unique train eeg_ids:", len(train_g))
print("train_g cols:", train_g.columns.tolist())
print(
    "target row-sum min/max:",
    train_g[CLASSES].sum(axis=1).min(),
    train_g[CLASSES].sum(axis=1).max(),
)



## === cell 5
try:
    import pyarrow.parquet as pq
except Exception as e:
    pq = None
    print(
        "[WARN] pyarrow.parquet unavailable; will fall back to pandas.read_parquet. Error:",
        repr(e),
    )


def spec_parquet_to_array(spec_path: Path) -> np.ndarray:
    if pq is not None:
        table = pq.read_table(spec_path)
        arr = (
            table.to_pandas()
            .fillna(0)
            .to_numpy(copy=False)[:, 1:]
            .T.astype("float32", copy=False)
        )
        return arr
    spec = pd.read_parquet(spec_path, engine="pyarrow")
    arr = spec.fillna(0).values[:, 1:].T.astype("float32")
    return arr


def build_npy_cache(ids: np.ndarray, src_dir: Path, dst_dir: Path, desc: str):
    print(
        f"[INFO] Skipping {desc} (on-demand Parquet loading is used to avoid 10+ minute preprocessing)."
    )
    return


build_npy_cache(
    train_g["spectrogram_id"].values,
    TRAIN_SPEC,
    TRAIN_SPEC_SPLIT,
    "Caching train spectrograms",
)
build_npy_cache(
    test["spectrogram_id"].values,
    TEST_SPEC,
    TEST_SPEC_SPLIT,
    "Caching test spectrograms",
)




## === cell 6
class HMSHBACSpecModel(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool, in_channels: int, num_classes: int
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        return self.model(x)


FilePath = tp.Union[str, Path]


class HMSHBACSpecDataset(Dataset):
    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Optional[tp.Sequence[np.ndarray]],
        transform: A.Compose,
    ):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else list(labels)
        self.transform = transform
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.image_paths)

    def _load_array(self, img_path: str) -> np.ndarray:
        arr = self._cache.get(img_path)
        if arr is not None:
            return arr
        p = Path(img_path)
        if p.suffix == ".npy" and p.exists():
            arr = np.load(p)
        else:
            spec_id = p.stem
            train_pq = TRAIN_SPEC / f"{spec_id}.parquet"
            test_pq = TEST_SPEC / f"{spec_id}.parquet"
            if train_pq.exists():
                arr = spec_parquet_to_array(train_pq)
            else:
                arr = spec_parquet_to_array(test_pq)
        self._cache[img_path] = arr
        return arr

    def __getitem__(self, index: int):
        img_path = str(self.image_paths[index])

        try:
            img = self._load_array(img_path)  # expected (freq, time)
        except Exception:
            img = np.zeros((CFG.img_size_h, CFG.img_size_w), dtype="float32")

        chunks = np.array_split(img, CFG.channels, axis=0)
        img = np.stack(chunks, axis=-1)  # (H,W,C)

        img = self.transform(image=img)["image"].float()  # (C,H,W)

        img = torch.clamp(img, float(np.exp(-4)), float(np.exp(8)))
        img = torch.log(img)

        eps = 1e-6
        img_mean = img.mean(dim=(1, 2), keepdim=True)
        img_std = img.std(dim=(1, 2), keepdim=True)
        img = (img - img_mean) / (img_std + eps)

        if self.labels is None:
            return {"data": img}
        target = torch.tensor(self.labels[index], dtype=torch.float32)
        return {"data": img, "target": target}


def get_transforms(CFG, train: bool):
    if train:
        return A.Compose(
            [
                A.Resize(p=1.0, height=CFG.img_size_h, width=CFG.img_size_w),
                ToTensorV2(p=1.0),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(p=1.0, height=CFG.img_size_h, width=CFG.img_size_w),
                ToTensorV2(p=1.0),
            ]
        )




## === cell 7
gss = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=CFG.seed)
idx = np.arange(len(train_g))
groups = train_g["patient_id"].fillna(-1).values
train_idx, val_idx = next(gss.split(idx, groups=groups))

train_df = train_g.iloc[train_idx].reset_index(drop=True)
val_df = train_g.iloc[val_idx].reset_index(drop=True)

eps = 1e-6
train_labels = train_df[CLASSES].values.astype("float32")
val_labels = val_df[CLASSES].values.astype("float32")
train_labels = np.clip(train_labels, eps, 1.0)
train_labels = train_labels / train_labels.sum(axis=1, keepdims=True)
val_labels = np.clip(val_labels, eps, 1.0)
val_labels = val_labels / val_labels.sum(axis=1, keepdims=True)


def make_paths_from_spec_ids(spec_ids: np.ndarray, split_dir: Path) -> list[Path]:
    return [split_dir / f"{sid}.npy" for sid in spec_ids.astype("int64")]


def sample_train_spec_ids_for_epoch(df: pd.DataFrame, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(CFG.seed + int(epoch))
    out = np.empty(len(df), dtype="int64")
    eegs = df["eeg_id"].values
    fallback = df["spectrogram_id"].values.astype("int64")
    for i, eeg_id in enumerate(eegs):
        cand = eeg_to_specs.get(int(eeg_id), None)
        if cand:
            out[i] = int(cand[int(rng.integers(0, len(cand)))])
        else:
            out[i] = int(fallback[i])
    return out


def build_multi_spec_df(
    df_eeg: pd.DataFrame,
    eeg_to_specs_map: dict,
    n_specs_per_eeg: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in df_eeg.iterrows():
        eeg_id = int(r["eeg_id"])
        fallback_sid = int(r["spectrogram_id"])
        cand = eeg_to_specs_map.get(eeg_id, None)
        if cand and len(cand) > 0:
            if len(cand) >= n_specs_per_eeg:
                pick = rng.choice(cand, size=n_specs_per_eeg, replace=False)
            else:
                pick = rng.choice(cand, size=n_specs_per_eeg, replace=True)
        else:
            pick = np.array([fallback_sid] * n_specs_per_eeg, dtype="int64")
        for sid in pick:
            rows.append({"eeg_id": eeg_id, "spectrogram_id": int(sid)})
    return pd.DataFrame(rows)


train_spec_ids_epoch0 = sample_train_spec_ids_for_epoch(train_df, epoch=0)
train_paths = make_paths_from_spec_ids(train_spec_ids_epoch0, TRAIN_SPEC_SPLIT)

val_ms = build_multi_spec_df(
    val_df[["eeg_id", "spectrogram_id"]],
    eeg_to_specs,
    n_specs_per_eeg=CFG.n_val_specs_per_eeg,
    seed=CFG.seed + 999,
)
val_ms_paths = make_paths_from_spec_ids(
    val_ms["spectrogram_id"].values, TRAIN_SPEC_SPLIT
)
val_ms_labels = (
    val_df.set_index("eeg_id")[CLASSES]
    .loc[val_ms["eeg_id"].values]
    .values.astype("float32")
)
val_ms_labels = np.clip(val_ms_labels, eps, 1.0)
val_ms_labels = val_ms_labels / val_ms_labels.sum(axis=1, keepdims=True)

train_ds = HMSHBACSpecDataset(
    train_paths, train_labels, transform=get_transforms(CFG, train=True)
)
val_ds = HMSHBACSpecDataset(
    val_ms_paths, val_ms_labels, transform=get_transforms(CFG, train=False)
)

num_workers = min(4, (os.cpu_count() or 2))
train_loader = DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)
val_loader = DataLoader(
    val_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

print(
    "train/val eeg sizes:",
    len(train_df),
    len(val_df),
    "val_ms samples:",
    len(val_ds),
    "num_workers:",
    num_workers,
)



## === cell 8
model = HMSHBACSpecModel(
    model_name=CFG.model_name,
    pretrained=True,
    in_channels=CFG.channels,
    num_classes=N_CLASSES,
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=(CFG.enable_amp and DEVICE.type == "cuda"))
criterion = nn.KLDivLoss(reduction="batchmean")


def aggregate_probs_by_eeg_geomean(
    probs: np.ndarray, eeg_ids: np.ndarray, eps: float = 1e-6
) -> pd.DataFrame:
    probs = np.clip(probs.astype("float32", copy=False), eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    logp = np.log(probs)
    df = pd.DataFrame(logp, columns=CLASSES)
    df.insert(0, "eeg_id", eeg_ids.astype("int64", copy=False))
    df = df.groupby("eeg_id", as_index=False)[CLASSES].mean()

    out = df.copy()
    out[CLASSES] = np.exp(out[CLASSES].values)
    out_vals = out[CLASSES].values.astype("float32", copy=False)
    out_vals = np.clip(out_vals, eps, 1.0)
    out_vals = out_vals / out_vals.sum(axis=1, keepdims=True)
    out[CLASSES] = out_vals
    return out


def evaluate_eeg_aggregated(
    model, loader, eeg_ids_for_loader: np.ndarray, eeg_targets_df: pd.DataFrame
):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["data"].to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=(CFG.enable_amp and DEVICE.type == "cuda")
            ):
                logits = model(x)
                prob = torch.softmax(logits, dim=1)
            all_probs.append(prob.float().cpu().numpy())
    if not all_probs:
        return np.nan
    probs = np.concatenate(all_probs, axis=0)

    pred_df = aggregate_probs_by_eeg_geomean(probs, eeg_ids_for_loader, eps=1e-6)

    merged = eeg_targets_df[["eeg_id"] + CLASSES].merge(
        pred_df, on="eeg_id", how="left", suffixes=("_t", "_p")
    )
    for c in CLASSES:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(1.0 / N_CLASSES)

    t = merged[[f"{c}_t" for c in CLASSES]].to_numpy(dtype="float32", copy=False)
    p = merged[CLASSES].to_numpy(dtype="float32", copy=False)

    t = np.clip(t, 1e-6, 1.0)
    t = t / t.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    kl = float(np.mean(np.sum(t * (np.log(t) - np.log(p)), axis=1)))
    return kl


best_val = float("inf")
best_path = WORKING / "best_model.pth"

val_targets_eeg = val_df[["eeg_id"] + CLASSES].copy()

for epoch in range(CFG.max_epoch):
    epoch_spec_ids = sample_train_spec_ids_for_epoch(train_df, epoch=epoch)
    train_ds.image_paths = [
        str(p) for p in make_paths_from_spec_ids(epoch_spec_ids, TRAIN_SPEC_SPLIT)
    ]
    train_ds._cache.clear()

    model.train()
    pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{CFG.max_epoch}")
    for batch in pbar:
        x = batch["data"].to(DEVICE, non_blocking=True)
        t = batch["target"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=(CFG.enable_amp and DEVICE.type == "cuda")
        ):
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, t)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=float(loss.item()))

    val_loss = evaluate_eeg_aggregated(
        model,
        val_loader,
        eeg_ids_for_loader=val_ms["eeg_id"].values,
        eeg_targets_df=val_targets_eeg,
    )
    print("val_eeg_kl:", val_loss)

    if (not np.isfinite(val_loss)) or (val_loss < best_val):
        best_val = val_loss if np.isfinite(val_loss) else best_val
        torch.save(model.state_dict(), best_path)

if not best_path.exists():
    torch.save(model.state_dict(), best_path)

print(
    "best_val_eeg_kl:", best_val, "saved to", best_path, "exists:", best_path.exists()
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'seizure_vote'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2764307244.py in <cell line: 0>()
    104         pbar.set_postfix(loss=float(loss.item()))
    105 
--> 106     val_loss = evaluate_eeg_aggregated(
    107         model,
    108         val_loader,

/tmp/ipykernel_55/2764307244.py in evaluate_eeg_aggregated(model, loader, eeg_ids_for_loader, eeg_targets_df)
     57     )
     58     for c in CLASSES:
---> 59         if merged[c].isna().any():
     60             merged[c] = merged[c].fillna(1.0 / N_CLASSES)
     61 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'seizure_vote'

## === cell 9
test_eeg_df = test[["eeg_id"]].drop_duplicates().reset_index(drop=True)

test_eeg_to_specs = (
    test.groupby("eeg_id")["spectrogram_id"]
    .apply(lambda s: s.dropna().astype("int64").unique().tolist())
    .to_dict()
)

rng = np.random.default_rng(CFG.seed + 2024)
test_rows = []
for _, r in test_eeg_df.iterrows():
    eeg_id = int(r["eeg_id"])
    cand = test_eeg_to_specs.get(eeg_id, None)
    if cand and len(cand) > 0:
        if len(cand) >= CFG.n_test_specs_per_eeg:
            pick = rng.choice(cand, size=CFG.n_test_specs_per_eeg, replace=False)
        else:
            pick = rng.choice(cand, size=CFG.n_test_specs_per_eeg, replace=True)
    else:
        pick = np.array([], dtype="int64")
    for sid in pick:
        test_rows.append({"eeg_id": eeg_id, "spectrogram_id": int(sid)})

test_ms = pd.DataFrame(test_rows)
test_paths = [
    TEST_SPEC_SPLIT / f"{sid}.npy"
    for sid in test_ms["spectrogram_id"].values.astype("int64")
]

test_ds = HMSHBACSpecDataset(
    test_paths,
    labels=None,
    transform=get_transforms(CFG, train=False),
)
test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

infer_model = HMSHBACSpecModel(
    model_name=CFG.model_name,
    pretrained=False,
    in_channels=CFG.channels,
    num_classes=N_CLASSES,
).to(DEVICE)

try:
    state = torch.load(best_path, map_location=DEVICE)
    infer_model.load_state_dict(state)
except Exception as e:
    print(
        "[WARN] Failed to load checkpoint; using current in-memory model weights. Error:",
        repr(e),
    )
    infer_model.load_state_dict(model.state_dict())

infer_model.eval()

preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="infer"):
        x = batch["data"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=(CFG.enable_amp and DEVICE.type == "cuda")
        ):
            logits = infer_model(x)
            prob = torch.softmax(logits, dim=1)
        preds.append(prob.float().cpu().numpy())

if len(test_ms) > 0:
    preds = np.concatenate(preds, axis=0)
    test_pred_df = aggregate_probs_by_eeg_geomean(
        preds, test_ms["eeg_id"].values, eps=1e-6
    )
else:
    test_pred_df = pd.DataFrame({"eeg_id": test_eeg_df["eeg_id"].values})
    for c in CLASSES:
        test_pred_df[c] = 1.0 / N_CLASSES



## === cell 10
sub = smpl_sub[["eeg_id"]].merge(test_pred_df, on="eeg_id", how="left")

for c in CLASSES:
    if sub[c].isna().any():
        sub[c] = sub[c].fillna(1.0 / N_CLASSES)

probs = sub[CLASSES].values.astype("float32")
probs = np.clip(probs, 1e-6, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
sub[CLASSES] = probs

assert len(sub) == len(smpl_sub), (len(sub), len(smpl_sub))
assert list(sub.columns) == ["eeg_id"] + CLASSES
row_sums = sub[CLASSES].sum(axis=1).values
print("row sum min/max:", row_sums.min(), row_sums.max())

out_path = WORKING / "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
