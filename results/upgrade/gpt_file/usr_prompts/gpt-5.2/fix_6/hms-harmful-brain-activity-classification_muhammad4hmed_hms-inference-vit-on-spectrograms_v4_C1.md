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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

1.1101675291376314

# 6. Current score

1.31036

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41) has done: 'Your current notebook doesn’t yield a Kaggle score mainly because it fails to reliably produce a valid `submission.csv`: the dataset indexing is unsafe (`loc[idx]` with non-0..N-1 indices) and the CSV write has a typo (`index =- False`). I make minimal fixes to ensure the DataLoader always reads the correct rows (using `iloc`), the spectrogram is converted to a proper numeric 2D array before applying the colormap, and the submission is written correctly. I also enforce the required KL-divergence constraint that each row sums to 1 by renormalizing the final probabilities (a safe post-process that improves validity and typically improves KL if any drift exists). Core model logic and inference remain unchanged.'
- What this solution (achieved 1.4016) has done: 'I remove the import that triggers the `MessageFactory.GetPrototype` protobuf error (it comes from `transformers` and isn’t used by your pipeline). Then I fix the missing checkpoint path by keeping the same load logic but adding a safe fallback: if the checkpoint file isn’t available in this environment, the code run an inference-only untrained model and still write a valid `submission.csv` (so you always get an uploadable file). Finally, I make the class-to-column mapping deterministic and robust by aligning model outputs to the required vote column order using the known 6-class set, and I keep the row-wise probability renormalization to satisfy the KL metric constraint.'
- What this solution (achieved 1.53632) has done: 'Your score (1.4016) is worse than the target (1.1102), so we should improve predictions while keeping the same core pipeline. The biggest low-risk gain here is to use ImageNet-pretrained ResNet18 weights (same architecture, just better initialization) because your current run typically uses untrained weights due to the missing checkpoint, which heavily hurts KL. I also keep your probability renormalization (required for valid KL submissions) and make DataLoader settings deterministic and safe. This should move the score significantly toward the target without changing the overall approach (spectrogram→colormap image→ResNet→softmax).'
- What this solution (achieved 1.31036) has done: 'The timeout is dominated by repeatedly reading/parsing ~10k Parquet spectrogram files in `__getitem__` using pandas, plus single-process data loading. I keep the exact same spectrogram→viridis RGBA→PIL→torchvision transform pipeline and the same model/training/inference logic, but replace pandas Parquet reads with much faster PyArrow reads and only read numeric columns. I also enable multi-worker DataLoaders with pinned memory and persistent workers to overlap CPU I/O/decoding with GPU compute, and add a tiny in-process LRU cache for recently used spectrograms (safe and exact) to avoid repeated reads when IDs repeat. These changes reduce wall time drastically without changing the algorithm, outputs semantics, or file paths.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from tqdm import tqdm
from PIL import Image
import matplotlib.cm as cm

import pyarrow.parquet as pq

from collections import OrderedDict
import multiprocessing as mp




## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 2
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
classes = np.array(vote_cols, dtype=object)
mapping = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)




## === cell 3
cmap = cm.get_cmap("viridis")




## === cell 4
class _LRUArrayCache:
    def __init__(self, max_items=64):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key, None)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


def _read_parquet_numeric_to_numpy(parquet_path: str) -> np.ndarray:
    table = pq.read_table(
        parquet_path
    )  # parquet files are numeric columns only in this dataset
    df = table.to_pandas(self_destruct=True)
    return df.to_numpy(dtype=np.float32, copy=False)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, data, spectrogram_dir, transform=None, with_target=False, cache_size=64
    ):
        self.data = data.reset_index(drop=True)
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.with_target = with_target
        self._cache = _LRUArrayCache(max_items=cache_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        specto_id = int(row["spectrogram_id"])
        specto_path = os.path.join(self.spectrogram_dir, f"{specto_id}.parquet")

        arr = self._cache.get(specto_path)
        if arr is None:
            arr = _read_parquet_numeric_to_numpy(specto_path)
            self._cache.put(specto_path, arr)

        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            arr_norm = np.zeros_like(arr, dtype=np.float32)
        else:
            arr_norm = (np.nan_to_num(arr, nan=mn) - mn) / (mx - mn)

        rgba = (cmap(arr_norm) * 255).astype(np.uint8)  # H x W x 4
        spectrogram = Image.fromarray(rgba, mode="RGBA").convert("RGB")

        if self.transform:
            spectrogram = self.transform(spectrogram)[:3, :, :]

        if self.with_target:
            y = row[vote_cols].to_numpy(dtype=np.float32)
            y = np.clip(y, 0.0, None)
            s = float(y.sum())
            if s <= 0:
                y[:] = 1.0 / len(y)
            else:
                y /= s
            return spectrogram, torch.from_numpy(y)
        return spectrogram




## === cell 5
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)




## === cell 6
train_agg = train.groupby("spectrogram_id", as_index=False)[vote_cols].sum()
train_agg[vote_cols] = (
    train_agg[vote_cols]
    .div(train_agg[vote_cols].sum(axis=1), axis=0)
    .fillna(1.0 / num_classes)
)

train_agg["eeg_id"] = (
    train.groupby("spectrogram_id", as_index=False)["eeg_id"].first()["eeg_id"].values
)

MAX_TRAIN = 12000
if len(train_agg) > MAX_TRAIN:
    train_agg = train_agg.sample(n=MAX_TRAIN, random_state=SEED).reset_index(drop=True)

train_dataset = ImageDataset(
    train_agg,
    spectrogram_dir="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms",
    transform=transform,
    with_target=True,
)

test_dataset = ImageDataset(
    test,
    spectrogram_dir="/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms",
    transform=transform,
    with_target=False,
)

num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
pin = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2,
)




## === cell 7
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = softmax(x, dim=1)
        return x




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(num_classes).to(device)

ckpt_path = "/kaggle/input/hms-train-spectogram-images/trained_model.pt"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    print(
        f"WARNING: checkpoint not found at {ckpt_path}. "
        "Training only the final FC layer on train_spectrograms to improve toward target score."
    )

    for p in model.resnet.parameters():
        p.requires_grad = False
    for p in model.resnet.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.resnet.fc.parameters(), lr=3e-3, weight_decay=1e-4
    )

    def soft_target_ce(probs, targets, eps=1e-8):
        probs = torch.clamp(probs, eps, 1.0)
        return -(targets * torch.log(probs)).sum(dim=1).mean()

    model.train()
    EPOCHS = 2  # unchanged
    for epoch in range(EPOCHS):
        losses = []
        pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            probs = model(images)
            loss = soft_target_ce(probs, targets)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            if len(losses) >= 10:
                pbar.set_postfix({"loss": sum(losses[-10:]) / 10.0})
        print(f"Epoch {epoch+1} mean loss: {sum(losses)/max(1,len(losses)):.5f}")

    model.eval()




## === cell 9
model.eval()
out = []
pbar = tqdm(test_loader, desc="infer")
for images in pbar:
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        outputs = model(images)
    out.append(outputs.detach().cpu().numpy())




## === cell 10
outputs = np.vstack(out)
outputs.shape




## === cell 11
mapping




## === cell 12
submission = test[["eeg_id"]].copy()

pred = outputs.astype(np.float64)

pred = np.clip(pred, 1e-12, None)
row_sum = pred.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
pred = pred / row_sum

for i, c in enumerate(vote_cols):
    submission[c] = pred[:, i]




## === cell 13
submission.to_csv("submission.csv", index=False)
submission
