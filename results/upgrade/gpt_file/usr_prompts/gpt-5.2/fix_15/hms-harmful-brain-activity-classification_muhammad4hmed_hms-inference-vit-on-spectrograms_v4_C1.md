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

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.50788) has done: 'Your pipeline doesn’t currently yield a Kaggle score because it fails to produce a valid submission CSV: there’s a typo in `to_csv(index =- False)` and the dataset loader uses `loc[idx]` which can break if the DataFrame index isn’t 0..N-1. I make the smallest fixes to (1) ensure the test DataFrame is reset so indexing is correct, (2) safely build the spectrogram image from parquet values, and (3) write a properly formatted `submission.csv` with probabilities that sum to 1 per row (via a tiny renormalization safeguard). This preserves your core logic (same model, same transforms, same inference loop, same class mapping usage) while ensuring a valid submission is generated.'
- What this solution (achieved 1.442) has done: 'I fix the runtime failure by removing the hard dependency on a missing external weights file and instead run the same ResNet18 inference with deterministic initialization, so the notebook always completes and writes `submission.csv`. I also make the class/probability column mapping consistent with the competition’s required 6 target columns (and independent of `expert_consensus` ordering), and ensure the model outputs exactly 6 probabilities. Finally, I keep your spectrogram→RGBA→RGB pipeline intact but add small safety normalization to prevent NaNs/Infs and guarantee each row sums to 1, which is required for a valid submission and should reduce KL failures.'
- What this solution (achieved 1.375) has done: 'The timeout is dominated by per-sample pandas/parquet reads plus a costly `matplotlib` colormap conversion inside `__getitem__`, all happening single-threaded (`num_workers=0`) for ~12k train + 9.85k test samples. I keep the exact same data flow and model/training logic, but (1) replace the `cmap(arr)` call with a mathematically equivalent, vectorized viridis LUT lookup (same RGBA result), and (2) add a bounded in-process cache of decoded tensors so each spectrogram parquet is read/converted at most once across training/inference. I also enable multi-worker DataLoaders with persistent workers and prefetching to overlap CPU preprocessing with GPU/CPU compute without changing any learning semantics. These changes preserve outputs up to negligible floating-point differences while cutting the constant factors that cause the 10-minute timeout.'
- What this solution (achieved 1.29868) has done: 'Your current code trains on a random subset of 12k spectrograms with a plain cross-entropy on vote distributions, which is noisy and tends to hurt KL on this competition; the smallest legitimate improvement toward your target is to reduce label noise without changing the model or training loop. I keep your exact architecture, transforms, loss, and 1-epoch training, but change the training sampling to aggregate the multiple overlapping rows per `spectrogram_id` by summing votes and normalizing once (a standard de-dup that better matches the evaluation targets). This keeps the same “spectrogram → image → ResNet18 → softmax” core logic, but provides cleaner targets and typically improves KL substantially (moving down from 1.375 toward 1.11). I also keep your caching/LUT fast path and ensure the submission remains correctly normalized and aligned.'
- What this solution (achieved 1.71936) has done: 'To move your KL score down toward the 1.110 target without changing the model/training “core logic”, I keep the same ResNet18 + 1-epoch training, but reduce label noise further by aggregating training labels at the same granularity as test (`eeg_id`) instead of `spectrogram_id` (still just summing votes then normalizing). I also remove the extra `softmax` inside the model and instead apply it only in the loss/inference step; this keeps the same predicted probabilities but makes optimization numerically better-conditioned (less saturation), which typically improves KL with minimal risk. Finally, I make the train sampling deterministic and avoid repeated `pd.read_parquet` work across workers by using per-worker caching safely (each worker keeps its own cache) while keeping your LUT/transform pipeline intact and still writing a valid normalized `submission.csv`.'
- What this solution (achieved 2.77306) has done: 'I fix the DataLoader crash by ensuring the transform pipeline receives a 3-channel RGB image (not 4-channel RGBA), which is what triggers the “size of tensor a (4) must match … (3)” error in `Normalize`. Then I make the inference and submission steps robust so they always run even if an earlier stage fails, and I guarantee the submission has exactly the required 6 target columns with per-row probabilities summing to 1. These changes preserve your core pipeline (parquet→viridis image→ResNet18→softmax, same loss/training loop) while unblocking end-to-end execution and producing a valid `submission.csv`.'
- What this solution (achieved 3.14333) has done: 'Your current score (2.77306, lower-is-better) is far worse than the target (1.11017), so we should improve prediction quality without changing the core pipeline. The biggest issue is a train/test granularity mismatch: you train labels aggregated by `spectrogram_id`, but the submission/evaluation is per `eeg_id`, and in test each `eeg_id` is unique; aligning training targets to `eeg_id` typically reduces KL a lot. I also add a tiny, metric-consistent probability floor + renormalization during inference (same semantics as your existing clamp in training) to avoid overconfident zeros that can blow up KL. Everything else (ResNet18, transforms, 1-epoch training loop, loss form, spectrogram parquet→viridis→RGB) stays the same.'
- What this solution (achieved 2.77306) has done: 'Your current KL (3.14333, lower-is-better) is far above the target (1.11017), so we should improve generalization while keeping your exact model/training loop and data→viridis→RGB pipeline intact. The smallest high-impact fix is to remove the train/test granularity mismatch you introduced: you aggregate labels by `eeg_id` but then attach an essentially arbitrary single `spectrogram_id` per `eeg_id`, which often points to a spectrogram window that doesn’t match the aggregated label distribution, injecting large noise. I keep everything else the same, but instead (a) aggregate votes at the `spectrogram_id` level (so the spectrogram you load matches the label you train on) and (b) deduplicate by `spectrogram_id` for the train set so each spectrogram contributes once. This should materially reduce KL toward your target without changing architecture, loss, epochs, transforms, or inference semantics, and it still writes a valid normalized `submission.csv`.'
- What this solution (achieved 2.77306) has done: 'Your current KL (2.77306, lower-is-better) is far above the target (1.11017), so we should improve prediction quality with the smallest changes that don’t alter your core pipeline (same ResNet18, transforms, 1-epoch loop, and KL-style loss). The biggest issue is that you train on only 12k randomly sampled spectrograms even though labels are already aggregated per `spectrogram_id`; for this competition, using all aggregated training spectrograms typically reduces KL substantially without changing semantics. I therefore remove the 12k cap (keeping the same grouping-by-`spectrogram_id`) and add a tiny, metric-consistent inference floor+renormalization (slightly larger epsilon) to prevent extreme near-zero probabilities that can disproportionately hurt KL. Everything else remains unchanged, and the script still writes a valid `submission.csv` with rows summing to 1.'

# 9. Code solution

## === cell 0
import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.cm as cm
from PIL import Image




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 2
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
test = test.reset_index(drop=True)

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
num_classes = len(TARGET_COLS)



## === cell 3
cmap = cm.get_cmap("viridis")

_VIRIDIS_LUT_RGBA_U8 = (
    (cmap(np.linspace(0.0, 1.0, 256, dtype=np.float32)) * 255.0)
    .round()
    .astype(np.uint8)
)  # (256, 4)




## === cell 4
class _LRUCache:
    def __init__(self, max_items: int = 2048):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, k):
        v = self._d.get(k, None)
        if v is not None:
            self._d.move_to_end(k)
        return v

    def put(self, k, v):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)




## === cell 5
class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        spectrogram_dir,
        transform=None,
        target_cols=None,
        cache_max_items=2048,
    ):
        self.data = data.reset_index(drop=True)
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.target_cols = target_cols
        self._cache = (
            _LRUCache(max_items=cache_max_items)
            if cache_max_items and cache_max_items > 0
            else None
        )

    def _arr_to_rgb_u8(self, arr: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
        a_min = float(arr.min())
        a_max = float(arr.max())
        if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
            idx = np.zeros(arr.shape, dtype=np.uint8)
        else:
            norm = (arr - a_min) / (a_max - a_min)
            norm = np.clip(norm, 0.0, 1.0)
            idx = (norm * 255.0).round().astype(np.uint8)

        rgba = _VIRIDIS_LUT_RGBA_U8[idx]  # H x W x 4 uint8
        rgb = rgba[..., :3]  # drop alpha -> H x W x 3
        return rgb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        specto_id = int(self.data.iloc[idx]["spectrogram_id"])

        if self._cache is not None:
            cached = self._cache.get(specto_id)
            if cached is not None:
                spectrogram_tensor = cached
            else:
                specto_path = f"{self.spectrogram_dir}/{specto_id}.parquet"
                specto = pd.read_parquet(specto_path)
                arr = specto.to_numpy(dtype=np.float32, copy=False)
                rgb = self._arr_to_rgb_u8(arr)
                spectrogram = Image.fromarray(rgb, mode="RGB")
                if self.transform:
                    spectrogram_tensor = self.transform(spectrogram)
                else:
                    spectrogram_tensor = transforms.ToTensor()(spectrogram)
                self._cache.put(specto_id, spectrogram_tensor)
        else:
            specto_path = f"{self.spectrogram_dir}/{specto_id}.parquet"
            specto = pd.read_parquet(specto_path)
            arr = specto.to_numpy(dtype=np.float32, copy=False)
            rgb = self._arr_to_rgb_u8(arr)
            spectrogram = Image.fromarray(rgb, mode="RGB")
            if self.transform:
                spectrogram_tensor = self.transform(spectrogram)
            else:
                spectrogram_tensor = transforms.ToTensor()(spectrogram)

        if self.target_cols is None:
            return spectrogram_tensor

        y = self.data.iloc[idx][self.target_cols].to_numpy(dtype=np.float32, copy=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        s = float(y.sum())
        if s <= 0:
            y[:] = 1.0 / len(y)
        else:
            y /= s
        return spectrogram_tensor, torch.from_numpy(y)




## === cell 6
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 7
train_used = train.groupby(["spectrogram_id"], as_index=False)[TARGET_COLS].sum()
train_used = train_used[["spectrogram_id"] + TARGET_COLS].reset_index(drop=True)


def _dl_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(2, min(8, cpu // 2))


NUM_WORKERS = _dl_workers()
PERSISTENT = NUM_WORKERS > 0

train_dataset = ImageDataset(
    train_used,
    spectrogram_dir="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms",
    transform=transform,
    target_cols=TARGET_COLS,
    cache_max_items=2048,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=PERSISTENT,
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
)

test_dataset = ImageDataset(
    test,
    spectrogram_dir="/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms",
    transform=transform,
    target_cols=None,
    cache_max_items=2048,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=PERSISTENT,
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
)




## === cell 8
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)




## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(num_classes).to(device)

weights_path = "/kaggle/input/hms-train-spectogram-images/trained_model.pt"
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)




## === cell 10
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    eps = 1e-12
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y = torch.clamp(y, eps, 1.0)
        y = y / y.sum(dim=1, keepdim=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        p = torch.softmax(logits, dim=1)
        p = torch.clamp(p, eps, 1.0)
        p = p / p.sum(dim=1, keepdim=True)

        loss = -(y * torch.log(p)).sum(dim=1).mean()

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


if not os.path.exists(weights_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    EPOCHS = 1
    for ep in range(EPOCHS):
        _ = train_one_epoch(model, train_loader, optimizer, device)



## === cell 11
model.eval()
out = []
pbar = tqdm(test_loader, desc="infer")
for images in pbar:
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(images)
        outputs = torch.softmax(logits, dim=1)
    out.append(outputs.detach().cpu().numpy())



## === cell 12
if len(out) == 0:
    outputs = np.zeros((len(test), num_classes), dtype=np.float64)
    outputs[:] = 1.0 / num_classes
else:
    outputs = np.vstack(out).astype(np.float64)

outputs = np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

pred_df = pd.DataFrame(outputs, columns=TARGET_COLS)
pred_df["eeg_id"] = test["eeg_id"].values
pred_df = pred_df.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

eps = 1e-6
probs = pred_df[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
pred_df[TARGET_COLS] = probs

submission = (
    test[["eeg_id"]].drop_duplicates("eeg_id").merge(pred_df, on="eeg_id", how="left")
)

missing = submission[TARGET_COLS].isna().any(axis=1)
if missing.any():
    submission.loc[missing, TARGET_COLS] = 1.0 / num_classes

probs = submission[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
submission[TARGET_COLS] = probs

submission.to_csv("submission.csv", index=False)
submission
