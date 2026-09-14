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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

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
pillow==11.3.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.8682799992482861

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the failing Albumentations dependency (it’s importing SciPy, which is broken in this environment) and replace it with an equivalent torchvision preprocessing pipeline (resize + normalize) so the dataset and dataloader work end-to-end. I also make the EfficientNet import robust by using torchvision’s built-in `efficientnet_b0` and load weights from your `MODEL_PATH` with safe `map_location`, while keeping the same “EfficientNet-B0 + replaced final FC to 2 classes” core architecture. I fix the submission logic to output probabilities (not hard-thresholded labels), since the competition metric is AUC and expects probabilities. Finally, I ensure the test ids come from `sample_submission.csv` to guarantee ordering/coverage and always write `submission.csv`.'
- What this solution (achieved 0.47665) has done: 'I fix the runtime failure by making model checkpoint loading robust: if the provided `MODEL_PATH` doesn’t exist, we automatically fall back to torchvision’s built-in EfficientNet-B0 ImageNet weights so inference can run end-to-end. This is a minimal change that preserves the same core architecture (EfficientNet-B0 with a replaced final linear layer) while avoiding the FileNotFoundError. Using pretrained backbone weights should also move the AUC materially up from ~0.5 toward your target, compared to random initialization. I also remove the unnecessary `pip install` cell to avoid network dependence and keep the rest of the pipeline (transforms, dataloader, probability submission) unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm

import torchvision
from torchvision import transforms



## === cell 1
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
TRAIN_CSV = f"{DATA_DIR}/train_labels.csv"

MODEL_PATH = "/kaggle/input/as-week-4-baseline-cnn-training/model_best.pth"  # Provided path (may be missing)
SUBMISSION_FILE = "submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET_SIZE = (96, 96)

BATCH_SIZE = 256
FALLBACK_BATCH_SIZE = 64

NUM_CLASSES = 2  # Kept as in the original solution (2-logit softmax)

EPOCHS = 1
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.1
SEED = 42

assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isfile(TRAIN_CSV), f"TRAIN_CSV not found: {TRAIN_CSV}"
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

Image.MAX_IMAGE_PIXELS = None



## === cell 2
default_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
weights_transforms = default_weights.transforms()

train_transforms = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights_transforms.mean, std=weights_transforms.std),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights_transforms.mean, std=weights_transforms.std),
    ]
)



## === cell 3
from torchvision.io import read_image, ImageReadMode


class HistologyDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_label: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_label = with_label

        self.ids = self.df["id"].astype(str).to_numpy()
        if self.with_label:
            self.labels = self.df["label"].astype(np.int64).to_numpy()
        else:
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = str(self.ids[idx])
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")

        img_t = read_image(img_path, mode=ImageReadMode.RGB)

        img = transforms.functional.to_pil_image(img_t)
        img = self.transform(img)

        if self.with_label:
            y = int(self.labels[idx])
            return img, y
        else:
            return img, img_id




## === cell 4
labels_df = pd.read_csv(TRAIN_CSV)
labels_df["id"] = labels_df["id"].astype(str)
labels_df["label"] = labels_df["label"].astype(int)
print("Train labels:", labels_df.shape, "pos_rate:", labels_df["label"].mean())

pos_df = (
    labels_df[labels_df["label"] == 1]
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
neg_df = (
    labels_df[labels_df["label"] == 0]
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
pos_val_n = int(len(pos_df) * VAL_FRAC)
neg_val_n = int(len(neg_df) * VAL_FRAC)

val_df = pd.concat([pos_df.iloc[:pos_val_n], neg_df.iloc[:neg_val_n]], axis=0).sample(
    frac=1.0, random_state=SEED
)
train_df = pd.concat([pos_df.iloc[pos_val_n:], neg_df.iloc[neg_val_n:]], axis=0).sample(
    frac=1.0, random_state=SEED
)

print("Train split:", train_df.shape, "pos_rate:", train_df["label"].mean())
print("Val split:", val_df.shape, "pos_rate:", val_df["label"].mean())

train_dataset = HistologyDataset(train_df, TRAIN_DIR, train_transforms, with_label=True)
val_dataset = HistologyDataset(val_df, TRAIN_DIR, test_transforms, with_label=True)

cpu_cnt = os.cpu_count() or 4
NUM_WORKERS = min(6, max(2, cpu_cnt - 1))


def _make_loader(ds, batch_size, shuffle, with_labels):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )


train_loader = _make_loader(train_dataset, BATCH_SIZE, True, True)
val_loader = _make_loader(val_dataset, BATCH_SIZE, False, True)

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_df = pd.read_csv(sample_path)
sample_df["id"] = sample_df["id"].astype(str)
test_img_ids = sample_df["id"].tolist()
print(f"Total test images (from sample_submission): {len(test_img_ids)}")

test_dataset = HistologyDataset(
    pd.DataFrame({"id": test_img_ids}), TEST_DIR, test_transforms, with_label=False
)
test_loader = _make_loader(test_dataset, BATCH_SIZE, False, False)




## === cell 5
class CancerClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, backbone_weights=None):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=backbone_weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


ckpt_exists = isinstance(MODEL_PATH, str) and os.path.isfile(MODEL_PATH)
if ckpt_exists:
    model = CancerClassifier(backbone_weights=None).to(device)
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from: {MODEL_PATH}")
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    model = CancerClassifier(backbone_weights=default_weights).to(device)
    print(f"WARNING: Checkpoint not found at '{MODEL_PATH}'.")
    print(
        "Falling back to torchvision EfficientNet-B0 ImageNet weights, then fine-tuning on train_labels.csv."
    )

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="max-autotune")
        print("torch.compile enabled.")
    except Exception as e:
        print(f"torch.compile not enabled (fallback to eager). Reason: {e}")



## === cell 6
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def binary_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    order = np.argsort(y_score)
    y_true = y_true[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.inference_mode()
def evaluate_auc(model, loader):
    model.eval()
    n = len(loader.dataset)
    y_true = np.empty(n, dtype=np.int32)
    y_score = np.empty(n, dtype=np.float64)
    o = 0
    for x, y in loader:
        if device.type == "cuda":
            x = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            x = x.to(device, non_blocking=True)
        logits = model(x)
        prob_pos = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        bs = prob_pos.shape[0]
        y_true[o : o + bs] = y.numpy()
        y_score[o : o + bs] = prob_pos
        o += bs
    return binary_auc_roc(y_true, y_score)


def _train_one_epoch():
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for x, y in tqdm(train_loader, desc=f"Training epoch {epoch}/{EPOCHS}"):
            if device.type == "cuda":
                x = x.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)
        val_auc = evaluate_auc(model, val_loader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}")


try:
    _train_one_epoch()
except torch.cuda.OutOfMemoryError:
    print(
        "CUDA OOM with BATCH_SIZE=",
        BATCH_SIZE,
        "-> falling back to",
        FALLBACK_BATCH_SIZE,
    )
    torch.cuda.empty_cache()
    BATCH_SIZE = FALLBACK_BATCH_SIZE
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    _train_one_epoch()

model.eval()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_53/1490993417.py in <cell line: 0>()
     79 
     80 try:
---> 81     _train_one_epoch()
     82 except torch.cuda.OutOfMemoryError:
     83     print(

/tmp/ipykernel_53/1490993417.py in _train_one_epoch()
     54         n_seen = 0
     55 
---> 56         for x, y in tqdm(train_loader, desc=f"Training epoch {epoch}/{EPOCHS}"):
     57             if device.type == "cuda":
     58                 x = x.to(device, non_blocking=True).contiguous(

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_53/3237891071.py", line 29, in __getitem__
    img_t = read_image(img_path, mode=ImageReadMode.RGB)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/io/image.py", line 337, in read_image
    return decode_image(data, mode, apply_exif_orientation=apply_exif_orientation)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/io/image.py", line 324, in decode_image
    output = torch.ops.image.decode_image(input, mode.value, apply_exif_orientation)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 1123, in __call__
    return self._op(*args, **(kwargs or {}))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Unsupported image file. Only jpeg, png, webp and gif are currently supported. For avif and heic format, please rely on `decode_avif` and `decode_heic` directly.


## === cell 7
@torch.inference_mode()
def generate_predictions(model, loader):
    """
    Generate predictions for the test set.
    Returns:
        ids: list[str]
        predictions: list[float] probabilities for the positive class
    """
    model.eval()

    n = len(loader.dataset)
    preds = np.empty(n, dtype=np.float32)
    ids = [None] * n
    o = 0

    for images, img_ids in tqdm(loader, desc="Generating predictions"):
        if device.type == "cuda":
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = (
            torch.softmax(outputs, dim=1)[:, 1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        bs = probs.shape[0]
        preds[o : o + bs] = probs
        ids[o : o + bs] = list(img_ids)
        o += bs

    return ids, preds.tolist()


img_ids, predictions = generate_predictions(model, test_loader)
print("Predictions generated:", len(predictions))




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_53/2743454743.py in <cell line: 0>()
     41 
     42 
---> 43 img_ids, predictions = generate_predictions(model, test_loader)
     44 print("Predictions generated:", len(predictions))
     45 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_53/2743454743.py in generate_predictions(model, loader)
     16     o = 0
     17 
---> 18     for images, img_ids in tqdm(loader, desc="Generating predictions"):
     19         if device.type == "cuda":
     20             images = images.to(device, non_blocking=True).contiguous(

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_53/3237891071.py", line 29, in __getitem__
    img_t = read_image(img_path, mode=ImageReadMode.RGB)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/io/image.py", line 337, in read_image
    return decode_image(data, mode, apply_exif_orientation=apply_exif_orientation)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/io/image.py", line 324, in decode_image
    output = torch.ops.image.decode_image(input, mode.value, apply_exif_orientation)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 1123, in __call__
    return self._op(*args, **(kwargs or {}))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Unsupported image file. Only jpeg, png, webp and gif are currently supported. For avif and heic format, please rely on `decode_avif` and `decode_heic` directly.


## === cell 8
def prepare_submission(img_ids, predictions):
    """
    For AUC evaluation, submit probabilities (floats), not hard-thresholded labels.
    """
    submission_df = pd.DataFrame({"id": img_ids, "label": predictions})
    submission_df["id"] = submission_df["id"].astype(str)

    submission_df = sample_df[["id"]].merge(submission_df, on="id", how="left")
    if submission_df["label"].isna().any():
        submission_df["label"] = submission_df["label"].fillna(0.5)

    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(
        f"Submission file '{SUBMISSION_FILE}' created with shape {submission_df.shape}."
    )
    return submission_df


sub_df = prepare_submission(img_ids, predictions)
print(sub_df.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_53/2608249447.py in <cell line: 0>()
     17 
     18 
---> 19 sub_df = prepare_submission(img_ids, predictions)
     20 print(sub_df.head())
     21 

NameError: name 'img_ids' is not defined

## === cell 9
assert os.path.isfile(SUBMISSION_FILE), "submission.csv was not created"
check = pd.read_csv(SUBMISSION_FILE)
assert list(check.columns) == ["id", "label"], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(sample_df), f"Bad row count: {len(check)} vs {len(sample_df)}"
assert check["label"].between(0, 1).all(), "Labels must be probabilities in [0,1]"
print("Submission looks valid.")
print(check.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_53/3877771216.py in <cell line: 0>()
----> 1 assert os.path.isfile(SUBMISSION_FILE), "submission.csv was not created"
      2 check = pd.read_csv(SUBMISSION_FILE)
      3 assert list(check.columns) == ["id", "label"], f"Bad columns: {check.columns.tolist()}"
      4 assert len(check) == len(sample_df), f"Bad row count: {len(check)} vs {len(sample_df)}"
      5 assert check["label"].between(0, 1).all(), "Labels must be probabilities in [0,1]"

AssertionError: submission.csv was not created
