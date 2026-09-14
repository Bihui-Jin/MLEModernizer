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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7565281625115422

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.31456) has done: 'I fix the missing `efficientnet_pytorch` dependency by switching to the built-in `torchvision.models.efficientnet_b0` (same EfficientNet-B0 family) so the model can be created reliably in the Kaggle environment. I also fix dataset parsing and label mapping so train/valid and inference use consistent class indices, and correct the image/CSV paths to point at `/kaggle/input/plant-pathology-2021-fgvc8/...`. Finally, I make inference device-safe (no unconditional `.cuda()`), ensure the submission includes **all** images in `sample_submission.csv` order, and always writes `/kaggle/working/submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.31709) has done: 'Your current pipeline is accidentally treating this as a single-label classification problem, but the competition is multi-label (space-delimited labels) and scored by mean F1, so this mismatch is the main reason for the low score. I keep the EfficientNet-B0 backbone and the overall training/inference flow, but change the target encoding to multi-hot vectors, switch the loss to `BCEWithLogitsLoss` (standard for multi-label), and adjust inference to output all labels whose sigmoid probability exceeds a fixed threshold. To keep changes minimal and stable, I derive the class list from the individual tokens in `train.csv` (not from whole strings) and use a simple default threshold (0.5) with a guaranteed non-empty fallback (choose top-1) to avoid blank predictions that hurt F1. The script still write `/kaggle/working/submission.csv` in the required `image,labels` format and keep image ordering identical to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import copy

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms
import torchvision


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV_PATH), f"Missing: {TRAIN_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TVIO = True
except Exception:
    _HAS_TVIO = False

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(480),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(512),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 2
df = pd.read_csv(TRAIN_CSV_PATH)
df["labels"] = df["labels"].astype(str).str.strip()
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

cnt = int(len(df) * 0.9)
train_df = df.iloc[:cnt].reset_index(drop=True)
valid_df = df.iloc[cnt:].reset_index(drop=True)

all_label_tokens = sorted(
    {tok for s in df["labels"].tolist() for tok in str(s).split() if tok}
)
label2idx = {lab: i for i, lab in enumerate(all_label_tokens)}
idx2label = {i: lab for lab, i in label2idx.items()}
num_classes = len(all_label_tokens)
print("num_classes:", num_classes)
print("classes:", all_label_tokens)


def _encode_multilabel_series(
    labels_series: pd.Series, label2idx: dict, num_classes: int
):
    y = torch.zeros((len(labels_series), num_classes), dtype=torch.float32)
    for i, s in enumerate(labels_series.astype(str).tolist()):
        for tok in s.split():
            j = label2idx.get(tok, None)
            if j is not None:
                y[i, j] = 1.0
    return y


train_targets = _encode_multilabel_series(train_df["labels"], label2idx, num_classes)
valid_targets = _encode_multilabel_series(valid_df["labels"], label2idx, num_classes)




## === cell 3
class torchvision_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        df,
        targets=None,
        label2idx=None,
        transforms=None,
        num_classes=0,
    ):
        self.df = df.reset_index(drop=True)
        self.image_path = data_root
        self.label2idx = label2idx
        self.transform = transforms
        self.num_classes = num_classes
        self.targets = targets  # Tensor [N, C] or None

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, path: str):
        if _HAS_TVIO:
            return read_image(path, mode=ImageReadMode.RGB)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image"]

        img = self._load_rgb(os.path.join(self.image_path, image_name))
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)

        if self.targets is not None:
            y = self.targets[idx]
        else:
            label_str = str(row["labels"]).strip()
            y = torch.zeros(self.num_classes, dtype=torch.float32)
            for tok in label_str.split():
                if tok in self.label2idx:
                    y[self.label2idx[tok]] = 1.0

        return x, y




## === cell 4
train_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR,
    train_df,
    targets=train_targets,
    label2idx=label2idx,
    transforms=transform_train,
    num_classes=num_classes,
)
valid_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR,
    valid_df,
    targets=valid_targets,
    label2idx=label2idx,
    transforms=transform_valid,
    num_classes=num_classes,
)




## === cell 5
pin = torch.cuda.is_available()


def _seed_worker(worker_id: int):
    base_seed = 42
    s = base_seed + worker_id
    random.seed(s)
    torch.manual_seed(s)


g = torch.Generator()
g.manual_seed(42)

train_dataloaders = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=min(8, os.cpu_count() or 4),
    pin_memory=pin,
    persistent_workers=True,
    prefetch_factor=4,
    worker_init_fn=_seed_worker,
    generator=g,
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=min(8, os.cpu_count() or 4),
    pin_memory=pin,
    persistent_workers=True,
    prefetch_factor=4,
    worker_init_fn=_seed_worker,
    generator=g,
)




## === cell 6
print(device)




## === cell 7
weights = None  # preserve core logic: no pretrained weights unless provided externally
model_ft = torchvision.models.efficientnet_b0(weights=weights)
in_features = model_ft.classifier[1].in_features
model_ft.classifier[1] = torch.nn.Linear(in_features, num_classes)
model_ft.to(device)




## === cell 8
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)




## === cell 9
criterion = torch.nn.BCEWithLogitsLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, 30, eta_min=0, last_epoch=-1
)
exp_lr_scheduler = GradualWarmupScheduler(
    optimizer_ft, multiplier=100, total_epoch=3, after_scheduler=cosine_scheduler
)




## === cell 10
def _batch_f1_multilabel_from_logits(
    logits, targets, thresh: float = 0.5, eps: float = 1e-9
):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).to(targets.dtype)

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return f1.item()


def _collect_logits_and_targets(model, dataloader):
    model.eval()
    logits_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            logits_list.append(logits.detach().cpu())
            targets_list.append(labels.detach().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)


def _f1_from_cached_logits(
    logits_cpu, targets_cpu, thresh: float = 0.5, eps: float = 1e-9
):
    probs = torch.sigmoid(logits_cpu)
    preds = (probs >= thresh).to(targets_cpu.dtype)

    tp = (preds * targets_cpu).sum()
    fp = (preds * (1.0 - targets_cpu)).sum()
    fn = ((1.0 - preds) * targets_cpu).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(f1.item())


def train_model(
    model, criterion, optimizer, scheduler, num_epochs=25, eval_thresh: float = 0.5
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = -1
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_data_cnt = 0

        for inputs, labels in tqdm(
            train_dataloaders, leave=False, desc=f"Train e{epoch+1}/{num_epochs}"
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)  # logits [B, C]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_data_cnt += inputs.size(0)

        scheduler.step()

        val_logits_cpu, val_targets_cpu = _collect_logits_and_targets(
            model, valid_dataloaders
        )
        epoch_f1 = _f1_from_cached_logits(
            val_logits_cpu, val_targets_cpu, thresh=eval_thresh
        )

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "outputs/best.pth")
            print(f"best epoch : {best_epoch} (valid_f1={best_f1:.5f})")

    return best_model_wts, best_f1, best_epoch




## === cell 11
TRAIN = True
EVAL_THRESH = 0.50  # keep simple default; used only for selecting best epoch & later can be tuned slightly from validation

best_valid_f1 = None
if TRAIN:
    best_wts, best_valid_f1, best_epoch = train_model(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=30,
        eval_thresh=EVAL_THRESH,
    )
    model_ft.load_state_dict(best_wts)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2191273029.py in <cell line: 0>()
      4 best_valid_f1 = None
      5 if TRAIN:
----> 6     best_wts, best_valid_f1, best_epoch = train_model(
      7         model_ft,
      8         criterion,

/tmp/ipykernel_55/1378010704.py in train_model(model, criterion, optimizer, scheduler, num_epochs, eval_thresh)
     61         train_data_cnt = 0
     62 
---> 63         for inputs, labels in tqdm(
     64             train_dataloaders, leave=False, desc=f"Train e{epoch+1}/{num_epochs}"
     65         ):

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

TypeError: Caught TypeError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/1583965.py", line 36, in __getitem__
    x = self.transform(img) if self.transform else transforms.ToTensor()(img)
        ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
          ^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 137, in __call__
    return F.to_tensor(pic)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py", line 142, in to_tensor
    raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")
TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>


## === cell 12
WEIGHT_PATH = "outputs/best.pth"
FALLBACK_EXTERNAL = "/kaggle/input/best-model/27.pth"

loaded = False
if os.path.exists(WEIGHT_PATH):
    state = torch.load(WEIGHT_PATH, map_location="cpu")
    missing, unexpected = model_ft.load_state_dict(state, strict=False)
    print("Loaded weights:", WEIGHT_PATH)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    loaded = True
elif os.path.exists(FALLBACK_EXTERNAL):
    state = torch.load(FALLBACK_EXTERNAL, map_location="cpu")
    missing, unexpected = model_ft.load_state_dict(state, strict=False)
    print("Loaded weights:", FALLBACK_EXTERNAL)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    loaded = True
else:
    print(
        "No weights found; using current model weights (this will likely score poorly)."
    )

model_ft.to(device)




## === cell 13
print("Example labels:", list(label2idx.items())[:5])
if best_valid_f1 is not None:
    print("Best valid F1 observed during training:", best_valid_f1)




## === cell 14
def pick_threshold_on_valid_from_cache(logits_cpu, targets_cpu, thresholds):
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        f1 = _f1_from_cached_logits(logits_cpu, targets_cpu, thresh=float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


THRESH = 0.50
if loaded:
    val_logits_cpu, val_targets_cpu = _collect_logits_and_targets(
        model_ft, valid_dataloaders
    )
    thresh_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    THRESH, val_f1_at_thresh = pick_threshold_on_valid_from_cache(
        val_logits_cpu, val_targets_cpu, thresh_grid
    )
    print(f"Picked THRESH={THRESH:.2f} from grid (valid_f1={val_f1_at_thresh:.5f})")
else:
    print(
        "Skipping threshold tuning because no reliable weights were loaded/trained; using THRESH=0.50"
    )




## === cell 15
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image"].tolist()


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_names, transform):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def _load_rgb(self, path: str):
        if _HAS_TVIO:
            return read_image(path, mode=ImageReadMode.RGB)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img = self._load_rgb(os.path.join(self.img_dir, name))
        x = self.transform(img)
        return name, x


test_dataset = TestDataset(TEST_IMG_DIR, test_images, transform_valid)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=min(8, os.cpu_count() or 4),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=4,
)

submit = []
model_ft.eval()

with torch.no_grad():
    for names, inputs in tqdm(test_loader, desc="Infer"):
        inputs = inputs.to(device, non_blocking=True)
        logits = model_ft(inputs)  # [B, C]
        probs = torch.sigmoid(logits)  # [B, C]
        preds_mask = probs >= THRESH  # bool [B, C]
        top1 = torch.argmax(probs, dim=1)  # [B]

        probs_cpu = probs.detach().cpu()
        preds_mask_cpu = preds_mask.detach().cpu()
        top1_cpu = top1.detach().cpu()

        for i, img_name in enumerate(names):
            nz = preds_mask_cpu[i].nonzero(as_tuple=False).view(-1)
            if nz.numel() == 0:
                nz = top1_cpu[i].view(1)
            pred_labels = " ".join(idx2label[int(j)] for j in nz.tolist())
            submit.append([img_name, pred_labels])

submission = pd.DataFrame(submit, columns=["image", "labels"])
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape", submission.shape)
print(submission.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1859375636.py in <cell line: 0>()
     41 # --- Speed: reduce redundant .cpu() copies and per-item tensor->list overhead; keep exact label selection logic.
     42 with torch.no_grad():
---> 43     for names, inputs in tqdm(test_loader, desc="Infer"):
     44         inputs = inputs.to(device, non_blocking=True)
     45         logits = model_ft(inputs)  # [B, C]

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

TypeError: Caught TypeError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/1859375636.py", line 23, in __getitem__
    x = self.transform(img)
        ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
          ^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 137, in __call__
    return F.to_tensor(pic)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py", line 142, in to_tensor
    raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")
TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>
