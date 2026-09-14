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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
seaborn==0.12.2
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

0.738670360110804

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24378) has done: 'I fix the two execution blockers: (1) the missing pretrained weights path by falling back to loading ImageNet weights when the custom `.pth` file isn’t available, so inference can run end-to-end; and (2) the deprecated `DataFrame.append` usage by collecting rows and building the submission DataFrame once (pandas 2.x compatible). I also make label extraction in the dataset robust to non-default indices by using `.iloc` instead of assuming the index equals `idx`, which prevents subtle misalignment bugs. These changes preserve the core approach (ResNet50 + sigmoid + thresholding) and ensure a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torchmetrics
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.models as models
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    cv2.setNumThreads(0)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171)  # image size (W, H)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
train_data["labels"] = train_data["labels"].apply(lambda string: string.split(" "))

s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)
labels_size = len(train_labels.columns)
train_df = pd.concat([train_data[["image"]], train_labels], axis=1)

sample_sub = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images_names = [
    os.path.join(test_images_path, img) for img in sample_sub["image"].tolist()
]
assert len(test_images_names) == len(
    sample_sub
), "Test list must match sample_submission rows"



## === cell 1
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_STD_CPU = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super(PlantDataSet, self).__init__()
        self.images_path = images_path
        self.transform = transform  # kept for API compatibility; not used

        if self.images_path is not None:
            self.image_names = dataset["image"].to_numpy()
            self.labels = dataset.drop(columns=["image"]).to_numpy(
                dtype=np.float32, copy=False
            )
            self.is_test = False
            self._base = self.images_path
            self._dummy_target = None
        else:
            self.image_paths = list(dataset)
            self.is_test = True
            self._dummy_target = torch.zeros((0,), dtype=torch.float32)

    def __getitem__(self, idx):
        if not self.is_test:
            img_name = self.image_names[idx]
            path = self._base + img_name
            labels = torch.from_numpy(self.labels[idx])
        else:
            path = self.image_paths[idx]
            labels = self._dummy_target

        img = read_image(path, mode=ImageReadMode.RGB)  # uint8, CHW, variable H/W
        return img, labels

    def __len__(self):
        return len(self.image_paths) if self.is_test else len(self.image_names)


transform = None  # normalization is done later (on device)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.1, random_state=42, shuffle=True
)
train_subset = train_df.iloc[train_idx].reset_index(drop=True)
val_subset = train_df.iloc[val_idx].reset_index(drop=True)

train_dataset = PlantDataSet(
    train_subset, "/kaggle/input/plant-pathology-2021-fgvc8/train_images/", transform
)
val_dataset = PlantDataSet(
    val_subset, "/kaggle/input/plant-pathology-2021-fgvc8/train_images/", transform
)

BS = 30


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

NUM_WORKERS = min(8, (os.cpu_count() or 2))
PIN = torch.cuda.is_available()
PREFETCH = 4 if NUM_WORKERS > 0 else None


def _collate_pad_to_tensor(batch):
    images, targets = zip(*batch)

    max_h = max(int(im.shape[1]) for im in images)
    max_w = max(int(im.shape[2]) for im in images)
    b = len(images)

    out = torch.zeros((b, 3, max_h, max_w), dtype=torch.uint8)
    for i, im in enumerate(images):
        _, h, w = im.shape
        out[i, :, :h, :w] = im

    if torch.is_tensor(targets[0]) and targets[0].numel() > 0:
        targets = torch.stack(targets, dim=0)
    else:
        targets = targets[0]
    return out, targets


train_loader = DataLoader(
    train_dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH,
    worker_init_fn=_seed_worker,
    generator=g,
    collate_fn=_collate_pad_to_tensor,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH,
    worker_init_fn=_seed_worker,
    generator=g,
    collate_fn=_collate_pad_to_tensor,
)

test_dataset = PlantDataSet(test_images_names, None, transform)
plants_test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH,
    worker_init_fn=_seed_worker,
    generator=g,
    collate_fn=_collate_pad_to_tensor,
)




## === cell 2
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)

        def _preload():
            nonlocal next_images, next_targets
            try:
                next_images, next_targets = next(it)
            except StopIteration:
                next_images = None
                next_targets = None
                return
            with torch.cuda.stream(self.stream):
                next_images = next_images.to(self.device, non_blocking=True)
                if torch.is_tensor(next_targets):
                    next_targets = next_targets.to(self.device, non_blocking=True)

        next_images = None
        next_targets = None
        _preload()
        while next_images is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            images = next_images
            targets = next_targets
            _preload()
            yield images, targets


_MEAN_CACHE = {}
_STD_CACHE = {}


def preprocess_on_device(images_u8: torch.Tensor) -> torch.Tensor:
    if images_u8.dtype != torch.uint8:
        images_u8 = images_u8.to(dtype=torch.uint8)

    images = images_u8.to(dtype=torch.float32).div_(255.0)
    images = F.interpolate(
        images,
        size=(DIMENTION[1], DIMENTION[0]),
        mode="bilinear",
        align_corners=False,
    )

    if device not in _MEAN_CACHE:
        _MEAN_CACHE[device] = _MEAN_CPU.to(device)
        _STD_CACHE[device] = _STD_CPU.to(device)
    images = images.sub_(_MEAN_CACHE[device]).div_(_STD_CACHE[device])

    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def test(test_dataloader, model):
    model.eval()
    model = model.to(device)
    preds = []
    dl = CUDAPrefetcher(test_dataloader, device)
    with torch.inference_mode():
        for images, _ in dl:
            images = preprocess_on_device(images)
            output = model(images)
            probabilities = torch.sigmoid(output)
            preds.append(probabilities.cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 3
weights_path = "/kaggle/input/resnet50-final/resnet50_final.pth"

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = resnet50.fc.in_features
resnet50.fc = torch.nn.Linear(in_features, labels_size)

if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    try:
        resnet50.load_state_dict(state, strict=True)
        print(f"Loaded custom weights from: {weights_path}")
    except Exception as e:
        print(
            f"Warning: failed to load custom weights strictly ({e}); using ImageNet backbone weights only."
        )
else:
    print(
        f"Warning: custom weights not found at {weights_path}; using ImageNet backbone weights only."
    )




## === cell 4
def create_submission(test_images_path, predictions, threshold=0.6):
    names = np.fromiter((os.path.basename(p) for p in test_images_path), dtype=object)
    cls = np.array(train_labels.columns.tolist(), dtype=object)

    pred_mask = predictions > threshold  # keep strict '>' for compatibility
    any_pos = pred_mask.any(axis=1)

    labels_out = np.empty(pred_mask.shape[0], dtype=object)

    pos_rows = np.flatnonzero(any_pos)
    for i in pos_rows:
        idx = np.flatnonzero(pred_mask[i])
        labels_out[i] = " ".join(cls[idx].tolist())

    neg_rows = np.flatnonzero(~any_pos)
    if neg_rows.size > 0:
        top1 = np.argmax(predictions[neg_rows], axis=1)
        labels_out[neg_rows] = cls[top1]

    submission_df = pd.DataFrame({"image": names, "labels": labels_out})
    submission_df.to_csv("submission.csv", index=False)
    return submission_df


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    dl = CUDAPrefetcher(loader, device)
    for images, targets in dl:
        images = preprocess_on_device(images)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    all_targets = []
    dl = CUDAPrefetcher(loader, device)
    for images, targets in dl:
        images = preprocess_on_device(images)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def macro_f1_from_probs(y_true, y_prob, thr):
    y_pred = y_prob >= thr
    yt = y_true.astype(np.int32) == 1
    yp = y_pred

    tp = np.sum(yt & yp, axis=0).astype(np.float64)
    fp = np.sum((~yt) & yp, axis=0).astype(np.float64)
    fn = np.sum(yt & (~yp), axis=0).astype(np.float64)

    eps = 1e-9
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(np.mean(f1))


resnet50 = resnet50.to(device)
if device.type == "cuda":
    resnet50 = resnet50.to(memory_format=torch.channels_last)

if device.type == "cuda":
    try:
        resnet50 = torch.compile(resnet50, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=2e-4)

EPOCHS = 2
for epoch in range(EPOCHS):
    loss = train_one_epoch(resnet50, train_loader, optimizer, criterion)
    print(f"epoch {epoch+1}/{EPOCHS} - train_loss: {loss:.5f}")

val_probs, val_true = predict_probs(resnet50, val_loader)
threshold_grid = np.round(np.arange(0.30, 0.71, 0.05), 2)
best_thr = 0.6
best_f1 = -1.0
for thr in threshold_grid:
    f1 = macro_f1_from_probs(val_true, val_probs, thr)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(thr)
print(f"selected_threshold: {best_thr} (val_macro_f1={best_f1:.5f})")

test_predictions = test(plants_test_data_loader, resnet50)
print(test_predictions.shape)
submission_df = create_submission(
    test_images_names, test_predictions, threshold=best_thr
)
print(submission_df.head())
print("Wrote submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_55/3931552495.py in <cell line: 0>()
     85 EPOCHS = 2
     86 for epoch in range(EPOCHS):
---> 87     loss = train_one_epoch(resnet50, train_loader, optimizer, criterion)
     88     print(f"epoch {epoch+1}/{EPOCHS} - train_loss: {loss:.5f}")
     89 

/tmp/ipykernel_55/3931552495.py in train_one_epoch(model, loader, optimizer, criterion)
     31         images = preprocess_on_device(images)
     32         optimizer.zero_grad(set_to_none=True)
---> 33         logits = model(images)
     34         loss = criterion(logits, targets)
     35         loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1099             # optimization to generate better code in a common case
   1100             self.add_output_instructions(
-> 1101                 self.compile_and_call_fx_graph(
   1102                     tx, list(reversed(stack_values)), root, output_replacements
   1103                 )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1550     if config_patches:
   1551         with config.patch(config_patches):
-> 1552             return compile_fx(
   1553                 model_,
   1554                 example_inputs_,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_autograd(flat_fn, flat_args, aot_config, fw_metadata)
    676 
    677             with TracingContext.report_output_strides() as fwd_output_strides:
--> 678                 compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)
    679 
    680             if not hasattr(compiled_fw_func, "_boxed_call"):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in __call__(self, gm, example_inputs)
    487         example_inputs: Sequence[InputType],
    488     ) -> OutputCode:
--> 489         return self.compiler_fn(gm, example_inputs)
    490 
    491 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fw_compiler_base(gm, example_inputs, is_inference)
   1739                     model_outputs_node.meta["user_visible_output_idxs"] = []
   1740 
-> 1741                 return inner_compile(
   1742                     gm,
   1743                     example_inputs,

/usr/lib/python3.11/contextlib.py in inner(*args, **kwds)
     79         def inner(*args, **kwds):
     80             with self._recreate_cm():
---> 81                 return func(*args, **kwds)
     82         return inner
     83 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx_inner(gm, example_inputs, **kwargs)
    567         )
    568 
--> 569         return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
    570             gm,
    571             example_inputs,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_aot.py in debug_wrapper(gm, example_inputs, **kwargs)
    100             # Call the compiler_fn - which is either aot_autograd or inductor
    101             # with fake inputs
--> 102             inner_compiled_fn = compiler_fn(gm, example_inputs)
    103         except Exception as e:
    104             # TODO: Failures here are troublesome because no real inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _compile_fx_inner(gm, example_inputs, **graph_kwargs)
    683             TritonBundler.begin_compile()
    684             try:
--> 685                 mb_compiled_graph = fx_codegen_and_compile(
    686                     gm, example_inputs, inputs_to_check, **graph_kwargs
    687                 )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fx_codegen_and_compile(gm, example_inputs, inputs_to_check, **graph_kwargs)
   1127     scheme: FxCompile = _InProcessFxCompile()
   1128 
-> 1129     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
   1130 
   1131 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in codegen_and_compile(self, gm, example_inputs, inputs_to_check, graph_kwargs)
    977                 metrics_helper = metrics.CachedMetricsHelper()
    978                 with V.set_graph_handler(graph):
--> 979                     graph.run(*example_inputs)
    980                     output_strides: List[Optional[Tuple[_StrideExprStr, ...]]] = []
    981                     if graph.graph_outputs is not None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in run(self, *args)
    853     def run(self, *args: Any) -> Any:  # type: ignore[override]
    854         with dynamo_timed("GraphLowering.run"):
--> 855             return super().run(*args)
    856 
    857     def register_operation(self, op: ir.Operation) -> str:

/usr/local/lib/python3.11/dist-packages/torch/fx/interpreter.py in run(self, initial_env, enable_io_processing, *args)
    165 
    166             try:
--> 167                 self.env[node] = self.run_node(node)
    168             except Exception as e:
    169                 if self.extra_traceback:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in run_node(self, n)
   1494             else:
   1495                 debug("")
-> 1496                 result = super().run_node(n)
   1497 
   1498             # require the same stride order for dense outputs,

/usr/local/lib/python3.11/dist-packages/torch/fx/interpreter.py in run_node(self, n)
    228             assert isinstance(args, tuple)
    229             assert isinstance(kwargs, dict)
--> 230             return getattr(self, n.op)(n.target, args, kwargs)
    231 
    232     # Main Node running APIs

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in call_function(self, target, args, kwargs)
   1141             return out
   1142         except Exception as e:
-> 1143             raise LoweringException(e, target, args, kwargs).with_traceback(
   1144                 e.__traceback__
   1145             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in call_function(self, target, args, kwargs)
   1131                 args, kwargs = layout_constraints(n, *args, **kwargs)
   1132 
-> 1133             out = lowerings[target](*args, **kwargs)  # type: ignore[index]
   1134 
   1135             if layout_constraints:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/lowering.py in wrapped(*args, **kwargs)
    400             ), "out= ops aren't yet supported"
    401 
--> 402         args, kwargs = transform_args(
    403             args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool
    404         )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/lowering.py in transform_args(args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool)
    314 
    315     if broadcast:
--> 316         broadcasted = broadcast_tensors(
    317             *list(
    318                 itertools.chain(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/lowering.py in wrapped(*args, **kwargs)
    407             args = [args]
    408 
--> 409         out = decomp_fn(*args, **kwargs)
    410         validate_ir(out)
    411 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/lowering.py in broadcast_tensors(*inputs)
    886     for x in inputs:
    887         sizes = x.get_size()
--> 888         if len(sizes) != len(target) or any(
    889             (
    890                 (

/usr/local/lib/python3.11/dist-packages/torch/_inductor/lowering.py in <genexpr>(.0)
    901                     )
    902                     and V.graph.sizevars.shape_env.evaluate_expr(
--> 903                         sympy.Eq(b, 1), size_oblivious=True
    904                     )
    905                 )

/usr/local/lib/python3.11/dist-packages/sympy/core/relational.py in __new__(cls, lhs, rhs, **options)
    621         rhs = _sympify(rhs)
    622         if evaluate:
--> 623             val = is_eq(lhs, rhs)
    624             if val is None:
    625                 return cls(lhs, rhs, evaluate=False)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     71         # This following call uses `waitid` with WNOHANG from C side. Therefore,
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:
     75             assert callable(previous_handler)

BackendCompilerFailed: backend='inductor' raised:
LoweringException: RuntimeError: DataLoader worker (pid 109) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
  target: aten.add.Tensor
  args[0]: TensorBox(StorageBox(
    Pointwise(
      'cuda',
      torch.float32,
      def inner_fn(index):
          i0, i1, i2, i3 = index
          tmp0 = ops.load(buf134, i1 + 128 * i3 + 8192 * i2 + 352256 * i0)
          tmp1 = ops.load(buf141, i1)
          tmp2 = tmp0 - tmp1
          tmp3 = ops.load(buf142, i1)
          tmp4 = ops.constant(0, torch.float32)
          tmp5 = ops.constant(82560, torch.float32)
          tmp6 = ops.constant(0, torch.float32)
          tmp7 = tmp5 - tmp4
          tmp8 = ops.maximum(tmp6, tmp7)
          tmp9 = tmp3 / tmp8
          tmp10 = ops.constant(1e-05, torch.float32)
          tmp11 = tmp9 + tmp10
          tmp12 = ops.rsqrt(tmp11)
          tmp13 = tmp2 * tmp12
          tmp14 = ops.load(primals_72, i1)
          tmp15 = tmp13 * tmp14
          return tmp15
      ,
      ranges=[30, 128, 43, 64],
      origin_node=mul_83,
      origins=OrderedSet([var_mean_11, sub_11, mul_83, add_59, mul_...
    )
  ))
  args[1]: TensorBox(
    ReinterpretView(
      StorageBox(
        InputBuffer(name='primals_73', layout=FixedLayout('cuda:0', torch.float32, size=[128], stride=[1]))
      ),
      FixedLayout('cuda:0', torch.float32, size=[128, 1, 1], stride=[1, 1, 1]),
      origins=OrderedSet([unsqueeze_47])
    )
  )

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
