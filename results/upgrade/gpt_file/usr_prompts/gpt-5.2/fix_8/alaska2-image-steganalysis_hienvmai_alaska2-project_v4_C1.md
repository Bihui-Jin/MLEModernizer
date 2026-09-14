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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.6894357242271227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm
import timm
import warnings
import gc
from collections import OrderedDict

warnings.filterwarnings("ignore")

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
except Exception:
    pass


class CFG:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "/kaggle/input/alaska2-image-steganalysis/"
    subset_size = 75000
    image_size = 512

    model_name = "tf_efficientnet_b2_ns"
    num_classes = 1

    n_folds = 5
    fold_to_train = 0
    epochs = 1
    train_batch_size = 16
    valid_batch_size = 32
    weight_decay = 1e-6

    lr = 1e-4
    T_0 = 5
    eta_min = 1e-6

    checkpoint_save_path = "/kaggle/working/latest_checkpoint.pth"
    best_model_save_path = f"/kaggle/working/best_model_fold_{fold_to_train}.pth"

    checkpoint_load_path = None

    img_cache_max_items = 4096


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
except Exception:
    pass




## === cell 1
def alaska_weighted_auc(y_true, y_pred):
    """
    Calculates the weighted AUC score for the ALASKA2 competition.
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    if len(fpr) < 2:
        return 0.5

    areas = np.array([0.0] * len(weights))
    for i, lower in enumerate(tpr_thresholds[:-1]):
        upper = tpr_thresholds[i + 1]
        mask = (tpr >= lower) & (tpr < upper)
        if np.any(mask):
            mask_indices = np.where(mask)[0]
            start_idx, end_idx = mask_indices[0], mask_indices[-1]
            tpr_slice = np.concatenate([[lower], tpr[start_idx : end_idx + 1], [upper]])
            fpr_slice = np.concatenate(
                [
                    [np.interp(lower, tpr, fpr)],
                    fpr[start_idx : end_idx + 1],
                    [np.interp(upper, tpr, fpr)],
                ]
            )
            tpr_slice, unique_indices = np.unique(tpr_slice, return_index=True)
            fpr_slice = fpr_slice[unique_indices]
            areas[i] = np.trapz(fpr_slice, tpr_slice)

    return np.sum(areas * weights) / np.sum(weights)




## === cell 2
image_folders = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]

per_folder = CFG.subset_size // len(image_folders)
remainder = CFG.subset_size - per_folder * len(image_folders)

subset_files = []
print(
    f"Creating a subset of {CFG.subset_size} total images ({per_folder} per folder + remainder {remainder})..."
)

rng = np.random.RandomState(CFG.seed)

for i, folder in enumerate(image_folders):
    folder_path = os.path.join(CFG.data_path, folder)
    files = [
        e.name
        for e in os.scandir(folder_path)
        if e.is_file() and e.name.endswith(".jpg")
    ]

    n_take = per_folder + (1 if i < remainder else 0)
    if n_take > len(files):
        n_take = len(files)

    idx = rng.choice(len(files), size=n_take, replace=False)
    taken = [os.path.join(folder_path, files[j]) for j in idx]
    subset_files.extend(taken)
    print(f"  - Took {len(taken)} images from {folder}")

df = pd.DataFrame({"image_path": np.array(subset_files, dtype=object)})

df["label"] = (~df["image_path"].astype(str).str.contains(r"/Cover/")).astype(np.int64)
df["image_id"] = df["image_path"].map(os.path.basename)

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = fold

print("\nSubset dataset distribution:")
print(df["label"].value_counts())
print("\nFold distribution:")
print(df.groupby("fold")["label"].value_counts())




## === cell 3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


def get_transforms(data_type="train"):
    def _transform(image_hwc_uint8: np.ndarray):
        if data_type == "train":
            if random.random() < 0.5:
                image_hwc_uint8 = np.ascontiguousarray(
                    np.flip(image_hwc_uint8, axis=1)
                )  # horizontal
            if random.random() < 0.5:
                image_hwc_uint8 = np.ascontiguousarray(
                    np.flip(image_hwc_uint8, axis=0)
                )  # vertical

        chw = np.transpose(image_hwc_uint8, (2, 0, 1))
        t = torch.from_numpy(chw).to(dtype=torch.float32).div_(255.0)
        t.sub_(_MEAN_T).div_(_STD_T)
        return t

    return _transform


TRAIN_TFMS = get_transforms("train")
VALID_TFMS = get_transforms("valid")
TEST_TFMS = get_transforms("test")


class _LRUImageCache:
    def __init__(self, max_items: int):
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

    def clear(self):
        self._d.clear()


_GLOBAL_IMG_CACHE = _LRUImageCache(CFG.img_cache_max_items)


class AlaskaDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df["image_path"].astype(str).to_numpy()
        self.labels = self.df["label"].astype(np.float32).to_numpy()
        self.transforms = transforms
        self._size = int(CFG.image_size)

    def __len__(self):
        return self.image_paths.shape[0]

    def _read_rgb_resized_uint8(self, image_path: str):
        key = (image_path, self._size)
        cached = _GLOBAL_IMG_CACHE.get(key)
        if cached is not None:
            return cached

        img_rgb = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR_RGB,
        )
        if img_rgb is None:
            img_bgr = cv2.imread(
                image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_rgb = cv2.resize(
            img_rgb, (self._size, self._size), interpolation=cv2.INTER_AREA
        )
        img_rgb = np.ascontiguousarray(img_rgb)

        _GLOBAL_IMG_CACHE.put(key, img_rgb)
        return img_rgb

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        image = self._read_rgb_resized_uint8(image_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label




## === cell 4
_TQDM_MININTERVAL = 0.5


def train_fn(loader, model, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False, mininterval=_TQDM_MININTERVAL)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        if images.is_cuda:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix(
            loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"])
        )

    return running_loss / max(1, len(loader))


def eval_fn(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0

    n = len(loader.dataset)
    all_preds = np.empty(n, dtype=np.float32)
    all_labels = np.empty(n, dtype=np.float32)
    offset = 0

    with torch.inference_mode():
        pbar = tqdm(
            loader, desc="Evaluating", leave=False, mininterval=_TQDM_MININTERVAL
        )
        for images, labels in pbar:
            bsz = images.size(0)
            images = images.to(device, non_blocking=True)
            if images.is_cuda:
                images = images.contiguous(memory_format=torch.channels_last)
            labels_t = labels.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels_t)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
            labs = labels_t.detach().cpu().numpy().ravel()
            all_preds[offset : offset + bsz] = probs
            all_labels[offset : offset + bsz] = labs
            offset += bsz

    val_loss = running_loss / max(1, len(loader))
    score = alaska_weighted_auc(all_labels, all_preds)
    return val_loss, score




## === cell 5
torch.cuda.empty_cache()
gc.collect()




## === cell 6
def _seed_worker(worker_id: int):
    worker_seed = (CFG.seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _recommended_num_workers():
    ncpu = os.cpu_count() or 4
    return int(max(2, min(8, ncpu // 2)))


def run_training(fold):
    print(f"========== Starting Training for Fold {fold} ==========")

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = AlaskaDataset(train_df, transforms=TRAIN_TFMS)
    valid_dataset = AlaskaDataset(valid_df, transforms=VALID_TFMS)

    num_workers = _recommended_num_workers()

    g = torch.Generator()
    g.manual_seed(CFG.seed)

    prefetch = 8 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    model = timm.create_model(
        CFG.model_name, pretrained=True, num_classes=CFG.num_classes
    ).to(CFG.device)
    if CFG.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.T_0 * max(1, len(train_loader)), eta_min=CFG.eta_min
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_score = 0.0
    if CFG.checkpoint_load_path and os.path.exists(CFG.checkpoint_load_path):
        print(f"Resuming training from checkpoint: {CFG.checkpoint_load_path}")
        checkpoint = torch.load(
            CFG.checkpoint_load_path, map_location=CFG.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", 0.0))
        print(
            f"Loaded model from epoch {start_epoch-1} with best score: {best_score:.4f}"
        )
    else:
        print("No checkpoint found, starting training from scratch.")

    for epoch in range(start_epoch, CFG.epochs):
        print(f"\n--- Epoch {epoch+1}/{CFG.epochs} ---")
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, CFG.device
        )
        val_loss, val_score = eval_fn(valid_loader, model, criterion, CFG.device)

        print(
            f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Weighted AUC: {val_score:.4f}"
        )

        if val_score > best_score:
            print(
                f"Validation score improved! ({best_score:.4f} -> {val_score:.4f}). Saving best model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), CFG.best_model_save_path)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(checkpoint, CFG.checkpoint_save_path)
        print(f"Epoch {epoch+1} state saved to checkpoint: {CFG.checkpoint_save_path}")

    print(
        f"\n========== Finished Training for Fold {fold}. Best Score: {best_score:.4f} =========="
    )


run_training(CFG.fold_to_train)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/2742274461.py in <cell line: 0>()
    119 
    120 
--> 121 run_training(CFG.fold_to_train)
    122 
    123 

/tmp/ipykernel_55/2742274461.py in run_training(fold)
     88     for epoch in range(start_epoch, CFG.epochs):
     89         print(f"\n--- Epoch {epoch+1}/{CFG.epochs} ---")
---> 90         train_loss = train_fn(
     91             train_loader, model, criterion, optimizer, scheduler, CFG.device
     92         )

/tmp/ipykernel_55/1410675447.py in train_fn(loader, model, criterion, optimizer, scheduler, device)
      6     running_loss = 0.0
      7     pbar = tqdm(loader, desc="Training", leave=False, mininterval=_TQDM_MININTERVAL)
----> 8     for images, labels in pbar:
      9         images = images.to(device, non_blocking=True)
     10         if images.is_cuda:

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

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

error: Caught error in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/521287490.py", line 106, in __getitem__
    image = self._read_rgb_resized_uint8(image_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/521287490.py", line 81, in _read_rgb_resized_uint8
    img_rgb = cv2.imread(
              ^^^^^^^^^^^
cv2.error: OpenCV(4.12.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:88: error: (-2:Unspecified error) in function 'int cv::calcType(int, int)'
> IMREAD_COLOR_BGR (IMREAD_COLOR) and IMREAD_COLOR_RGB can not be set at the same time. (expected: 'flags & (IMREAD_COLOR_BGR | IMREAD_COLOR_RGB) != IMREAD_COLOR_BGR | IMREAD_COLOR_RGB'), where
>     'flags & (IMREAD_COLOR_BGR | IMREAD_COLOR_RGB)' is 257
> must be not equal to
>     'IMREAD_COLOR_BGR | IMREAD_COLOR_RGB' is 257



## === cell 7
def create_submission():
    print("\nStarting inference on the test set...")

    test_folder = os.path.join(CFG.data_path, "Test")

    test_image_ids = sorted(
        [
            e.name
            for e in os.scandir(test_folder)
            if e.is_file() and e.name.endswith(".jpg")
        ]
    )

    test_paths = [os.path.join(test_folder, x) for x in test_image_ids]
    test_df = pd.DataFrame({"image_id": np.array(test_image_ids, dtype=object)})
    test_df["image_path"] = np.array(test_paths, dtype=object)
    test_df["label"] = 0  # dummy

    test_dataset = AlaskaDataset(test_df, transforms=TEST_TFMS)

    num_workers = _recommended_num_workers()
    prefetch = 8 if num_workers > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    model = timm.create_model(
        CFG.model_name, pretrained=False, num_classes=CFG.num_classes
    )
    model.load_state_dict(
        torch.load(CFG.best_model_save_path, map_location="cpu", weights_only=False)
    )
    model.to(CFG.device)
    if CFG.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    predictions = np.empty(len(test_dataset), dtype=np.float32)
    offset = 0

    with torch.inference_mode():
        pbar = tqdm(
            test_loader, desc="Predicting", leave=False, mininterval=_TQDM_MININTERVAL
        )
        for images, _ in pbar:
            bsz = images.size(0)
            images = images.to(CFG.device, non_blocking=True)
            if images.is_cuda:
                images = images.contiguous(memory_format=torch.channels_last)
            outputs = model(images)
            probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
            predictions[offset : offset + bsz] = probs
            offset += bsz

    submission_df = (
        pd.DataFrame({"Id": test_image_ids, "Label": predictions.tolist()})
        .sort_values("Id")
        .reset_index(drop=True)
    )
    submission_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission file created successfully at: {submission_path}")
    print(submission_df.head())
    return submission_df


_ = create_submission()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2794366882.py in <cell line: 0>()
     75 
     76 
---> 77 _ = create_submission()

/tmp/ipykernel_55/2794366882.py in create_submission()
     38     )
     39     model.load_state_dict(
---> 40         torch.load(CFG.best_model_save_path, map_location="cpu", weights_only=False)
     41     )
     42     model.to(CFG.device)

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/best_model_fold_0.pth'
