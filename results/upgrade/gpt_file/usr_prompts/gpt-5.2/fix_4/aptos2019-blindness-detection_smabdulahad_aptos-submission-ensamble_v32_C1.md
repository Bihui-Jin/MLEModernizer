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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.12

# 3. Installed packages

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7956236328847175

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.42663) has done: 'I fix the immediate runtime break in image resizing by replacing the removed `Image.ANTIALIAS` with the Pillow 11+ equivalent (`Image.Resampling.LANCZOS`) and add missing imports used in `fast_image_resize`. Next, I make the resized-image folder path consistent (you save into `.../images_resized/` but were reading from `.../images_resized/` without the extra `test/` subfolder) and ensure images are always converted to RGB before transforms. Finally, since the referenced ensemble weight files are not present in this environment, I fall back to a standard pretrained timm model (same core inference pipeline) so the notebook runs end-to-end and produces a valid `submission.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import os
import warnings
import shutil
from multiprocessing import Pool, set_start_method

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

try:
    set_start_method("spawn", force=True)
except Exception:
    pass

torch.manual_seed(42)
np.random.seed(42)




## === cell 1
try:
    shutil.rmtree("/kaggle/working/train")
except Exception:
    pass

try:
    shutil.rmtree("/kaggle/working/test")
except Exception:
    pass

try:
    os.remove("/kaggle/working/submission.csv")
except Exception:
    pass




## === cell 2
def load_data(data_dir):
    test_csv = os.path.join(data_dir, "test.csv")
    test = pd.read_csv(test_csv)

    test_dir = os.path.join(data_dir, "test_images")
    test["file_path"] = test["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    test["file_name"] = test["id_code"] + ".png"
    return test




## === cell 3
data_dir = "/kaggle/input/aptos2019-blindness-detection/"
test_df = load_data(data_dir)
test_df.head()




## === cell 4
def crop_img(img, percentage):
    img_arr = np.array(img.convert("RGB"))
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    nonzero = img_gray[img_gray != 0]
    if nonzero.size == 0:
        return img.convert("RGB")

    thresh_val = 0.1 * np.mean(nonzero)
    threshold = img_gray > thresh_val

    row_sums = np.sum(threshold, axis=1)
    col_sums = np.sum(threshold, axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return img.convert("RGB")

    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)

    cropped = img_arr[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(cropped)




## === cell 5
def resize_maintain_aspect(img, desired_size):
    resample = getattr(Image, "Resampling", Image).LANCZOS

    old_width, old_height = img.size
    if old_height == 0 or old_width == 0:
        return Image.new("RGB", (desired_size, desired_size))

    aspect_ratio = old_width / old_height

    if aspect_ratio > 1:
        new_width = desired_size
        new_height = max(1, int(desired_size / aspect_ratio))
    else:
        new_height = desired_size
        new_width = max(1, int(desired_size * aspect_ratio))

    resized_img = img.resize((new_width, new_height), resample=resample)

    padded_image = Image.new("RGB", (desired_size, desired_size))
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2
    padded_image.paste(resized_img, (x_offset, y_offset))

    return padded_image




## === cell 6
def save_single(args):
    image_path, output_path_folder, percentage, output_size = args

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    cropped_img = crop_img(image, percentage)
    image_resized = resize_maintain_aspect(cropped_img, desired_size=output_size[0])

    output_image_name = os.path.basename(image_path)
    output_file_path = os.path.join(output_path_folder, output_image_name)
    image_resized.save(output_file_path)




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Uses multiprocessing to make it fast"""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=(100,100)")
        return

    os.makedirs(output_path_folder, exist_ok=True)

    jobs = []
    for i in range(len(df)):
        image_path = df.file_path.iloc[i]
        jobs.append((image_path, output_path_folder, percentage, output_size))

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))




## === cell 8
percentage = 0.01  # kept for consistency with existing crop logic (used in dataset)
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

print("Using test_root_dir:", test_root_dir)
print("Using train_root_dir:", train_root_dir)




## === cell 9
class BlindnessDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, test=False, crop_percentage=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.crop_percentage = crop_percentage

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        with Image.open(img_name) as im:
            image = im.convert("RGB")

        if self.crop_percentage is not None:
            image = crop_img(image, self.crop_percentage)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 10
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("resnet18", pretrained=True, num_classes=5)
model.to(device)

train_df_full = pd.read_csv(train_csv_file)

rng = np.random.RandomState(42)
tr_parts = []
va_parts = []
for cls in sorted(train_df_full["diagnosis"].unique()):
    cls_df = train_df_full[train_df_full["diagnosis"] == cls]
    cls_idx = cls_df.index.values.copy()
    rng.shuffle(cls_idx)
    split = int(0.9 * len(cls_idx))
    tr_parts.append(train_df_full.loc[cls_idx[:split]])
    va_parts.append(train_df_full.loc[cls_idx[split:]])

train_split_df = (
    pd.concat(tr_parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
)
val_split_df = (
    pd.concat(va_parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
)

train_split_df.to_csv("/kaggle/working/_train_split.csv", index=False)
val_split_df.to_csv("/kaggle/working/_val_split.csv", index=False)

train_dataset = BlindnessDataset(
    "/kaggle/working/_train_split.csv",
    train_root_dir,
    transform=transform,
    test=False,
    crop_percentage=percentage,
)
val_dataset = BlindnessDataset(
    "/kaggle/working/_val_split.csv",
    train_root_dir,
    transform=transform,
    test=False,
    crop_percentage=percentage,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0
    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E / E.sum() * O.sum()
    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def predict_proba(loader):
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            probs_all.append(probs)
            y_all.append(labels.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(y_all, axis=0)


EPOCHS = 2
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for images, labels in tqdm(train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()

    val_probs, val_y = predict_proba(val_loader)
    val_pred = np.argmax(val_probs, axis=1)
    qwk = quadratic_weighted_kappa(val_y, val_pred, n_classes=5)
    print(
        f"Epoch {epoch+1}: train_loss={running/max(1,len(train_loader)):.4f} val_qwk={qwk:.4f}"
    )




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 132) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/27837965.py in <cell line: 0>()
    103     model.train()
    104     running = 0.0
--> 105     for images, labels in tqdm(train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}"):
    106         images = images.to(device, non_blocking=True)
    107         labels = labels.to(device, non_blocking=True)

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
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 132) exited unexpectedly

## === cell 12
def apply_thresholds(x, thr):
    return np.digitize(x, bins=thr).astype(int)


def fit_thresholds(x, y, n_classes=5, iters=60):
    thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    best = quadratic_weighted_kappa(y, apply_thresholds(x, thr), n_classes=n_classes)

    for _ in range(iters):
        improved = False
        for k in range(len(thr)):
            for delta in (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05):
                cand = thr.copy()
                cand[k] = cand[k] + delta
                cand = np.sort(cand)
                cand = np.clip(cand, 0.0, 4.0)
                score = quadratic_weighted_kappa(
                    y, apply_thresholds(x, cand), n_classes=n_classes
                )
                if score > best:
                    thr, best = cand, score
                    improved = True
        if not improved:
            break
    return thr, best


val_probs, val_y = predict_proba(val_loader)
val_x = (val_probs * np.arange(5)[None, :]).sum(axis=1)  # expected class
thr, best_qwk = fit_thresholds(val_x, val_y, n_classes=5)
print("Fitted thresholds:", thr, "val_qwk_thr:", best_qwk)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 138) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2277780297.py in <cell line: 0>()
     26 
     27 
---> 28 val_probs, val_y = predict_proba(val_loader)
     29 val_x = (val_probs * np.arange(5)[None, :]).sum(axis=1)  # expected class
     30 thr, best_qwk = fit_thresholds(val_x, val_y, n_classes=5)

/tmp/ipykernel_55/27837965.py in predict_proba(loader)
     89     y_all = []
     90     with torch.no_grad():
---> 91         for batch in loader:
     92             images, labels = batch
     93             images = images.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 138) exited unexpectedly

## === cell 13
test_dataset = BlindnessDataset(
    test_csv_file,
    test_root_dir,
    transform=transform,
    test=True,
    crop_percentage=percentage,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.eval()
all_outputs = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="infer test"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = nn.functional.softmax(logits, dim=1)
        all_outputs.append(probs.cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
test_x = (all_outputs * np.arange(5)[None, :]).sum(axis=1)
final_predictions = apply_thresholds(test_x, thr)

print("Pred shape:", final_predictions.shape, "Unique:", np.unique(final_predictions))




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 143) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4255286163.py in <cell line: 0>()
     17 all_outputs = []
     18 with torch.no_grad():
---> 19     for images in tqdm(test_loader, desc="infer test"):
     20         images = images.to(device, non_blocking=True)
     21         logits = model(images)

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
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 143) exited unexpectedly

## === cell 14
test_ids = pd.read_csv(test_csv_file)["id_code"].values
assert len(test_ids) == len(
    final_predictions
), f"Mismatch: ids={len(test_ids)} preds={len(final_predictions)}"

submission_df = pd.DataFrame(
    {
        "id_code": test_ids,
        "diagnosis": final_predictions.astype(int),
    }
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1481425517.py in <cell line: 0>()
      3 test_ids = pd.read_csv(test_csv_file)["id_code"].values
      4 assert len(test_ids) == len(
----> 5     final_predictions
      6 ), f"Mismatch: ids={len(test_ids)} preds={len(final_predictions)}"
      7 

NameError: name 'final_predictions' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
