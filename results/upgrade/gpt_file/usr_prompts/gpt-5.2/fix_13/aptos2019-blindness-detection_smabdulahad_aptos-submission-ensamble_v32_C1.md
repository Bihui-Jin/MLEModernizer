# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import warnings
import shutil
from multiprocessing import set_start_method

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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")



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
    img_arr = np.asarray(img.convert("RGB"))
    img_gray = (
        0.299 * img_arr[..., 0] + 0.587 * img_arr[..., 1] + 0.114 * img_arr[..., 2]
    ).astype(np.float32)

    nonzero = img_gray[img_gray != 0]
    if nonzero.size == 0:
        return img.convert("RGB")

    thresh_val = 0.1 * float(nonzero.mean())
    threshold = img_gray > thresh_val

    row_sums = threshold.sum(axis=1)
    col_sums = threshold.sum(axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return img.convert("RGB")

    min_row, min_col = int(rows.min()), int(cols.min())
    max_row, max_col = int(rows.max()), int(cols.max())

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
def _crop_bounds_from_rgb(img_rgb, percentage):
    rgb_f = img_rgb.astype(np.float32, copy=False)
    img_gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]

    nonzero = img_gray[img_gray != 0]
    if nonzero.size == 0:
        return (0, img_rgb.shape[0] - 1, 0, img_rgb.shape[1] - 1)

    thresh_val = 0.1 * float(nonzero.mean())
    threshold = img_gray > thresh_val

    row_sums = threshold.sum(axis=1)
    col_sums = threshold.sum(axis=0)

    rows = np.where(row_sums > img_rgb.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_rgb.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return (0, img_rgb.shape[0] - 1, 0, img_rgb.shape[1] - 1)

    return (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))


def _resize_maintain_aspect_cv(img_rgb, desired_size):
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = desired_size
        new_h = max(1, int(desired_size / aspect_ratio))
    else:
        new_h = desired_size
        new_w = max(1, int(desired_size * aspect_ratio))

    interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

    out = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    x0 = (desired_size - new_w) // 2
    y0 = (desired_size - new_h) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def build_image_cache(
    csv_file, root_dir, output_dir, percentage, output_size=(224, 224), max_workers=None
):
    os.makedirs(output_dir, exist_ok=True)
    return 0




## === cell 8
percentage = 0.01  # kept for consistency with existing crop logic (used in dataset)
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

print("Using test_root_dir:", test_root_dir)
print("Using train_root_dir:", train_root_dir)



## === cell 9
from collections import OrderedDict


class _LRUCache:
    def __init__(self, max_items=512):
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


class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        test=False,
        crop_percentage=None,
        desired=224,
        lru_items=512,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.crop_percentage = crop_percentage
        self.desired = int(desired)

        if "id_code" not in self.annotations.columns:
            raise ValueError(
                f"Expected 'id_code' in {csv_file}, got columns={self.annotations.columns.tolist()}"
            )
        if (not self.test) and ("diagnosis" not in self.annotations.columns):
            raise ValueError(
                f"Expected 'diagnosis' in {csv_file} for train/val, got columns={self.annotations.columns.tolist()}"
            )

        self.ids = self.annotations["id_code"].astype(str).to_numpy()
        self.img_paths = [
            os.path.join(self.root_dir, f"{id_code}.png") for id_code in self.ids
        ]

        if not self.test:
            self.labels = self.annotations["diagnosis"].astype(np.int64).to_numpy()
        else:
            self.labels = None

        self._cache = _LRUCache(max_items=lru_items)

    def __len__(self):
        return len(self.ids)

    def _safe_open_rgb_cv(self, img_path):
        cached = self._cache.get(img_path)
        if cached is not None:
            return cached

        try:
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imread failed")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = np.zeros((self.desired, self.desired, 3), dtype=np.uint8)

        if self.crop_percentage is not None:
            try:
                r0, r1, c0, c1 = _crop_bounds_from_rgb(rgb, self.crop_percentage)
                rgb = rgb[r0 : r1 + 1, c0 : c1 + 1]
            except Exception:
                pass

        rgb = _resize_maintain_aspect_cv(rgb, self.desired)

        self._cache.put(img_path, rgb)
        return rgb

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        img_path = self.img_paths[idx]

        rgb = self._safe_open_rgb_cv(img_path)
        image = Image.fromarray(rgb)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image, id_code
        else:
            label = int(self.labels[idx])
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

if device.type == "cuda":
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        pass

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

train_split_csv = "/kaggle/working/_train_split.csv"
val_split_csv = "/kaggle/working/_val_split.csv"
train_split_df.to_csv(train_split_csv, index=False)
val_split_df.to_csv(val_split_csv, index=False)

train_dataset = BlindnessDataset(
    train_split_csv,
    train_root_dir,
    transform=transform,
    test=False,
    crop_percentage=percentage,  # apply same crop+resize as the cache would have
    desired=224,
    lru_items=512,
)
val_dataset = BlindnessDataset(
    val_split_csv,
    train_root_dir,
    transform=transform,
    test=False,
    crop_percentage=percentage,
    desired=224,
    lru_items=512,
)


def _seed_worker(worker_id):
    base_seed = 42
    np.random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)


g = torch.Generator()
g.manual_seed(42)

num_workers = min(4, max(1, os.cpu_count() or 1))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def _qwk_weight_matrix(n_classes=5):
    w = (np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :]) ** 2
    return w.astype(np.float64) / float((n_classes - 1) ** 2)


_QWK_W5 = _qwk_weight_matrix(5)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    idx = y_true * n_classes + y_pred
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64)
    )

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E / E.sum() * O.sum()

    W = _QWK_W5 if n_classes == 5 else _qwk_weight_matrix(n_classes)
    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def predict_proba(loader):
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for images, labels in loader:
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


try:
    val_probs, val_y = predict_proba(val_loader)
    val_x = (val_probs * np.arange(5)[None, :]).sum(axis=1)  # expected class
    thr, best_qwk = fit_thresholds(val_x, val_y, n_classes=5)
    print("Fitted thresholds:", thr, "val_qwk_thr:", best_qwk)
except Exception as e:
    thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    print("Threshold fitting failed; using default thresholds. Error:", repr(e))



## === cell 13
test_dataset = BlindnessDataset(
    test_csv_file,
    test_root_dir,
    transform=transform,
    test=True,
    crop_percentage=percentage,
    desired=224,
    lru_items=256,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

model.eval()
all_outputs = []
all_ids = []
with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="infer test"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = nn.functional.softmax(logits, dim=1)
        all_outputs.append(probs.cpu().numpy())
        all_ids.extend(list(ids))

all_outputs = np.concatenate(all_outputs, axis=0)
test_x = (all_outputs * np.arange(5)[None, :]).sum(axis=1)
final_predictions = apply_thresholds(test_x, thr)

print("Pred shape:", final_predictions.shape, "Unique:", np.unique(final_predictions))
print("IDs:", len(all_ids), "Unique IDs:", len(set(all_ids)))



## === cell 14
test_order_df = pd.read_csv(test_csv_file)
test_order_ids = test_order_df["id_code"].astype(str).str.strip().to_numpy()

all_ids_arr = pd.Series(all_ids, dtype="string").str.strip().astype(str).to_numpy()

if len(all_ids_arr) != len(test_order_ids):
    raise ValueError(
        f"Prediction count mismatch: got {len(all_ids_arr)} ids, expected {len(test_order_ids)}"
    )

if not np.array_equal(all_ids_arr, test_order_ids):
    pos = {k: i for i, k in enumerate(all_ids_arr.tolist())}
    missing = [i for i in test_order_ids if i not in pos]
    if missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} id_codes (e.g. {missing[:5]})"
        )
    ordered_idx = np.fromiter(
        (pos[i] for i in test_order_ids), dtype=np.int64, count=len(test_order_ids)
    )
    ordered_pred = final_predictions[ordered_idx].astype(np.int64)
else:
    ordered_pred = final_predictions.astype(np.int64)

submission_df = pd.DataFrame(
    {"id_code": test_order_ids.astype(str), "diagnosis": ordered_pred}
)
submission_df = submission_df[["id_code", "diagnosis"]]

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
print(
    "Diagnosis value counts:\n", submission_df["diagnosis"].value_counts().sort_index()
)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
