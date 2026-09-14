# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import sys
import json
import time
import random
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
import torchvision
from torchvision import models

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "torch:",
    torch.__version__,
    "| torchvision:",
    torchvision.__version__,
    "| device:",
    DEVICE,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)
except Exception:
    pass
torch.set_num_threads(1)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True



## === cell 1
try:
    from efficientnet_pytorch import model as enet  # type: ignore

    HAVE_EFFNET_PYTORCH = True
    print("Using efficientnet_pytorch backend.")
except Exception as e:
    enet = None
    HAVE_EFFNET_PYTORCH = False
    print(
        "efficientnet_pytorch not available; will use torchvision EfficientNet backend."
    )
    print("Import error:", repr(e))



## === cell 2
TEST = True
VER = "v3"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4
TTAS = [0, 1, 2]
FOLDS = [0, 1, 3]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

print("DATA_PATH:", DATA_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("IMGS_PATH:", IMGS_PATH)



## === cell 3
params_path = f"{MDLS_PATH}/params.json"
have_external_models = os.path.exists(params_path)

if have_external_models:
    with open(params_path) as file:
        params = json.load(file)

    LABELS_ = params["labels_"]
    LABELS = params["labels"]

    if KAGGLE:
        cpu_cnt = os.cpu_count() or 4
        WORKERS = int(min(8, max(2, cpu_cnt // 2)))
    else:
        WORKERS = int(params.get("workers", 2))

    print("loaded params keys:", sorted(list(params.keys())))
    print("num classes:", len(LABELS_))
    print(
        "img_size:", params.get("img_size"), "| batch_size:", params.get("batch_size")
    )
    print("WORKERS:", WORKERS)
else:
    print(
        f"WARNING: Missing model params at {params_path}. Falling back to local training."
    )

    train_csv_path = f"{DATA_PATH}/train.csv"
    df_train_all = pd.read_csv(train_csv_path)
    all_labels = sorted(
        {l for s in df_train_all["labels"].fillna("").values for l in str(s).split()}
    )
    LABELS_ = all_labels
    LABELS = {str(i): lab for i, lab in enumerate(LABELS_)}
    LABELS_TO_IDX = {lab: i for i, lab in enumerate(LABELS_)}

    params = {
        "backbone": "efficientnet_b0",
        "dropout": 0.3,
        "img_size": 256,
        "batch_size": 16,
        "workers": 2,
        "labels_": LABELS_,
        "labels": LABELS,
    }
    if KAGGLE:
        cpu_cnt = os.cpu_count() or 4
        WORKERS = int(min(8, max(2, cpu_cnt // 2)))
    else:
        WORKERS = 2

    print("fallback num classes:", len(LABELS_))
    print(
        "fallback img_size:", params["img_size"], "| batch_size:", params["batch_size"]
    )
    print("WORKERS:", WORKERS)



## === cell 4
sub_template_path = f"{DATA_PATH}/sample_submission.csv"
if not os.path.exists(sub_template_path):
    alt = "../input/sample_submission.csv"
    if os.path.exists(alt):
        sub_template_path = alt
    else:
        raise FileNotFoundError(
            f"sample_submission.csv not found at {DATA_PATH} or ../input"
        )

df_sub = pd.read_csv(sub_template_path)[["image", "labels"]].copy()
df_sub["labels"] = "healthy"
print(df_sub.head())
print("n_images(submission template):", len(df_sub))

if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Image directory not found: {IMGS_PATH}")




## === cell 5
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


from collections import OrderedDict


class ImageCache:
    def __init__(self, size: int, max_items: int = 4096):
        self.size = int(size)
        self.max_items = int(max_items)
        self._cache = OrderedDict()  # img_path -> np.ndarray (H,W,3) float32 in [0,1]

    def get(self, img_path: str) -> np.ndarray:
        arr = self._cache.get(img_path, None)
        if arr is not None:
            self._cache.move_to_end(img_path)
            return arr

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = img[:, :, ::-1]  # BGR->RGB
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        self._cache[img_path] = img
        self._cache.move_to_end(img_path)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return img


class TorchCHWImageCache:
    def __init__(
        self,
        base_cache: ImageCache,
        max_items: int = 4096,
        pin_memory: bool = False,
    ):
        self.base_cache = base_cache
        self.max_items = int(max_items)
        self.pin_memory = bool(pin_memory)
        self._cache = OrderedDict()  # img_path -> torch.FloatTensor (3,H,W) on CPU

    def get(self, img_path: str) -> torch.Tensor:
        t = self._cache.get(img_path, None)
        if t is not None:
            self._cache.move_to_end(img_path)
            return t

        img = self.base_cache.get(img_path)  # (H,W,3) float32
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,H,W) float32 CPU
        if self.pin_memory:
            t = t.pin_memory()

        self._cache[img_path] = t
        self._cache.move_to_end(img_path)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return t


GLOBAL_IMG_CACHE = ImageCache(size=params["img_size"], max_items=2048)
GLOBAL_CHW_CACHE = TorchCHWImageCache(
    GLOBAL_IMG_CACHE, max_items=2048, pin_memory=(DEVICE.type == "cuda")
)


class PlantDataset(data.Dataset):
    def __init__(
        self,
        df,
        size,
        labels,
        transform=None,
        tta=0,
        labels_to_idx=None,
        return_ttas: bool = False,
        ttas=(0, 1, 2),
        img_cache: ImageCache = None,
    ):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.labels_to_idx = labels_to_idx
        self.transform = transform
        self.tta = tta
        self.return_ttas = return_ttas
        self.ttas = tuple(ttas)
        self.img_cache = img_cache  # may be None => direct read

    def __len__(self):
        return self.df.shape[0]

    def _read_base_img(self, img_path: str) -> np.ndarray:
        if self.img_cache is not None:
            img = self.img_cache.get(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            img = img[:, :, ::-1]
            img = cv2.resize(
                img, (self.size, self.size), interpolation=cv2.INTER_LINEAR
            )
            img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"

        img = self._read_base_img(img_path)

        if self.labels:
            img_chw = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels), dtype=np.float32)
            mapper = (
                self.labels_to_idx if self.labels_to_idx is not None else self.labels
            )
            for lbl in str(row.labels).split():
                if lbl in mapper:
                    label[mapper[lbl]] = 1.0
            return torch.from_numpy(img_chw), torch.from_numpy(label)

        if self.return_ttas:
            return img
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.from_numpy(img)




## === cell 6
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params["backbone"]

        if HAVE_EFFNET_PYTORCH:
            self.backend = "efficientnet_pytorch"
            self.enet = enet.EfficientNet.from_name(backbone_name)
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
            self.extract = self.enet
            self.pool = None
        else:
            self.backend = "torchvision"
            tv_map = {
                "efficientnet-b0": models.efficientnet_b0,
                "efficientnet-b1": models.efficientnet_b1,
                "efficientnet-b2": models.efficientnet_b2,
                "efficientnet-b3": models.efficientnet_b3,
                "efficientnet-b4": models.efficientnet_b4,
                "efficientnet-b5": models.efficientnet_b5,
                "efficientnet-b6": models.efficientnet_b6,
                "efficientnet-b7": models.efficientnet_b7,
                "efficientnet_v2_s": models.efficientnet_v2_s,
                "efficientnet_v2_m": models.efficientnet_v2_m,
                "efficientnet_v2_l": models.efficientnet_v2_l,
                "efficientnet_b0": models.efficientnet_b0,
                "efficientnet_b1": models.efficientnet_b1,
                "efficientnet_b2": models.efficientnet_b2,
                "efficientnet_b3": models.efficientnet_b3,
                "efficientnet_b4": models.efficientnet_b4,
                "efficientnet_b5": models.efficientnet_b5,
                "efficientnet_b6": models.efficientnet_b6,
                "efficientnet_b7": models.efficientnet_b7,
            }
            if backbone_name not in tv_map:
                raise ValueError(
                    f"Backbone '{backbone_name}' not supported by torchvision fallback. "
                    f"Supported: {sorted(tv_map.keys())}"
                )
            self.enet = tv_map[backbone_name](weights=None)

            if not hasattr(self.enet, "features") or not hasattr(self.enet, "avgpool"):
                raise RuntimeError(
                    "Unexpected torchvision EfficientNet structure; missing features/avgpool."
                )
            self.extract = self.enet.features
            self.pool = self.enet.avgpool

            if hasattr(self.enet, "classifier") and isinstance(
                self.enet.classifier, nn.Sequential
            ):
                nc = self.enet.classifier[-1].in_features
            else:
                raise RuntimeError(
                    "Unexpected torchvision EfficientNet structure; cannot locate classifier head."
                )

        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def forward(self, x):
        x = self.extract(x)
        if self.pool is not None:
            x = self.pool(x)
            x = torch.flatten(x, 1)
        x = self.myfc(x)
        return x




## === cell 7
def train_fallback_models_if_needed():
    global models_ens

    if have_external_models:
        return

    train_csv_path = f"{DATA_PATH}/train.csv"
    df_train = pd.read_csv(train_csv_path)

    img_size = params["img_size"]
    batch_size = params["batch_size"]

    df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_train["fold"] = np.arange(len(df_train)) % 5

    models_ens = []
    n_epochs = 1  # keep within runtime; main goal here is a valid pipeline (no external weights available).
    lr = 1e-3

    for n_fold in FOLDS:
        trn = df_train[df_train.fold != n_fold].reset_index(drop=True)
        val = df_train[df_train.fold == n_fold].reset_index(drop=True)

        global IMGS_PATH
        old_imgs_path = IMGS_PATH
        IMGS_PATH = f"{DATA_PATH}/train_images"

        ds_trn = PlantDataset(
            trn,
            size=img_size,
            labels=LABELS_TO_IDX,
            transform=None,
            tta=0,
            labels_to_idx=LABELS_TO_IDX,
            img_cache=GLOBAL_IMG_CACHE,  # exact pixels; less disk I/O
        )

        dl_trn = torch.utils.data.DataLoader(
            ds_trn,
            batch_size=batch_size,
            sampler=SequentialSampler(ds_trn),
            num_workers=WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(WORKERS > 0),
            prefetch_factor=2 if WORKERS > 0 else None,
        )

        model = EffNet(params, out_dim=len(LABELS_)).to(DEVICE)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(n_epochs):
            running = 0.0
            for xb, yb in dl_trn:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                running += float(loss.detach().cpu().item())
            print(
                f"fallback train fold {n_fold} epoch {epoch+1}/{n_epochs} loss {running/max(1,len(dl_trn)):.4f}"
            )

        model.eval()
        models_ens.append(model)

        IMGS_PATH = old_imgs_path

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


train_fallback_models_if_needed()



## === cell 8
models_ens = [] if "models_ens" not in globals() else models_ens

if have_external_models:
    models_ens = []
    for n_fold in FOLDS:
        model = EffNet(params, out_dim=len(LABELS_))
        path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fold weight file: {path}")
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.float()
        model.eval()
        model.to(DEVICE)

        if DEVICE.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        models_ens.append(model)
        print("loaded:", path, "| backend:", getattr(model, "backend", "unknown"))

    del state_dict, model
    gc.collect()

print("ensemble size:", len(models_ens))




## === cell 9
class InferenceCHWDataset(data.Dataset):
    def __init__(self, df, chw_cache: TorchCHWImageCache):
        self.images = df["image"].values
        self.chw_cache = chw_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = f"{IMGS_PATH}/{img_name}"
        return self.chw_cache.get(img_path)


if KAGGLE:
    cpu_cnt = os.cpu_count() or 4
    INF_WORKERS = (
        int(min(8, max(2, cpu_cnt // 2)))
        if DEVICE.type == "cuda"
        else int(min(4, max(1, cpu_cnt // 4)))
    )
else:
    INF_WORKERS = 2

INF_WORKERS = int(max(0, INF_WORKERS))
print("INF_WORKERS:", INF_WORKERS)

infer_ds = InferenceCHWDataset(df_sub, GLOBAL_CHW_CACHE)
infer_dl = torch.utils.data.DataLoader(
    infer_ds,
    batch_size=int(params["batch_size"]),
    sampler=SequentialSampler(infer_ds),
    num_workers=INF_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(INF_WORKERS > 0),
    prefetch_factor=4 if INF_WORKERS > 0 else None,
    drop_last=False,
)

print(
    "inference loader batches:",
    len(infer_dl),
    "| batch_size:",
    int(params["batch_size"]),
)




## === cell 10
def get_labels(row, labels, th):
    idx = [i for i, x in enumerate(row) if x > th]
    labs = [labels[str(i)] for i in idx]
    return "healthy" if ("healthy" in labs or len(labs) == 0) else " ".join(labs)


if len(models_ens) == 0:
    raise RuntimeError(
        "No models available for inference (external models missing and fallback training failed)."
    )

if DEVICE.type == "cuda":
    try:
        compiled = []
        for m in models_ens:
            compiled.append(torch.compile(m, mode="reduce-overhead", fullgraph=False))
        models_ens = compiled
        del compiled
        print("torch.compile enabled for inference.")
    except Exception as e:
        print("torch.compile not used:", repr(e))

n_images = len(df_sub)
n_classes = len(LABELS_)
n_ttas = len(TTAS)

sum_preds = np.zeros((n_images, n_classes), dtype=np.float32)


def _apply_ttas(x: torch.Tensor, ttas):
    outs = []
    for ax in ttas:
        if ax == 0:
            outs.append(x)
        elif ax == 1:
            outs.append(torch.flip(x, dims=(2,)))  # vertical flip (H)
        elif ax == 2:
            outs.append(torch.flip(x, dims=(3,)))  # horizontal flip (W)
        elif ax == 3:
            outs.append(torch.flip(x, dims=(2, 3)))
        else:
            outs.append(x)
    return outs


bs = int(params["batch_size"])
with torch.inference_mode():
    for i, model in enumerate(models_ens):
        ofs = 0
        for xb_cpu in infer_dl:
            b = xb_cpu.shape[0]
            xb = xb_cpu.to(DEVICE, non_blocking=True)
            if DEVICE.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)

            tta_views = _apply_ttas(xb, TTAS)  # list of (B,3,H,W)
            x = torch.cat(tta_views, dim=0)  # (B*T,3,H,W)

            out = model(x).sigmoid()  # (B*T, C)
            out = out.view(len(TTAS), b, n_classes).mean(dim=0)  # (B,C)

            sum_preds[ofs : ofs + b] += out.detach().cpu().numpy()
            ofs += b

        print(f"model {i} done")

logits = sum_preds / float(len(models_ens))

idx_to_label = [LABELS[str(i)] for i in range(n_classes)]
healthy_idx = None
for j, lab in enumerate(idx_to_label):
    if lab == "healthy":
        healthy_idx = j
        break

above = logits > TH  # (N,C) bool
has_any = above.any(axis=1)
if healthy_idx is not None:
    has_healthy = above[:, healthy_idx]
else:
    has_healthy = np.zeros((n_images,), dtype=bool)

labels_out = np.full((n_images,), "healthy", dtype=object)
need_labels = has_any & (~has_healthy)
need_idx = np.flatnonzero(need_labels)
for r in need_idx:
    cols = np.flatnonzero(above[r])
    labels_out[r] = " ".join(idx_to_label[int(k)] for k in cols)

df_sub["labels"] = labels_out.tolist()

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 11
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 12
sub_path = "submission.csv"
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv(sub_path, index=False)
print("wrote:", sub_path, "| shape:", df_sub.shape)
print(df_sub.head())
print("expected rows (template):", len(pd.read_csv(sub_template_path)))
