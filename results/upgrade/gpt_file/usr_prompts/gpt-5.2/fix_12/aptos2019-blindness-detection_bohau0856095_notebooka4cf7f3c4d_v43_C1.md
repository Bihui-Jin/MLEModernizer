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
pillow==11.3.0
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
import glob
import random
import time
import math
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)
    thr = torch.tensor(threshold, dtype=out.dtype, device=out.device)
    return (out[:, None] >= thr[None, :]).sum(dim=1).to(torch.long).detach().cpu()


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 3
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)

        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size

        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w

        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h

        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image


class CachedTrimCrop(object):
    def __init__(self, cache_dir: str, key_salt: str = "v1"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.key_salt = key_salt
        self._trim = trim()
        self._crop = cropTo4_3()

    def _key(self, img_path: str) -> str:
        s = f"{self.key_salt}|{img_path}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def get_cached_path(self, img_path: str) -> str:
        k = self._key(img_path)
        return os.path.join(self.cache_dir, f"{k}.png")

    def __call__(self, image, img_path: str = None):
        if img_path is None:
            image = self._trim(image)
            image = self._crop(image)
            return image

        out_path = self.get_cached_path(img_path)
        if os.path.exists(out_path):
            im = Image.open(out_path).convert("RGB")
            im.load()
            return im

        image = self._trim(image)
        image = self._crop(image)
        image.save(out_path, format="PNG", optimize=False)
        return image




## === cell 4
BASE1 = "../input/aptos2019-blindness-detection"
BASE2 = "/kaggle/data/aptos2019-blindness-detection"
BASE3 = "/kaggle/input/aptos2019-blindness-detection"


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


train_csv = first_existing(
    os.path.join(BASE1, "train.csv"),
    os.path.join(BASE2, "train.csv"),
    os.path.join(BASE3, "train.csv"),
)
train_img_dir = first_existing(
    os.path.join(BASE1, "train_images"),
    os.path.join(BASE2, "train_images"),
    os.path.join(BASE3, "train_images"),
)

test_csv = first_existing(
    os.path.join(BASE1, "test.csv"),
    os.path.join(BASE2, "test.csv"),
    os.path.join(BASE3, "test.csv"),
)
test_img_dir = first_existing(
    os.path.join(BASE1, "test_images"),
    os.path.join(BASE2, "test_images"),
    os.path.join(BASE3, "test_images"),
)

if (
    test_csv is None
    or test_img_dir is None
    or train_csv is None
    or train_img_dir is None
):
    raise FileNotFoundError(
        "Could not locate train/test csv or image directories in expected Kaggle paths."
    )

train_df = pd.read_csv(train_csv)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

test_ids = pd.read_csv(test_csv)["id_code"].astype(str).values

input_size = 380

_pre_cache_dir = os.path.join("/kaggle/working", "trimcrop_cache")
_cached_trimcrop = CachedTrimCrop(_pre_cache_dir, key_salt=f"trimcrop_{input_size}")


class _TrimCropWithPath(object):
    def __call__(self, img, img_path=None):
        return _cached_trimcrop(img, img_path=img_path)


post_trimcrop_transform = transforms.Compose(
    [
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

post_trimcrop_train_transform = transforms.Compose(
    [
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model().to(device)




## === cell 5
def _unwrap_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]
    if isinstance(state, dict) and "net" in state and isinstance(state["net"], dict):
        state = state["net"]
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
        and isinstance(state["model_state_dict"], dict)
    ):
        state = state["model_state_dict"]
    if (
        isinstance(state, dict)
        and "module" in state
        and isinstance(state["module"], dict)
    ):
        state = state["module"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if isinstance(k, str) and k.startswith("module."):
                new_state[k[len("module.") :]] = v
            else:
                new_state[k] = v
        return new_state
    return state


def _state_dict_match_ratio(model: nn.Module, state: dict) -> float:
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    if not model_keys:
        return 0.0
    return len(model_keys & state_keys) / float(len(model_keys))


def try_load_checkpoint(model: nn.Module):
    candidates = [
        "../input/weights/B4_3stage_50epoch_CLAHE.pkl",
        "/kaggle/input/weights/B4_3stage_50epoch_CLAHE.pkl",
    ]

    best_path = None
    best_ratio = 0.0
    for ckpt_path in candidates:
        if not os.path.isfile(ckpt_path):
            continue
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            state = _unwrap_state_dict(state)
            if not isinstance(state, dict):
                continue
            ratio = _state_dict_match_ratio(model, state)
            if ratio < 0.92:
                continue
            model.load_state_dict(state, strict=False)
            best_ratio = ratio
            best_path = ckpt_path
            break
        except Exception:
            continue

    if best_path is not None:
        print(f"Loaded checkpoint: {best_path} (key match ratio={best_ratio:.3f})")
        return True, best_path
    else:
        print("No highly compatible checkpoint found in the expected weights paths.")
        return False, None


loaded_ckpt, ckpt_path = try_load_checkpoint(net)




## === cell 6
def _warm_trimcrop_cache_for_ids(
    id_codes, img_dir: str, desc: str, max_workers: int = None, chunk_size: int = 64
):
    import concurrent.futures

    id_codes = [str(x) for x in id_codes]
    img_paths = [os.path.join(img_dir, f"{idc}.png") for idc in id_codes]
    cached_paths = [_cached_trimcrop.get_cached_path(p) for p in img_paths]

    missing_img_paths = [
        ip for ip, cp in zip(img_paths, cached_paths) if not os.path.exists(cp)
    ]
    if not missing_img_paths:
        print(f"Cache warmup [{desc}]: nothing to do (all cached).")
        return

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(12, max(4, cpu))

    t0 = time.time()
    total = len(missing_img_paths)
    print(
        f"Cache warmup [{desc}]: building {total}/{len(img_paths)} cached trim+crop images with {max_workers} workers..."
    )

    def _process_chunk(paths_chunk):
        done = 0
        for img_path in paths_chunk:
            out_path = _cached_trimcrop.get_cached_path(img_path)
            if os.path.exists(out_path):
                done += 1
                continue
            img = Image.open(img_path).convert("RGB")
            img.load()
            _ = _cached_trimcrop(img, img_path=img_path)
            done += 1
        return done

    chunks = [
        missing_img_paths[i : i + chunk_size] for i in range(0, total, chunk_size)
    ]

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_chunk, ch) for ch in chunks]
        for f in concurrent.futures.as_completed(futures):
            done += int(f.result())
            if done % 256 == 0 or done >= total:
                print(
                    f"Cache warmup [{desc}]: {done}/{total} done ({time.time()-t0:.1f}s)"
                )

    print(f"Cache warmup [{desc}] finished in {time.time()-t0:.1f}s")


if loaded_ckpt:
    _warm_trimcrop_cache_for_ids(test_ids, test_img_dir, desc="test")
else:
    _warm_trimcrop_cache_for_ids(
        train_df["id_code"].values, train_img_dir, desc="train"
    )
    _warm_trimcrop_cache_for_ids(test_ids, test_img_dir, desc="test")




## === cell 7
class APTOSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform=None,
        has_label: bool = True,
        use_cache_trimcrop: bool = True,
        is_train: bool = False,
    ):
        df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label
        self.use_cache_trimcrop = use_cache_trimcrop
        self.is_train = is_train

        self.id_codes = df["id_code"].astype(str).to_numpy()
        self.labels = None
        if has_label:
            self.labels = df["diagnosis"].astype(np.int64).to_numpy()

        self.img_paths = [
            os.path.join(self.img_dir, f"{idc}.png") for idc in self.id_codes
        ]
        if self.use_cache_trimcrop:
            self.cached_paths = [
                _cached_trimcrop.get_cached_path(p) for p in self.img_paths
            ]
        else:
            self.cached_paths = None

    def __len__(self):
        return len(self.id_codes)

    def __getitem__(self, i):
        img_path = self.img_paths[i]

        if self.use_cache_trimcrop:
            cached_path = self.cached_paths[i]
            if os.path.exists(cached_path):
                img = Image.open(cached_path).convert("RGB")
                img.load()
            else:
                img = Image.open(img_path).convert("RGB")
                img.load()
                img = _cached_trimcrop(img, img_path=img_path)
        else:
            img = Image.open(img_path).convert("RGB")
            img.load()

        if self.transform is not None:
            img = self.transform(img)

        if self.has_label:
            y = int(self.labels[i])
            return img, torch.tensor([float(y)], dtype=torch.float32)
        else:
            return img, str(self.id_codes[i])


def run_inference_regression(model: nn.Module, loader: DataLoader):
    model.eval()
    preds = []
    targs = []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x, final=True).view(-1)  # [bs]
            preds.append(out.detach().cpu().numpy())
            targs.append(y.view(-1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targs = np.concatenate(targs, axis=0)
    return preds, targs


def qwk_from_preds(y_true, y_pred_class):
    return cohen_kappa_score(
        y_true.astype(int), y_pred_class.astype(int), weights="quadratic"
    )


def apply_thresholds(y_reg, thr):
    y_reg = np.asarray(y_reg, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)
    y_cls = (y_reg[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)
    return np.clip(y_cls, 0, 4)


def calibrate_thresholds(y_true, y_reg, init_thr=None):
    if init_thr is None:
        thr = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        thr = np.array(init_thr, dtype=np.float32)

    y_true_i64 = np.asarray(y_true, dtype=np.int64)
    y_reg_f32 = np.asarray(y_reg, dtype=np.float32)

    def _score(thr_arr):
        return qwk_from_preds(y_true_i64, apply_thresholds(y_reg_f32, thr_arr))

    best = _score(thr)

    step_schedule = [0.25, 0.10, 0.05]
    for step in step_schedule:
        improved = True
        it = 0
        while improved and it < 20:
            improved = False
            it += 1
            for k in range(4):
                base = float(thr[k])

                cands = []
                for direction in (-1.0, 1.0):
                    cand = thr.copy()
                    cand[k] = base + direction * step
                    if not (0.0 <= cand[0] < cand[1] < cand[2] < cand[3] <= 4.5):
                        continue
                    cands.append(cand)

                for cand in cands:
                    score = _score(cand)
                    if score > best + 1e-8:
                        thr = cand
                        best = score
                        improved = True
    return thr.tolist(), float(best)




## === cell 8
def _seed_worker(worker_id):
    base_seed = 42
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _make_loader(ds, batch_size, shuffle, num_workers):
    if num_workers is None:
        cpu = os.cpu_count() or 4
        num_workers = min(8, max(2, cpu // 2))
    g = torch.Generator()
    g.manual_seed(42)

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    try:
        kwargs["multiprocessing_context"] = "fork"
    except Exception:
        pass

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return DataLoader(ds, **kwargs)


def train_if_needed(model: nn.Module, train_df: pd.DataFrame):
    if loaded_ckpt:
        print("Checkpoint loaded; skipping training.")
        return None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, va_idx = next(sss.split(train_df["id_code"], train_df["diagnosis"]))
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_ds = APTOSDataset(
        tr_df,
        train_img_dir,
        transform=post_trimcrop_train_transform,
        has_label=True,
        use_cache_trimcrop=True,
        is_train=True,
    )
    val_ds = APTOSDataset(
        va_df,
        train_img_dir,
        transform=post_trimcrop_transform,
        has_label=True,
        use_cache_trimcrop=True,
        is_train=False,
    )

    train_loader = _make_loader(train_ds, batch_size=6, shuffle=True, num_workers=None)
    val_loader = _make_loader(val_ds, batch_size=16, shuffle=False, num_workers=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = nn.MSELoss()

    start = time.time()
    best_qwk = -1.0
    best_state = None

    max_epochs = 4
    for ep in range(max_epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)  # [bs]
            out = model(x, final=True).view(-1)
            loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu().item()) * x.size(0)
            n += x.size(0)

        y_reg, y_true = run_inference_regression(model, val_loader)
        y_cls = apply_thresholds(y_reg, threshold)
        qwk = qwk_from_preds(y_true, y_cls)

        if qwk > best_qwk + 1e-8:
            best_qwk = qwk
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        elapsed = time.time() - start
        print(
            f"Epoch {ep+1}/{max_epochs} train_mse={running/max(n,1):.4f} val_qwk(default_thr)={qwk:.4f} best={best_qwk:.4f} elapsed={elapsed:.1f}s"
        )
        if elapsed > 520:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return (tr_df, va_df)


split_info = train_if_needed(net, train_df)
net.eval()



## === cell 9
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(sss.split(train_df["id_code"], train_df["diagnosis"]))
va_df = train_df.iloc[va_idx].reset_index(drop=True)

if loaded_ckpt:
    _warm_trimcrop_cache_for_ids(va_df["id_code"].values, train_img_dir, desc="val")

val_ds = APTOSDataset(
    va_df,
    train_img_dir,
    transform=post_trimcrop_transform,
    has_label=True,
    use_cache_trimcrop=True,
)
val_loader = _make_loader(val_ds, batch_size=32, shuffle=False, num_workers=None)

y_reg, y_true = run_inference_regression(net, val_loader)
best_thr, best_qwk = calibrate_thresholds(y_true, y_reg, init_thr=threshold)
threshold = best_thr
print("Calibrated thresholds:", threshold, "val_qwk:", best_qwk)



## === cell 10
test_df = pd.DataFrame({"id_code": test_ids})
test_ds = APTOSDataset(
    test_df,
    test_img_dir,
    transform=post_trimcrop_transform,
    has_label=False,
    use_cache_trimcrop=True,
)
test_loader = _make_loader(test_ds, batch_size=32, shuffle=False, num_workers=None)

all_ids = [None] * len(test_ids)
all_pred = np.empty(len(test_ids), dtype=np.int64)

net.eval()
with torch.inference_mode():
    idx = 0
    for x, ids in test_loader:
        bs = len(ids)
        x = x.to(device, non_blocking=True)
        out = net(x, final=True).view(-1)  # [bs]
        pred = regress2class(out).numpy().astype(np.int64)
        all_pred[idx : idx + bs] = pred
        all_ids[idx : idx + bs] = list(ids)
        idx += bs
        if idx % 64 == 0:
            print(f"Inferencing {idx}/{len(test_ids)}")

df = pd.DataFrame({"id_code": np.asarray(all_ids, dtype=str), "diagnosis": all_pred})
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
