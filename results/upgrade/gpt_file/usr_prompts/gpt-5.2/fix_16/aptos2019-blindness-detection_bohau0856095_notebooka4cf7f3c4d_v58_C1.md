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
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

try:
    Image.MAX_IMAGE_PIXELS = None
    Image.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    thr = threshold if thr is None else thr
    thr_t = torch.as_tensor(thr, device=out.device, dtype=out.dtype).view(1, -1)
    out2d = out.view(-1, 1)
    pred = (out2d >= thr_t).sum(dim=1).to("cpu")
    return pred


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    out_clamped = torch.clamp(out, 0.0, 4.0)
    l1 = torch.floor(out_clamped).to(torch.long)
    l2 = torch.ceil(out_clamped).to(torch.long)
    w2 = out_clamped - l1.to(out.dtype)
    w1 = 1.0 - w2
    pred_prob.scatter_(1, l1.view(-1, 1), w1.view(-1, 1))
    pred_prob.scatter_add_(
        1, l2.view(-1, 1), (1.0 - (l2.to(out.dtype) - out_clamped)).view(-1, 1)
    )
    ge4 = out >= 4.0
    if ge4.any():
        pred_prob[ge4] = 0
        pred_prob[ge4, 4] = 1.0
    return pred_prob




## === cell 1
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
    def __init__(self, pretrained_backbone=False):
        super(Regressor, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=pretrained_backbone, num_classes=1000
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=pretrained_backbone, num_classes=1000
        )
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




## === cell 2
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




## === cell 3
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

WEIGHT_PATH = "../input/weights/B4_3stage_57epoch_CLAHE.pkl"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].values

input_size = 380

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

_use_three_stage_weights_exist = os.path.exists(WEIGHT_PATH)
if _use_three_stage_weights_exist:
    net = ThreeStage_Model(pretrained_backbone=False)
    state = torch.load(WEIGHT_PATH, map_location="cpu")
    net.load_state_dict(state)
    use_three_stage = True
else:
    print(
        f"WARNING: Weights not found at {WEIGHT_PATH}. Will train the Regressor on train.csv to get meaningful predictions."
    )
    net = Regressor(pretrained_backbone=True)
    use_three_stage = False

net = net.to(device)
net.eval()



## === cell 4
from torchvision.io import read_image, ImageReadMode

_CACHE_DIR = "../working"
TRAIN_TENSOR_CACHE = os.path.join(_CACHE_DIR, "train_tensor_cache_380.pt")
TEST_TENSOR_CACHE = os.path.join(_CACHE_DIR, "test_tensor_cache_380.pt")


def _seed_worker(worker_id: int):
    base_seed = 42
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _make_loader(dataset, batch_size, shuffle, device, num_workers):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
    if device == "cuda":
        kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _trim_bbox_from_tensor_rgb_u8(img_u8: torch.Tensor):
    bg = img_u8[:, 0, 0]  # (3,)
    diff = (img_u8.to(torch.int32) - bg.view(3, 1, 1).to(torch.int32)).abs_()  # (3,H,W)
    mask = diff.amax(dim=0) > 5  # (H,W)
    if not bool(mask.any()):
        return None
    ys = torch.where(mask.any(dim=1))[0]
    xs = torch.where(mask.any(dim=0))[0]
    y0, y1 = int(ys[0].item()), int(ys[-1].item()) + 1
    x0, x1 = int(xs[0].item()), int(xs[-1].item()) + 1
    return x0, y0, x1, y1


def _center_crop_to_4_3(img: torch.Tensor):
    _, h, w = img.shape
    if (w / h) >= (4 / 3):
        new_h = h
        new_w = int(h * 4 / 3)
    else:
        new_h = int(w * 3 / 4)
        new_w = w
    left = int(round((w - new_w) / 2.0))
    top = int(round((h - new_h) / 2.0))
    return img[:, top : top + new_h, left : left + new_w]


_MEAN = torch.tensor([0.384, 0.258, 0.174], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.124, 0.089, 0.094], dtype=torch.float32).view(3, 1, 1)


def _fast_preprocess_png(path: str, out_h: int, out_w: int):
    img = read_image(path, mode=ImageReadMode.RGB)  # uint8 (3,H,W)
    bbox = _trim_bbox_from_tensor_rgb_u8(img)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        img = img[:, y0:y1, x0:x1]
    img = _center_crop_to_4_3(img)
    img = img.to(torch.float32).mul_(1.0 / 255.0)
    img = FT.resize(
        img, [out_h, out_w], interpolation=FT.InterpolationMode.BILINEAR, antialias=True
    )
    img.sub_(_MEAN).div_(_STD)
    return img


class _AptosTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id_code"].values
        self.y = df["diagnosis"].values.astype(np.float32)
        self.img_dir = img_dir
        self.transform = transform
        self._out_h = input_size * 3 // 4
        self._out_w = input_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        x = _fast_preprocess_png(image_name, self._out_h, self._out_w)
        y = self.y[i]
        return x, y


class _AptosTestDataset(torch.utils.data.Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = np.asarray(ids)
        self.img_dir = img_dir
        self.transform = transform
        self._out_h = input_size * 3 // 4
        self._out_w = input_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        x = _fast_preprocess_png(image_name, self._out_h, self._out_w)
        return x, idx


class _AptosTensorCacheDataset(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y=None, ids=None):
        self.x = x_tensor
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)
        self.ids = None if ids is None else np.asarray(ids)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if self.y is None and self.ids is None:
            return self.x[i]
        if self.y is None:
            return self.x[i], self.ids[i]
        if self.ids is None:
            return self.x[i], self.y[i]
        return self.x[i], self.y[i], self.ids[i]


def _build_or_load_tensor_cache_from_ids(
    ids, img_dir, transform, cache_path, kind="cache"
):
    ids = np.asarray(ids)
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        if isinstance(obj, dict) and "ids" in obj and "x" in obj:
            cached_ids = obj["ids"]
            if (
                len(cached_ids) == len(ids)
                and np.asarray(cached_ids).dtype == ids.dtype
            ):
                if np.asarray(cached_ids).shape == ids.shape and np.array_equal(
                    np.asarray(cached_ids), ids
                ):
                    return obj["x"]

    t0 = time.time()
    ds = _AptosTestDataset(ids, img_dir, transform)

    cpu = os.cpu_count() or 4
    cpu_workers = min(12, max(4, cpu - 2))
    loader = _make_loader(
        ds,
        batch_size=64,
        shuffle=False,  # must preserve ID order for cache correctness
        device=device,
        num_workers=cpu_workers,
    )

    x_chunks = []
    with torch.inference_mode():
        for xb, _ in loader:
            x_chunks.append(xb)
    x = torch.cat(x_chunks, dim=0)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"ids": ids, "x": x}, cache_path)
    print(
        f"Built {kind} tensor cache: {cache_path} shape={tuple(x.shape)} time={time.time()-t0:.1f}s workers={cpu_workers}"
    )
    return x


def _build_or_load_train_tensor_cache(train_csv, img_dir, transform, cache_path):
    df = pd.read_csv(train_csv).reset_index(drop=True)
    ids = df["id_code"].values
    y = df["diagnosis"].values.astype(np.int64)

    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        if isinstance(obj, dict) and "ids" in obj and "x" in obj:
            cached_ids = obj["ids"]
            if (
                len(cached_ids) == len(ids)
                and np.asarray(cached_ids).shape == ids.shape
                and np.array_equal(np.asarray(cached_ids), ids)
            ):
                x = obj["x"]
                return df, ids, y, x

    x = _build_or_load_tensor_cache_from_ids(
        ids, img_dir, transform, cache_path, kind="train"
    )
    return df, ids, y, x


def _predict_from_tensor_dataset_regression(x_tensor, ids_subset_idx, batch_size):
    ds = _AptosTensorCacheDataset(x_tensor[ids_subset_idx])
    loader = _make_loader(
        ds, batch_size=batch_size, shuffle=False, device=device, num_workers=0
    )

    preds = np.empty(len(ds), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for xb in loader:
            xb = xb.to(device, non_blocking=(device == "cuda"))
            if use_three_stage:
                _, r_out, _ = net(xb)
                out = r_out.squeeze(1)
            else:
                out = net(xb).squeeze(1)
            out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
            preds[offset : offset + len(out_np)] = out_np
            offset += len(out_np)
    return preds


def _predict_from_tensor_dataset_twohead(x_tensor, ids_subset_idx, batch_size):
    ds = _AptosTensorCacheDataset(x_tensor[ids_subset_idx])
    loader = _make_loader(
        ds, batch_size=batch_size, shuffle=False, device=device, num_workers=0
    )

    r_preds = np.empty(len(ds), dtype=np.float32)
    cexp_preds = np.empty(len(ds), dtype=np.float32)
    offset = 0
    arange5 = torch.arange(5, device=device)
    with torch.inference_mode():
        for xb in loader:
            xb = xb.to(device, non_blocking=(device == "cuda"))
            c_out, r_out, _ = net(xb)
            r_out = r_out.squeeze(1)
            probs = F.softmax(c_out, dim=1)
            ev = (probs * arange5.to(dtype=probs.dtype)).sum(dim=1)

            r_np = r_out.detach().cpu().numpy().astype(np.float32, copy=False)
            ev_np = ev.detach().cpu().numpy().astype(np.float32, copy=False)
            r_preds[offset : offset + len(r_np)] = r_np
            cexp_preds[offset : offset + len(ev_np)] = ev_np
            offset += len(r_np)
    return r_preds, cexp_preds


def _apply_thresholds(preds_float, thr):
    preds_float = np.asarray(preds_float, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32).reshape(1, -1)
    out = (preds_float.reshape(-1, 1) >= thr).sum(axis=1).astype(np.int64, copy=False)
    return np.clip(out, 0, 4)


def _sanitize_thresholds(thr):
    thr = np.asarray(thr, dtype=np.float32).copy()
    thr = np.clip(thr, 0.0, 4.5)
    thr = np.sort(thr)
    for j in range(1, 4):
        if thr[j] <= thr[j - 1]:
            thr[j] = min(4.5, thr[j - 1] + 1e-3)
    return thr


def _tune_thresholds_qwk(y_true, y_pred_float, init_thr=None, n_rounds=2):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32)

    thr = np.array(
        init_thr if init_thr is not None else [0.75, 1.5, 2.5, 3.5], dtype=np.float32
    )
    thr = _sanitize_thresholds(thr)

    def score_for(thr_vec):
        y_pred = _apply_thresholds(y_pred_float, thr_vec)
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    def golden_search_on_k(thr_vec, k, lo, hi, iters=18):
        phi = (1 + 5**0.5) / 2
        invphi = 1 / phi
        invphi2 = invphi * invphi

        a, b = float(lo), float(hi)
        if b - a < 1e-6:
            return thr_vec, score_for(thr_vec)

        h = b - a
        c = a + invphi2 * h
        d = a + invphi * h

        def eval_at(x):
            cand = thr_vec.copy()
            cand[k] = x
            cand = _sanitize_thresholds(cand)
            return cand, score_for(cand)

        cand_c, sc_c = eval_at(c)
        cand_d, sc_d = eval_at(d)

        for _ in range(iters):
            if sc_c < sc_d:
                a = c
                c = d
                sc_c = sc_d
                cand_c = cand_d
                h = b - a
                d = a + invphi * h
                cand_d, sc_d = eval_at(d)
            else:
                b = d
                d = c
                sc_d = sc_c
                cand_d = cand_c
                h = b - a
                c = a + invphi2 * h
                cand_c, sc_c = eval_at(c)

        if sc_c >= sc_d:
            return cand_c, float(sc_c)
        return cand_d, float(sc_d)

    best = float(score_for(thr))

    for _ in range(n_rounds):
        for k in range(4):
            lo = 0.0 if k == 0 else float(thr[k - 1] + 1e-3)
            hi = 4.5 if k == 3 else float(thr[k + 1] - 1e-3)
            if hi <= lo + 1e-4:
                continue
            thr_candidate, sc = golden_search_on_k(thr, k, lo, hi, iters=18)
            if sc >= best:
                thr = thr_candidate
                best = sc

    thr = _sanitize_thresholds(thr)
    return thr.tolist(), float(best)


def _train_regressor_on_traincsv(net, train_csv, img_dir, transform, device):
    df = pd.read_csv(train_csv).reset_index(drop=True)

    ds = _AptosTrainDataset(df, img_dir, transform)
    batch_size = 12 if device == "cuda" else 6
    loader = _make_loader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        num_workers=min(8, max(2, (os.cpu_count() or 4) // 2)),
    )

    net.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    epochs = 4 if device == "cuda" else 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device == "cuda"))
            yb = yb.to(
                device, dtype=torch.float32, non_blocking=(device == "cuda")
            ).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs

        scheduler.step()
        print(
            f"Regressor training epoch {ep}/{epochs} - loss={loss_sum/max(n,1):.5f} - lr={scheduler.get_last_lr()[0]:.2e} - time={time.time()-t0:.1f}s"
        )

    net.eval()
    return net


def _tune_thresholds_qwk_oof_regressor_from_cache(
    train_df, ids, y, x_tensor, batch_size, init_thr
):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train_df), dtype=np.float32)

    for fold, (_, val_idx) in enumerate(skf.split(ids, y), 1):
        oof[val_idx] = _predict_from_tensor_dataset_regression(
            x_tensor, val_idx, batch_size
        )
        print(f"OOF preds: fold {fold}/5 done ({len(val_idx)} samples)")

    tuned_thr, tuned_qwk = _tune_thresholds_qwk(y, oof, init_thr=init_thr, n_rounds=3)
    return tuned_thr, tuned_qwk


float_source_for_test = "regressor"  # default for regressor-only model

if (not use_three_stage) and os.path.exists(TRAIN_CSV) and os.path.isdir(TRAIN_IMG_DIR):
    net = _train_regressor_on_traincsv(net, TRAIN_CSV, TRAIN_IMG_DIR, transform, device)

if os.path.exists(TRAIN_CSV) and os.path.isdir(TRAIN_IMG_DIR):
    train_df, train_ids, train_y, train_x = _build_or_load_train_tensor_cache(
        TRAIN_CSV, TRAIN_IMG_DIR, transform, TRAIN_TENSOR_CACHE
    )

    BATCH_SIZE_FIT = 96 if device == "cuda" else 6
    if use_three_stage:
        y = train_y
        ids = train_ids

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, val_idx = next(splitter.split(ids, y))

        y_val = y[val_idx]

        val_pred_r, val_pred_cexp = _predict_from_tensor_dataset_twohead(
            train_x, val_idx, BATCH_SIZE_FIT
        )
        tuned_thr_r, tuned_qwk_r = _tune_thresholds_qwk(
            y_val, val_pred_r, init_thr=threshold, n_rounds=3
        )
        tuned_thr_c, tuned_qwk_c = _tune_thresholds_qwk(
            y_val, val_pred_cexp, init_thr=threshold, n_rounds=3
        )

        if tuned_qwk_c > tuned_qwk_r:
            tuned_threshold = tuned_thr_c
            float_source_for_test = "classifier_expected_value"
            tuned_qwk = tuned_qwk_c
        else:
            tuned_threshold = tuned_thr_r
            float_source_for_test = "regressor_head"
            tuned_qwk = tuned_qwk_r

        print(
            f"Tuned thresholds (holdout, source={float_source_for_test}): {tuned_threshold} (holdout QWK={tuned_qwk:.5f})"
        )
    else:
        tuned_threshold, tuned_qwk = _tune_thresholds_qwk_oof_regressor_from_cache(
            train_df, train_ids, train_y, train_x, BATCH_SIZE_FIT, threshold
        )
        float_source_for_test = "regressor"
        print(
            f"Tuned thresholds (OOF 5-fold): {tuned_threshold} (OOF QWK={tuned_qwk:.5f})"
        )
else:
    tuned_threshold = threshold
    print("WARNING: Train data not found; using default thresholds.")
    float_source_for_test = "regressor"



## === cell 5
test_x = _build_or_load_tensor_cache_from_ids(
    test_ids, TEST_IMG_DIR, transform, TEST_TENSOR_CACHE, kind="test"
)

BATCH_SIZE = 96 if device == "cuda" else 6

test_ds = _AptosTensorCacheDataset(test_x, y=None, ids=test_ids)
test_loader = _make_loader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    device=device,
    num_workers=0,  # tensors already in memory; avoid worker overhead
)

n_test = len(test_ds)
submission_ids = list(test_ids)  # already in correct order due to shuffle=False
submission_preds = np.empty(n_test, dtype=np.int64)

arange5 = torch.arange(5, device=device)
offset = 0

with torch.inference_mode():
    for xb, idb in test_loader:
        xb = xb.to(device, non_blocking=(device == "cuda"))

        if use_three_stage:
            c_out, r_out, _ = net(xb)
            r_out = r_out.squeeze(1)

            if float_source_for_test == "classifier_expected_value":
                probs = F.softmax(c_out, dim=1)
                preds_float = (probs * arange5.to(dtype=probs.dtype)).sum(dim=1)
            else:
                preds_float = r_out
        else:
            preds_float = net(xb).squeeze(1)

        preds_np = (
            regress2class(preds_float, thr=tuned_threshold)
            .numpy()
            .astype(np.int64, copy=False)
        )
        submission_preds[offset : offset + len(preds_np)] = preds_np
        offset += len(preds_np)

submission_df = pd.DataFrame(
    {"id_code": submission_ids, "diagnosis": submission_preds.astype(int).clip(0, 4)}
)

assert len(submission_df) == len(
    test_ids
), f"Submission rows {len(submission_df)} != test rows {len(test_ids)}"
submission_df["diagnosis"] = submission_df["diagnosis"].astype(int).clip(0, 4)



## === cell 6
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape {submission_df.shape}")
print(submission_df.head())
