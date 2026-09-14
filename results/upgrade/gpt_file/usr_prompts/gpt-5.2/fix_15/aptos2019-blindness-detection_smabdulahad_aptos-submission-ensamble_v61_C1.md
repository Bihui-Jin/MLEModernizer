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
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm
from timm.data import resolve_data_config, create_transform



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False, return_id=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.return_id = return_id

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        id_code = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, id_code + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            if self.return_id:
                return image, id_code
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            if self.return_id:
                return image, label, id_code
            return image, label




## === cell 3
_fallback_model_name_for_transform = "tf_efficientnet_b5.ns_jft_in1k"
_tmp_model = timm.create_model(
    _fallback_model_name_for_transform, pretrained=True, num_classes=0
)
_data_cfg = resolve_data_config({}, model=_tmp_model)
transform = create_transform(**_data_cfg, is_training=False)

_raw_data_cfg = dict(_data_cfg)
_raw_data_cfg["mean"] = (0.0, 0.0, 0.0)
_raw_data_cfg["std"] = (1.0, 1.0, 1.0)
raw_transform = create_transform(**_raw_data_cfg, is_training=False)

del _tmp_model



## === cell 4
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True, return_id=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 5
"""
model_paths = {
    'resnet18': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    'efficientnet_b5': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/efficientnet_b5.pth",
    'inception_resnet_v2': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    'inception_v4': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_v4.pth",
    'seresnext50_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext50_32x4d.pth",
    'seresnext101_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext101_32x4d.pth"
}
"""
model_paths = {
    "efficientnet_b0": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b0.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}




## === cell 6
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


class ImageInformedFallback(nn.Module):
    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b5.ns_jft_in1k",
        image_root: str = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.image_root = image_root

        self.register_buffer(
            "grade_centers",
            torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0], dtype=torch.float32).view(1, 5),
        )
        self.register_buffer("sigma", torch.tensor(0.16, dtype=torch.float32))

        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

        self.register_buffer("energy_scale", torch.tensor(1.25, dtype=torch.float32))

    def _raw_tensor_from_ids(self, id_codes, device, dtype):
        raws = []
        sizes = []
        for idc in id_codes:
            p = os.path.join(self.image_root, f"{idc}.png")
            im = Image.open(p).convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # [H,W,3] in [0,1]
            t = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
            raws.append(t)
            sizes.append((t.shape[1], t.shape[2]))
        max_h = max(h for h, w in sizes)
        max_w = max(w for h, w in sizes)
        out = torch.zeros((len(raws), 3, max_h, max_w), dtype=dtype, device=device)
        for i, t in enumerate(raws):
            _, h, w = t.shape
            out[i, :, :h, :w] = t.to(device=device, dtype=dtype)
        return out

    def forward(self, x, id_codes=None):
        b = x.shape[0]

        if (id_codes is not None) and (self.image_root is not None):
            x_raw_for_feats = self._raw_tensor_from_ids(
                id_codes=id_codes, device=x.device, dtype=x.dtype
            )
        else:
            mean = torch.tensor(_data_cfg["mean"], device=x.device, dtype=x.dtype).view(
                1, 3, 1, 1
            )
            std = torch.tensor(_data_cfg["std"], device=x.device, dtype=x.dtype).view(
                1, 3, 1, 1
            )
            x_raw_for_feats = (x * std + mean).clamp(0.0, 1.0)

        r = x_raw_for_feats[:, 0:1]
        g = x_raw_for_feats[:, 1:2]
        bl = x_raw_for_feats[:, 2:3]

        mean_intensity = x_raw_for_feats.mean(dim=(1, 2, 3))  # [B]
        std_intensity = x_raw_for_feats.std(dim=(1, 2, 3))  # [B]

        green_dom = g.mean(dim=(1, 2, 3)) - 0.5 * (
            r.mean(dim=(1, 2, 3)) + bl.mean(dim=(1, 2, 3))
        )
        red_excess = (r - g).clamp_min(0.0).mean(dim=(1, 2, 3))  # [B]
        blue_deficit = (g - bl).clamp_min(0.0).mean(dim=(1, 2, 3))  # [B]

        gx = torch.nn.functional.conv2d(g, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(g, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(gx * gx + gy * gy + 1e-12).mean(dim=(1, 2, 3))  # [B]

        feat = self.backbone(x)  # [B,C]
        feat_energy = torch.log1p(feat.abs().mean(dim=1))  # [B]
        feat_energy01 = torch.sigmoid(self.energy_scale * (feat_energy - 0.7))

        s_hand = (
            0.42 * (1.0 - mean_intensity)
            + 0.22 * std_intensity
            + 0.18 * red_excess
            + 0.10 * blue_deficit
            + 0.10 * edge_mag
            - 0.08 * green_dom
        )
        s_hand = torch.clamp(s_hand, 0.0, 1.0)

        s = torch.clamp(0.82 * s_hand + 0.18 * feat_energy01, 0.0, 1.0).view(b, 1)
        logits = -((s - self.grade_centers) ** 2) / (2.0 * (self.sigma**2))
        return logits


models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        models_list.append(model.to(device).eval())
        loaded_model_keys.append(model_key)
    else:
        print(f"WARNING: missing checkpoint for {model_key} at {path}. Skipping.")

using_fallback = False
if len(models_list) == 0:
    using_fallback = True
    print(
        "WARNING: no checkpoints found; using image-informed fallback model for inference."
    )
    model = ImageInformedFallback(
        backbone_name=_fallback_model_name_for_transform,
        image_root="/kaggle/input/aptos2019-blindness-detection/train_images",  # default, overridden per use
    )
    models_list = [model.to(device).eval()]
    loaded_model_keys = [_fallback_model_name_for_transform]



## === cell 7
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}



## === cell 8
if using_fallback and loaded_model_keys[0] not in validation_scores:
    validation_scores[loaded_model_keys[0]] = validation_scores.get(
        "efficientnet_b5", 0.870
    )

total_score = sum(validation_scores[k] for k in loaded_model_keys)
weights = {k: validation_scores[k] / total_score for k in loaded_model_keys}




## === cell 9
def _quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    N = n_classes
    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)
    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _apply_thresholds(scores, thr):
    thr = np.asarray(thr, dtype=np.float64)
    thr = np.sort(thr)
    return np.digitize(scores, thr).astype(np.int64)


def _optimize_thresholds(scores, y_true):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    q_inits = [
        np.quantile(scores, [0.2, 0.4, 0.6, 0.8]).astype(np.float64),
        np.quantile(scores, [0.15, 0.35, 0.65, 0.85]).astype(np.float64),
        np.quantile(scores, [0.25, 0.45, 0.65, 0.85]).astype(np.float64),
    ]
    smin, smax = float(scores.min()), float(scores.max())
    lin_init = np.array(
        [smin + (smax - smin) * p for p in (0.2, 0.4, 0.6, 0.8)], dtype=np.float64
    )
    inits = q_inits + [lin_init]

    grid = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 81))).astype(
        np.float64
    )

    best_thr = None
    best_kappa = -1e9

    for thr0 in inits:
        thr = np.sort(thr0.copy())
        k0 = _quadratic_weighted_kappa(y_true, _apply_thresholds(scores, thr))
        if k0 > best_kappa:
            best_kappa = k0
            best_thr = thr.copy()

        cur_thr = thr.copy()
        cur_kappa = k0
        for _ in range(4):
            improved = False
            for t in range(4):
                cand_best_thr = cur_thr.copy()
                cand_best_kappa = cur_kappa
                for v in grid:
                    cand = cur_thr.copy()
                    cand[t] = v
                    cand = np.sort(cand)
                    pred = _apply_thresholds(scores, cand)
                    k = _quadratic_weighted_kappa(y_true, pred)
                    if k > cand_best_kappa:
                        cand_best_kappa = k
                        cand_best_thr = cand
                if cand_best_kappa > cur_kappa + 1e-12:
                    cur_kappa = cand_best_kappa
                    cur_thr = cand_best_thr
                    improved = True
            if not improved:
                break

        if cur_kappa > best_kappa:
            best_kappa = cur_kappa
            best_thr = cur_thr.copy()

    return best_thr, float(best_kappa)


def _tta_probs(model, images, id_codes=None):
    def _call(m, x, ids):
        try:
            return m(x, ids)
        except TypeError:
            return m(x)

    logits0 = _call(model, images, id_codes)
    logits_h = _call(model, torch.flip(images, dims=[3]), id_codes)
    logits_v = _call(model, torch.flip(images, dims=[2]), id_codes)
    logits_hv = _call(model, torch.flip(images, dims=[2, 3]), id_codes)

    probs0 = nn.functional.softmax(logits0, dim=1)
    probsh = nn.functional.softmax(logits_h, dim=1)
    probsv = nn.functional.softmax(logits_v, dim=1)
    probshv = nn.functional.softmax(logits_hv, dim=1)
    return 0.25 * (probs0 + probsh + probsv + probshv)


def _probs_to_expected_grade(probs):
    g = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, -1)
    return (probs * g).sum(dim=1)  # [B]




## === cell 10
calibrated_thresholds = None

if using_fallback:
    train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
    train_df = pd.read_csv(train_csv_file)

    rng = np.random.default_rng(SEED)
    idx_by_class = {}
    for cls in sorted(train_df["diagnosis"].unique()):
        cls_idx = np.where(train_df["diagnosis"].values == cls)[0]
        rng.shuffle(cls_idx)
        idx_by_class[int(cls)] = cls_idx

    folds = [[], []]
    for cls, cls_idx in idx_by_class.items():
        split = len(cls_idx) // 2
        folds[0].append(cls_idx[:split])
        folds[1].append(cls_idx[split:])
    folds = [
        np.concatenate(f) if len(f) else np.array([], dtype=np.int64) for f in folds
    ]

    full_train_ds = BlindnessDataset(
        csv_file=train_csv_file,
        root_dir=train_root_dir,
        transform=transform,
        test=False,
        return_id=True,
    )

    class _SubsetDataset(Dataset):
        def __init__(self, base_ds, indices):
            self.base_ds = base_ds
            self.indices = np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.base_ds[int(self.indices[i])]

    fallback_model = models_list[0]
    fallback_model.eval()
    if isinstance(fallback_model, ImageInformedFallback):
        fallback_model.image_root = train_root_dir

    thr_list = []
    kappa_list = []

    for fi in range(2):
        val_idx = folds[fi]
        train_idx = folds[1 - fi]

        train_loader = DataLoader(
            _SubsetDataset(full_train_ds, train_idx),
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            _SubsetDataset(full_train_ds, val_idx),
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        tr_scores = []
        tr_labels = []
        with torch.no_grad():
            for images, labels, id_codes in tqdm(
                train_loader,
                desc=f"Collecting OOF-threshold fit scores (fold {fi+1}/2)",
            ):
                images = images.to(device, non_blocking=True)
                probs_avg = _tta_probs(fallback_model, images, id_codes=id_codes)
                s = _probs_to_expected_grade(probs_avg).detach().cpu().numpy()
                tr_scores.append(s)
                tr_labels.append(labels.numpy())
        tr_scores = np.concatenate(tr_scores, axis=0)
        tr_labels = np.concatenate(tr_labels, axis=0).astype(np.int64)

        thr, _ = _optimize_thresholds(tr_scores, tr_labels)

        val_scores = []
        val_labels = []
        with torch.no_grad():
            for images, labels, id_codes in tqdm(
                val_loader, desc=f"Evaluating OOF thresholds (fold {fi+1}/2)"
            ):
                images = images.to(device, non_blocking=True)
                probs_avg = _tta_probs(fallback_model, images, id_codes=id_codes)
                s = _probs_to_expected_grade(probs_avg).detach().cpu().numpy()
                val_scores.append(s)
                val_labels.append(labels.numpy())
        val_scores = np.concatenate(val_scores, axis=0)
        val_labels = np.concatenate(val_labels, axis=0).astype(np.int64)

        val_pred = _apply_thresholds(val_scores, thr)
        kappa = _quadratic_weighted_kappa(val_labels, val_pred)

        thr_list.append(thr)
        kappa_list.append(float(kappa))
        print(f"Fold {fi+1} thresholds:", thr)
        print(f"Fold {fi+1} QWK (OOF val):", float(kappa))

    calibrated_thresholds = np.mean(np.stack(thr_list, axis=0), axis=0)
    calibrated_thresholds = np.sort(calibrated_thresholds.astype(np.float64))
    print("Averaged calibrated thresholds:", calibrated_thresholds)
    print("Mean fold QWK (OOF val):", float(np.mean(kappa_list)))



## === cell 11
all_outputs = []

if using_fallback and isinstance(models_list[0], ImageInformedFallback):
    models_list[0].image_root = test_root_dir

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting test"):
        images, id_codes = batch
        images = images.to(device, non_blocking=True)

        outputs = []
        for model_key, model in zip(loaded_model_keys, models_list):
            probs_avg = _tta_probs(
                model, images, id_codes=id_codes if using_fallback else None
            )
            outputs.append(weights[model_key] * probs_avg.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        weighted_outputs = torch.sum(outputs, dim=0)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

if using_fallback and calibrated_thresholds is not None:
    expected_grade = (all_outputs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
    final_predictions = _apply_thresholds(expected_grade, calibrated_thresholds).astype(
        int
    )
else:
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 12
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
print(
    "Pred distribution:",
    pd.Series(final_predictions).value_counts().sort_index().to_dict(),
)
