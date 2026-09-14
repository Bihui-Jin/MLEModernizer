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
import glob
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)

_num_workers = min(4, (os.cpu_count() or 4))
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/resnet18.pth",
    "efficientnet_b0": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b0.pth",
    "efficientnet_b1": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "efficientnet_b4": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b4.pth",
    "efficientnet_b5": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
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



## === cell 5
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}
total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_deterministic(seed: int = 1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_set_deterministic(1337)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if isinstance(ckpt, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        tensor_like = any(torch.is_tensor(v) for v in ckpt.values())
        if tensor_like:
            return ckpt

    return ckpt


def _normalize_state_dict_keys(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "student.", "encoder."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _candidate_checkpoint_filenames(model_key: str, original_path: str) -> list[str]:
    base = os.path.basename(original_path)
    stem, ext = os.path.splitext(base)
    exts = [ext] if ext else [".pth"]
    exts = list(dict.fromkeys(exts + [".pth", ".pt", ".bin"]))

    cands = set()
    for e in exts:
        cands.add(stem + e)

    suffixes = [
        "",
        "_best",
        "-best",
        "_final",
        "-final",
        "_last",
        "-last",
        "_best_kappa",
        "-best_kappa",
    ]
    fold_suffixes = [
        "",
        "_fold0",
        "_fold1",
        "_fold2",
        "_fold3",
        "_fold4",
        "-fold0",
        "-fold1",
        "-fold2",
        "-fold3",
        "-fold4",
        "_fold_0",
        "_fold_1",
        "_fold_2",
        "_fold_3",
        "_fold_4",
    ]
    for e in exts:
        for s in suffixes:
            for f in fold_suffixes:
                cands.add(f"{stem}{s}{f}{e}")

    if model_key.startswith("efficientnet_b"):
        b = model_key.split("_")[-1]  # b0..b5
        bases = [
            f"efficientnet_{b}",
            f"efficientnet-{b}",
            f"efficentNet_{b}",
            f"efficentnet_{b}",
            f"EfficientNet_{b}",
            f"EfficientNet-{b}",
            f"efficientNet_{b}",
        ]
        for bstem in bases:
            for e in exts:
                cands.add(bstem + e)
                for s in suffixes:
                    for f in fold_suffixes:
                        cands.add(f"{bstem}{s}{f}{e}")

    return list(cands)


def _pick_best_checkpoint_hit(
    hits: list[str], original_path: str, model_key: str
) -> str | None:
    if not hits:
        return None

    target_base = os.path.basename(original_path).lower()

    def score(p: str) -> tuple:
        pl = p.lower()
        preferred_ds = (
            ("aptos-ensamble-models" in pl)
            or ("aptos_ensamble_models" in pl)
            or ("aptos-ensemble-models" in pl)
            or ("aptos_ensemble_models" in pl)
            or ("ensamble_v2" in pl)
            or ("ensemble" in pl)
        )
        same_base = os.path.basename(p).lower() == target_base
        contains_model = model_key.replace("_", "") in pl.replace("_", "")
        return (preferred_ds, same_base, contains_model, -len(pl))

    return sorted(hits, key=score, reverse=True)[0]


def _discover_checkpoint_roots() -> list[str]:
    roots = []
    for base in ("/kaggle/input", "/kaggle/data"):
        if not os.path.exists(base):
            continue
        try:
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if os.path.isdir(p):
                    dl = d.lower()
                    if "aptos" in dl and (
                        "ensam" in dl or "ensemble" in dl or "model" in dl
                    ):
                        roots.append(p)
        except Exception:
            pass

    preferred = [
        "/kaggle/input/aptos-ensamble-models",
        "/kaggle/input/aptos_ensamble_models",
        "/kaggle/input/aptos-ensemble-models",
        "/kaggle/input/aptos_ensemble_models",
        "/kaggle/data/aptos-ensamble-models",
        "/kaggle/data/aptos_ensamble_models",
        "/kaggle/data/aptos-ensemble-models",
        "/kaggle/data/aptos_ensemble_models",
    ]
    ordered = []
    for p in preferred + roots:
        if p not in ordered and os.path.exists(p):
            ordered.append(p)
    return ordered


_DISCOVERED_CKPT_ROOTS = _discover_checkpoint_roots()

_CKPT_BY_BASENAME: dict[str, list[str]] = {}
_CKPT_INDEX = []
for root in _DISCOVERED_CKPT_ROOTS:
    try:
        hits = []
        for ext in ("*.pth", "*.pt", "*.bin"):
            hits.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        _CKPT_INDEX.extend(hits)
        for p in hits:
            b = os.path.basename(p)
            _CKPT_BY_BASENAME.setdefault(b, []).append(p)
    except Exception:
        pass


def _resolve_checkpoint_path(original_path: str, model_key: str) -> str | None:
    if os.path.exists(original_path):
        return original_path

    orig_dir = os.path.dirname(original_path)
    for fname in _candidate_checkpoint_filenames(model_key, original_path):
        trial = os.path.join(orig_dir, fname)
        if os.path.exists(trial):
            return trial

    for fname in _candidate_checkpoint_filenames(model_key, original_path):
        hits = _CKPT_BY_BASENAME.get(fname, [])
        best = _pick_best_checkpoint_hit(hits, original_path, model_key)
        if best is not None:
            return best

    return None


def _remap_head_keys_if_needed(sd: dict, model: nn.Module) -> dict:
    if not isinstance(sd, dict):
        return sd

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    sd_keys = set(sd.keys())

    if len(sd_keys & model_keys) / max(1, len(model_keys)) > 0.90:
        return sd

    remap_candidates = [
        ("classifier.", "fc."),
        ("fc.", "classifier."),
        ("head.", "fc."),
        ("fc.", "head."),
        ("head.fc.", "fc."),
        ("model.classifier.", "classifier."),
        ("model.fc.", "fc."),
        ("classifier.fc.", "classifier."),
        ("head.fc.", "classifier."),
        ("classifier.", "head."),
    ]

    best_sd = sd
    best_overlap = len(sd_keys & model_keys)

    for src, dst in remap_candidates:
        trial = {}
        for k, v in sd.items():
            if k.startswith(src):
                trial[dst + k[len(src) :]] = v
            else:
                trial[k] = v
        overlap = len(set(trial.keys()) & model_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sd = trial

    return best_sd


def _init_5class_head_deterministically(model: nn.Module, seed: int = 1337):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for _, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, mean=0.0, std=0.02, generator=g)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)




## === cell 7
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"


def _kappa_quadratic_np(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            O[a, b] += 1.0
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def _train_one_model(
    model_name: str, train_loader, val_loader, epochs: int = 1, lr: float = 3e-4
):
    model = timm.create_model(model_name, pretrained=True, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_p.append(pred)
            all_t.append(yb.numpy())
    kp = _kappa_quadratic_np(
        np.concatenate(all_t), np.concatenate(all_p), num_classes=5
    )
    return model, float(kp)


full_df = pd.read_csv(train_csv_file)
perm = np.random.RandomState(1337).permutation(len(full_df))
val_size = max(1, int(0.15 * len(full_df)))
val_idx = perm[:val_size]
tr_idx = perm[val_size:]

train_df = full_df.iloc[tr_idx].reset_index(drop=True)
val_df = full_df.iloc[val_idx].reset_index(drop=True)

_tmp_train_csv = "/kaggle/working/_train_split.csv"
_tmp_val_csv = "/kaggle/working/_val_split.csv"
train_df.to_csv(_tmp_train_csv, index=False)
val_df.to_csv(_tmp_val_csv, index=False)

train_ds = BlindnessDataset(
    _tmp_train_csv, train_root_dir, transform=transform, test=False
)
val_ds = BlindnessDataset(_tmp_val_csv, train_root_dir, transform=transform, test=False)

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)
val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)



## === cell 8
models_list = []
active_model_keys = []
loaded_paths = {}

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    resolved = _resolve_checkpoint_path(path, model_key)

    if resolved is not None:
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        ckpt = torch.load(resolved, map_location="cpu")
        sd = _extract_state_dict(ckpt)

        if isinstance(sd, dict):
            sd = _normalize_state_dict_keys(sd)
            sd = _remap_head_keys_if_needed(sd, model)

        model_keys = set(model.state_dict().keys())
        sd_keys = set(sd.keys()) if isinstance(sd, dict) else set()
        overlap_ratio = (
            (len(model_keys & sd_keys) / max(1, len(model_keys)))
            if isinstance(sd, dict)
            else 0.0
        )

        if overlap_ratio >= 0.20:
            try:
                model.load_state_dict(sd, strict=True)
            except Exception:
                model.load_state_dict(sd, strict=False)
            loaded_paths[model_key] = resolved
        else:
            resolved = None  # force fallback path

    if resolved is None:
        model, val_kappa = _train_one_model(
            model_name, train_loader, val_loader, epochs=1, lr=3e-4
        )

    model.to(device)
    model.eval()
    models_list.append(model)
    active_model_keys.append(model_key)

active_total = sum(weights[k] for k in active_model_keys)
active_weights = {k: (weights[k] / active_total) for k in active_model_keys}

print("Discovered checkpoint roots:", _DISCOVERED_CKPT_ROOTS)
print("Indexed checkpoints:", len(_CKPT_INDEX))
print("Active models:", active_model_keys)
print("Loaded checkpoints:", {k: loaded_paths.get(k, None) for k in active_model_keys})
print(
    "Num models:",
    len(models_list),
    "| weight sum:",
    float(sum(active_weights.values())),
)



## === cell 9
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)

        per_model = []
        for model_key, model in zip(active_model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            per_model.append(active_weights[model_key] * probs)

        weighted_outputs = torch.stack(per_model, dim=0).sum(dim=0)
        all_outputs.append(weighted_outputs.cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 10
test_ids = pd.read_csv(test_csv_file)["id_code"].values

submission_df = pd.DataFrame(
    {
        "id_code": test_ids,
        "diagnosis": final_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("rows:", len(submission_df), "cols:", list(submission_df.columns))

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
