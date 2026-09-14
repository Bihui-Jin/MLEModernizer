# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8332698961136675

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.04608) has done: 'I fix the image path resolution bug that’s causing `FileNotFoundError` by making `resolve_images_dir()` robust to nested `test_images/test_images` (and similar) structures. I also harden the dataset `__getitem__` to try a small set of common fallback locations so the DataLoader won’t crash if a directory layout is slightly different. Finally, I ensure inference always produces `final_predictions` and then write a valid `submission.csv` aligned to `test.csv` order. These changes are execution-stability focused (score-neutral) and preserve the existing model/inference logic.'
- What this solution (achieved 0.64003) has done: 'Your current negative kappa is mainly because the fallback model is ImageNet-pretrained with a random 5-class head (since no checkpoints load), so predictions are essentially noise. To move the score toward your target with minimal changes, I keep the exact inference semantics (softmax → weighted average → argmax) and keep the same model family, but I add a tiny training step to fit only the final classification head on the provided `train.csv` images (backbone frozen). This preserves the core model architecture and avoids changing loss/feature extraction; it simply makes the 5-class head non-random so predictions become meaningful. I also keep your robust path resolution and ensure the submission stays aligned to `test.csv`.'
- What this solution (achieved 0.73393) has done: 'Your current score is well below the target (0.64003 vs 0.83327), so we should improve performance while keeping your model/inference semantics unchanged. The biggest low-risk win is to make the “fit only the head” step actually learn better features for QWK by (1) using a higher-resolution input for EfficientNet (its native 224 is OK but bumping modestly helps) and (2) using light, standard train-time augmentation (flip/rotate/color jitter) while keeping test preprocessing unchanged. To better match the ordinal nature of QWK without changing loss/model, we also replace the final `argmax` with an OOF-optimized set of 4 thresholds on the model’s expected value (softmax-weighted mean), which is a minimal post-processing change commonly used for this competition. All changes preserve the core backbone+head architecture, keep CrossEntropy training, keep “softmax → weighted average” inference, and still write a valid `submission.csv`.'

# 9. Code solution

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
from torchvision import transforms
import timm




## === cell 1
def find_aptos_root():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c

    for base in ["/kaggle/input", "/kaggle/data"]:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            if "train.csv" in files and "test.csv" in files:
                return root

    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset root under /kaggle/input or /kaggle/data."
    )


APTOS_ROOT = find_aptos_root()


def resolve_images_dir(root, split):
    base = "train_images" if split == "train" else "test_images"
    d = os.path.join(root, base)

    candidates = [
        os.path.join(root, base, base),
        os.path.join(root, base),
        os.path.join(root, "aptos2019-blindness-detection", base, base),
        os.path.join(root, "aptos2019-blindness-detection", base),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c

    return d




## === cell 2
from collections import OrderedDict


class _LRUCache:
    def __init__(self, max_items: int = 2048):
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


_GLOBAL_TENSOR_CACHE = _LRUCache(max_items=4096)


class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        test=False,
        cache_dir=None,
        cache_key=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

        self.cache_dir = cache_dir
        self.cache_key = cache_key or ("test" if test else "train")

        self._alt_roots = []
        parent = os.path.dirname(root_dir)
        if parent and parent != root_dir:
            self._alt_roots.append(parent)
        grandparent = os.path.dirname(parent) if parent else ""
        if grandparent and grandparent not in (parent, root_dir):
            self._alt_roots.append(grandparent)

    def __len__(self):
        return len(self.annotations)

    def _open_image(self, img_id):
        rel = img_id + ".png"
        candidates = [os.path.join(self.root_dir, rel)]
        for r in self._alt_roots:
            candidates.append(os.path.join(r, rel))

        if os.path.basename(self.root_dir) in ("train_images", "test_images"):
            parent = os.path.dirname(self.root_dir)
            candidates.append(os.path.join(parent, rel))

        for p in candidates:
            if os.path.exists(p):
                with Image.open(p) as im:
                    return im.convert("RGB")

        raise FileNotFoundError(
            f"Could not find image for id_code={img_id}. Tried: {candidates}"
        )

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]

        ck = (self.cache_key, img_id)
        cached = _GLOBAL_TENSOR_CACHE.get(ck)
        if cached is not None:
            image = cached
        else:
            image = self._open_image(img_id)
            if self.transform:
                image = self.transform(image)
            _GLOBAL_TENSOR_CACHE.put(ck, image)

        if self.test:
            return image

        label = int(self.annotations.iloc[idx, 1])
        return image, label




## === cell 3
IMG_SIZE = 320  # keep identical core setting

test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 4
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

test_csv_file = os.path.join(APTOS_ROOT, "test.csv")
test_root_dir = resolve_images_dir(APTOS_ROOT, "test")

CACHE_DIR = "/kaggle/working/_tensor_cache"

test_dataset = BlindnessDataset(
    test_csv_file,
    test_root_dir,
    transform=test_transform,
    test=True,
    cache_dir=CACHE_DIR,
    cache_key="test",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_num_workers = min(4, (os.cpu_count() or 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)



## === cell 5
"""
Keep the same inference semantics, but if checkpoints are missing we must avoid a random 5-class head.
Train ONLY the final classification head on train.csv with frozen backbone (unchanged core logic).

Change rationale (score toward target): ensure the intended checkpoint path is treated as optional so the
pipeline always trains a non-random head and produces a valid submission instead of failing or being noise.
"""

model_paths = {
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/seresnext50_32x4d.pth",
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




## === cell 7
def _get_classifier_params(model: nn.Module):
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module):
            return list(head.parameters())
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        return list(model.classifier.parameters())
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        return list(model.fc.parameters())
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        return list(model.head.parameters())
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            return list(m.parameters())
    raise AttributeError("Could not locate classifier head parameters for this model.")


def freeze_backbone_only_head_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for p in _get_classifier_params(model):
        p.requires_grad = True


def fit_head_on_train(
    model: nn.Module,
    train_csv: str,
    train_images_dir: str,
    device: torch.device,
    train_indices=None,
    val_df=None,
):
    full_df = pd.read_csv(train_csv)
    if train_indices is not None:
        df = full_df.iloc[train_indices].reset_index(drop=True)
    else:
        df = full_df

    tmp_train_csv = "/kaggle/working/_train_head_fit.csv"
    df.to_csv(tmp_train_csv, index=False)

    train_dataset = BlindnessDataset(
        tmp_train_csv,
        train_images_dir,
        transform=train_transform,
        test=False,
        cache_dir=CACHE_DIR,
        cache_key="train_aug",
    )

    _num_workers_tr = min(4, (os.cpu_count() or 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=_num_workers_tr,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(_num_workers_tr > 0),
        prefetch_factor=2 if _num_workers_tr > 0 else None,
    )

    freeze_backbone_only_head_trainable(model)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=3e-3,
        weight_decay=1e-4,
    )

    for epoch in range(2):
        for images, labels in tqdm(
            train_loader, desc=f"Fitting head (epoch {epoch+1}/2)", leave=False
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()

    val_probs = None
    if val_df is not None and len(val_df) > 0:
        tmp_val_csv = "/kaggle/working/_val_for_thr_from_fit.csv"
        val_df.to_csv(tmp_val_csv, index=False)
        val_dataset = BlindnessDataset(
            tmp_val_csv,
            train_images_dir,
            transform=test_transform,
            test=False,
            cache_dir=CACHE_DIR,
            cache_key="val",
        )
        _num_workers_val = min(4, (os.cpu_count() or 2))
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=_num_workers_val,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(_num_workers_val > 0),
            prefetch_factor=2 if _num_workers_val > 0 else None,
        )
        probs_list = []
        with torch.no_grad():
            for images, _y in tqdm(
                val_loader,
                desc="Val forward (cached) for threshold tuning",
                leave=False,
            ):
                images = images.to(device, non_blocking=True)
                probs = nn.functional.softmax(model(images), dim=1)
                probs_list.append(probs.detach().cpu().numpy())
        val_probs = np.concatenate(probs_list, axis=0)

    return model, val_probs




## === cell 8
models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        continue
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        new_ckpt = {}
        for k, v in ckpt.items():
            nk = k[7:] if k.startswith("module.") else k
            new_ckpt[nk] = v
        ckpt = new_ckpt

    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()

    models_list.append(model)
    loaded_model_keys.append(model_key)



## === cell 9
train_csv_file = os.path.join(APTOS_ROOT, "train.csv")
train_root_dir = resolve_images_dir(APTOS_ROOT, "train")
train_df = pd.read_csv(train_csv_file)

labels_all = train_df["diagnosis"].astype(int).values
idx_all = np.arange(len(train_df))

rng = np.random.RandomState(42)
val_mask = np.zeros(len(train_df), dtype=bool)
for c in range(5):
    cls_idx = idx_all[labels_all == c]
    rng.shuffle(cls_idx)
    n_val = max(1, int(round(0.15 * len(cls_idx))))
    val_mask[cls_idx[:n_val]] = True

train_indices = idx_all[~val_mask]
val_indices = idx_all[val_mask]
val_df = train_df.loc[val_indices, ["id_code", "diagnosis"]].reset_index(drop=True)

val_probs_from_fit = None

if len(models_list) == 0:
    fallback_key = "efficientnet_b0"
    fallback_name = model_names[fallback_key]
    model = timm.create_model(fallback_name, pretrained=True, num_classes=5)
    model.to(device)

    model, val_probs_from_fit = fit_head_on_train(
        model,
        train_csv_file,
        train_root_dir,
        device,
        train_indices=train_indices,
        val_df=val_df,
    )

    models_list = [model]
    loaded_model_keys = [fallback_key]



## === cell 10
scores = []
for k in loaded_model_keys:
    scores.append(validation_scores.get(k, 1.0))
total_score = float(np.sum(scores))
if total_score <= 0:
    weights = {k: 1.0 / len(loaded_model_keys) for k in loaded_model_keys}
else:
    weights = {
        k: float(validation_scores.get(k, 1.0)) / total_score for k in loaded_model_keys
    }




## === cell 11
def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    N = num_classes
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

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def apply_thresholds(preds_cont: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.digitize(preds_cont, thr).astype(int)


def optimize_thresholds_oof(
    preds_cont: np.ndarray, y_true: np.ndarray, init=None, iters: int = 60
) -> np.ndarray:
    if init is None:
        thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    else:
        thr = np.array(init, dtype=np.float64)

    thr.sort()
    best = quadratic_weighted_kappa(y_true, apply_thresholds(preds_cont, thr))

    for step in [0.20, 0.10, 0.05, 0.02, 0.01]:
        for _ in range(iters):
            improved = False
            for i in range(4):
                for delta in (-step, step):
                    cand = thr.copy()
                    cand[i] += delta
                    cand.sort()
                    if cand[0] <= -0.5 or cand[-1] >= 4.5:
                        continue
                    score = quadratic_weighted_kappa(
                        y_true, apply_thresholds(preds_cont, cand)
                    )
                    if score > best:
                        best = score
                        thr = cand
                        improved = True
            if not improved:
                break
    return thr




## === cell 12
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        batch_probs = None

        for model_key, model in zip(loaded_model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            w = weights.get(model_key, 1.0 / len(loaded_model_keys))
            probs = probs * w
            batch_probs = probs if batch_probs is None else (batch_probs + probs)

        all_outputs.append(batch_probs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)



## === cell 13
class_values = np.arange(5, dtype=np.float64)

val_labels = val_df["diagnosis"].astype(int).values

if (
    val_probs_from_fit is not None
    and len(val_probs_from_fit) == len(val_labels)
    and len(models_list) == 1
):
    val_probs = val_probs_from_fit
else:
    tmp_csv = "/kaggle/working/_val_split.csv"
    val_df.to_csv(tmp_csv, index=False)

    val_dataset = BlindnessDataset(
        tmp_csv,
        train_root_dir,
        transform=test_transform,
        test=False,
        cache_dir=CACHE_DIR,
        cache_key="val",
    )
    _num_workers_val2 = min(4, (os.cpu_count() or 2))
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=_num_workers_val2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(_num_workers_val2 > 0),
        prefetch_factor=2 if _num_workers_val2 > 0 else None,
    )

    val_probs_list = []
    with torch.no_grad():
        for images, _y in tqdm(
            val_loader, desc="Val forward for threshold tuning", leave=False
        ):
            images = images.to(device, non_blocking=True)
            batch_probs = None
            for model_key, model in zip(loaded_model_keys, models_list):
                probs = nn.functional.softmax(model(images), dim=1)
                w = weights.get(model_key, 1.0 / len(loaded_model_keys))
                probs = probs * w
                batch_probs = probs if batch_probs is None else (batch_probs + probs)
            val_probs_list.append(batch_probs.detach().cpu().numpy())
    val_probs = np.concatenate(val_probs_list, axis=0)

val_pred_cont = (val_probs * class_values[None, :]).sum(axis=1)
thr = optimize_thresholds_oof(val_pred_cont, val_labels)
print("Optimized thresholds (val split):", thr)

test_pred_cont = (all_outputs * class_values[None, :]).sum(axis=1)
final_predictions = apply_thresholds(test_pred_cont, thr)
final_predictions = np.clip(final_predictions, 0, 4).astype(int)



## === cell 14
test_ids = pd.read_csv(test_csv_file)["id_code"].astype(str).values
if len(final_predictions) != len(test_ids):
    raise RuntimeError(
        f"Prediction length mismatch: preds={len(final_predictions)} vs test_ids={len(test_ids)}"
    )

submission_df = pd.DataFrame(
    {"id_code": test_ids, "diagnosis": final_predictions.astype(int)}
)
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Using APTOS_ROOT:", APTOS_ROOT)
print("Resolved test_root_dir:", test_root_dir)
print("Loaded models:", loaded_model_keys)
print("Weights:", weights)
print("CACHE_DIR (unused for disk now; kept for compatibility):", CACHE_DIR)
