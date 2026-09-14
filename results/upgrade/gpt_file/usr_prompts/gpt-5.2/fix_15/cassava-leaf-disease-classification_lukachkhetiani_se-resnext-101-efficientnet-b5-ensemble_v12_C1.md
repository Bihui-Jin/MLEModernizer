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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8757932910244787

# 6. Current score

0.65546

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the notebook-only `cd`/`pip` cells that won’t work in a normal Kaggle Python runtime and instead rely on the environment’s preinstalled PyTorch/TIMM/OpenCV. I replace the `efficientnet_pytorch` dependency with a TIMM EfficientNet-B5 created the same way as the other TIMM model, which fixes the import error while keeping the ensemble logic identical. I also fix two inference-time bugs that would crash or mispredict: `crop_image()` uses an undefined variable (`img` instead of `image`), and `softmax(total)` is missing the `dim` argument. Finally, I ensure the submission is aligned to `sample_submission.csv` order and always writes a valid `submission.csv`.'
- What this solution (achieved 0.11622) has done: 'I fix the immediate runtime blocker by making the model weight loading robust to missing `../input/ensemblev5/*.pth` files, so the script still runs end-to-end in this Kaggle environment. To keep the core ensemble logic intact, the same two-model averaging be used; if weights are unavailable, the models fall back to pretrained ImageNet initialization (or random init as last resort) rather than crashing. I also avoid `torch.hub` network access (which is typically disabled) by gracefully skipping MiDaS and using a no-op depth mask so cropping doesn’t break. Finally, I ensure the submission is aligned to `sample_submission.csv` and always written as `submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your low score is mainly because the ensemble is running with generic ImageNet-pretrained/random weights (the custom `../input/ensemblev5/*.pth` aren’t present), so predictions are essentially meaningless for this dataset. The smallest change that legitimately improves accuracy toward your target is to keep the exact same two-model averaging logic, but quickly fine-tune both models on the provided `train.csv` + `train_images` for a small number of epochs, then run the same inference code. I also disable MiDaS (already effectively disabled) and keep full-image processing to avoid unstable cropping behavior that can hurt accuracy. Finally, the submission is still aligned to `sample_submission.csv` order and written as `submission.csv`.'
- What this solution (achieved 0.74439) has done: 'The main timeout driver here is `torch.compile`: compiling two large backbones plus the compiled Python ensemble function adds significant one-time overhead that can exceed the 600s limit, while not changing the algorithmic work. I disable compilation (keeping the same models, weights, preprocessing, and training loop) and add a couple of safe throughput tweaks: set threads explicitly and use `channels_last` + non-blocking GPU transfer to speed convolution-heavy inference/training without changing outputs. I also keep determinism settings intact and avoid any changes to epochs, data volume, augmentations, or loss. All file paths and core logic remain the same.'
- What this solution (achieved 0.72907) has done: 'The timeout is dominated by (1) caching every training image with OpenCV resize/normalize on CPU (done twice for train/val) and (2) using heavy multi-worker DataLoaders while the dataset is already fully cached (duplicating memory and adding IPC overhead). To preserve identical training/inference semantics, the main speed fix is to remove the expensive full-image cache build and instead keep preprocessing inside `__getitem__` while using a small, safe number of workers and tuned prefetching/pinning. Additionally, we reduce repeated work by batching path joins ahead of time, enabling `inference_mode()` for eval/predict, and using `torch.set_float32_matmul_precision("high")` (no reduced precision) plus more appropriate thread counts on CPU-only runs. No model architecture, loss, epochs, dataset split, or evaluation logic is changed.'
- What this solution (achieved 0.65546) has done: 'Your current gap to the target is large (0.72907 → 0.87579), so the smallest legitimate way to move toward the target is to improve the fine-tuning effectiveness without changing the model architectures, loss, or overall training loop. I keep the same two-model ensemble and 2-epoch head-only fine-tune, but fix two high-impact issues that currently limit learning: (1) `ensemble_forward` averages logits, which weakens gradients for each head during training; we compute per-model loss and average the losses (same objective, better signal) while keeping inference unchanged, and (2) BatchNorm “train-only” currently updates running stats (bad for small fine-tune); we set BN layers to use batch stats without updating running stats. Finally, I add a minimal, deterministic train-time augmentation (horizontal flip) inside the training dataset only, which generally boosts generalization for leaf images while preserving the same preprocessing pipeline and evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    _cpu = os.cpu_count() or 1
    torch.set_num_threads(min(4, _cpu))
    torch.set_num_interop_threads(min(2, _cpu))
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"


## === cell 1
USE_MIDAS = False
midas = None
transform = None


def _safe_load_weights(model: torch.nn.Module, weight_path: str) -> bool:
    if weight_path is None or (not os.path.exists(weight_path)):
        return False
    state = torch.load(weight_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    return True


efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)
eff_loaded = _safe_load_weights(efficient, "../input/ensemblev5/eff_best.pth")
efficient.to(device)

seresnext = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)
se_loaded = _safe_load_weights(seresnext, "../input/ensemblev5/seresnext_best.pth")
seresnext.to(device)

_COMPILED = False

print(
    "Models ready. Custom weights loaded:",
    {"efficient": bool(eff_loaded), "seresnext": bool(se_loaded)},
    " MiDaS:",
    USE_MIDAS,
    " torch.compile:",
    _COMPILED,
)


## === cell 2
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

_IMREAD_FLAG = cv2.IMREAD_COLOR


def processor(image_bgr: np.ndarray, device_override=None) -> torch.Tensor:
    img = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32, copy=False)
    img *= 1.0 / 255.0
    img -= _MEAN
    img /= _STD
    chw = np.ascontiguousarray(img.transpose(2, 0, 1))  # (C,H,W) float32 contiguous
    t = torch.from_numpy(chw).unsqueeze(0)  # CPU float32
    if device_override is not None:
        t = t.to(device_override, non_blocking=True)
    return t


@torch.no_grad()
def get_depth(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    return np.ones((h, w), dtype=bool)


def crop_image(image_bgr: np.ndarray, depth_mask: np.ndarray) -> np.ndarray:
    if depth_mask is None:
        return image_bgr
    if (
        isinstance(depth_mask, np.ndarray)
        and depth_mask.dtype == bool
        and depth_mask.ndim == 2
    ):
        if depth_mask.shape[:2] == image_bgr.shape[:2] and depth_mask.all():
            return image_bgr

    m = depth_mask.astype(bool, copy=False)
    if m.ndim != 2 or m.shape[:2] != image_bgr.shape[:2]:
        return image_bgr
    if m.all():
        return image_bgr
    ys, xs = np.nonzero(m)
    if ys.size == 0 or xs.size == 0:
        return image_bgr
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    h, w = image_bgr.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)
    if x_max <= x_min or y_max <= y_min:
        return image_bgr
    cropped = image_bgr[y_min : y_max + 1, x_min : x_max + 1]
    cropped_mask = m[y_min : y_max + 1, x_min : x_max + 1]
    if cropped_mask.all():
        return cropped
    out = cropped.copy()
    out[~cropped_mask] = 0
    return out




## === cell 3
class CassavaTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, augment: bool = False):
        df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.image_ids = df["image_id"].astype(str).to_numpy()
        self.labels = df["label"].astype(np.int64).to_numpy()

        self.augment = bool(augment)

        self._cache_x = None

    def enable_cache(self):
        return

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.image_ids[idx])
        img = cv2.imread(img_path, _IMREAD_FLAG)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.augment and ((idx & 1) == 1):
            img = cv2.flip(img, 1)

        depth = get_depth(img)
        cropped = crop_image(img, depth)
        x = processor(cropped, device_override=None).squeeze(0)  # CPU tensor (C,H,W)
        y = int(self.labels[idx])
        return x, y


def _get_classifier_params(m: torch.nn.Module):
    if hasattr(m, "get_classifier"):
        head = m.get_classifier()
        if isinstance(head, torch.nn.Module):
            return list(head.parameters())
    if hasattr(m, "classifier") and isinstance(m.classifier, torch.nn.Module):
        return list(m.classifier.parameters())
    if hasattr(m, "fc") and isinstance(m.fc, torch.nn.Module):
        return list(m.fc.parameters())
    if hasattr(m, "head") and isinstance(m.head, torch.nn.Module):
        return list(m.head.parameters())
    return list(m.parameters())


def ensemble_forward(x):
    return (efficient(x) + seresnext(x)) / 2.0


def _stratified_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    val_idx = []
    for lab, g in df.groupby("label"):
        idxs = g.index.to_numpy()
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac)))
        val_idx.append(idxs[:n_val])
    val_idx = np.concatenate(val_idx)
    val_mask = df.index.isin(val_idx)
    train_df = df.loc[~val_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    return train_df, val_df


@torch.inference_mode()
def _eval_acc(loader):
    efficient.eval()
    seresnext.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = ensemble_forward(xb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(1, total)


def _set_bn_train_only(m: torch.nn.Module):
    for mod in m.modules():
        if isinstance(
            mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        ):
            mod.train()
            mod.track_running_stats = False


def _make_loader(ds, batch_size, shuffle, nw):
    pf = 2 if nw > 0 else None
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=pf,
        drop_last=False,
    )


def _fine_tune_if_needed():
    need_train = (not eff_loaded) or (not se_loaded)
    if not need_train:
        efficient.eval()
        seresnext.eval()
        return

    train_df_full = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = _stratified_split(train_df_full, val_frac=0.1, seed=42)

    ds_tr = CassavaTrainDataset(train_df, TRAIN_DIR, augment=True)
    ds_va = CassavaTrainDataset(val_df, TRAIN_DIR, augment=False)

    ds_tr.enable_cache()
    ds_va.enable_cache()

    _cpu = os.cpu_count() or 1
    nw = 2 if device.type == "cuda" else 4
    nw = int(max(0, min(nw, _cpu)))

    try:
        tr_loader = _make_loader(ds_tr, batch_size=16, shuffle=True, nw=nw)
        va_loader = _make_loader(ds_va, batch_size=32, shuffle=False, nw=nw)
        _ = next(iter(tr_loader))
    except Exception as e:
        print(
            "DataLoader worker issue detected; falling back to num_workers=0. Error:",
            repr(e),
        )
        nw = 0
        tr_loader = _make_loader(ds_tr, batch_size=16, shuffle=True, nw=nw)
        va_loader = _make_loader(ds_va, batch_size=32, shuffle=False, nw=nw)

    efficient.train()
    seresnext.train()

    if device.type == "cuda":
        efficient.to(memory_format=torch.channels_last)
        seresnext.to(memory_format=torch.channels_last)

    for p in efficient.parameters():
        p.requires_grad = False
    for p in seresnext.parameters():
        p.requires_grad = False

    eff_head_params = _get_classifier_params(efficient)
    se_head_params = _get_classifier_params(seresnext)
    for p in eff_head_params:
        p.requires_grad = True
    for p in se_head_params:
        p.requires_grad = True

    params = eff_head_params + se_head_params
    optim = torch.optim.Adam(params, lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    best_eff = None
    best_se = None

    epochs = 2  # unchanged
    for ep in range(epochs):
        total_loss = 0.0
        n = 0

        efficient.train()
        seresnext.train()
        _set_bn_train_only(efficient)
        _set_bn_train_only(seresnext)

        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)

            optim.zero_grad(set_to_none=True)

            out_eff = efficient(xb)
            out_se = seresnext(xb)
            loss = 0.5 * criterion(out_eff, yb) + 0.5 * criterion(out_se, yb)

            loss.backward()
            optim.step()

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        val_acc = _eval_acc(va_loader)
        print(
            f"finetune epoch {ep+1}/{epochs} loss={total_loss/max(n,1):.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_eff = {
                k: v.detach().cpu().clone() for k, v in efficient.state_dict().items()
            }
            best_se = {
                k: v.detach().cpu().clone() for k, v in seresnext.state_dict().items()
            }

    if best_eff is not None and best_se is not None:
        efficient.load_state_dict(best_eff, strict=True)
        seresnext.load_state_dict(best_se, strict=True)

    for m in efficient.modules():
        if isinstance(
            m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        ):
            m.track_running_stats = True
    for m in seresnext.modules():
        if isinstance(
            m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        ):
            m.track_running_stats = True

    efficient.eval()
    seresnext.eval()


_fine_tune_if_needed()


## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
names = sample_sub["image_id"].astype(str).tolist()

files = [os.path.join(TEST_DIR, n) for n in names]
labels = []


class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        fp = self.file_paths[idx]
        img = cv2.imread(fp, _IMREAD_FLAG)
        if img is None:
            x = torch.zeros((3, 512, 512), dtype=torch.float32)
            return x
        depth = get_depth(img)
        cropped = crop_image(img, depth)
        x = processor(cropped, device_override=None).squeeze(0)  # CPU tensor
        return x


test_ds = CassavaTestDataset(files)

_cpu = os.cpu_count() or 1
nw = 2 if device.type == "cuda" else 4
nw = int(max(0, min(nw, _cpu)))
pf = 2 if nw > 0 else None

try:
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=pf,
    )
    _ = next(iter(test_loader))
except Exception as e:
    print(
        "Test DataLoader worker issue detected; falling back to num_workers=0. Error:",
        repr(e),
    )
    nw = 0
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

efficient.eval()
seresnext.eval()

if device.type == "cuda":
    efficient.to(memory_format=torch.channels_last)
    seresnext.to(memory_format=torch.channels_last)

with torch.inference_mode():
    for xb in test_loader:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        total = ensemble_forward(xb)
        pred = torch.argmax(total, dim=1).detach().cpu().numpy().astype(int).tolist()
        labels.extend(pred)

print(f"Predicted {len(labels)} test images. Expected: {len(names)}")


## === cell 5
if len(labels) != len(names):
    print(
        "Warning: labels length mismatch; padding/truncating to match sample submission."
    )
    if len(labels) < len(names):
        fill_val = int(pd.Series(labels).mode().iloc[0]) if len(labels) else 0
        labels = labels + [fill_val] * (len(names) - len(labels))
    else:
        labels = labels[: len(names)]

sub = pd.DataFrame({"image_id": names, "label": labels})

merged = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")

if merged["label"].isna().any():
    fill_val = int(pd.Series(labels).mode().iloc[0]) if len(labels) else 0
    merged["label"] = merged["label"].fillna(fill_val).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", merged.shape)
print(merged.head())
