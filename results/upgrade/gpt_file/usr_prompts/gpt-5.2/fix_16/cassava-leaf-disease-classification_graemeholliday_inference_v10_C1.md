# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os
import gc
import time
import random
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm

import torchvision



## === cell 1
image_size = 380



## === cell 2
DATA_ROOT = "/kaggle/data/cassava-leaf-disease-classification"
ALT_DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.isdir(DATA_ROOT) and os.path.isdir(ALT_DATA_ROOT):
    DATA_ROOT = ALT_DATA_ROOT

CASSAVA_MEAN = [0.43216, 0.50340, 0.31323]
CASSAVA_STD = [0.21506, 0.24055, 0.18584]

config = dict(
    seed=22,
    experiment_name="enb7",
    test_location=os.path.join(DATA_ROOT, "test_images"),
    train_location=os.path.join(DATA_ROOT, "train_images"),
    train_csv=os.path.join(DATA_ROOT, "train.csv"),
    checkpoint_path=DATA_ROOT,
    checkpoint="enb7best.pt",
    model="efficientnet-b7",
    epochs=10,
    finetune_epochs=2,
    lr=3e-4,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(
            name="RandomResizedCrop",
            params=dict(
                height=image_size,
                width=image_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
                p=1.0,
            ),
        ),
        dict(
            name="Resize",
            params=dict(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_LINEAR,
                p=1.0,
            ),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=CASSAVA_MEAN,
                std=CASSAVA_STD,
                max_pixel_value=1.0,
                p=1.0,
            ),
        ),
    ],
)



## === cell 3
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path)
test = sample[["image_id"]].copy()
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("DATA_ROOT:", DATA_ROOT)
print(
    "Test images dir exists:",
    os.path.isdir(config["test_location"]),
    "|",
    config["test_location"],
)

if not os.path.isdir(config["test_location"]):
    raise FileNotFoundError(
        f"test_images directory not found: {config['test_location']}"
    )
_test_files = [
    f for f in os.listdir(config["test_location"]) if f.lower().endswith(".jpg")
]
print("Test images found (jpg count):", len(_test_files))
if len(_test_files) == 0:
    raise RuntimeError(f"No .jpg files found in: {config['test_location']}")

print(
    "Train images dir exists:",
    os.path.isdir(config["train_location"]),
    "|",
    config["train_location"],
)
if not os.path.isdir(config["train_location"]):
    raise FileNotFoundError(
        f"train_images directory not found: {config['train_location']}"
    )
if not os.path.isfile(config["train_csv"]):
    raise FileNotFoundError(f"train.csv not found: {config['train_csv']}")




## === cell 6
def _build_model_torchvision(use_imagenet_weights: bool = False):
    weights = (
        torchvision.models.EfficientNet_B7_Weights.DEFAULT
        if use_imagenet_weights
        else None
    )
    model = torchvision.models.efficientnet_b7(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 5)
    return model


def _strip_known_prefixes(state_dict: dict):
    """
    Change rationale (score toward target): if the checkpoint was saved under wrappers (DataParallel/Lightning),
    keys often have prefixes like 'module.' or 'model.'; stripping them allows strict loading of real weights.
    """
    prefixes = ["module.", "model.", "net.", "backbone.", "student.", "ema_model."]
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _extract_state_dict(ckpt_obj):
    """
    Change rationale (score toward target): robustly extract actual model weights dict from common
    checkpoint formats so we don't silently run with mostly-random weights.
    """
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if len(ckpt_obj) > 0 and all(
            hasattr(v, "shape") or torch.is_tensor(v) for v in ckpt_obj.values()
        ):
            return ckpt_obj
    return None


def _looks_like_5class_head_tensor(t):
    try:
        return torch.is_tensor(t) and t.ndim in (1, 2) and t.shape[0] == 5
    except Exception:
        return False


def _looks_like_enb7_5class_state_dict(state: dict) -> bool:
    """
    Change rationale (score toward target): accept common head key variants so we don't reject
    the real fine-tuned checkpoint and fall back to ImageNet (near-random on cassava).
    """
    if not isinstance(state, dict) or len(state) == 0:
        return False

    if "classifier.1.weight" in state and "classifier.1.bias" in state:
        w = state["classifier.1.weight"]
        b = state["classifier.1.bias"]
        return _looks_like_5class_head_tensor(w) and _looks_like_5class_head_tensor(b)

    candidates = [
        ("classifier.weight", "classifier.bias"),
        ("head.weight", "head.bias"),
        ("fc.weight", "fc.bias"),
        ("last_linear.weight", "last_linear.bias"),
        ("_fc.weight", "_fc.bias"),
    ]
    for wk, bk in candidates:
        if wk in state and bk in state:
            w = state[wk]
            b = state[bk]
            if _looks_like_5class_head_tensor(w) and _looks_like_5class_head_tensor(b):
                return True

    return False


def _remap_efficientnet_keys_to_torchvision(state: dict) -> dict:
    """
    Change rationale (score toward target): handle common non-torchvision EfficientNet key layouts.
    Minimal remapping only (no architecture change), to make strict loading succeed.
    """
    if not isinstance(state, dict):
        return state

    remapped = {}
    for k, v in state.items():
        nk = k

        if nk.startswith("state_dict."):
            nk = nk[len("state_dict.") :]

        if nk.startswith("encoder."):
            nk = "features." + nk[len("encoder.") :]
        if nk.startswith("backbone."):
            nk = "features." + nk[len("backbone.") :]

        if nk in ("_fc.weight", "model._fc.weight"):
            nk = "classifier.1.weight"
        if nk in ("_fc.bias", "model._fc.bias"):
            nk = "classifier.1.bias"

        if nk in (
            "classifier.weight",
            "head.weight",
            "fc.weight",
            "last_linear.weight",
        ):
            nk = "classifier.1.weight"
        if nk in (
            "classifier.bias",
            "head.bias",
            "fc.bias",
            "last_linear.bias",
        ):
            nk = "classifier.1.bias"

        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]

        if nk.startswith("."):
            nk = nk[1:]

        remapped[nk] = v
    return remapped


def _match_ratio(model, state: dict) -> float:
    msd = model.state_dict()
    if not isinstance(state, dict) or len(state) == 0:
        return 0.0
    matched = 0
    total = 0
    for k, v in state.items():
        if k in msd:
            total += 1
            try:
                if tuple(msd[k].shape) == tuple(v.shape):
                    matched += 1
            except Exception:
                pass
    return float(matched) / float(max(1, total))


def _load_state_dict_flex(model, ckpt_obj):
    state = _extract_state_dict(ckpt_obj)
    if state is None:
        raise ValueError(
            "Could not extract a state_dict from the checkpoint object. "
            "Expected keys like 'state_dict'/'model_state_dict'/'model'."
        )

    state = _strip_known_prefixes(state)
    state = _remap_efficientnet_keys_to_torchvision(state)

    if not _looks_like_enb7_5class_state_dict(state):
        raise ValueError(
            "Checkpoint state_dict does not look like a 5-class classifier head (various common key names)."
        )

    ratio = _match_ratio(model, state)
    print(f"Checkpoint key-shape match ratio (approx): {ratio:.3f}")

    if ratio < 0.70:
        raise ValueError(
            f"Checkpoint matches too few model keys (ratio={ratio:.3f}); likely incompatible architecture/keying."
        )

    model.load_state_dict(state, strict=True)
    print("Loaded checkpoint with strict=True.")
    return model


def _candidate_checkpoint_files(root: str):
    exts = {".pt", ".pth", ".ckpt", ".bin"}
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                out.append(os.path.join(dirpath, fn))
    return out


def _find_checkpoint_path(checkpoint_name: str, preferred_dir: str):
    """
    Change rationale (score toward target): ensure we actually find cassava fine-tuned weights if present.
    Search the cassava dataset tree (including nested copies) rather than relying on one exact filename.
    """
    if preferred_dir:
        p = os.path.join(preferred_dir, checkpoint_name)
        if os.path.isfile(p):
            return p

    search_roots = []
    for root in [
        preferred_dir,
        DATA_ROOT,
        "/kaggle/data",
        "/kaggle/input",
        os.path.join(DATA_ROOT, "cassava-leaf-disease-classification"),
        os.path.join("/kaggle/data", "cassava-leaf-disease-classification"),
        os.path.join("/kaggle/input", "cassava-leaf-disease-classification"),
    ]:
        if root and os.path.isdir(root) and root not in search_roots:
            search_roots.append(root)

    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            if checkpoint_name in filenames:
                return os.path.join(dirpath, checkpoint_name)

    common_names = [
        "enb7best.pt",
        "enb7_best.pt",
        "enb7_best.pth",
        "best.pth",
        "best.pt",
        "model.pth",
        "model.pt",
        "checkpoint.pth",
        "checkpoint.pt",
        "enb7.pth",
        "enb7.pt",
        "efficientnet_b7.pth",
        "efficientnet-b7.pth",
        "fold0.pth",
        "fold0.pt",
        "weights.pth",
        "weights.pt",
    ]
    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for nm in common_names:
                if nm in filenames:
                    return os.path.join(dirpath, nm)

    all_ckpts = []
    for root in search_roots:
        all_ckpts.extend(_candidate_checkpoint_files(root))

    if len(all_ckpts) == 0:
        return None

    probe_model = _build_model_torchvision(use_imagenet_weights=False)

    def score_path(p: str) -> int:
        name = os.path.basename(p).lower()
        s = 0
        if "enb7" in name or "b7" in name or "efficientnet" in name:
            s += 5
        if "cassava" in p.lower():
            s += 2
        if "best" in name:
            s += 3
        if name.endswith(".pt") or name.endswith(".pth"):
            s += 1
        try:
            sz = os.path.getsize(p)
            if sz > 50_000_000:
                s += 2
            elif sz > 5_000_000:
                s += 1
        except OSError:
            pass
        return s

    all_ckpts_sorted = sorted(all_ckpts, key=lambda p: (score_path(p), p), reverse=True)

    best = None
    best_ratio = -1.0
    for chosen in all_ckpts_sorted[:120]:
        try:
            ckpt = torch.load(chosen, map_location="cpu")
            st = _extract_state_dict(ckpt)
            if st is None:
                continue
            st = _strip_known_prefixes(st)
            st = _remap_efficientnet_keys_to_torchvision(st)
            if not _looks_like_enb7_5class_state_dict(st):
                continue
            r = _match_ratio(probe_model, st)
            if r > best_ratio:
                best_ratio = r
                best = chosen
        except Exception:
            continue

    if best is not None:
        print(
            f"WARNING: Exact checkpoint name '{checkpoint_name}' not found. "
            f"Auto-selected best-matching compatible checkpoint candidate: {best} (match_ratio={best_ratio:.3f})"
        )
        return best

    chosen = all_ckpts_sorted[0]
    print(
        f"WARNING: No compatible checkpoint found during validation. "
        f"Falling back to top heuristic candidate (may fail to load): {chosen}"
    )
    return chosen


def load_model():
    ckpt_path = _find_checkpoint_path(
        config["checkpoint"], config.get("checkpoint_path")
    )

    if ckpt_path is None:
        print(
            f"WARNING: No checkpoint found (looked for '{config['checkpoint']}').\n"
            f"Proceeding with ImageNet-pretrained EfficientNet-B7 (architecture unchanged)."
        )
        model = _build_model_torchvision(use_imagenet_weights=True)
        model = model.to(device)
        model.eval()
        return model, None

    print("Found checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for k in ["epoch", "train_loss", "val_loss", "metrics", "lr"]:
            if k in checkpoint:
                print(f"{k}:", checkpoint[k])

    model = _build_model_torchvision(use_imagenet_weights=False)
    try:
        model = _load_state_dict_flex(model, checkpoint)
    except Exception as e:
        print(
            "WARNING: Failed to load weights from checkpoint; falling back to ImageNet weights.\n"
            f"Reason: {repr(e)}"
        )
        model = _build_model_torchvision(use_imagenet_weights=True)

    with torch.no_grad():
        w = model.classifier[1].weight.detach().float().cpu()
        print("Classifier weight norm (sanity):", float(w.norm().item()))

    model = model.to(device)
    model.eval()
    return model, ckpt_path




## === cell 7
def _make_albu_transform(name: str, params: dict):
    cls = getattr(A, name)

    p = dict(params) if params is not None else {}

    if name == "RandomResizedCrop":
        if "size" not in p and ("height" in p and "width" in p):
            h, w = p.pop("height"), p.pop("width")
            p["size"] = (h, w)

        try:
            return cls(**p)
        except Exception:
            p2 = dict(params)
            if "size" in p2 and ("height" not in p2 and "width" not in p2):
                h, w = p2["size"]
                p2.pop("size")
                p2["height"] = h
                p2["width"] = w
            return cls(**p2)

    return cls(**p)


def get_transforms():
    transforms = [
        _make_albu_transform(item["name"], item["params"])
        for item in config["inference_augmentations"]
    ]
    comp = A.Compose(transforms)
    return comp


def get_train_transforms():
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.8, 1.25),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.Normalize(mean=CASSAVA_MEAN, std=CASSAVA_STD, max_pixel_value=1.0, p=1.0),
        ]
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_LINEAR,
                p=1.0,
            ),
            A.Normalize(mean=CASSAVA_MEAN, std=CASSAVA_STD, max_pixel_value=1.0, p=1.0),
        ]
    )




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        img_path = os.path.join(config["test_location"], self.images[n])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        image = self.transforms(image=image)["image"]  # HWC, float32 after Normalize
        image = np.moveaxis(image, -1, 0)  # CHW
        image = torch.from_numpy(image).float()
        return image

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __getitem__(self, n):
        image_id = self.df.loc[n, "image_id"]
        label = int(self.df.loc[n, "label"])
        img_path = os.path.join(config["train_location"], image_id)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).float()
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"])
    data = CassavaDataset(test_data, transforms)

    workers = config["workers"]
    if device.type == "cpu":
        workers = 0

    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=(device.type == "cuda"),
        num_workers=workers,
    )
    return dataloader


def get_train_valid_loaders():
    df = pd.read_csv(config["train_csv"])
    df = df[["image_id", "label"]].copy()

    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=config["seed"])
    tr_idx, va_idx = next(sss.split(df["image_id"], df["label"]))
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    workers = config["workers"]
    if device.type == "cpu":
        workers = 0

    train_ds = CassavaTrainDataset(df_tr, get_train_transforms())
    valid_ds = CassavaTrainDataset(df_va, get_valid_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, valid_loader




## === cell 10
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            batch_hat = model(batch)
            predictions.append(batch_hat.detach().cpu())

    return torch.cat(predictions, dim=0)


def finetune_if_needed(model, ckpt_path):
    if ckpt_path is not None:
        print("Checkpoint loaded; skipping in-notebook fine-tuning.")
        return model

    print(
        "No compatible checkpoint found/loaded; starting fixed-epoch fine-tuning on train.csv to move score toward target."
    )
    train_loader, valid_loader = get_train_valid_loaders()

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(config["finetune_epochs"]):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for xb, yb in tqdm(
            train_loader, desc=f"finetune train ep {ep+1}/{config['finetune_epochs']}"
        ):
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            tr_loss += float(loss.detach().cpu().item()) * bs
            tr_n += bs

        model.eval()
        va_correct = 0
        va_total = 0
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(
                valid_loader,
                desc=f"finetune valid ep {ep+1}/{config['finetune_epochs']}",
            ):
                xb = xb.to(device, non_blocking=(device.type == "cuda"))
                yb = yb.to(device, non_blocking=(device.type == "cuda"))
                logits = model(xb)
                loss = criterion(logits, yb)
                preds = torch.argmax(logits, dim=1)
                va_correct += int((preds == yb).sum().item())
                va_total += int(yb.numel())
                va_loss += float(loss.detach().cpu().item()) * xb.size(0)

        print(
            f"Fine-tune ep {ep+1}: "
            f"train_loss={tr_loss/max(1,tr_n):.4f} | "
            f"valid_loss={va_loss/max(1,va_total):.4f} | "
            f"valid_acc={va_correct/max(1,va_total):.4f}"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    model.eval()
    torch.save(model.state_dict(), "finetuned_enb7_state_dict.pth")
    print("Saved fine-tuned weights to finetuned_enb7_state_dict.pth")
    return model




## === cell 11
if __name__ == "__main__":
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, ckpt_path = load_model()
    model = finetune_if_needed(model, ckpt_path)

    predictions = None
    print("Inferring experiment", config["experiment_name"])

    for epoch in range(config["epochs"]):
        print("Epoch (TTA pass):", epoch)
        start_time = time.time()

        dataloader = get_dataloader()

        if epoch == 0:
            predictions = infer(model, dataloader)
        else:
            predictions += infer(model, dataloader)

        print("Time:", time.time() - start_time)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.numpy()
    test["label"] = np.argmax(results, axis=-1).astype(int)

    submission = sample[["image_id"]].merge(
        test[["image_id", "label"]], on="image_id", how="left"
    )
    if submission["label"].isna().any():
        missing = submission[submission["label"].isna()]["image_id"].head().tolist()
        raise RuntimeError(f"Missing predictions for some test images, e.g.: {missing}")

    submission["label"] = submission["label"].astype(int)
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Wrote submission.csv with rows:", len(submission))
    print("Saved at:", os.path.abspath("submission.csv"))
