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

geopandas==0.14.4
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

0.8773043215472952

# 6. Current score

0.75374

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'The crash comes from moving tensors to CUDA inside `Dataset.__getitem__` while using `num_workers>0`, which forks subprocesses and triggers “Cannot re-initialize CUDA”. I keep the model and transforms intact, but change the dataset to return CPU tensors only, move the batch to `device` in the inference loop, and set `num_workers=0` for maximum compatibility in Kaggle notebooks. I also fix the missing weights path by searching common Kaggle input locations for a `.pth` file named like your checkpoint, so you get a non-random model when available (score improvement vs. random). Finally, the script always write `submission.csv` and validate its format/length.'
- What this solution (achieved 0.10762) has done: 'Your current score is extremely low because the script often runs with randomly initialized weights when it can’t find `weight_epoch_14.pth`; the smallest safe way to move toward the target is to reliably load a good pretrained cassava checkpoint if it exists anywhere under the provided `/kaggle/input` tree. I keep the exact model (ResNet50 + linear(5)) and the exact inference logic, but I improve the weight-file discovery to also match common cassava checkpoint names (not just `epoch_14`) and to correctly handle typical checkpoint dict formats (`state_dict`, `model_state_dict`, `net`, and `module.` prefixes). This should substantially increase accuracy versus random init without changing architecture or training semantics, and still always writes a valid `submission.csv`. I also ensure the loaded tensors are moved to the right device safely and keep `num_workers=0` for Kaggle CUDA stability.'
- What this solution (achieved 0.10762) has done: 'Your score is still near random, which strongly suggests the model is not actually loading compatible trained weights (or is loading an incompatible checkpoint silently due to `strict=False`). To move the score toward the target with minimal changes and identical model/inference logic, I (1) make weight discovery more robust but more conservative (prefer resnet50-sized checkpoints and cassava-specific names), (2) harden checkpoint parsing to correctly extract the tensor state dict (including nested keys and common Lightning formats) and filter to only keys that match your model’s shapes so `fc` weights load when possible, and (3) print a clear load report so you can verify that most layers (and `fc`) were loaded rather than running effectively random. This keeps architecture, transforms, and prediction logic unchanged; it just increases the chance that your intended trained weights are correctly found and loaded, which should substantially improve accuracy toward your target.'
- What this solution (achieved 0.10389) has done: 'Your current score is near-random, so the smallest change that materially move you toward the target is to stop accidentally running with random weights by ensuring we can load standard ImageNet ResNet50 weights (the same architecture) when your cassava checkpoint isn’t found. This keeps your model (ResNet50 + `Linear(…,5)`) and inference logic identical, but provides strong generic features so accuracy rises substantially versus random. I also make checkpoint loading stricter/safer: try to load the full state dict first (so you don’t silently drop most layers), then fall back to shape-filtered partial loading only if needed, and print a clear load summary. Submission writing and ordering stay unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.77578) has done: 'Your score is near-random because even with ImageNet ResNet50 weights, the 5-class `fc` head is still random when no cassava-trained checkpoint is found/loaded, so predictions collapse. To move toward the target with minimal changes and identical inference semantics, I keep your ResNet50+Linear(5) model and single-pass test inference, but add a lightweight train-on-`train.csv` step to fit only the final `fc` layer using the provided `train_images` (no architecture/loss changes; just standard cross-entropy). I also ensure the dataset uses the correct train image directory and keep your submission ordering/merge logic unchanged. This should materially increase accuracy from ~0.10 toward the target without changing the core model or introducing approximations.'
- What this solution (achieved 0.75374) has done: 'The timeout is most likely dominated by slow image decoding/transforms (PIL conversion + single-worker DataLoaders) and by the expensive weight-file fallback search that recursively scans all of `/kaggle/input`. I keep the exact same model, loss, and train/infer loops, but speed up data input by removing PIL from the transform pipeline (using torchvision’s tensor-native ops) and enabling multi-worker DataLoaders with persistent workers and prefetching (deterministic). I also make the checkpoint fallback search non-recursive and limited to likely locations to avoid minutes of filesystem traversal while preserving the same “load if available” behavior. These changes reduce overhead without changing evaluation semantics (only negligible floating-point differences possible).'

# 9. Code solution

## === cell 0
import os
import sys
import copy
import datetime
import random
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

import cv2


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

DATA_ROOT = "../input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.exists(DATA_ROOT):
    alt_root = "/kaggle/input/cassava-leaf-disease-classification"
    if os.path.exists(alt_root):
        DATA_ROOT = alt_root
        TEST_DIR = os.path.join(DATA_ROOT, "test_images")
        TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
        TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
        SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing test directory: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing train directory: {TRAIN_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
)
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model = model.to(device)

weights_path = "../input/wwwwww/weight_epoch_14.pth"


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(ckpt):
    """
    Why: Many Kaggle checkpoints are wrapped (Lightning, custom dicts).
    Extract the actual parameter dict without changing model/inference semantics.
    """
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        tensor_like = 0
        total = 0
        for k, v in ckpt.items():
            total += 1
            if isinstance(k, str) and torch.is_tensor(v):
                tensor_like += 1
        if total > 0 and tensor_like / total > 0.8:
            return ckpt
    return ckpt


def _filter_state_by_shape(state, model_state):
    """
    Why: If the found checkpoint is close but has extra keys or mismatched heads,
    we still want to load all matching layers (including fc when shapes match).
    This is safer than loading random weights and keeps architecture unchanged.
    """
    if not isinstance(state, dict):
        return state, [], list(model_state.keys())

    kept = {}
    dropped = []
    for k, v in state.items():
        if (
            k in model_state
            and torch.is_tensor(v)
            and torch.is_tensor(model_state[k])
            and v.shape == model_state[k].shape
        ):
            kept[k] = v
        else:
            dropped.append(k)
    missing = [k for k in model_state.keys() if k not in kept]
    return kept, dropped, missing


def find_weight_fallback():
    """
    Why: The previous recursive scan over /kaggle/input/**/*.pth can be extremely slow.
    This bounded search is functionally equivalent for Kaggle: it checks common dataset mounts
    and only inspects a small number of directories/files.
    """
    candidates = []
    for base in ("/kaggle/input", "../input"):
        if os.path.isdir(base):
            try:
                for d in os.listdir(base):
                    p = os.path.join(base, d, "weight_epoch_14.pth")
                    if os.path.exists(p):
                        return p
                    candidates.append(os.path.join(base, d))
            except Exception:
                pass

    exts = (".pth", ".pt", ".bin")
    picked = None
    picked_score = -(10**9)

    def score_path(p):
        b = os.path.basename(p).lower()
        s = 0
        if "cassava" in b:
            s += 8
        if "resnet50" in b or ("resnet" in b and "50" in b) or "r50" in b:
            s += 6
        if "epoch_14" in b or "epoch14" in b:
            s += 4
        if "best" in b:
            s += 4
        if "checkpoint" in b or "ckpt" in b:
            s += 1
        if "effnet" in b or "efficientnet" in b:
            s -= 6
        if "tf" in b or "tflite" in b or "keras" in b:
            s -= 8
        try:
            sz = os.path.getsize(p)
            if sz < 200_000:
                s -= 10
            elif 60_000_000 <= sz <= 140_000_000:
                s += 8
            elif 20_000_000 <= sz <= 400_000_000:
                s += 2
            else:
                s -= 1
        except Exception:
            pass
        return s

    for ds_dir in candidates:
        if not os.path.isdir(ds_dir):
            continue
        try:
            for fn in os.listdir(ds_dir):
                p = os.path.join(ds_dir, fn)
                if os.path.isfile(p) and p.lower().endswith(exts):
                    sc = score_path(p)
                    if sc > picked_score:
                        picked_score, picked = sc, p
            for sub in (
                "models",
                "model",
                "weights",
                "checkpoints",
                "checkpoint",
                "ckpt",
            ):
                subdir = os.path.join(ds_dir, sub)
                if os.path.isdir(subdir):
                    for fn in os.listdir(subdir):
                        p = os.path.join(subdir, fn)
                        if os.path.isfile(p) and p.lower().endswith(exts):
                            sc = score_path(p)
                            if sc > picked_score:
                                picked_score, picked = sc, p
        except Exception:
            continue

    if picked is not None and picked_score >= 10:
        return picked
    return None


loaded = False
loaded_full = False
loaded_filtered = False

if not os.path.exists(weights_path):
    fb = find_weight_fallback()
    if fb is not None:
        weights_path = fb

if os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix(state)

    model_state = model.state_dict()

    try:
        model.load_state_dict(state, strict=True)
        loaded = True
        loaded_full = True
    except Exception:
        filtered_state, dropped_keys, missing_keys = _filter_state_by_shape(
            state, model_state
        )
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        loaded = True
        loaded_filtered = True

    fc_w = model_state["fc.weight"].shape
    fc_b = model_state["fc.bias"].shape
    print(f"Loaded checkpoint: {weights_path}")
    print(f"Load mode: {'STRICT_FULL' if loaded_full else 'FILTERED_PARTIAL'}")
    if loaded_filtered and isinstance(state, dict):
        print(f"Filtered state_dict keys kept: {len(filtered_state)} / {len(state)}")
        print(
            f"load_state_dict reports Missing={len(missing)} Unexpected={len(unexpected)}"
        )
        fc_w_loaded = ("fc.weight" in filtered_state) and (
            filtered_state["fc.weight"].shape == fc_w
        )
        fc_b_loaded = ("fc.bias" in filtered_state) and (
            filtered_state["fc.bias"].shape == fc_b
        )
        print(f"fc loaded? weight={fc_w_loaded} bias={fc_b_loaded}")
        if len(filtered_state) < 50:
            print(
                "WARNING: Very few keys matched your model; checkpoint likely different architecture."
            )
else:
    print(
        f"Cassava weights not found at {weights_path}. Using ImageNet-pretrained ResNet50 backbone + random 5-class head."
    )

model.eval()



## === cell 2
img_size = 512

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FastTensorTransform:
    def __init__(self, size, mean, std):
        self.size = (size, size)
        self.mean = mean
        self.std = std

    def __call__(self, img_rgb_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).div_(255.0)
        x = F.resize(
            x, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True
        )
        x = F.normalize(x, mean=self.mean, std=self.std)
        return x


transforms = FastTensorTransform(img_size, IMAGENET_MEAN, IMAGENET_STD)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].astype(str).tolist()
len(test_image_ids), test_image_ids[:3]



## === cell 3
train_df = pd.read_csv(TRAIN_CSV_PATH)
assert {"image_id", "label"}.issubset(train_df.columns)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)


class TrainSet(Dataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, transforms):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        path = os.path.join(self.img_dir, fn)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = (
            self.transforms(img)
            if self.transforms is not None
            else torch.from_numpy(img)
        )
        return x, y


do_fc_train = not loaded_full

if do_fc_train:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
    model.train()

    rng = np.random.RandomState(42)
    idxs = np.arange(len(train_df))
    rng.shuffle(idxs)
    val_frac = 0.1
    val_n = int(round(len(idxs) * val_frac))
    val_idxs = idxs[:val_n]
    tr_idxs = idxs[val_n:]

    tr_df = train_df.iloc[tr_idxs].reset_index(drop=True)
    va_df = train_df.iloc[val_idxs].reset_index(drop=True)

    train_set = TrainSet(TRAIN_DIR, tr_df, transforms)
    val_set = TrainSet(TRAIN_DIR, va_df, transforms)

    def _seed_worker(worker_id):
        base_seed = 42
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    g = torch.Generator()
    g.manual_seed(42)

    num_workers = min(4, (os.cpu_count() or 2))
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )

    class_counts = (
        train_df["label"].value_counts().reindex(range(5), fill_value=1).values
    )
    class_weights = (class_counts.sum() / class_counts).astype(np.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, device=device, dtype=torch.float32)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    epochs = 3
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            epoch_loss += float(loss.detach().cpu().item()) * bs
            n += bs

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = torch.as_tensor(yb, device=device, dtype=torch.long)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        val_acc = correct / max(1, total)
        avg_loss = epoch_loss / max(1, n)
        print(
            f"fc train ep={ep}/{epochs}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(
        f"Selected best fc by val_acc={best_val_acc:.4f} (split {len(tr_df)}/{len(va_df)})"
    )
else:
    print(
        "Skipping fc training because a full checkpoint was loaded (assumed cassava-trained)."
    )

model.eval()




## === cell 4
class InferSet(Dataset):
    def __init__(self, img_dir: str, image_ids, transforms):
        self.img_dir = img_dir
        self.image_ids = list(image_ids)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        path = os.path.join(self.img_dir, fn)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = (
            self.transforms(img)
            if self.transforms is not None
            else torch.from_numpy(img)
        )
        return x, fn


infer_set = InferSet(TEST_DIR, test_image_ids, transforms)

num_workers = min(4, (os.cpu_count() or 2))
infer_loader = DataLoader(
    infer_set,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

xb, fnb = next(iter(infer_loader))
xb.shape, fnb[:3]



## === cell 5
preds_all = []
fns_all = []

with torch.no_grad():
    model.eval()
    for xb, fns in infer_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
        preds_all.append(preds)
        fns_all.extend(list(fns))

preds_all = np.concatenate(preds_all, axis=0)
assert len(preds_all) == len(test_image_ids) == len(fns_all)

sub = pd.DataFrame({"image_id": fns_all, "label": preds_all})
sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
assert (
    sub["label"].notna().all()
), "Some predictions are missing after merge; check filenames."

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
out_path, sub.shape, sub.head()



## === cell 6
check = pd.read_csv("./submission.csv")
print(check.columns.tolist(), len(check))
print(check.head())
assert check.columns.tolist() == ["image_id", "label"]
assert len(check) == len(
    sample_sub
), "Submission must match sample_submission row count."
print("submission.csv is ready.")
