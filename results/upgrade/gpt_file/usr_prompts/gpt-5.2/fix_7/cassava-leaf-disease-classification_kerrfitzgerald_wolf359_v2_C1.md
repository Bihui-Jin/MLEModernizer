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

0.8152009670595346

# 6. Current score

0.11883

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14088) has done: 'I fix the missing weight-file crash by making pretrained weight loading conditional on the file actually existing, so the notebook runs in any Kaggle environment. I also fix the “no NVIDIA driver” runtime error by selecting `cuda` only if available and moving both model and tensors to the same device. Finally, I make the submission deterministic and valid by using `sample_submission.csv` order (instead of shuffled `os.listdir`) and by replacing the deprecated/slow `DataFrame.append` loop with list accumulation, then writing `submission.csv` with the required columns.'
- What this solution (achieved 0.14088) has done: 'I fix the crash caused by the missing fine-tuned weight file by making the code robust: it auto-discover the checkpoint in the input directory if present, and if not present it fall back to using the base ViT weights (so the notebook still runs end-to-end and produces a valid submission). I also fix the `load_state_dict` call (it incorrectly expects a `(missing, unexpected)` return tuple) and make checkpoint loading tolerant to common key-prefix patterns like `module.`. Finally, I ensure `cassava_gpu_model` is always defined so inference runs and `submission.csv` is written in the exact `sample_submission.csv` order and format.'
- What this solution (achieved 0.14088) has done: 'Your current score (0.14088) is far below the target (0.8152), and the most likely cause is that the fine-tuned Cassava checkpoint isn’t actually being loaded (or is being loaded incorrectly), so you’re effectively predicting with a near-random head. I make the checkpoint loading robust to common formats (e.g., keys nested under `state_dict`, `model`, `net`, EMA variants) and, crucially, allow a safe `strict=False` fallback when only the classifier head shape differs—this keeps the same ViT architecture and inference semantics but prevents silently failing to load useful backbone weights. I also infer `n_classes` from the checkpoint head when possible (still the same model, just correct class count), and ensure we never proceed with an uninitialized/random head if the checkpoint contains a trained head. These are minimal, directly score-relevant changes and should move accuracy sharply upward toward the target when the provided Cassava weights are present.'
- What this solution (achieved 0.14088) has done: 'Your low score strongly suggests the fine-tuned checkpoint still isn’t being applied to the actual model weights (most commonly because the checkpoint keys don’t match: e.g., missing/extra `model.` prefix), leaving you with a mostly random classifier head. I make a minimal, score-relevant change: automatically remap checkpoint keys between `head.*` ↔ `model.head.*` and `model.*` ↔ no-prefix forms before `load_state_dict`, and verify that the head weights truly loaded (otherwise we fail fast instead of silently submitting near-random predictions). I also remove the unused optimizer/criterion (no effect on predictions) and switch inference to `torch.inference_mode()` (same semantics, slightly safer/faster). These changes preserve the same ViT architecture and inference logic, but greatly increase the chance the trained head/backbone weights are actually used, moving accuracy toward your target.'
- What this solution (achieved 0.11883) has done: 'Your current accuracy (0.14088) is far below the target (0.8152), so we need a small change that legitimately improves predictions without changing the core model/training logic. The most likely issue is a preprocessing mismatch: ViT models in `timm` typically expect images normalized to `[-1, 1]` (mean=0.5/std=0.5), while your code uses ImageNet normalization; this can crush performance even with correct weights. I keep the same ViT model and inference loop, but switch the test transform normalization to the ViT-default and add a tiny sanity-check print of the model’s `default_cfg` to confirm. This is minimal, metric-aligned, and should move the score substantially upward toward the target if the fine-tuned checkpoint is being loaded.'

# 9. Code solution

## === cell 0
import os
import time
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.utils import make_grid



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 2
wheel_path = "../input/timm034/timm-0.3.4-py3-none-any.whl"
if os.path.exists(wheel_path):
    os.system(f"pip -q install '{wheel_path}'")
else:
    print(
        "timm wheel not found at",
        wheel_path,
        "- relying on existing timm installation (if available).",
    )



## === cell 3
try:
    import timm
except Exception as e:
    raise RuntimeError(
        "timm is required but could not be imported. Ensure timm is available in the environment."
    ) from e



## === cell 4
print("Available ViT Models (subset):")
print(timm.list_models("vit*")[:10], "... total:", len(timm.list_models("vit*")))



## === cell 5
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"


def _find_first_file(root_dir, exts):
    if (root_dir is None) or (not os.path.exists(root_dir)):
        return None
    for r, _, files in os.walk(root_dir):
        for fn in files:
            low = fn.lower()
            if any(low.endswith(ext) for ext in exts):
                return os.path.join(r, fn)
    return None


model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = "../input/cassavapretrained4epoch/CassavaViT4epoch.pt"

if not os.path.exists(model_path):
    candidate = _find_first_file("../input/vitbase16224", exts=[".pth", ".pt", ".bin"])
    if candidate is not None:
        print("Base ViT weight file not found at fixed path. Auto-found:", candidate)
        model_path = candidate

if not os.path.exists(Cassava_model):
    candidate = _find_first_file(
        "../input/cassavapretrained4epoch", exts=[".pt", ".pth", ".bin"]
    )
    if candidate is not None:
        print(
            "Cassava fine-tuned weight file not found at fixed path. Auto-found:",
            candidate,
        )
        Cassava_model = candidate

sample_sub_path = os.path.join(data_path, "sample_submission.csv")
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.isdir(test_path), f"Missing test_images directory at {test_path}"

print("Resolved base weight path:", model_path, "| exists:", os.path.exists(model_path))
print(
    "Resolved cassava ft path:",
    Cassava_model,
    "| exists:",
    os.path.exists(Cassava_model),
)




## === cell 6
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        if pretrained:
            if os.path.exists(model_path):
                state = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state, strict=True)
                print("Loaded base ViT weights from:", model_path)
            else:
                print("WARNING: base ViT weight file not found, skipping:", model_path)

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 7
def _unwrap_checkpoint(state):
    """
    Robustly unwrap common checkpoint containers so we actually load the fine-tuned weights.
    """
    if not isinstance(state, dict):
        return state

    for k in ["state_dict", "model", "net", "model_state_dict", "params", "weights"]:
        if k in state and isinstance(state[k], dict):
            print(
                f"Loaded checkpoint dict with key '{k}'. Using that for load_state_dict."
            )
            return state[k]

    for k in ["state_dict_ema", "model_ema", "ema", "ema_state_dict"]:
        if k in state and isinstance(state[k], dict):
            print(
                f"Loaded checkpoint dict with key '{k}'. Using that for load_state_dict."
            )
            return state[k]

    return state


def _strip_prefix_from_state_dict(sd, prefix="module."):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith(prefix) for k in sd.keys()):
        return {
            k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()
        }
    return sd


def _maybe_infer_num_classes_from_state_dict(sd, default=5):
    """
    Infer class count from the checkpoint head weight when available.
    """
    if not isinstance(sd, dict):
        return default
    for head_key in ["model.head.weight", "head.weight"]:
        if (
            head_key in sd
            and hasattr(sd[head_key], "shape")
            and len(sd[head_key].shape) == 2
        ):
            return int(sd[head_key].shape[0])
    return default


def _remap_state_dict_keys_for_vit(sd, model):
    """
    Change (score-critical, minimal): remap common key naming mismatches so the fine-tuned weights
    (especially the classifier head) actually load instead of being skipped.
    Handles:
      - 'head.*' <-> 'model.head.*'
      - 'model.*' prefix present/absent
    """
    if not isinstance(sd, dict):
        return sd

    model_keys = set(model.state_dict().keys())

    if len(model_keys.intersection(sd.keys())) > 0:
        return sd

    candidates = []

    sd1 = {
        ("model." + k) if not k.startswith("model.") else k: v for k, v in sd.items()
    }
    candidates.append(sd1)

    sd2 = {
        k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()
    }
    candidates.append(sd2)

    sd3 = {("model." + k) if k.startswith("head.") else k: v for k, v in sd.items()}
    candidates.append(sd3)

    sd4 = {
        k[len("model.") :] if k.startswith("model.head.") else k: v
        for k, v in sd.items()
    }
    candidates.append(sd4)

    sd5_tmp = {
        k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()
    }
    sd5 = {
        ("model." + k) if k.startswith("head.") else k: v for k, v in sd5_tmp.items()
    }
    candidates.append(sd5)

    best_sd = sd
    best_inter = len(model_keys.intersection(sd.keys()))
    for i, cand in enumerate(candidates, 1):
        inter = len(model_keys.intersection(cand.keys()))
        if inter > best_inter:
            best_inter = inter
            best_sd = cand

    if best_sd is not sd:
        print(
            f"Remapped checkpoint keys to improve match: matched {best_inter} tensors."
        )
    else:
        print(
            "No key remap improved matching; proceeding with original checkpoint keys."
        )
    return best_sd


def _head_loaded_ok(model, sd_loaded):
    """
    Change (safety): ensure we didn't silently miss the classifier head. If the checkpoint contains
    a head weight, verify the model's head.weight equals it after loading.
    """
    if not isinstance(sd_loaded, dict):
        return True

    possible_keys = ["model.head.weight", "head.weight"]
    for k in possible_keys:
        if k in sd_loaded:
            with torch.no_grad():
                target = sd_loaded[k].detach().cpu()
                model_w = model.state_dict().get(k, None)
                if model_w is None:
                    alt = (
                        "head.weight"
                        if k == "model.head.weight"
                        else "model.head.weight"
                    )
                    model_w = model.state_dict().get(alt, None)
                if model_w is None:
                    return False
                model_w = model_w.detach().cpu()
                if model_w.shape != target.shape:
                    return False
                return bool(torch.allclose(model_w, target, atol=0, rtol=0))
    return True


ckpt_state_raw = None
ckpt_sd = None
if os.path.exists(Cassava_model):
    ckpt_state_raw = torch.load(Cassava_model, map_location="cpu")
    ckpt_sd = _unwrap_checkpoint(ckpt_state_raw)
    ckpt_sd = _strip_prefix_from_state_dict(ckpt_sd, prefix="module.")
    inferred_classes = _maybe_infer_num_classes_from_state_dict(ckpt_sd, default=5)
else:
    inferred_classes = 5

cassava_model = ViTBase16(n_classes=inferred_classes, pretrained=True)

if ckpt_sd is not None:
    ckpt_sd = _remap_state_dict_keys_for_vit(ckpt_sd, cassava_model)

    try:
        incompatible = cassava_model.load_state_dict(ckpt_sd, strict=True)
        print("Loaded Cassava fine-tuned weights from:", Cassava_model, "(strict=True)")
    except RuntimeError as e:
        print("Strict load failed:", str(e).split("\n")[0])
        incompatible = cassava_model.load_state_dict(ckpt_sd, strict=False)
        print(
            "Loaded Cassava fine-tuned weights from:",
            Cassava_model,
            "(strict=False fallback)",
        )

    if hasattr(incompatible, "missing_keys") and hasattr(
        incompatible, "unexpected_keys"
    ):
        if len(incompatible.missing_keys) or len(incompatible.unexpected_keys):
            print(
                "load_state_dict incompatibilities:",
                "missing:",
                len(incompatible.missing_keys),
                "unexpected:",
                len(incompatible.unexpected_keys),
            )

    assert _head_loaded_ok(cassava_model, ckpt_sd), (
        "Checkpoint appears to include classifier head weights, but they were not loaded into the model. "
        "This typically means key mismatch (e.g., different prefixes)."
    )
else:
    print(
        "WARNING: Cassava fine-tuned model weights not found; falling back to base ViT head weights.\n"
        f"Tried path: {Cassava_model}\n"
        "Attach the dataset ../input/cassavapretrained4epoch (or correct path) for competitive score."
    )

cassava_gpu_model = cassava_model.to(device)
cassava_gpu_model.eval()

try:
    print(
        "ViT default_cfg mean/std:",
        cassava_gpu_model.model.default_cfg.get("mean"),
        cassava_gpu_model.model.default_cfg.get("std"),
    )
except Exception as _:
    pass

cassava_gpu_model



## === cell 8
sample_sub = pd.read_csv(sample_sub_path)
test_img_names = sample_sub["image_id"].tolist()
print("Testing Images (from sample_submission):", len(test_img_names))
print("First 5:", test_img_names[:5])




## === cell 9
class TestSet2(Dataset):
    """Cassava Disease Test Dataset (ordered by provided image_id list)."""

    def __init__(self, test_dir, image_names, transform=None):
        super().__init__()
        self.test_dir = test_dir
        self.image_names = list(image_names)
        self.transform = transform
        print("Cassava Disease Test Dataset Length = ", len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_path = os.path.join(self.test_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(img)
        else:
            image = transforms.ToTensor()(img)
        return image, image_name




## === cell 10
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 11
testset = TestSet2(
    test_dir=test_path, image_names=test_img_names, transform=test_transform
)
print(testset)



## === cell 12
test_batch_size = 32
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 13
images, names = next(iter(test_loader))
im = make_grid(images[:8], nrow=4)
print("Batch example names:", list(names[:8]))
plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()



## === cell 14
pred_rows = []
tic = time.time()

with torch.inference_mode():
    for X_test, names in test_loader:
        X_test = X_test.to(device, non_blocking=True)
        logits = cassava_gpu_model(X_test)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)

        for img_name, label in zip(names, preds):
            pred_rows.append((img_name, int(label)))

toc = time.time() - tic
print("Time for test inference:", toc, "seconds")

submission_df = pd.DataFrame(pred_rows, columns=["image_id", "label"])

submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
assert submission_df["label"].isna().sum() == 0, "Some test images were not predicted."

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())



## === cell 15
print("submission.csv rows/cols:", submission_df.shape)
print("Columns:", submission_df.columns.tolist())
print("Unique labels:", sorted(submission_df["label"].unique().tolist()))
print("\nDirectory listing:")
os.system("ls -la")
