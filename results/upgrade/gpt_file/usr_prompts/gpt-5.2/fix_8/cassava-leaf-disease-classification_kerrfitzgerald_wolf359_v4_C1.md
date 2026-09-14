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

0.8354487760652766

# 6. Current score

0.10874

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10874) has done: 'I fix the runtime failure by removing the hard dependency on an external checkpoint that isn’t present in this environment and instead fall back to a timm ImageNet-pretrained ViT (same architecture) when the custom `.pt` file can’t be found. This keeps the core model and inference logic intact while ensuring the notebook runs end-to-end and always defines `cassava_gpu_model`. I also make the weight-loading more robust (handle common checkpoint key formats) and ensure the submission is written as `submission.csv` with the required columns and row count aligned to the sample submission ordering.'
- What this solution (achieved 0.10949) has done: 'Your current score (0.10874) is far below the target (0.83545), and the biggest likely cause is that you are not actually loading the trained Cassava fine-tuned checkpoint (so predictions are essentially from a generic ImageNet ViT head). I make a minimal, robust checkpoint-loading fix that correctly maps common key patterns (including `state_dict` with `model.` prefixes and mismatched `head.*` keys) so the 5-class head weights load when available, without changing the model architecture or inference logic. I also switch inference to batched prediction (larger `batch_size`) and enable AMP autocast to stay well under the time limit; this should not change evaluation semantics beyond negligible floating-point differences. Finally, I keep the submission alignment to `sample_submission.csv` exactly as you already do.'
- What this solution (achieved 0.10949) has done: 'Your score is far below the target, and the most likely cause is that the Cassava fine-tuned checkpoint is not actually being applied to the model you run inference with (the current key-remapping logic is inconsistent with how the checkpoint keys are typically structured for this class). I make a minimal fix to checkpoint loading by cleanly handling both cases: checkpoints saved from the raw `ViTBase16` module (keys like `model.*`) and checkpoints saved from the inner timm model (keys like `head.*`, `blocks.*`, etc.), remapping them to the exact keys expected by `cassava_model.state_dict()`. This preserves your architecture and inference flow, but should move the accuracy sharply upward toward the target if the checkpoint exists and is compatible. I also keep your submission alignment to `sample_submission.csv` unchanged to avoid any ordering mistakes that can crater accuracy.'
- What this solution (achieved 0.10874) has done: 'Your score is far below the target, and the most likely reason is that the Cassava fine-tuned checkpoint still isn’t being loaded into the exact model you run inference with (especially the classifier `head.*` weights), leaving you effectively predicting with an untrained 5-class head. I make a minimal checkpoint-loading fix that (1) detects whether the checkpoint keys correspond to the wrapper (`model.*`) or the inner timm model, and (2) forcibly maps the classifier weights into `model.head.*` when they exist under common alternative names. I also disable AMP during inference to avoid any potential numerical quirks (this should be negligible for accuracy, but it removes a variable while we debug the low score). No architecture, transforms, or inference semantics change beyond the checkpoint key mapping.'
- What this solution (achieved 0.10874) has done: 'Your current score is far below the target, so the smallest likely way to move accuracy upward is to ensure we’re actually loading a compatible fine-tuned Cassava checkpoint (and not silently running with a random 5-class head). I keep your ViT architecture and inference pipeline intact, but (1) fix the `timm` dependency so it reliably uses a built-in installed version if the wheel isn’t present, and (2) make checkpoint loading robust to common Kaggle `.pt` formats by detecting when keys belong to the wrapper vs the inner timm model and by mapping head keys more comprehensively (including `model.model.*` / `backbone.*` patterns). Finally, I add a small sanity print of head weight stats after loading (no behavior change) to confirm the head is non-random, and keep submission alignment to `sample_submission.csv` exactly as you already do.'
- What this solution (achieved 0.10874) has done: 'Your low score indicates the fine-tuned Cassava checkpoint is still not being loaded into the exact parameters used at inference (especially the classifier head), so the model behaves close to random. I keep your model/inference pipeline intact, but make checkpoint loading more robust by (1) handling checkpoints that store a full model object, (2) removing common extra prefixes like `model.`/`net.`/`backbone.` in addition to `module.`, and (3) explicitly trying to load into the inner `timm` model (`cassava_model.model`) when the checkpoint keys match that namespace better. This is the smallest change that can plausibly move accuracy sharply upward toward the target without changing architecture, transforms, or inference semantics. The submission writing and ordering logic stays the same.'

# 9. Code solution

## === cell 0
import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)



## === cell 1
import sys
import subprocess

wheel_path = "../input/timm034/timm-0.3.4-py3-none-any.whl"
try:
    if os.path.exists(wheel_path):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", wheel_path]
        )
        print("Installed timm from wheel:", wheel_path)
    else:
        print(f"Wheel not found at {wheel_path}; will use existing installed timm.")
except Exception as e:
    print(
        "pip install attempt failed; will use existing installed timm. Error:", repr(e)
    )



## === cell 2
import timm

print("timm version:", getattr(timm, "__version__", "unknown"))



## === cell 3
print("Available ViT Models (first 20):")
print(timm.list_models("vit*")[:20])



## === cell 4
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"
model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavaaugmtp98epochs1lr175/CassavaViT_Augm_TP98_Epochs1_LR1-75e05.pt"
)

print("Paths:")
print(" data_path :", data_path)
print(" train_path:", train_path)
print(" test_path :", test_path)
print(" model_path exists?  ", os.path.exists(model_path), model_path)
print(" Cassava_model exists?", os.path.exists(Cassava_model), Cassava_model)




## === cell 5
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()

        timm_pretrained = False
        if pretrained and (not os.path.exists(model_path)):
            timm_pretrained = True
            print(
                f"WARNING: base model weights not found at {model_path}. "
                f"Falling back to timm pretrained weights."
            )

        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=timm_pretrained
        )

        if pretrained and os.path.exists(model_path) and (not timm_pretrained):
            state = torch.load(model_path, map_location="cpu")
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
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"Loaded base weights with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                )

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 6
def _unwrap_checkpoint(ckpt_obj):
    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict()
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj

    for key in ["state_dict", "model_state_dict", "model", "net", "checkpoint"]:
        if key in ckpt_obj:
            inner = ckpt_obj[key]
            if isinstance(inner, nn.Module):
                return inner.state_dict()
            if isinstance(inner, dict):
                return inner
    return ckpt_obj


def _clean_state_dict_keys(state):
    cleaned = {}
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in ("module.", "model.", "net.", "backbone."):
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        cleaned[nk] = v
    return cleaned


def _best_prefix_variant(sd, target_keys):
    """
    why: checkpoints might be saved from different module nesting.
    Try a few common prefixes and pick the variant with max overlap to our target keys.
    """
    variants = []
    variants.append(sd)
    if not any(k.startswith("model.") for k in sd.keys()):
        variants.append({f"model.{k}": v for k, v in sd.items()})
    if not any(k.startswith("model.model.") for k in sd.keys()):
        variants.append({f"model.model.{k}": v for k, v in sd.items()})
    if any(k.startswith("model.") for k in sd.keys()):
        variants.append(
            {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
        )
    if any(k.startswith("model.model.") for k in sd.keys()):
        variants.append(
            {
                k[len("model.model.") :]: v
                for k, v in sd.items()
                if k.startswith("model.model.")
            }
        )

    best = sd
    best_overlap = -1
    for cand in variants:
        overlap = len(set(cand.keys()) & target_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best = cand
    return best, best_overlap


def _force_map_head(sd, target_sd):
    """
    why: low score commonly happens if classifier head weights aren't loaded.
    Map common naming variants into 'model.head.(weight|bias)' when shapes match.
    """
    sd = dict(sd)  # copy

    target_w = target_sd.get("model.head.weight", None)
    target_b = target_sd.get("model.head.bias", None)
    if target_w is None or target_b is None:
        return sd

    cand_w_keys = [
        "model.head.weight",
        "head.weight",
        "classifier.weight",
        "fc.weight",
        "model.classifier.weight",
        "model.fc.weight",
        "model.model.head.weight",
        "backbone.head.weight",
        "backbone.model.head.weight",
    ]
    cand_b_keys = [
        "model.head.bias",
        "head.bias",
        "classifier.bias",
        "fc.bias",
        "model.classifier.bias",
        "model.fc.bias",
        "model.model.head.bias",
        "backbone.head.bias",
        "backbone.model.head.bias",
    ]

    found_w_key = next((k for k in cand_w_keys if k in sd), None)
    found_b_key = next((k for k in cand_b_keys if k in sd), None)

    if found_w_key is not None and tuple(sd[found_w_key].shape) == tuple(
        target_w.shape
    ):
        sd["model.head.weight"] = sd[found_w_key]
    if found_b_key is not None and tuple(sd[found_b_key].shape) == tuple(
        target_b.shape
    ):
        sd["model.head.bias"] = sd[found_b_key]

    return sd


cassava_model = ViTBase16(n_classes=5, pretrained=True)

if os.path.exists(Cassava_model):
    ckpt = torch.load(Cassava_model, map_location="cpu")
    ckpt = _unwrap_checkpoint(ckpt)

    if isinstance(ckpt, dict):
        ckpt = _clean_state_dict_keys(ckpt)

        target_sd_outer = cassava_model.state_dict()
        target_keys_outer = set(target_sd_outer.keys())

        target_sd_inner = cassava_model.model.state_dict()
        target_keys_inner = set(target_sd_inner.keys())

        ckpt_outer, overlap_outer = _best_prefix_variant(ckpt, target_keys_outer)
        ckpt_inner, overlap_inner = _best_prefix_variant(ckpt, target_keys_inner)

        if overlap_inner > overlap_outer:
            missing, unexpected = cassava_model.model.load_state_dict(
                ckpt_inner, strict=False
            )
            print(
                f"Loaded trained Cassava checkpoint INTO inner timm model. Key-overlap={overlap_inner}. "
                f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )
            head_loaded = ("head.weight" not in missing) and (
                "head.bias" not in missing
            )
            print("Inner head weights loaded?", head_loaded)
        else:
            ckpt_outer = _force_map_head(ckpt_outer, target_sd_outer)
            missing, unexpected = cassava_model.load_state_dict(
                ckpt_outer, strict=False
            )
            print(
                f"Loaded trained Cassava checkpoint INTO wrapper model. Key-overlap={overlap_outer}. "
                f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )
            head_loaded = ("model.head.weight" not in missing) and (
                "model.head.bias" not in missing
            )
            print("Wrapper head weights loaded?", head_loaded)

        with torch.no_grad():
            w = cassava_model.model.state_dict()["head.weight"].float()
            b = cassava_model.model.state_dict()["head.bias"].float()
            print("Head weight mean/std:", float(w.mean()), float(w.std()))
            print("Head bias mean/std  :", float(b.mean()), float(b.std()))

        if not head_loaded:
            print(
                "WARNING: Classifier head did not load; score may remain low. "
                "Checkpoint may be incompatible or for different class count."
            )
else:
    print(
        f"WARNING: Trained Cassava checkpoint not found: {Cassava_model}\n"
        "Proceeding with ViT backbone initialization (timm pretrained if available)."
    )

cassava_gpu_model = cassava_model.to(DEVICE).eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_gpu_model.parameters(), lr=1.5e-05)

print(cassava_gpu_model.__class__.__name__, "ready on", DEVICE)



## === cell 7
test_img_names = []
for folder, subfolders, filenames in os.walk(test_path):
    for img in filenames:
        if img.lower().endswith(".jpg"):
            test_img_names.append(img)

print("Testing Images:", len(test_img_names))
print("First 5 test images:", sorted(test_img_names)[:5])




## === cell 8
class TestSet2(Dataset):
    """Cassava Disease Test Dataset"""

    def __init__(self, root_dir, test_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.transform = transform

        self.files = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")]
        )

        print(root_dir)
        print(test_dir)
        print("Cassava Disease Test Dataset Length = ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.test_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(img)
        else:
            image = transforms.ToTensor()(img)

        return (image, fname)




## === cell 9
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 10
testset = TestSet2(root_dir="", test_dir=test_path, transform=test_transform)
print(testset)



## === cell 11
test_batch_size = 32 if torch.cuda.is_available() else 8
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=2,
)



## === cell 12
print(test_loader)

for images, names in test_loader:
    break

im = make_grid(images[: min(len(images), 16)], nrow=4)
print("Example batch names:", names[:4])

plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()



## === cell 13
rows = []
tic = time.time()

cassava_gpu_model.eval()

use_amp = False

with torch.no_grad():
    for b, (X_test, names) in enumerate(test_loader):
        X_test = X_test.to(DEVICE, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                y_test_pred = cassava_gpu_model(X_test)
        else:
            y_test_pred = cassava_gpu_model(X_test)

        predicted = torch.argmax(y_test_pred, dim=1).detach().cpu().numpy().astype(int)
        for fname, lab in zip(list(names), list(predicted)):
            rows.append((fname, int(lab)))

toc = time.time() - tic
print("Time for test inference is", toc, "seconds")

submission_df = pd.DataFrame(rows, columns=["image_id", "label"])

assert submission_df.shape[0] == len(
    testset
), "Submission rows != number of test images"
assert list(submission_df.columns) == ["image_id", "label"]

sample_path = os.path.join(data_path, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_df = pd.read_csv(sample_path)
    submission_df = sample_df[["image_id"]].merge(
        submission_df, on="image_id", how="left"
    )
    if submission_df["label"].isna().any():
        submission_df["label"] = submission_df["label"].fillna(0).astype(int)
else:
    submission_df = submission_df.sort_values("image_id").reset_index(drop=True)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

df_sub_test = pd.read_csv(submission_path)
print(df_sub_test.head())
print("Wrote:", submission_path, "rows:", len(df_sub_test))

print("Files in current directory:")
print(os.listdir("."))
