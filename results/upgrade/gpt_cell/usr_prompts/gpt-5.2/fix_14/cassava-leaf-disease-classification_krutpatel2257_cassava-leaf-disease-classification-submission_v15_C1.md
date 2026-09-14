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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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

0.8856

# 6. Current score

0.65433

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15396) has done: 'Diagnosis: Cell 3 crashes because `albumentations.Cutout` was removed/relocated in Albumentations v2.x, so accessing it as `albumentations.Cutout` raises `AttributeError`. The rest of the pipeline expects an augmentation object in the Compose list that performs cutout-style masking. Albumentations v2 provides the equivalent functionality via `albumentations.augmentations.dropout.cutout.Cutout` (and related dropout transforms), so we can import it and keep the same semantics.

Patch summary: Modify only cell 3 to use a version-compatible Cutout import. Try the v2 import path first and fall back to the older `albumentations.Cutout` if present, preserving the original augmentation behavior and interface.

Updated cells: Only cell 3 is updated below.

Compatibility notes for cell k+1: `sub_aug` remains an `albumentations.Compose` and `sub_aug(image=image)["image"]` continues to return a normalized HWC numpy array, so cell 4 works unchanged.

Assumptions: The installed Albumentations version is 2.0.8 (as provided), where Cutout exists under the new module path.'
- What this solution (achieved 0.05531) has done: 'Your very low score is most consistent with the pretrained weights not actually being loaded, so the model is effectively random; the current code silently proceeds with random initialization if it can’t find the checkpoint. I make a minimal, execution-safe change that (1) searches specifically for `model(11).pth` first (then any `.pth/.pt`), (2) fails fast if no weights are found (so you don’t unknowingly submit random predictions again), and (3) loads checkpoints robustly whether they were saved as a plain `state_dict` or wrapped under keys like `state_dict`. These changes preserve the same model architecture and TTA logic, but ensure the intended trained weights are actually used, which should move accuracy sharply toward your target.'
- What this solution (achieved 0.05531) has done: 'The crash happens because `model_path` points to a Kaggle dataset (`rn-tta-calr-ft-ofasf`) that is not present in this environment, and the current logic raises `FileNotFoundError` when it can’t find any `.pth/.pt` checkpoint. To unblock execution while preserving the rest of the notebook’s behavior, I keep the checkpoint search but, if nothing is found, fall back to running with randomly initialized weights instead of hard-failing. This keeps the model architecture and inference code unchanged and ensures `model` is still placed on the correct device and set to `eval()` for cell 3+.'
- What this solution (achieved 0.12855) has done: 'Your score is far below target because this inference notebook is almost certainly running with randomly initialized weights (the referenced checkpoint dataset isn’t available here), and additionally the current “TTA” applies heavy *training* augmentations (random crop/HSV/cutout) at test time, which severely degrades accuracy even with good weights. To move sharply toward the target while preserving the same model and inference loop, I (1) stop silently using random weights by loading ImageNet pretrained weights when no checkpoint is found, and (2) replace the test-time augmentation pipeline with standard deterministic validation preprocessing (resize/center-crop/normalize) so averaging 10 passes no longer injects label-destroying noise. These are minimal changes that keep the same architecture, loss semantics (none here), and prediction logic, but should raise accuracy substantially toward ~0.8856. The script still produces a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.17339) has done: 'Your current score suggests predictions are still effectively untrained for this dataset (ImageNet backbone + random 5-class head), so the smallest legitimate move toward the target is to load *Cassava-specific* weights when present and ensure they actually match the modified `fc` layer. I keep the same ResNeXt50 architecture and single-model inference loop, but (1) improve checkpoint discovery to prefer any Cassava-trained `*.pth/*.pt` under the competition input folder, (2) enforce a strict load for the classification head when possible (to avoid silently using a random head), and (3) fix the preprocessing to use the model’s expected input size (224) consistently (resize + center crop) to reduce distribution shift. This keeps your logic intact (10-pass averaging, argmax labels, same submission schema) while making the model much more likely to produce meaningful predictions and push accuracy toward 0.8856.'
- What this solution (achieved 0.11248) has done: 'Your score is far below the target because the notebook is almost certainly still not loading any Cassava-trained checkpoint (it searches only for `.pth/.pt` and your original reference was a Kaggle dataset that isn’t available here), so the 5-class head remains random. I make the smallest change that’s most likely to move accuracy toward the target: expand checkpoint discovery to include common Kaggle formats (`.pth/.pt/.bin/.ckpt`) and prefer any file whose name hints it’s a trained “best/model” checkpoint. I also make weight loading tolerant of checkpoints saved from different wrappers (including Lightning) while keeping the same ResNeXt50 model, preprocessing, and inference loop. This keeps core logic intact but greatly increases the chance you actually load trained 5-class weights, which should raise accuracy substantially.'
- What this solution (achieved 0.37444) has done: 'Your score is far below target because the model is almost certainly still running with a mostly random 5-class head (either no Cassava checkpoint is found, or the checkpoint loads without matching `fc`), so predictions are near chance. I make the smallest change that improves accuracy without changing the model/loop: (1) explicitly look for common Cassava-trained ResNeXt50 checkpoints under all Kaggle input roots, and (2) when a checkpoint is found but `strict=True` fails, still force-load the `fc.*` weights if present so the classifier head isn’t left random. I also fix a minor preprocessing bug (double `np.array`/tensor conversion) to ensure the normalized float image is converted to a tensor consistently (no semantic change intended, just correctness). The rest (architecture, 10-pass averaging loop, resize/center-crop/normalize, argmax, submission format) stays the same.'
- What this solution (achieved 0.15919) has done: 'Your current gap to the target is large (0.37444 vs 0.8856), and the most likely cause is still that no Cassava-trained checkpoint is being loaded, so the 5-class head remains effectively random. I keep the same ResNeXt50 architecture and the same inference/TTA loop, but make checkpoint discovery/loading more robust by (1) explicitly prioritizing common Cassava checkpoints like `best.pth`, `best_model.pth`, `*_fold*.pth` and known dataset filenames, and (2) improving key-remapping so checkpoints that store `fc.*` under `model.fc.*`, `classifier.*`, or Lightning-style prefixes actually populate your `model.fc` weights. This is a minimal change focused on getting the intended trained weights loaded (which should move accuracy sharply upward toward your target) while keeping preprocessing and submission formatting unchanged. The script still always write a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.65433) has done: 'Your score is far below the target, and the current inference loop is almost certainly using an ImageNet backbone with a randomly initialized 5-class head because no Cassava-trained checkpoint exists in this environment. The smallest legitimate way to move accuracy sharply upward (without changing the model architecture or inference semantics) is to add a short, standard fine-tuning step on `train.csv` using the same preprocessing you already use for test, then run the exact same 10-pass averaging inference to create `submission.csv`. To keep runtime under 600s and changes minimal, this trains only the final `fc` layer for a few epochs (backbone frozen), which is a minimal extension to your existing approach and should move accuracy substantially toward your target. All paths remain unchanged and the output submission format stays `image_id,label`.'

# 9. Code solution

## === cell 0
import os

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms



## === cell 1
model_path = "../input/rn-tta-calr-ft-ofasf/model(11).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"



## === cell 2
weights = None
try:
    weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
except Exception:
    weights = None

model = models.resnext50_32x4d(weights=weights)
model.fc = nn.Linear(2048, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def _find_checkpoint(initial_path: str) -> str:
    if initial_path and os.path.exists(initial_path):
        return initial_path

    search_roots = [
        "../input",
        "/kaggle/input",
        "../kaggle/input",
        "input",
        "/kaggle/data",
        "data",
        "/kaggle/working",
    ]

    exts = (".pth", ".pt", ".bin", ".ckpt")

    strong_name_hits = [
        "model(11).pth",
        "best.pth",
        "best_model.pth",
        "best-model.pth",
        "best.ckpt",
        "final.pth",
        "final_model.pth",
        "checkpoint.pth",
        "ckpt.pth",
    ]
    strong_substrings = [
        "cassava",
        "leaf",
        "disease",
        "cldc",
        "resnext50",
        "resnext",
        "32x4d",
        "x50",
        "fold",
        "best",
        "final",
    ]

    best_candidates = []
    cassava_candidates = []
    any_ckpt = None

    for root_dir in search_roots:
        if not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            for f in files:
                lf = f.lower()
                if not lf.endswith(exts):
                    continue
                p = os.path.join(root, f)
                if any_ckpt is None:
                    any_ckpt = p

                if lf in strong_name_hits:
                    best_candidates.append(p)
                    continue

                path_l = p.lower()
                if any(s in lf or s in path_l for s in strong_substrings):
                    cassava_candidates.append(p)

    if len(best_candidates) > 0:
        best_candidates = sorted(best_candidates, key=lambda x: (len(x), x))
        return best_candidates[0]

    if len(cassava_candidates) > 0:

        def _score(p: str) -> int:
            pl = p.lower()
            s = 0
            if "best" in pl:
                s += 5
            if "final" in pl:
                s += 3
            if "fold" in pl:
                s += 2
            if "resnext" in pl:
                s += 2
            if "cassava" in pl or "disease" in pl or "leaf" in pl:
                s += 2
            return -s  # negative for ascending sort

        cassava_candidates = sorted(
            cassava_candidates, key=lambda x: (_score(x), len(x), x)
        )
        return cassava_candidates[0]

    return any_ckpt


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if "checkpoint" in ckpt and isinstance(ckpt["checkpoint"], dict):
            inner = ckpt["checkpoint"]
            for key in ["state_dict", "model_state_dict", "model", "net"]:
                if key in inner and isinstance(inner[key], dict):
                    return inner[key]
        return ckpt
    return ckpt


def _clean_keys(state):
    cleaned = {}
    for k, v in state.items():
        nk = k

        for pref in ["module.", "model.", "net.", "backbone."]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]

        if nk.startswith("ema."):
            nk = nk[len("ema.") :]
        if nk.startswith("student."):
            nk = nk[len("student.") :]
        if nk.startswith("teacher."):
            nk = nk[len("teacher.") :]

        cleaned[nk] = v
    return cleaned


def _alias_classifier_keys_to_fc(state: dict, model: nn.Module) -> dict:
    if not isinstance(state, dict):
        return state

    model_sd = model.state_dict()
    if "fc.weight" in state and "fc.bias" in state:
        return state

    candidate_weight_keys = [
        "classifier.weight",
        "head.weight",
        "linear.weight",
        "model.fc.weight",
        "net.fc.weight",
        "backbone.fc.weight",
        "model.classifier.weight",
        "model.head.weight",
    ]
    candidate_bias_keys = [
        "classifier.bias",
        "head.bias",
        "linear.bias",
        "model.fc.bias",
        "net.fc.bias",
        "backbone.fc.bias",
        "model.classifier.bias",
        "model.head.bias",
    ]

    w_key = next(
        (
            k
            for k in candidate_weight_keys
            if k in state
            and hasattr(state[k], "shape")
            and state[k].shape == model_sd["fc.weight"].shape
        ),
        None,
    )
    b_key = next(
        (
            k
            for k in candidate_bias_keys
            if k in state
            and hasattr(state[k], "shape")
            and state[k].shape == model_sd["fc.bias"].shape
        ),
        None,
    )

    if w_key is not None:
        state["fc.weight"] = state[w_key]
    if b_key is not None:
        state["fc.bias"] = state[b_key]
    return state


def _load_checkpoint_weights(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(ckpt)
    if isinstance(state, dict):
        state = _clean_keys(state)
        state = _alias_classifier_keys_to_fc(state, model)

    try:
        model.load_state_dict(state, strict=True)
        print("Loaded checkpoint with strict=True (fc weights should match).")
        return True
    except Exception as e:
        print(
            f"WARNING: strict=True load failed ({type(e).__name__}: {e}). Trying targeted fc alias + strict=False."
        )

    if isinstance(state, dict):
        state = _alias_classifier_keys_to_fc(state, model)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"WARNING: Missing keys (up to 20): {missing[:20]}")
    if len(unexpected) > 0:
        print(f"WARNING: Unexpected keys (up to 20): {unexpected[:20]}")
    return True


model_path_found = _find_checkpoint(model_path)

loaded_any_checkpoint = False

if model_path_found is None:
    print(
        "WARNING: No checkpoint (.pth/.pt/.bin/.ckpt) found; using ImageNet pretrained backbone weights "
        "and randomly initialized classification head."
    )
else:
    model_path = model_path_found
    print(f"Loading model weights from: {model_path}")
    loaded_any_checkpoint = _load_checkpoint_weights(model, model_path, device)

model.eval()



## === cell 3
sub_aug = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


def _maybe_finetune_fc(
    model: nn.Module,
    device,
    train_csv_path: str,
    train_images_path: str,
    aug,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-3,
    num_workers: int = 2,
    max_steps_per_epoch: int | None = None,
):
    if not (os.path.isfile(train_csv_path) and os.path.isdir(train_images_path)):
        print("Training data not found; skipping fine-tuning.")
        return

    df = pd.read_csv(train_csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        print("train.csv schema unexpected; skipping fine-tuning.")
        return

    class CassavaTrainDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, aug):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.aug = aug
            self.to_tensor = transforms.ToTensor()

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row.image_id)
            image = np.array(Image.open(img_path).convert("RGB"))
            image = self.aug(image=image)["image"]
            x = self.to_tensor(image)  # CHW float32
            y = int(row.label)
            return x, y

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    ds = CassavaTrainDataset(df, train_images_path, aug)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        seen = 0
        steps = 0

        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = out.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            seen += xb.size(0)
            steps += 1

            if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                break

        print(
            f"Fine-tune epoch {ep+1}/{epochs} - loss: {running_loss/max(1,seen):.4f} - acc: {correct/max(1,seen):.4f}"
        )

    model.eval()


if model_path_found is None:
    _maybe_finetune_fc(
        model,
        device,
        train_csv_path=train_csv_path,
        train_images_path=train_images_path,
        aug=sub_aug,
        epochs=3,
        batch_size=32,
        lr=3e-3,
        num_workers=2,
        max_steps_per_epoch=250,  # bounded runtime; still a legitimate training run on real labels
    )



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
to_tensor = transforms.ToTensor()

for _, sample_row in sample_sub.iterrows():
    image_pred = 0
    img_path = os.path.join(test_images_path, sample_row.image_id)

    for j in range(10):
        image = np.array(Image.open(img_path).convert("RGB"))
        image = sub_aug(image=image)["image"]

        image_t = to_tensor(image).to(device)

        with torch.no_grad():
            outputs = model(image_t.unsqueeze(0))
        image_pred += outputs

    image_pred /= 10
    _, pred_label = torch.max(image_pred, 1)
    predictions.append([sample_row.image_id, int(pred_label.item())])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
