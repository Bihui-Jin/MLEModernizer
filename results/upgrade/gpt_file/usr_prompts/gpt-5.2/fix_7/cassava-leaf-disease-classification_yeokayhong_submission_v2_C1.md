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

3.13

# 3. Installed packages

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

# 5. Target score

0.8845572680568148

# 6. Current score

0.15658

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25673) has done: 'I make the pipeline run end-to-end without requiring an external weights dataset by falling back to a torchvision pretrained ViT when the requested `.pth` file isn’t available, keeping the same ViT family and inference flow. I also fix the device/dtype mismatch that caused `Input type ... and weight type ... should be the same` by ensuring the model is moved to `device` after any potential weight loading and that inputs use the same device. Finally, I keep the submission formatting aligned to `sample_submission.csv` and always write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.25673) is far below the target (0.88456), and the biggest issue is that your fallback path uses an ImageNet-pretrained ViT with a freshly initialized 5-class head, so predictions are essentially near-random. To move the score sharply toward the target while keeping the same core ViT inference pipeline, the minimal fix is to *require* a valid cassava fine-tuned `.pth/.pt` and load it correctly (including common checkpoint key-prefix patterns like `module.`). I also switch test-time preprocessing to the official ViT_H_14 weights’ transforms when using torchvision weights (kept as a last-resort fallback), and I add a lightweight batch inference (no change to semantics) to reduce overhead and stay within runtime. The output submission format/path stays identical and still always write `submission.csv`.'
- What this solution (achieved 0.05792) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end and always writes a valid `submission.csv`. The main bug is that you currently hard-fail when the external cassava weights dataset isn’t present, leaving `weights_path=None` and causing the later `torch.load(None)` crash; I restore a safe fallback to torchvision ViT weights while still attempting to auto-discover a 5-class cassava checkpoint if one exists. I also fix the CUDA dtype/device mismatch by explicitly moving the model to `device` *after* loading weights and ensuring inputs are on the same device. Finally, I align the preprocessing with the selected torchvision ViT weights transforms when falling back, which should improve score compared to the generic normalization while keeping the same inference semantics.'
- What this solution (achieved 0.15658) has done: 'Your current score is far below the target because the code almost certainly falls back to an ImageNet-pretrained ViT with a randomly initialized 5-class head (no cassava fine-tuning weights found), which yields near-random predictions. The smallest change that moves accuracy sharply upward (toward your 0.8846 target) while keeping the same ViT inference core is to reliably find and load *any* cassava 5-class checkpoint if present, by widening the search to include `/kaggle/data` and `/kaggle/working` (not just `/kaggle/input`) and prioritizing likely cassava checkpoints. I also make the checkpoint filtering slightly more permissive for common head key names so the 5-class head weights actually load when available, while keeping the architecture, transforms, and inference loop intact. If no suitable checkpoint exists anywhere, the fallback behavior remains unchanged and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models
from torchvision.models import ViT_H_14_Weights
from tqdm import tqdm
from PIL import Image

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 518
num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def torch_load_state_dict(path: str, map_location):
    """
    Compatibility: environments differ on torch.load(weights_only=...).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(state):
    """
    Handle common checkpoint wrappers & key prefixes.
    """
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state_dict = state["state_dict"]
    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state_dict = state["model"]
    else:
        state_dict = state

    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_sd[nk] = v
        state_dict = new_sd

    return state_dict


def state_dict_head_out_features(sd: dict) -> int | None:
    """
    Prefer checkpoints that clearly have a 5-class head.
    (Expanded key list to catch more common ViT head naming variants so the
    cassava head weights actually get detected and loaded when present.)
    """
    for k in (
        "heads.head.weight",
        "heads.head.bias",
        "head.weight",
        "head.bias",
        "heads.weight",
        "heads.bias",
        "classifier.weight",
        "classifier.bias",
        "fc.weight",
        "fc.bias",
    ):
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
            return int(sd[k].shape[0])
    return None


def find_best_cassava_weights_path(preferred_path: str) -> str | None:
    """
    Try to find a cassava fine-tuned checkpoint.
    Minimal score-impacting fix: also search /kaggle/data and /kaggle/working
    (some environments place datasets/extracted weights there), not only /kaggle/input.
    """
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    search_roots = ["/kaggle/input", "/kaggle/data", "/kaggle/working"]
    candidates = []
    for root in search_roots:
        for pattern in [
            f"{root}/**/*.pth",
            f"{root}/**/*.pt",
            f"{root}/**/*.ckpt",
        ]:
            candidates.extend(glob.glob(pattern, recursive=True))

    candidates = sorted(set(candidates))

    def quick_text_score(p: str) -> int:
        base = os.path.basename(p).lower()
        full = p.lower()
        s = 0
        if "cassava" in full:
            s += 80
        if "leaf" in full:
            s += 15
        if "disease" in full:
            s += 10
        if "vit" in full:
            s += 15
        if "efficientnet" in full or "resnet" in full:
            s -= 5
        if "imagenet" in full:
            s -= 30
        if "swa" in full:
            s += 2
        if "best" in base:
            s += 3
        if "fold" in full:
            s += 1
        return s

    prefiltered = sorted(
        candidates, key=lambda p: (quick_text_score(p), p), reverse=True
    )[:80]

    best = None
    best_score = -(10**9)

    for p in prefiltered:
        try:
            state = torch_load_state_dict(p, map_location="cpu")
            sd = unwrap_state_dict(state)
            if not isinstance(sd, dict):
                continue

            out_features = state_dict_head_out_features(sd)
            s = quick_text_score(p)

            if out_features == num_classes:
                s += 2000
            elif out_features is None:
                s -= 100
            else:
                s -= 500

            if s > best_score:
                best_score = s
                best = p
        except Exception:
            continue

    return best


weights_path = find_best_cassava_weights_path(model_path)
if weights_path is None:
    print(
        "Warning: No cassava fine-tuned checkpoint found under /kaggle/input,/kaggle/data,/kaggle/working. "
        "Falling back to torchvision ImageNet-pretrained ViT-H/14 weights."
    )
else:
    print(f"Using weights: {weights_path}")




## === cell 1
tv_weights = ViT_H_14_Weights.DEFAULT

if weights_path is None:
    val_transforms = tv_weights.transforms()
else:
    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

if weights_path is None:
    model = models.vit_h_14(weights=tv_weights, image_size=image_size)
else:
    model = models.vit_h_14(weights=None, image_size=image_size)

model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

if weights_path is not None:
    state = torch_load_state_dict(weights_path, map_location="cpu")
    state_dict = unwrap_state_dict(state)

    model_sd = model.state_dict()
    filtered_sd = {}
    for k, v in state_dict.items():
        if (
            k in model_sd
            and hasattr(v, "shape")
            and hasattr(model_sd[k], "shape")
            and tuple(v.shape) == tuple(model_sd[k].shape)
        ):
            filtered_sd[k] = v

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if missing or unexpected:
        print(
            f"Warning: non-strict load_state_dict after shape-filter. "
            f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

model = model.to(device)
model.eval()




## === cell 2
sample_sub = pd.read_csv(sample_submission_path)
image_ids = sample_sub["image_id"].tolist()

batch_size = 16 if device.type == "cuda" else 8

test_predictions = []
batch_imgs = []

with torch.no_grad():
    for image_name in tqdm(image_ids, desc="Test"):
        image_path = os.path.join(test_data_directory, image_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Test image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_t = val_transforms(img)
        batch_imgs.append(img_t)

        if len(batch_imgs) == batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            preds = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
            )
            test_predictions.extend(preds)
            batch_imgs = []

    if batch_imgs:
        x = torch.stack(batch_imgs, dim=0).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        test_predictions.extend(preds)

if len(test_predictions) != len(image_ids):
    raise RuntimeError(
        f"Prediction length mismatch: {len(test_predictions)} vs {len(image_ids)}"
    )

submission_df = pd.DataFrame({"image_id": image_ids, "label": test_predictions})
submission_df = submission_df[["image_id", "label"]]
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df))
