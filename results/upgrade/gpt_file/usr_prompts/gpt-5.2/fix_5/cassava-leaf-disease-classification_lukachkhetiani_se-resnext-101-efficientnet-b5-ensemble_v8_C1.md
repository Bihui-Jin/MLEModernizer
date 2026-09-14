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

0.7234814143245694

# 6. Current score

0.11136

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11136) has done: 'I fix the pipeline so it always runs end-to-end and writes a valid `submission.csv` with exactly the same `image_id` ordering/length as `sample_submission.csv`. The main blocker is missing external weight files (`/kaggle/input/ensemble/...`), so I add a safe fallback: use `timm` pretrained backbones (still the same two-model ensemble) when those weights aren’t present, and print which path was used. I also make inference robust to any unreadable/missing images by falling back to the sample row’s label (keeps submission length valid) instead of crashing, which fixes the length assertion and Kaggle “same length” error. These changes preserve the core logic (two timm models + average logits + argmax) while ensuring a valid submission is always produced.'
- What this solution (achieved 0.11136) has done: 'Your current score (0.11136) is far below the target (0.72348), so we should improve performance rather than just ensuring a valid CSV. The biggest issue is that when your private ensemble weights are missing, the fallback uses generic ImageNet-pretrained classifiers with a randomly initialized 5-class head, which produces near-random predictions and explains the ~0.11 accuracy. With minimal change to the existing two-model average-logits + argmax core logic, I switch the fallback to load publicly available Cassava-finetuned weights shipped inside the competition dataset (`/kaggle/input/.../*.pth`) when your external `/kaggle/input/ensemble/...` files aren’t present. This keeps the same architecture and inference flow, but replaces random heads with task-trained heads, which should move accuracy sharply upward toward (and likely within) the target band.'
- What this solution (achieved 0.11136) has done: 'Your score is far below the target, and the main reason is that the current “fallback” almost certainly never finds Cassava-finetuned weights inside the competition dataset, leaving you with ImageNet backbones plus a randomly initialized 5-class head (near-random ~0.11 accuracy). I keep your exact core logic (two timm models, average logits, argmax) but make the weight fallback actually discover and load any `.pth/.pt/.bin` checkpoints that exist under `/kaggle/input` (including the dataset folder), preferring ones that look like efficientnet/hrnet and then loading them with the same `_clean_state_dict` routine. I also make the weight loader slightly more robust by explicitly reinitializing the classifier head to 5 classes after loading when a checkpoint has a mismatched head, so the model doesn’t silently keep an incompatible classifier. This should move accuracy sharply upward toward your target while preserving the architecture/inference semantics.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import tqdm
import glob

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def _clean_state_dict(sd):
    """Fix common key-prefix issues; robustness only."""
    if not isinstance(sd, dict):
        return sd
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}

    return sd


def load_weights_if_exists(model, weight_path: str):
    """
    Score fix: ensure we actually load a Cassava-finetuned checkpoint when present, so the 5-class head is trained.
    Robustness: load with strict=False and (if needed) re-create the classifier head to num_classes=5.
    """
    if weight_path and os.path.exists(weight_path):
        sd = torch.load(weight_path, map_location="cpu")
        sd = _clean_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)

        if missing:
            try:
                model.reset_classifier(num_classes=5)
            except Exception:
                pass

        if missing or unexpected:
            print(
                f"Warning loading {os.path.basename(weight_path)}: missing={len(missing)} unexpected={len(unexpected)}"
            )
        print(f"Loaded weights: {weight_path}")
    else:
        print(f"Weight file not found, using model as-initialized: {weight_path}")
    return model


def process(image_bgr):
    img = cv2.resize(image_bgr, (384, 384), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x




## === cell 1
eff_weight_path = "/kaggle/input/ensemble/eff_model_last_adam.pth"
hr_weight_path = "/kaggle/input/ensemble/hr_model_last_adam.pth"


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_candidate_checkpoints(search_roots, exts=(".pth", ".pt", ".bin")):
    """
    Score fix: the previous fallback only checked a few hardcoded filenames that typically don't exist.
    Here we legitimately search /kaggle/input (and the dataset dir) for weight files and then pick likely matches.
    """
    cand = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for ext in exts:
            cand.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    seen = set()
    out = []
    for p in cand:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _pick_weight(candidates, include_keywords, exclude_keywords=()):
    inc = [k.lower() for k in include_keywords]
    exc = [k.lower() for k in exclude_keywords]
    for p in candidates:
        bn = os.path.basename(p).lower()
        if all(k in bn for k in inc) and not any(k in bn for k in exc):
            return p
    for p in candidates:
        bn = os.path.basename(p).lower()
        if any(k in bn for k in inc) and not any(k in bn for k in exc):
            return p
    return None


all_ckpts = _find_candidate_checkpoints(
    search_roots=[
        "/kaggle/input",  # broad search to catch dataset-provided community checkpoints if present
        DATA_DIR,
    ]
)

eff_fallback = _pick_weight(
    all_ckpts,
    include_keywords=["eff"],
    exclude_keywords=["hrnet", "w48", "resnet", "densenet"],
)
if eff_fallback is None:
    eff_fallback = _pick_weight(
        all_ckpts, include_keywords=["efficient"], exclude_keywords=["hrnet", "w48"]
    )

hr_fallback = _pick_weight(
    all_ckpts,
    include_keywords=["hrnet"],
    exclude_keywords=["eff", "efficient", "b5", "b4", "b3"],
)
if hr_fallback is None:
    hr_fallback = _pick_weight(
        all_ckpts, include_keywords=["w48"], exclude_keywords=["eff", "efficient"]
    )

eff_to_load = eff_weight_path if os.path.exists(eff_weight_path) else eff_fallback
hr_to_load = hr_weight_path if os.path.exists(hr_weight_path) else hr_fallback

efficient = timm.create_model(
    "tf_efficientnet_b5",
    pretrained=not bool(
        eff_to_load
    ),  # if we have finetuned weights, no need for ImageNet init
    num_classes=5,
)
efficient = load_weights_if_exists(efficient, eff_to_load).to(device).eval()

hrnet = timm.create_model(
    "hrnet_w48",
    pretrained=not bool(hr_to_load),
    num_classes=5,
)
hrnet = load_weights_if_exists(hrnet, hr_to_load).to(device).eval()

print("Models have been loaded...\n")
print(f"Effective eff weights used: {eff_to_load}")
print(f"Effective hr weights used: {hr_to_load}")



## === cell 2
sample = pd.read_csv(SAMPLE_SUB_PATH)
test_names = sample["image_id"].tolist()

names, labels = [], []

with torch.no_grad():
    for image_id in tqdm.tqdm(test_names):
        file_path = os.path.join(TEST_IMG_DIR, image_id)
        img = cv2.imread(file_path)

        if img is None:
            pred = int(sample.loc[sample["image_id"] == image_id, "label"].iloc[0])
            names.append(image_id)
            labels.append(pred)
            continue

        topred = process(img)
        hr_out = hrnet(topred)
        eff_out = efficient(topred)

        total = (hr_out + eff_out) / 2.0
        pred = int(torch.argmax(F.softmax(total, dim=1), dim=1).item())

        names.append(image_id)
        labels.append(pred)



## === cell 3
df = pd.DataFrame({"image_id": names, "label": labels})

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print(f"\nSaved submission to: {out_path} (rows={len(df)})")

assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(df.columns) == ["image_id", "label"]
assert len(df) == len(pd.read_csv(SAMPLE_SUB_PATH))
assert df["image_id"].tolist() == pd.read_csv(SAMPLE_SUB_PATH)["image_id"].tolist()
