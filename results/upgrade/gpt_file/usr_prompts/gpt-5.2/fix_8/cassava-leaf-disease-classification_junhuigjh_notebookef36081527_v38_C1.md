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

0.7660924750679964

# 6. Current score

0.58184

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the immediate runtime blocker by making the model loading robust: the notebook currently points to a missing input checkpoint, so I search common Kaggle input locations and, if none exist, fall back to a built-in torchvision ResNet50 to still produce a valid submission. I also ensure the script always defines `image_ids`/`prediction` by moving submission creation after successful inference, and I keep the preprocessing and prediction loop unchanged. Finally, I add a small safeguard to align the submission to `sample_submission.csv` ordering (score-neutral but prevents accidental misalignment) and always write `submission.csv`.'
- What this solution (achieved 0.09417) has done: 'Your current score (0.10762) is far below the target (0.7661), and the main reason is that the model is effectively untrained because you fall back to `weights=None` (random ResNet50) when the checkpoint isn’t found. To move the score upward with minimal logic change, I keep the exact inference loop and architecture, but switch the fallback to use ImageNet-pretrained ResNet50 weights, which should immediately yield a large accuracy jump on this 5-class leaf task. I also align normalization to the standard ImageNet mean/std when using pretrained weights (still the same resize→tensor→normalize pipeline), and keep submission ordering aligned to `sample_submission.csv` as you already do. These changes are directly targeted to improve accuracy without altering the overall approach.'
- What this solution (achieved 0.28812) has done: 'Your score is far below the target because when no checkpoint is found you currently attach a *randomly initialized* 5-class head to an ImageNet backbone, which makes predictions nearly arbitrary. To move accuracy toward the target with minimal semantic change, I keep the same ResNet50 inference pipeline but replace the fallback head initialization with a deterministic, stronger zero-shot baseline: initialize the 5-way `fc` using the pretrained ImageNet `fc` weights by selecting the most relevant ImageNet classes and copying their weights/biases into the 5 outputs. I also fix checkpoint-loading to start from ImageNet weights before applying a state_dict (so partial checkpoints don’t leave large parts random), while keeping preprocessing and the prediction loop unchanged. This should substantially increase accuracy without changing the architecture or adding training.'
- What this solution (achieved 0.4275) has done: 'Your score is far below the target, so we should increase accuracy with the smallest change that preserves your current ResNet50 inference-only core logic. The biggest issue is that your “leaf-ish ImageNet fc row copy” for a 5-class head is a weak heuristic; replacing it with a simple, deterministic “class-prototype head” built from the pretrained backbone’s own penultimate features on a small, fixed set of training images usually jump accuracy significantly without changing architecture or adding any training loop. Concretely, we keep your exact preprocessing and prediction loop style, but we (1) load `train.csv`, (2) compute mean feature vectors per class from a small capped number of training images using the ResNet50 backbone, then (3) set the final `fc` weights/bias from these prototypes. If a real checkpoint exists, we keep your checkpoint path logic and use it as before.'
- What this solution (achieved 0.54821) has done: 'Your current score (0.4275) is still far below the target (0.7661), so we should improve accuracy while keeping your inference-only ResNet50 pipeline intact. The simplest high-impact fix is to keep your prototype head idea but make it substantially stronger and less noisy by (1) computing prototypes from more labeled images per class and (2) using L2-normalized features with a cosine-similarity style classifier scale, which is still just setting `fc.weight`/`fc.bias` (no training loop added). To avoid runtime blowups while increasing quality, we batch feature extraction with a DataLoader and cap images/class at a higher (but still bounded) value. Everything else (paths, preprocessing resize/normalize, prediction loop behavior, submission alignment) remains the same and still writes `submission.csv`.'
- What this solution (achieved 0.58184) has done: 'Your current score (0.54821) is below the target (0.76609), so we should increase accuracy with the smallest change that preserves your inference-only ResNet50 + prototype-head core logic. The most impactful minimal fix is to match train/test preprocessing by using the same “center-crop after resize” pipeline used by standard ImageNet models, which usually improves transfer accuracy without changing the model. I keep your prototype initialization exactly as-is, but (1) switch to Resize(256)+CenterCrop(224) and (2) run test inference in small batches via a DataLoader to reduce per-image overhead and keep execution stable under the time limit. Submission ordering/format and all paths remain unchanged, and it still always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, models

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

main_model_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_test_images_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
        "/kaggle/working/cassava-leaf-disease-classification/test_images",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Could not find test_images directory in expected Kaggle paths."
    )


def find_train_images_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/cassava-leaf-disease-classification/train_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
        "/kaggle/working/cassava-leaf-disease-classification/train_images",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def find_train_csv_path():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
        "/kaggle/working/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_sample_submission_path():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/working/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_checkpoint_path():
    """
    Keep existing robust checkpoint search logic.
    """
    original = (
        "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
    )
    if os.path.isfile(original):
        return original

    patterns = [
        "/kaggle/input/**/Resnet50_70_512x512.pth",
        "/kaggle/input/**/resnet50*.pth",
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            matches = sorted(set(matches))
            return matches[0]
    return None


class _ProtoDataset(torch.utils.data.Dataset):
    def __init__(self, rows, images_dir, preprocess):
        self.rows = rows
        self.images_dir = images_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_id, label = self.rows[idx]
        img_path = os.path.join(self.images_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img)
        y = int(label)
        return x, y


def init_fc_from_train_prototypes(
    model_5cls: torch.nn.Module,
    preprocess,
    device,
    max_images_per_class: int = 256,
    batch_size: int = 32,
    scale: float = 12.0,
):
    """
    Inference-only prototype head initialization (kept as core logic).
    """
    train_csv_path = find_train_csv_path()
    train_images_dir = find_train_images_dir()
    if train_csv_path is None or train_images_dir is None:
        return {"used": False, "reason": "train.csv or train_images dir not found"}

    df = pd.read_csv(train_csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        return {"used": False, "reason": "train.csv missing required columns"}

    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["label", "image_id"]).reset_index(drop=True)

    rows = []
    counts = []
    for lbl in range(5):
        sub = df[df["label"] == lbl].head(max_images_per_class)
        counts.append(int(len(sub)))
        for r in sub.itertuples(index=False):
            img_path = os.path.join(train_images_dir, r.image_id)
            if os.path.isfile(img_path):
                rows.append((r.image_id, int(r.label)))

    if len(rows) == 0 or min(counts) == 0:
        return {
            "used": False,
            "reason": f"insufficient images per class (counts={counts})",
        }

    feat_extractor = torch.nn.Sequential(*(list(model_5cls.children())[:-1])).to(device)
    feat_extractor.eval()

    ds = _ProtoDataset(rows, train_images_dir, preprocess)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    feats_sum = torch.zeros(
        (5, model_5cls.fc.in_features), dtype=torch.float32, device=device
    )
    feats_cnt = torch.zeros((5,), dtype=torch.long, device=device)

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            f = feat_extractor(xb).flatten(1)  # [B, 2048]
            f = torch.nn.functional.normalize(f, dim=1)
            for c in range(5):
                m = yb == c
                if m.any():
                    feats_sum[c] += f[m].sum(dim=0)
                    feats_cnt[c] += int(m.sum().item())

    cnt_cpu = feats_cnt.detach().cpu().numpy().tolist()
    if min(cnt_cpu) == 0:
        return {
            "used": False,
            "reason": f"insufficient usable images per class (counts={cnt_cpu})",
        }

    protos = feats_sum / feats_cnt.unsqueeze(1).float()
    protos = torch.nn.functional.normalize(protos, dim=1)

    with torch.no_grad():
        model_5cls.fc.weight.copy_(protos * float(scale))
        model_5cls.fc.bias.zero_()

    return {
        "used": True,
        "counts": cnt_cpu,
        "max_images_per_class": max_images_per_class,
        "batch_size": batch_size,
        "scale": scale,
    }


ckpt_path = find_checkpoint_path()

model = None
if ckpt_path is not None:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        state_dict = None
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state_dict = obj["model_state_dict"]
        elif all(isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()):
            state_dict = obj

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, 5)

        if state_dict is not None:
            cleaned = {}
            for k, v in state_dict.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                cleaned[nk] = v
            backbone.load_state_dict(cleaned, strict=False)
        model = backbone
    else:
        model = None

prototype_init_info = None
if model is None:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)

    prototype_init_info = init_fc_from_train_prototypes(
        model,
        main_model_preprocess,
        device,
        max_images_per_class=256,
        batch_size=32,
        scale=12.0,
    )

model.to(device)
model.eval()

test_images_dir = find_test_images_dir()
sample_sub_path = find_sample_submission_path()

print("Device:", device)
print("Test images dir:", test_images_dir)
print("Sample submission path:", sample_sub_path)
print("Checkpoint used:", ckpt_path)
print("Prototype init info (None if ckpt used):", prototype_init_info)



## === cell 1
if sample_sub_path is not None:
    sample_df = pd.read_csv(sample_sub_path)
    image_ids = sample_df["image_id"].astype(str).tolist()
else:
    image_ids = sorted(
        [f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")]
    )


class _TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, images_dir, preprocess):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img)
        return x


test_ds = _TestDataset(image_ids, test_images_dir, main_model_preprocess)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

prediction = []
with torch.no_grad():
    for xb in test_dl:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        pred = torch.argmax(out, dim=1).detach().cpu().numpy().astype(np.int64).tolist()
        prediction.extend(pred)

print("Predictions computed:", len(prediction))



## === cell 2
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission["label"] = submission["label"].astype(int)

if submission.isna().any().any():
    raise ValueError("Submission contains NaNs.")
if submission.shape[0] == 0:
    raise ValueError("Empty submission; no test images found or inference failed.")

submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")
