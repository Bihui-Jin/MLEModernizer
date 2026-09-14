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

0.28812

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the immediate runtime blocker by making the model loading robust: the notebook currently points to a missing input checkpoint, so I search common Kaggle input locations and, if none exist, fall back to a built-in torchvision ResNet50 to still produce a valid submission. I also ensure the script always defines `image_ids`/`prediction` by moving submission creation after successful inference, and I keep the preprocessing and prediction loop unchanged. Finally, I add a small safeguard to align the submission to `sample_submission.csv` ordering (score-neutral but prevents accidental misalignment) and always write `submission.csv`.'
- What this solution (achieved 0.09417) has done: 'Your current score (0.10762) is far below the target (0.7661), and the main reason is that the model is effectively untrained because you fall back to `weights=None` (random ResNet50) when the checkpoint isn’t found. To move the score upward with minimal logic change, I keep the exact inference loop and architecture, but switch the fallback to use ImageNet-pretrained ResNet50 weights, which should immediately yield a large accuracy jump on this 5-class leaf task. I also align normalization to the standard ImageNet mean/std when using pretrained weights (still the same resize→tensor→normalize pipeline), and keep submission ordering aligned to `sample_submission.csv` as you already do. These changes are directly targeted to improve accuracy without altering the overall approach.'
- What this solution (achieved 0.28812) has done: 'Your score is far below the target because when no checkpoint is found you currently attach a *randomly initialized* 5-class head to an ImageNet backbone, which makes predictions nearly arbitrary. To move accuracy toward the target with minimal semantic change, I keep the same ResNet50 inference pipeline but replace the fallback head initialization with a deterministic, stronger zero-shot baseline: initialize the 5-way `fc` using the pretrained ImageNet `fc` weights by selecting the most relevant ImageNet classes and copying their weights/biases into the 5 outputs. I also fix checkpoint-loading to start from ImageNet weights before applying a state_dict (so partial checkpoints don’t leave large parts random), while keeping preprocessing and the prediction loop unchanged. This should substantially increase accuracy without changing the architecture or adding training.'

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
        transforms.Resize((224, 224)),
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
    Bug fix: original hardcoded path doesn't exist in this environment.
    We search common Kaggle input locations for .pth/.pt checkpoints.
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


def init_fc_from_imagenet_leafish(model_5cls: torch.nn.Module):
    """
    Score-improvement change (minimal semantics): if we have no task checkpoint,
    we should not leave the 5-class head randomly initialized (near-random accuracy).
    We initialize the 5 outputs by copying rows from the pretrained ImageNet fc
    corresponding to leaf-ish classes. This keeps the same ResNet50 architecture
    and inference loop, but provides a much stronger deterministic baseline.
    """
    w = models.ResNet50_Weights.IMAGENET1K_V2
    ref = models.resnet50(weights=w)
    ref.eval()

    keywords = [
        "leaf",
        "leaves",
        "corn",
        "maize",
        "cabbage",
        "broccoli",
        "cauliflower",
        "mushroom",
        "strawberry",
        "banana",
        "lemon",
        "orange",
        "fig",
        "pineapple",
        "gourd",
        "squash",
        "cucumber",
        "artichoke",
        "cardoon",
        "head cabbage",
        "acorn squash",
        "zucchini",
        "rapeseed",
        "sunflower",
        "thistle",
        "potato",
        "eggplant",
        "pepper",
        "bell pepper",
        "tomato",
        "pomegranate",
        "jackfruit",
        "custard apple",
    ]

    try:
        categories = w.meta.get("categories", None)
    except Exception:
        categories = None

    selected = []
    if isinstance(categories, (list, tuple)) and len(categories) == 1000:
        lower = [c.lower() for c in categories]
        for kw in keywords:
            kwl = kw.lower()
            for idx, name in enumerate(lower):
                if kwl in name:
                    selected.append(idx)
            if len(selected) >= 5:
                break

    if len(selected) < 5:
        fallback = [971, 938, 937, 940, 943]  # generic high-index picks; stable
        selected = (selected + fallback)[:5]
    else:
        selected = selected[:5]

    with torch.no_grad():
        model_5cls.fc.weight.copy_(ref.fc.weight[selected, :])
        model_5cls.fc.bias.copy_(ref.fc.bias[selected])

    return selected


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

selected_imagenet_rows = None
if model is None:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    selected_imagenet_rows = init_fc_from_imagenet_leafish(model)

model.to(device)
model.eval()

test_images_dir = find_test_images_dir()
sample_sub_path = find_sample_submission_path()

print("Device:", device)
print("Test images dir:", test_images_dir)
print("Sample submission path:", sample_sub_path)
print("Checkpoint used:", ckpt_path)
print(
    "Selected ImageNet fc rows for 5-class init (None if ckpt used):",
    selected_imagenet_rows,
)



## === cell 1
if sample_sub_path is not None:
    sample_df = pd.read_csv(sample_sub_path)
    image_ids = sample_df["image_id"].astype(str).tolist()
else:
    image_ids = sorted(
        [f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")]
    )

prediction = []
with torch.no_grad():
    for image_name in image_ids:
        img_path = os.path.join(test_images_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        image_tensor_py = main_model_preprocess(img).unsqueeze(0).to(device)
        output_py = model(image_tensor_py)

        pred = int(torch.argmax(output_py, dim=1).item())
        prediction.append(pred)

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
