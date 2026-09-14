# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.12182

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12182) has done: 'The failure is because the notebook references a private `/kaggle/input/vit_l_cassava/...` checkpoint that is not present in your environment, and the fallback search finds no `.pth` anywhere under `/kaggle/input`, so inference cannot start. To make the pipeline run end-to-end and still keep the core inference logic (ViT + argmax) intact, the minimal fix is to stop hard-failing when no checkpoint exists and instead load an official torchvision ViT-H pretrained weight (ImageNet) as a safe fallback. Since the torchvision ViT-H pretrained head is 1000-way, we keep your 5-class head and simply leave it randomly initialized when no cassava checkpoint is available; this yield a valid submission (score likely low, but at least it runs). The code also adds deterministic settings and ensures the submission matches `sample_submission.csv` ordering exactly.'
- What this solution (achieved 0.12182) has done: 'Your current score is far below the target because the model usually runs with a randomly initialized 5-class head (no cassava checkpoint found), which makes predictions close to random. The smallest change that legitimately improves accuracy (without changing the model architecture or inference semantics) is to add a lightweight “zero-training” head initialization using the provided `train.csv`: compute per-class mean features from the frozen ViT backbone and classify by nearest class mean (a simple prototypical classifier). This keeps the same ViT model and argmax decision rule, but replaces random head weights with data-driven weights derived from the competition’s training images. The rest of the pipeline (paths, transforms, CSV ordering/format) stays the same, and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 518
num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

val_transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 1
def _find_checkpoint_fallback(preferred_path: str) -> str | None:
    """
    Return None if no checkpoint is available under /kaggle/input.
    """
    if os.path.isfile(preferred_path):
        return preferred_path

    candidates = []
    for root, _, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn.lower().endswith(".pth"):
                fp = os.path.join(root, fn)
                score = 0
                low = fp.lower()
                if "cassava" in low:
                    score += 5
                if "vit" in low:
                    score += 3
                if "weight" in low or "model" in low:
                    score += 1
                candidates.append((score, fp))

    if not candidates:
        print(
            f"[WARN] Checkpoint not found at '{preferred_path}', and no .pth files were found under /kaggle/input. "
            "Falling back to torchvision pretrained ViT weights."
        )
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen = candidates[0][1]
    print(
        f"[WARN] Checkpoint not found at '{preferred_path}'. Using fallback checkpoint: {chosen}"
    )
    return chosen


def _load_state_dict_robust(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> None:
    """
    Handle checkpoints saved as raw state_dict or wrapped dict (e.g., {'state_dict': ...}),
    stripping common prefixes like 'module.' and 'model.'.
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
    else:
        raise RuntimeError(
            f"Unsupported checkpoint object type: {type(obj)} at {ckpt_path}"
        )

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys when loading checkpoint (showing up to 20): {missing[:20]}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading checkpoint (showing up to 20): {unexpected[:20]}"
        )


def _vit_forward_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Minimal feature extractor for torchvision ViT:
    returns the embedding before the classification head.
    This preserves the same backbone; we only reuse its penultimate features.
    """
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if isinstance(feats, dict):
            for key in ("x", "features", "last_hidden_state"):
                if key in feats:
                    feats = feats[key]
                    break
            else:
                raise RuntimeError(
                    f"Unsupported forward_features dict keys: {list(feats.keys())}"
                )
        return feats
    raise RuntimeError("This torchvision ViT model does not expose forward_features().")


@torch.no_grad()
def _compute_class_prototypes(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    train_dir: str,
    tfm,
    device: torch.device,
    max_per_class: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Score-improving change (core logic preserved): when no cassava checkpoint is available,
    build a simple nearest-class-mean (prototype) classifier from training images using the frozen ViT backbone.
    This avoids a random 5-class head and typically moves accuracy far upward toward the target.

    Returns:
      prototypes: [C, D] normalized mean features
      counts: [C] number of samples used per class
    """
    model.eval()

    rng = np.random.default_rng(seed)

    feats_sum = None
    counts = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        sub = train_df[train_df["label"] == c]
        if len(sub) == 0:
            continue
        take = min(max_per_class, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        chosen = sub.iloc[idx]["image_id"].tolist()

        for image_id in chosen:
            image_path = os.path.join(train_dir, image_id)
            if not os.path.isfile(image_path):
                continue
            img = Image.open(image_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            f = _vit_forward_features(model, x).detach()

            if f.ndim != 2 or f.shape[0] != 1:
                raise RuntimeError(f"Unexpected feature shape: {tuple(f.shape)}")

            if feats_sum is None:
                feats_sum = torch.zeros(
                    (num_classes, f.shape[1]), device=device, dtype=f.dtype
                )

            feats_sum[c] += f[0]
            counts[c] += 1

    if feats_sum is None or (counts == 0).any():
        print(
            "[WARN] Prototype computation incomplete; falling back to model head predictions."
        )
        return None, counts.to(device)

    prototypes = feats_sum / counts.to(device).unsqueeze(1)
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    return prototypes, counts.to(device)




## === cell 2
resolved_model_path = _find_checkpoint_fallback(model_path)

if resolved_model_path is None:
    try:
        weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    except Exception:
        weights = models.ViT_H_14_Weights.DEFAULT

    model = models.vit_h_14(weights=weights, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    print(
        "[INFO] Using torchvision pretrained backbone; 5-class head is randomly initialized (no cassava checkpoint)."
    )
else:
    model = models.vit_h_14(weights=None, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    _load_state_dict_robust(model, resolved_model_path, device)
    print(f"[INFO] Loaded cassava checkpoint from: {resolved_model_path}")

model.to(device)
model.eval()

prototypes = None
if resolved_model_path is None:
    train_df = pd.read_csv(train_csv_path)
    train_df["image_id"] = train_df["image_id"].astype(str)
    train_df["label"] = train_df["label"].astype(int)

    prototypes, proto_counts = _compute_class_prototypes(
        model=model,
        train_df=train_df,
        train_dir=train_data_directory,
        tfm=val_transforms,
        device=device,
        max_per_class=200,  # small, fast, and usually enough to move score a lot toward target
    )
    if prototypes is not None:
        print(
            "[INFO] Using prototype-based classifier (nearest class mean in ViT feature space)."
        )
        print(
            "[INFO] Prototype counts per class:", proto_counts.detach().cpu().tolist()
        )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/2374698243.py in <cell line: 0>()
     32     train_df["label"] = train_df["label"].astype(int)
     33 
---> 34     prototypes, proto_counts = _compute_class_prototypes(
     35         model=model,
     36         train_df=train_df,

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_56/3176460540.py in _compute_class_prototypes(model, train_df, train_dir, tfm, device, max_per_class)
    142             img = Image.open(image_path).convert("RGB")
    143             x = tfm(img).unsqueeze(0).to(device)
--> 144             f = _vit_forward_features(model, x).detach()
    145 
    146             if f.ndim != 2 or f.shape[0] != 1:

/tmp/ipykernel_56/3176460540.py in _vit_forward_features(model, x)
     98                 )
     99         return feats
--> 100     raise RuntimeError("This torchvision ViT model does not expose forward_features().")
    101 
    102 

RuntimeError: This torchvision ViT model does not expose forward_features().

## === cell 3
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].astype(str).tolist()

missing_files = [
    img_id
    for img_id in test_image_ids
    if not os.path.isfile(os.path.join(test_data_directory, img_id))
]
if missing_files:
    raise FileNotFoundError(
        f"Some test images listed in sample_submission are missing under '{test_data_directory}'. "
        f"Missing count={len(missing_files)}. Example: {missing_files[:5]}"
    )

test_predictions = []

for image_id in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_id)
    image = Image.open(image_path).convert("RGB")
    x = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if prototypes is None:
            output = model(x)
            predicted_class = int(output.argmax(dim=1).item())
        else:
            f = _vit_forward_features(model, x)
            f = torch.nn.functional.normalize(f, dim=1)  # [1, D]
            sims = f @ prototypes.T  # [1, C]
            predicted_class = int(sims.argmax(dim=1).item())

    test_predictions.append(predicted_class)

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": test_predictions})
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
