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
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

0.6128739800543971

# 6. Current score

0.75187

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08034) has done: 'I fix the missing-weight crash by making the script robust to the dataset environment: it search for the provided `.pth` file if present, otherwise fall back to an ImageNet-pretrained EfficientNet-B0 with the same head (so it still runs end-to-end and yields a valid submission). I also fix the CUDA/CPU dtype mismatch by ensuring the loaded state dict (or fallback model) is moved onto the same `device` before inference. Finally, I guarantee that `prediction` length matches `image_ids` by appending a safe fallback label if any image fails to load, preventing the submission निर्माण error and ensuring `submission.csv` is always written.'
- What this solution (achieved 0.20777) has done: 'Your current score is extremely low because the fallback model uses an untrained 5-class head and your preprocessing normalization doesn’t match EfficientNet’s ImageNet training, so predictions are nearly random. To move the score toward the target with minimal semantic changes, I (1) switch preprocessing to EfficientNet-B0’s official ImageNet transforms (size + normalization) and (2) ensure the ImageNet-pretrained weights are always used for the backbone (still the same architecture/head), only replacing the classifier head to 5 classes as you already do. This keeps the same inference-only pipeline and submission semantics, but should substantially increase accuracy compared to the current mismatch/random-head behavior. The rest (paths, ordering via sample_submission, CSV writing) stays the same to preserve stability.'
- What this solution (achieved 0.20777) has done: 'Your score is far below the target because the model is effectively doing near-random 5-class classification when the custom finetuned `.pth` weights aren’t found/loaded, since the classifier head is newly initialized. To move accuracy upward with minimal change while preserving your inference-only EfficientNet-B0 pipeline, I make the checkpoint loading more tolerant by allowing partial loading when the head shape doesn’t match (common when checkpoints were saved with a different classifier definition), so we can still reuse the finetuned backbone weights instead of falling back to ImageNet features. I also add a small key-remapping for common Lightning-style prefixes (`model.`/`net.`) to improve the chance of matching keys without changing architecture. This should legitimately increase performance toward the target while keeping the same model, preprocessing, and submission semantics.'
- What this solution (achieved 0.75187) has done: 'Your score is far below the target because when the finetuned checkpoint is missing/incompatible, the 5-class classifier head is randomly initialized, making predictions close to random. To move accuracy upward with minimal disruption to your current inference-only EfficientNet-B0 pipeline, I keep the exact same architecture and preprocessing but add a tiny, deterministic “fallback head fit” step: run the frozen backbone once over the provided `train.csv` images to extract features, then train only the final `Linear` layer with a few fast epochs. This preserves the core logic (EfficientNet-B0 + linear 5-class head + argmax) while turning the head from random into meaningful, which should move the score substantially toward your target. The submission ordering and CSV schema are kept unchanged.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

effnet_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
main_model_preprocess = effnet_weights.transforms()

model = models.efficientnet_b0(weights=effnet_weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 5)

preferred_weights_path = "/kaggle/input/efficient_60_512x512/pytorch/default/1/best_model_Efficient_60_512x512.pth"
weights_path = None
if os.path.isfile(preferred_weights_path):
    weights_path = preferred_weights_path
else:
    target_name = os.path.basename(preferred_weights_path)
    for root, _, files in os.walk("/kaggle/input"):
        if target_name in files:
            weights_path = os.path.join(root, target_name)
            break


def _extract_state_dict(state_obj):
    if (
        isinstance(state_obj, dict)
        and "state_dict" in state_obj
        and isinstance(state_obj["state_dict"], dict)
    ):
        return state_obj["state_dict"]
    return state_obj


def _strip_prefix(k: str) -> str:
    for pref in ("module.", "model.", "net."):
        if k.startswith(pref):
            return k[len(pref) :]
    return k


loaded_full_head = False

weights_info = "custom weights not found; using ImageNet-pretrained EfficientNet-B0 backbone with new 5-class head"
if weights_path is not None and os.path.isfile(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(state)

    if isinstance(state, dict):
        cleaned = {_strip_prefix(k): v for k, v in state.items()}
        state = cleaned

        try:
            model.load_state_dict(state, strict=True)
            weights_info = f"custom weights loaded strictly from: {weights_path}"
            loaded_full_head = True
        except Exception as e:
            model_sd = model.state_dict()
            filtered = {}
            for k, v in state.items():
                if (
                    k in model_sd
                    and hasattr(v, "shape")
                    and hasattr(model_sd[k], "shape")
                    and v.shape == model_sd[k].shape
                ):
                    filtered[k] = v
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            weights_info = (
                f"custom weights partially loaded from: {weights_path} "
                f"(kept {len(filtered)}/{len(state)} tensors; strict-load failed: {type(e).__name__})"
            )
            if len(filtered) == 0:
                print(
                    "Warning: partial-load kept 0 tensors; checkpoint likely incompatible. Falling back effectively to ImageNet + new head."
                )
            else:
                print(
                    f"Partial-load diagnostics: missing={len(missing)}, unexpected={len(unexpected)}"
                )
            loaded_full_head = ("classifier.1.weight" in filtered) and (
                "classifier.1.bias" in filtered
            )

model.to(device)
model.eval()

print("Device:", device)
print(weights_info)
print("Head loaded from checkpoint:", loaded_full_head)




## === cell 1
test_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
]
test_dir = None
for p in test_dir_candidates:
    if os.path.isdir(p):
        test_dir = p
        break
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {test_dir_candidates}"
    )

sample_sub_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_sub_path = None
for p in sample_sub_candidates:
    if os.path.isfile(p):
        sample_sub_path = p
        break

if sample_sub_path is not None:
    sample_submission = pd.read_csv(sample_sub_path)
    image_ids = sample_submission["image_id"].tolist()
else:
    image_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])

print("Test dir:", test_dir)
print("Num test images:", len(image_ids))
if sample_sub_path:
    print("Using sample_submission ordering from:", sample_sub_path)




## === cell 2
train_csv_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
]
train_img_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
]

train_csv_path = next((p for p in train_csv_candidates if os.path.isfile(p)), None)
train_img_dir = next((p for p in train_img_dir_candidates if os.path.isdir(p)), None)


def _fit_head_on_train_if_needed(model: nn.Module) -> None:
    global loaded_full_head
    if loaded_full_head:
        return
    if train_csv_path is None or train_img_dir is None:
        print(
            "Train data not found; cannot fit head. Proceeding with random head (may score low)."
        )
        return

    t0 = time.time()
    df = pd.read_csv(train_csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        print(
            "train.csv schema unexpected; cannot fit head. Proceeding without head-fit."
        )
        return

    MAX_TRAIN_IMAGES = 6000
    df = df.iloc[: min(len(df), MAX_TRAIN_IMAGES)].reset_index(drop=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier[1].parameters():
        p.requires_grad = True

    features = []
    labels = []
    failed = 0

    with torch.no_grad():
        for i, row in df.iterrows():
            img_path = os.path.join(train_img_dir, str(row["image_id"]))
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    x = main_model_preprocess(im).unsqueeze(0).to(device)
                f = model.features(x)
                f = model.avgpool(f)
                f = torch.flatten(f, 1).detach().cpu()
                features.append(f)
                labels.append(int(row["label"]))
            except Exception:
                failed += 1
            if (i + 1) % 1000 == 0:
                print(f"Head-fit: extracted {i+1}/{len(df)} train features")

    if len(features) < 100:
        print(
            f"Head-fit: too few features ({len(features)}), failed={failed}. Skipping head-fit."
        )
        return

    X = torch.cat(features, dim=0)  # [N, 1280]
    y = torch.tensor(labels, dtype=torch.long)

    head = nn.Linear(X.shape[1], 5)
    head.load_state_dict(
        {
            k.replace("classifier.1.", ""): v.detach().cpu()
            for k, v in model.classifier[1].state_dict().items()
        }
    )
    head.train()

    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 256
    epochs = 6

    for ep in range(epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size]
            yb = y[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(xb)
            total += len(xb)
            correct += int((logits.argmax(1) == yb).sum().item())

        print(
            f"Head-fit epoch {ep+1}/{epochs}: loss={total_loss/total:.4f} acc={correct/total:.4f}"
        )

    model.classifier[1].load_state_dict(
        {k: v.to(device) for k, v in head.state_dict().items()}
    )
    loaded_full_head = True
    model.eval()
    print(
        f"Head-fit done on {len(X)} samples (failed loads={failed}) in {time.time()-t0:.1f}s"
    )


_fit_head_on_train_if_needed(model)




## === cell 3
prediction = []
t0 = time.time()

DEFAULT_LABEL_ON_ERROR = 0

with torch.no_grad():
    for i, image_name in enumerate(image_ids):
        img_path = os.path.join(test_dir, image_name)
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                x = main_model_preprocess(im).unsqueeze(0).to(device)

            out = model(x)
            pred = int(torch.argmax(out, dim=1).item())
        except Exception as e:
            pred = DEFAULT_LABEL_ON_ERROR
            if (i < 3) or ((i + 1) % 500 == 0):
                print(
                    f"Warning: failed on {image_name} ({e}); using label={DEFAULT_LABEL_ON_ERROR}"
                )

        prediction.append(pred)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"Inferred {i+1}/{len(image_ids)} images in {elapsed:.1f}s")

print(f"Done inference on {len(image_ids)} images in {time.time()-t0:.1f}s")
print("Predictions:", len(prediction), "Image IDs:", len(image_ids))




## === cell 4
if len(prediction) != len(image_ids):
    raise RuntimeError(
        f"Prediction length mismatch: {len(prediction)} preds vs {len(image_ids)} image_ids"
    )

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

if submission.isna().any().any():
    raise ValueError("Submission contains NaNs.")
if len(submission) == 0:
    raise ValueError("Empty submission produced.")
if submission["image_id"].duplicated().any():
    submission = submission.drop_duplicates(
        subset=["image_id"], keep="first"
    ).reset_index(drop=True)

submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
