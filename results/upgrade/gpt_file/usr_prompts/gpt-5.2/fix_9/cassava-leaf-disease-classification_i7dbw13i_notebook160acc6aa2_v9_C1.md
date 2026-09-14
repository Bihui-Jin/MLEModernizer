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

3.11

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
sentence-transformers==4.1.0
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
timm==1.0.19
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
transformers==4.53.3

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

0.8827440314294349

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime errors so the notebook runs end-to-end and reliably writes `submission.csv`. The main blockers are (1) a protobuf/TensorFlow import crash (not needed for this PyTorch-only inference pipeline) and (2) using Python sets for `DataFrame(columns=...)`, which pandas rejects. I also make model weight loading more robust (`map_location` + common checkpoint key handling), ensure `image_id` ordering matches `sample_submission.csv`, and guard against missing optional model files so you still get a valid submission. These changes are execution/correctness focused and should improve score vs. a broken pipeline by enabling the intended ensembling to run.'
- What this solution (achieved 0.05531) has done: 'I fix the protobuf/Transformers import crash by removing the Transformers dependency (it’s not needed for a valid end-to-end run and currently breaks at import time in this environment). To move accuracy toward the target, I also fix a calibration/logic issue where ViT and MobileNet probabilities were unintentionally down-weighted by `* 0.5`, which severely hurts the ensemble. Finally, I make checkpoint loading slightly more robust (handle `module.` prefixes and non-strict loading for MobileNet if needed) while keeping the same model architectures and inference flow, and ensure the submission keeps the exact `sample_submission.csv` ordering.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.88274), so we should make the smallest changes that restore the intended ensemble signal without changing the modeling approach. The biggest issue is that the MobileNet head is left at the default `num_classes=1000`, then you slice `output[:, :5]`, which makes predictions essentially meaningless; changing MobileNet to `num_classes=5` preserves the architecture but fixes the classification head to match the task. I also add an explicit float32 stacking of probability vectors and a safe fallback for any missing probs to avoid NaNs silently turning into class 0. These changes keep the same inference/ensemble flow but should move accuracy sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the smallest high-impact fix is to ensure the per-image probability vectors actually survive the pandas merge intact. Right now you store `np.ndarray` objects inside DataFrame cells and then merge; this commonly turns them into `object` cells that can become missing/NaN-like and silently fall back to all-zeros in your ensemble, producing near-constant predictions and very low accuracy. I change the prediction storage to write 5 explicit float columns per model (`resnext_0..4`, `mobilenet_0..4`) and ensemble those columns directly (same averaging/summing logic, just robustly represented). This preserves the same models, transforms, inference loops, and “sum probs then argmax” semantics, but prevents the merge/object/NaN issue that is likely causing the 0.055 score.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so the smallest likely high-impact improvement is to restore the missing ViT leg of your ensemble (it is currently entirely skipped), while keeping the same “sum probs then argmax” semantics. I implement ViT inference using `timm` (already installed) instead of `transformers`, load weights as robustly as your other models, and output 5 explicit `vit_0..4` probability columns to avoid any object/merge issues. I also align ViT preprocessing to standard ImageNet normalization and use a reasonable fixed input size (384) commonly used for ViT models, without changing the overall pipeline structure. This should substantially increase accuracy toward the target by adding back a strong model, while leaving your ResNeXt/MobileNet logic intact.'
- What this solution (achieved 0.05531) has done: 'Your score (0.055) suggests predictions are effectively misaligned or near-constant; the most likely minimal, high-impact fix is to ensure every model uses the exact same input normalization it was trained with, because your MobileNet currently uses `mean=std=0.5` while the other legs use ImageNet stats. I switch MobileNet’s validation normalization to ImageNet mean/std (keeping the same MobileNet architecture and inference loop), which typically restores meaningful logits and should move accuracy sharply upward toward your target. I also add a small safeguard to verify probability rows sum to ~1 (debug-only print) and ensure merges preserve the `sample_submission.csv` order (already mostly correct), without changing the ensemble semantics (still plain sum then argmax). No training, no new models, no architectural changes—just correcting preprocessing to match expected evaluation behavior.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so we should make a minimal, high-impact correctness fix rather than tuning. The most likely cause of ~0.055 accuracy here is that the ViT branch is loaded with the wrong head / mismatched checkpoint format, so it either fails silently (all-random) or contributes harmful probabilities to the ensemble. I keep the exact same ensemble semantics (sum probabilities then argmax), but I (1) make ViT weight loading correctly handle checkpoints that include a classifier head of a different size by dropping mismatched head keys, and (2) add a very small “gate” that includes a model in the ensemble only if its probabilities look valid (finite and ~sum to 1 on average), preventing a broken leg from dragging the whole submission down. This preserves your architecture choices and inference flow and should move accuracy sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the most likely “small but high-impact” fix is correcting a preprocessing mismatch: your ResNeXt branch uses `Resize(512,512)` even though these cassava ResNeXt checkpoints are almost always trained with 224/256-ish crops and ImageNet normalization. I keep the exact same model, checkpoint loading, inference loop, and ensemble semantics, but change only the ResNeXt test-time transform to `Resize(256,256)` (still deterministic, no cropping/augmentation). This should make the ResNeXt probabilities meaningful again and move accuracy sharply upward toward your target. I also add a small safeguard to assert the number of produced predictions matches `sample_submission.csv` to prevent silent misalignment.'

# 9. Code solution

## === cell 0
import os
import random
import json
import gc
import math

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
image_path = os.path.join(path, "test_images") + "/"

sample_path = os.path.join(path, "sample_submission.csv")
submission_df = pd.read_csv(sample_path)

assert set(submission_df.columns) == {"image_id", "label"}
print("n_test:", len(submission_df))
print(submission_df.head())



## === cell 2
used_models_pytorch = {
    "resnext": [f"../input/models/resnext50_32x4d_fold{fold}_best.pth" for fold in [1]],
    "vit": "../input/model-vit/original_save_pretrained",
    "mobilenet": "../input/model-mobilenet/mn3_bt20_ep5_lr1.pth",
}


def _exists_any(paths):
    if isinstance(paths, (list, tuple)):
        return all(os.path.exists(p) for p in paths)
    return os.path.exists(paths)


available_models = {k: v for k, v in used_models_pytorch.items() if _exists_any(v)}
missing_models = [k for k in used_models_pytorch.keys() if k not in available_models]
print("available_models (files exist):", list(available_models.keys()))
print("missing_models:", missing_models)




## === cell 3
def load_checkpoint_state_dict(ckpt):
    """
    Handle common checkpoint formats:
    - {'model': state_dict}
    - {'state_dict': state_dict}
    - raw state_dict
    Also strips 'model.' and 'module.' prefixes when present.
    """
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_sd[nk] = v
        return new_sd
    return sd


def drop_mismatched_head_keys(
    state_dict, model, head_key_prefixes=("head.", "classifier.", "fc.")
):
    if not isinstance(state_dict, dict):
        return state_dict
    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state_dict.items():
        if k in model_sd:
            if (
                hasattr(v, "shape")
                and hasattr(model_sd[k], "shape")
                and tuple(v.shape) != tuple(model_sd[k].shape)
            ):
                dropped += 1
                continue
            filtered[k] = v
        else:
            if any(k.startswith(p) for p in head_key_prefixes):
                dropped += 1
                continue
            filtered[k] = v
    return filtered, dropped




## === cell 4
class CustomResNext(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 5)

    def forward(self, x):
        return self.model(x)


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.file_names = self.df["image_path_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = cv2.imread(file_name)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {file_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image


def get_transforms_resnext():
    return Compose(
        [
            Resize(256, 256),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def inference_resnext(model, state_dicts, test_loader, device):
    model.to(device)
    probabilities = []
    for images in tqdm(test_loader, desc="resnext infer"):
        images = images.to(device, non_blocking=True)
        avg_preds = []
        for sd in state_dicts:
            model.load_state_dict(sd, strict=True)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).detach().cpu().numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probabilities.append(avg_preds)
    return np.concatenate(probabilities, axis=0)




## === cell 5
predictions_resnext = None

if "resnext" in available_models:
    predictions_resnext = pd.DataFrame({"image_id": submission_df["image_id"].values})
    predictions_resnext["image_path_id"] = image_path + predictions_resnext[
        "image_id"
    ].astype(str)

    model = CustomResNext("resnext50_32x4d", pretrained=False)

    ckpts = [torch.load(f, map_location="cpu") for f in available_models["resnext"]]
    state_dicts = [load_checkpoint_state_dict(c) for c in ckpts]

    test_dataset = TestDataset(predictions_resnext, transform=get_transforms_resnext())
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    preds = inference_resnext(model, state_dicts, test_loader, device)

    assert preds.shape[0] == len(
        submission_df
    ), f"resnext preds rows {preds.shape[0]} != n_test {len(submission_df)}"

    for c in range(5):
        predictions_resnext[f"resnext_{c}"] = preds[:, c].astype(np.float32)

    predictions_resnext = predictions_resnext.drop(columns=["image_path_id"])

    torch.cuda.empty_cache()
    del model, ckpts, state_dicts, test_dataset, test_loader, preds
    gc.collect()



## === cell 6
predictions_vit = None


def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


if "vit" in available_models:
    vit_dir = available_models["vit"]

    vit_ckpt_path = None
    if os.path.isdir(vit_dir):
        vit_ckpt_path = _find_first_existing(
            [
                os.path.join(vit_dir, "pytorch_model.bin"),
                os.path.join(vit_dir, "model.pth"),
                os.path.join(vit_dir, "model.bin"),
                os.path.join(vit_dir, "checkpoint.pth"),
                os.path.join(vit_dir, "vit.pth"),
            ]
        )
    elif os.path.isfile(vit_dir):
        vit_ckpt_path = vit_dir

    if vit_ckpt_path is None:
        print("vit path exists but no known checkpoint file found; skipping vit.")
        predictions_vit = None
    else:
        print("vit checkpoint:", vit_ckpt_path)

        vit_model = None
        vit_last_err = None
        vit_input_size = None

        for vit_name, vit_img_size in [
            ("vit_base_patch16_384", 384),
            ("vit_base_patch16_224", 224),
        ]:
            try:
                vit_model = timm.create_model(vit_name, pretrained=False, num_classes=5)
                ckpt = torch.load(vit_ckpt_path, map_location="cpu")
                sd = load_checkpoint_state_dict(ckpt)

                sd, dropped = drop_mismatched_head_keys(sd, vit_model)
                if dropped:
                    print(f"vit: dropped {dropped} mismatched head key(s) before load")

                missing, unexpected = vit_model.load_state_dict(sd, strict=False)
                print(
                    f"vit model={vit_name} img_size={vit_img_size} "
                    f"load_state_dict strict=False; missing={len(missing)} unexpected={len(unexpected)}"
                )
                vit_model.eval()
                vit_model.to(device)
                vit_input_size = vit_img_size
                break
            except Exception as e:
                vit_last_err = e
                vit_model = None

        if vit_model is None:
            print(
                "vit could not be constructed/loaded with tried configs; last error:",
                repr(vit_last_err),
            )
            predictions_vit = None
        else:
            vit_transform = Compose(
                [
                    Resize(vit_input_size, vit_input_size),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

            predictions_vit = pd.DataFrame(
                {"image_id": submission_df["image_id"].values}
            )
            predictions_vit["image_path_id"] = image_path + predictions_vit[
                "image_id"
            ].astype(str)

            vit_dataset = TestDataset(predictions_vit, transform=vit_transform)
            vit_loader = DataLoader(
                vit_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

            vit_probs = []
            for images in tqdm(vit_loader, desc="vit infer"):
                images = images.to(device, non_blocking=True)
                with torch.no_grad():
                    logits = vit_model(images)
                    probs = logits.softmax(1).detach().cpu().numpy()
                vit_probs.append(probs)
            vit_probs = np.concatenate(vit_probs, axis=0).astype(np.float32)

            assert vit_probs.shape[0] == len(
                submission_df
            ), f"vit probs rows {vit_probs.shape[0]} != n_test {len(submission_df)}"

            for c in range(5):
                predictions_vit[f"vit_{c}"] = vit_probs[:, c].astype(np.float32)
            predictions_vit = predictions_vit.drop(columns=["image_path_id"])

            torch.cuda.empty_cache()
            del vit_model, vit_dataset, vit_loader, vit_probs, ckpt, sd
            gc.collect()



## === cell 7
predictions_mobilenet = None

if "mobilenet" in available_models:
    __all__ = ["mobilenetv3_large", "mobilenetv3_small"]

    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super(h_sigmoid, self).__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super(h_swish, self).__init__()
            self.sigmoid = h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    class SELayer(nn.Module):
        def __init__(self, channel, reduction=4):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y

    def conv_3x3_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            h_swish(),
        )

    def conv_1x1_bn(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
        )

    class InvertedResidual(nn.Module):
        def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
            super(InvertedResidual, self).__init__()
            assert stride in [1, 2]
            self.identity = stride == 1 and inp == oup

            if inp == hidden_dim:
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        (kernel_size - 1) // 2,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        (kernel_size - 1) // 2,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )

        def forward(self, x):
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)

    class MobileNetV3(nn.Module):
        def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.0):
            super(MobileNetV3, self).__init__()
            self.cfgs = cfgs
            assert mode in ["large", "small"]

            input_channel = _make_divisible(16 * width_mult, 8)
            layers = [conv_3x3_bn(3, input_channel, 2)]
            block = InvertedResidual
            for k, t, c, use_se, use_hs, s in self.cfgs:
                output_channel = _make_divisible(c * width_mult, 8)
                exp_size = _make_divisible(input_channel * t, 8)
                layers.append(
                    block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
                )
                input_channel = output_channel
            self.features = nn.Sequential(*layers)
            self.conv = conv_1x1_bn(input_channel, exp_size)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            output_channel = {"large": 1280, "small": 1024}
            output_channel = (
                _make_divisible(output_channel[mode] * width_mult, 8)
                if width_mult > 1.0
                else output_channel[mode]
            )
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, output_channel),
                h_swish(),
                nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes),
            )
            self._initialize_weights()

        def forward(self, x):
            x = self.features(x)
            x = self.conv(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def mobilenetv3_large(**kwargs):
        cfgs = [
            [3, 1, 16, 0, 0, 1],
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1],
        ]
        return MobileNetV3(cfgs, mode="large", **kwargs)

    class LeafDatasetMN(torch.utils.data.Dataset):
        def __init__(self, df, data_path, mode="train", transforms=None):
            super().__init__()
            self.df_data = df.values
            self.data_path = data_path
            self.transforms = transforms
            self.data_dir = "train_images" if mode == "train" else "test_images"

        def __len__(self):
            return len(self.df_data)

        def __getitem__(self, index):
            img_name = self.df_data[index][0]
            img_path = os.path.join(self.data_path, self.data_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img

    def predict_mobilenet(model, test_dataset):
        preds = []
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=20,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        model.eval()
        for test_images in tqdm(test_dataloader, desc="mobilenet infer"):
            test_images = test_images.to(device, non_blocking=True)
            with torch.no_grad():
                output = model(test_images)
            preds.extend(output.softmax(1).detach().cpu().numpy())
        return np.asarray(preds, dtype=np.float32)

    transforms_val_mn = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    predictions_mobilenet = pd.DataFrame({"image_id": submission_df["image_id"].values})

    model = mobilenetv3_large(num_classes=5)
    model.to(device)

    ckpt = torch.load(available_models["mobilenet"], map_location="cpu")
    sd = load_checkpoint_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        print(
            f"mobilenet load_state_dict strict=False; missing={len(missing)} unexpected={len(unexpected)}"
        )

    test_dataset = LeafDatasetMN(
        df=predictions_mobilenet,
        data_path=path,
        mode="test",
        transforms=transforms_val_mn,
    )
    predictions_raw_mobilenet = predict_mobilenet(model, test_dataset)

    assert predictions_raw_mobilenet.shape[0] == len(
        submission_df
    ), f"mobilenet preds rows {predictions_raw_mobilenet.shape[0]} != n_test {len(submission_df)}"

    for c in range(5):
        predictions_mobilenet[f"mobilenet_{c}"] = predictions_raw_mobilenet[
            :, c
        ].astype(np.float32)

    torch.cuda.empty_cache()
    del model, test_dataset, predictions_raw_mobilenet, ckpt, sd
    gc.collect()




## === cell 8
def prob_block_is_valid(df, cols, name, sample_n=256):
    block = df[cols].to_numpy(dtype=np.float32, copy=True)
    if block.shape[0] == 0:
        print(f"{name}: empty block -> invalid")
        return False
    if not np.isfinite(block).all():
        print(f"{name}: non-finite values -> invalid")
        return False
    n = min(sample_n, block.shape[0])
    row_sums = block[:n].sum(axis=1)
    mean_sum = float(np.mean(row_sums))
    if not (0.95 <= mean_sum <= 1.05):
        print(f"{name}: mean prob row-sum={mean_sum:.4f} (expected ~1) -> invalid")
        return False
    return True


submission_pred = submission_df.copy()
submission_pred["label"] = 0  # placeholder

if predictions_resnext is not None:
    submission_pred = submission_pred.merge(
        predictions_resnext, on="image_id", how="left"
    )
if predictions_mobilenet is not None:
    submission_pred = submission_pred.merge(
        predictions_mobilenet, on="image_id", how="left"
    )
if predictions_vit is not None:
    submission_pred = submission_pred.merge(predictions_vit, on="image_id", how="left")

resnext_cols = [
    f"resnext_{i}" for i in range(5) if f"resnext_{i}" in submission_pred.columns
]
mobilenet_cols = [
    f"mobilenet_{i}" for i in range(5) if f"mobilenet_{i}" in submission_pred.columns
]
vit_cols = [f"vit_{i}" for i in range(5) if f"vit_{i}" in submission_pred.columns]

model_prob_cols = []
if len(resnext_cols) == 5 and prob_block_is_valid(
    submission_pred, resnext_cols, "resnext"
):
    model_prob_cols.append(resnext_cols)
if len(mobilenet_cols) == 5 and prob_block_is_valid(
    submission_pred, mobilenet_cols, "mobilenet"
):
    model_prob_cols.append(mobilenet_cols)
if len(vit_cols) == 5 and prob_block_is_valid(submission_pred, vit_cols, "vit"):
    model_prob_cols.append(vit_cols)

print(
    "merged prob groups (kept for ensemble):",
    {
        "resnext": resnext_cols if resnext_cols in model_prob_cols else [],
        "mobilenet": mobilenet_cols if mobilenet_cols in model_prob_cols else [],
        "vit": vit_cols if vit_cols in model_prob_cols else [],
    },
)

if len(model_prob_cols) == 0:
    submission_pred["label"] = 0
else:
    summed = np.zeros((len(submission_pred), 5), dtype=np.float32)
    for cols in model_prob_cols:
        block = submission_pred[cols].to_numpy(dtype=np.float32, copy=True)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        summed += block

    submission_pred["label"] = np.argmax(summed, axis=1).astype(int)

submission_pred = submission_df[["image_id"]].merge(
    submission_pred[["image_id", "label"]], on="image_id", how="left"
)
submission_pred["label"] = submission_pred["label"].fillna(0).astype(int)

print(submission_pred.head())



## === cell 9
out_path = "submission.csv"
submission_pred[["image_id", "label"]].to_csv(out_path, index=False)

print("wrote:", out_path, "rows:", len(submission_pred))
print(pd.read_csv(out_path).head())
print(
    "label value counts:\n",
    pd.read_csv(out_path)["label"].value_counts(dropna=False).sort_index(),
)
