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

0.8396796615291628

# 6. Current score

0.11099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the import-time crash caused by the protobuf/TensorFlow stack (your pipeline is PyTorch-only here, so we can safely remove the TensorFlow import that triggers the `MessageFactory.GetPrototype` error). Then I fix the pandas `DataFrame(columns={...})` bugs by using lists instead of sets, which currently prevents `submission_df` and `predictions_vit` from being created and cascades into the later `NameError`s. Finally, I make submission generation robust by aligning `image_id` ordering to `sample_submission.csv` and ensuring the prediction columns exist before ensembling, so a valid `submission.csv` is always written.'
- What this solution (achieved 0.13004) has done: 'I remove the TensorFlow/transformers import-time crash by avoiding `transformers` entirely (it is what pulls in the protobuf stack here) and keep the pipeline PyTorch-only as intended. Then I fix the ViT loading logic so it loads a local timm ViT checkpoint (or falls back to a standard timm ViT weights if the expected local directory is missing), which should move accuracy substantially toward the target instead of producing near-random outputs. Finally, I make submission generation robust by ensuring the test image order matches `sample_submission.csv`, and by safely ensembling available model probability vectors without pandas row-wise `apply` overhead or missing-column failures.'
- What this solution (achieved 0.1988) has done: 'The current score is far below the target, so we should increase accuracy with minimal risk while keeping your overall pipeline (timm ViT inference + optional ResNeXt + mean ensembling) unchanged. The biggest likely issue is that your ViT loader often falls back to generic ImageNet-pretrained weights (or partially-mismatched weights due to `strict=False`), which yields near-random/low-accuracy predictions on Cassava. I make checkpoint discovery and loading stricter and more compatible with common timm training wrappers (e.g., `model`, `model_state_dict`, `state_dict`, EMA), and only fall back to ImageNet weights if absolutely no checkpoint exists. I also ensure we use timm’s model-specific normalization/config (still same semantics: resize->tensor->normalize) to avoid a subtle train/test preprocessing mismatch that can materially hurt accuracy.'
- What this solution (achieved 0.24813) has done: 'Your current score (0.1988) is far below the target (0.83968), so we should increase accuracy with minimal, low-risk fixes while keeping your core approach (timm ViT inference + optional ResNeXt + mean ensembling) unchanged. The biggest likely issue is a preprocessing mismatch: you hardcode `Resize+Normalize(mean,std)` instead of using timm’s model-specific `create_transform` (which can include `center_crop_pct`, interpolation, and exact normalization), and this alone can crater ViT accuracy. Second, your ViT checkpoint loader may be silently loading an incompatible checkpoint (wrong head) and then using `strict=False`, leaving a random classification head; we (a) adapt head weights when shapes differ and (b) explicitly error only if we truly have no usable weights (still falling back to ImageNet as last resort). Finally, we add `inference_mode()` and ensure deterministic ordering stays aligned to `sample_submission.csv` (no semantic change, just safer execution).'
- What this solution (achieved 0.24813) has done: 'We keep your pipeline (timm ViT inference + optional ResNeXt + mean ensemble) unchanged, but fix the most likely cause of the very low accuracy: the ViT checkpoint is probably not being found/loaded, so you often fall back to ImageNet-pretrained weights with a random 5-class head. I add robust checkpoint discovery that searches both the provided directory and common Kaggle input mount points, and I correctly remap common checkpoint key patterns (including `head.*`/`fc.*` and `classifier.*`) so the 5-class head loads when present (instead of being dropped and left random). Finally, I ensure the ViT preprocessing uses timm’s resolved config but forces the correct `input_size` for this ViT variant to avoid accidental resize/crop mismatches. These are minimal changes focused on getting the intended fine-tuned weights + correct eval preprocessing, which should move accuracy substantially toward your target.'
- What this solution (achieved 0.24813) has done: 'The current score is far below the target, so we should increase accuracy with minimal, low-risk changes while preserving your core pipeline (timm ViT inference + optional ResNeXt + mean ensembling). The biggest likely issue is that the ViT is not actually loading a Cassava-finetuned checkpoint (so predictions are effectively near-random); I make checkpoint discovery search common Kaggle input locations and, critically, allow loading when the checkpoint head shape differs by auto-dropping only the classifier weights (still the same model/forward/inference logic). Next, I ensure we use timm’s own inference preprocessing from the resolved config (including crop/interpolation) without forcing an incompatible `input_size`, which can significantly affect ViT accuracy. Finally, I keep submission ordering aligned to `sample_submission.csv` and keep the ensembling logic identical, just making it more robust to missing predictions.'
- What this solution (achieved 0.1293) has done: 'We keep your inference-only timm ViT + optional ResNeXt + mean-ensemble logic unchanged, and focus on the most likely reason the score is stuck near random: the ViT checkpoint is being loaded but the classification head often remains random due to key mismatches (e.g., `head` vs `head_dist`) and shape incompatibilities. I extend the checkpoint key remapping to handle common DeiT/ViT distilled heads and also copy `head_dist.*` into `head.*` when appropriate, so the 5-class head weights are actually used instead of silently dropped. Additionally, I ensure we always use the model’s resolved inference transform config directly from the final model (not a temporary model) to avoid subtle config mismatches. These are minimal changes aimed at moving accuracy substantially upward toward your target without changing architecture, loops, or ensembling semantics.'
- What this solution (achieved 0.12519) has done: 'Your score is far below the target, so we should improve accuracy by fixing the most likely inference bug while keeping your overall approach (timm ViT inference + optional ResNeXt + mean ensembling) unchanged. The biggest issue is that your ViT checkpoint loader may be loading weights but leaving the classifier head random due to key/shape mismatches (very common with Cassava DeiT/ViT distilled checkpoints), which drives accuracy toward chance. I minimally extend the checkpoint remapping to correctly handle `head_dist.*` vs `head.*` and also handle checkpoints that store weights under nested dict keys; this should let the actual 5-class head weights load when present. Finally, I make the ViT transform use timm’s resolved config but explicitly enforce `crop_pct=1.0` (no center-crop) to better match how Cassava models are typically trained, without changing model architecture or the inference loop.'
- What this solution (achieved 0.1293) has done: 'Your current score is far below the target, so the most likely issue is that the ViT is not actually using the fine-tuned Cassava checkpoint and/or is applying an inference transform that doesn’t match how the checkpoint was trained, leading to near-random accuracy. I keep your exact modeling/inference approach (timm ViT forward + softmax + optional mean-ensemble) but make the ViT checkpoint loading stricter and more compatible: properly handle common timm/DeiT checkpoint key patterns (including `head_dist.*`) and, critically, automatically resize/interpolate the position embedding if the checkpoint was trained at a different input resolution. I also stop forcing `crop_pct=1.0` and instead use timm’s resolved inference config as-is (this is a minimal preprocessing fix that often matters a lot for ViT). These are small, targeted changes intended to move accuracy substantially upward toward your target without changing the core pipeline.'
- What this solution (achieved 0.60688) has done: 'The current score is far below the target, so we should increase accuracy by ensuring the model is actually producing meaningful Cassava predictions rather than effectively random outputs. The most likely issue is that `used_models_pytorch` is overwritten to only include `"vit"` pointing to a directory that often has no valid checkpoint, causing a fallback to ImageNet-pretrained weights with a randomly-initialized 5-class head (very low accuracy). I (1) stop overwriting `used_models_pytorch`, (2) make the ViT loader search for a usable checkpoint under common Kaggle input locations (including your `../input/models` area) and prefer Cassava-like checkpoints, and (3) if we still must fall back to ImageNet weights, at least keep a valid head by using timm’s `reset_classifier(num_classes=5)` after loading pretrained features (same architecture/inference semantics, just avoids a random head). These are minimal, targeted fixes that should move accuracy substantially toward your target without changing the overall pipeline (timm model inference + softmax + mean ensembling + submission alignment).'
- What this solution (achieved 0.11099) has done: 'We keep your exact inference-and-mean-ensemble pipeline, but fix the most score-critical issue: the ViT checkpoint path is almost certainly not found (so you fall back to ImageNet weights with a random 5-class head), which caps accuracy around where you are now. The smallest safe change is to (1) also treat `used_models_pytorch["vit"]` as a *prefix* and search for any checkpoint files under that directory (recursively) and (2) prefer Cassava-like finetuned checkpoints by scoring filenames, so we actually load a 5-class head when available. Additionally, we ensure the ViT model is created as the distilled variant if the checkpoint contains `head_dist.*`, which is a common Cassava DeiT setup and otherwise leads to a mismatched/random head. These changes preserve architecture/inference semantics (timm model forward + softmax + mean ensembling) and should move accuracy materially toward your target.'
- What this solution (achieved 0.11099) has done: 'Your current score is far below the target, so we should increase accuracy by fixing the most likely remaining inference bug while preserving your exact model/inference/ensembling structure. The highest-impact minimal fix is to ensure the ViT preprocessing is correct for PIL input: `create_transform` can include a final normalization step expecting a tensor, but without `ToTensor()` in the pipeline you can end up feeding a PIL image through an incomplete transform (or a transform that doesn’t match timm’s expected tensor normalization), which can collapse performance. We also make checkpoint discovery slightly more robust by allowing direct `.pth/.pt/.bin` paths and preferring the *best-scoring* checkpoint among all found candidates rather than returning the first match in a directory. These changes keep the same timm model forward + softmax + mean-ensemble semantics, but should move the score materially upward toward the target.'
- What this solution (achieved 0.11099) has done: 'We keep your inference-only timm ViT + optional ResNeXt + mean-ensemble core logic unchanged, but fix the most score-critical bug: your ViT preprocessing currently *adds* `transforms.ToTensor()` even though timm’s `create_transform` already returns a tensor pipeline, which can break normalization and make predictions near-random. We instead use the `create_transform` output as-is and only wrap it if it isn’t already a `Compose`, avoiding double tensor conversion. Additionally, we make ViT checkpoint discovery consider common “best/last” filenames *recursively* and prefer them over just “largest file”, which increases the chance you load the intended fine-tuned Cassava weights. These are minimal changes targeted at moving accuracy upward toward the target without altering architecture, inference loops, or ensembling semantics.'
- What this solution (achieved 0.11099) has done: 'Your score is far below the target, so we should increase accuracy by fixing the most likely remaining “near-random” inference cause while keeping your pipeline unchanged (timm ViT + optional ResNeXt + mean-probability ensembling). The key minimal fix is to stop wrapping `create_transform(...)` inside another `transforms.Compose([ ... ])`: when `create_transform` returns an `albumentations`-style/other callable (not a torchvision `Compose`), your current `isinstance(..., transforms.Compose)` check can mis-detect and double-wrap/alter behavior; we instead use the transform as returned and only fall back to a safe timm `transforms_factory` default when needed. Additionally, we ensure the ViT input size matches the model’s resolved config (often 224) and that the dataset never applies an incorrect extra conversion (no extra `ToTensor()`), keeping evaluation semantics identical but making preprocessing consistent with timm expectations. These changes are small, localized to preprocessing and should move predictions away from chance toward your target without changing model architecture, ensembling, or training loops.'
- What this solution (achieved 0.11099) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest changes that fix likely “near-random” inference causes while keeping your exact ViT/ResNeXt inference + mean-probability ensembling intact. The most impactful minimal fix is to ensure the ViT inference transform is **definitely** the correct timm tensor pipeline (avoiding any accidental PIL-only path or missing `ToTensor`/Normalize), by building it from `timm.data.create_transform` + `resolve_model_data_config` and verifying it returns a tensor. Next, we slightly harden the ViT checkpoint loader to handle common HuggingFace-style keys (`"model"`/`"state_dict"` nesting and `"base_model"` prefixes) without changing the model architecture. These changes are localized to preprocessing and checkpoint key-cleaning and should move the score upward toward your target while preserving the core logic and submission semantics.'

# 9. Code solution

## === cell 0
import sys
import os
import random
import json
import gc
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
image_path = os.path.join(path, "test_images") + "/"

IMAGE_SIZE = (512, 512)

sample_sub_path = os.path.join(path, "sample_submission.csv")
if os.path.exists(sample_sub_path):
    submission_df = pd.read_csv(sample_sub_path)
else:
    submission_df = pd.DataFrame(columns=["image_id", "label"])
    submission_df["image_id"] = sorted(os.listdir(image_path))
    submission_df["label"] = 0

submission_df = submission_df[["image_id", "label"]].copy()
submission_df["label"] = submission_df["label"].fillna(0).astype(int)



## === cell 2
onlykeras = False

used_models_pytorch = {
    "resnext": [f"../input/models/resnext50_32x4d_fold{fold}_best.pth" for fold in [1]],
    "vit": "../input/model-vit/original_save_pretrained",
}

used_models_keras = {}

stacked_mean = False




## === cell 3
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
        self.df = df
        self.file_names = df["image_path_id"].values
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


if "resnext" in used_models_pytorch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_transforms():
        return Compose(
            [
                Resize(512, 512),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def inference(model, states, test_loader, device):
        model.to(device)

        probabilities = []
        for images in tqdm(test_loader, total=len(test_loader)):
            images = images.to(device)
            avg_preds = []
            for state in states:
                model.load_state_dict(state["model"])
                model.eval()
                with torch.inference_mode():
                    y_preds = model(images)
                avg_preds.append(y_preds.softmax(1).to("cpu").numpy())
            avg_preds = np.mean(avg_preds, axis=0)
            probabilities.append(avg_preds)
        return np.concatenate(probabilities, axis=0)

    predictions_resnext = pd.DataFrame(columns=["image_id", "image_path_id"])
    predictions_resnext["image_id"] = submission_df["image_id"].values
    predictions_resnext["image_path_id"] = image_path + predictions_resnext[
        "image_id"
    ].astype(str)

    resnext_ckpts = []
    for f in used_models_pytorch["resnext"]:
        if os.path.exists(f):
            resnext_ckpts.append(f)
    if len(resnext_ckpts) == 0:
        print("WARNING: No resnext checkpoints found; skipping resnext inference.")
        predictions_resnext = predictions_resnext.drop(columns=["image_path_id"])
    else:
        model = CustomResNext("resnext50_32x4d", pretrained=False)
        states = [torch.load(f, map_location="cpu") for f in resnext_ckpts]

        test_dataset = TestDataset(predictions_resnext, transform=get_transforms())
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        predictions = inference(model, states, test_loader, device)

        predictions_resnext["resnext"] = [np.squeeze(p) for p in predictions]
        predictions_resnext = predictions_resnext.drop(["image_path_id"], axis=1)

        torch.cuda.empty_cache()
        try:
            del model
            del states
        except Exception:
            pass
        gc.collect()



## === cell 4
if "vit" in used_models_pytorch:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    num_classes = 5

    vit_arch = "vit_base_patch16_224"

    from timm.data import resolve_model_data_config, create_transform

    class LeafDataset(torch.utils.data.Dataset):
        def __init__(self, df, data_path, mode="train", transforms=None):
            super().__init__()
            self.df_data = df.values
            self.data_path = data_path
            self.transforms = transforms
            self.mode = mode
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

    def _find_checkpoint_in_dir(d):
        if not os.path.isdir(d):
            return None
        preferred = [
            "pytorch_model.bin",
            "model.pth",
            "checkpoint.pth",
            "best.pth",
            "best.pt",
            "last.pth",
            "last.pt",
        ]
        for fn in preferred:
            p = os.path.join(d, fn)
            if os.path.exists(p) and os.path.isfile(p):
                return p

        candidates = []
        for fn in os.listdir(d):
            if fn.endswith((".pth", ".pt", ".bin")):
                candidates.append(os.path.join(d, fn))
        candidates = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)
        return candidates[0] if candidates else None

    def _list_ckpt_files_recursively(
        root_dir, exts=(".pth", ".pt", ".bin"), max_files=2000
    ):
        out = []
        if not os.path.isdir(root_dir):
            return out
        for r, _, files in os.walk(root_dir):
            for fn in files:
                if fn.endswith(exts):
                    out.append(os.path.join(r, fn))
                    if len(out) >= max_files:
                        return out
        return out

    def _score_ckpt_path(p):
        name = os.path.basename(p).lower()
        score = 0.0
        for tok, w in [
            ("best", 6.0),
            ("last", 4.0),
            ("final", 3.0),
            ("fold", 2.0),
            ("cassava", 5.0),
            ("leaf", 2.0),
            ("vit", 1.0),
            ("deit", 1.0),
            ("distill", 1.0),
            ("224", 0.5),
            ("patch16", 0.5),
        ]:
            if tok in name:
                score += w
        try:
            score += min(os.path.getsize(p) / 1e8, 4.0)
        except Exception:
            pass
        return score

    def _locate_ckpt_path(model_dir_or_ckpt):
        candidates = []

        if isinstance(model_dir_or_ckpt, str):
            candidates.append(model_dir_or_ckpt)

            if model_dir_or_ckpt.startswith("../input/"):
                candidates.append(
                    os.path.join("/kaggle/input", model_dir_or_ckpt[len("../input/") :])
                )

            if model_dir_or_ckpt.startswith("/kaggle/input/"):
                rel = "../input/" + model_dir_or_ckpt[len("/kaggle/input/") :]
                candidates.append(rel)

        file_hits = [c for c in candidates if c and os.path.isfile(c)]
        if file_hits:
            file_hits = sorted(file_hits, key=_score_ckpt_path, reverse=True)
            return file_hits[0]

        dir_hits = [c for c in candidates if c and os.path.isdir(c)]
        for c in dir_hits:
            ckpts = []
            p = _find_checkpoint_in_dir(c)
            if p is not None:
                ckpts.append(p)
            ckpts.extend(_list_ckpt_files_recursively(root_dir=c, max_files=2000))
            if ckpts:
                ckpts = sorted(list(set(ckpts)), key=_score_ckpt_path, reverse=True)
                return ckpts[0]

        base = None
        if isinstance(model_dir_or_ckpt, str):
            base = os.path.basename(model_dir_or_ckpt.rstrip("/"))
        if base:
            for root in [
                "/kaggle/input",
                "/kaggle/data/input",
                "/kaggle/data",
                "/kaggle/working",
            ]:
                if not os.path.isdir(root):
                    continue
                try:
                    for dn in os.listdir(root):
                        if dn == base:
                            cand = os.path.join(root, dn)
                            if os.path.isfile(cand):
                                return cand
                            if os.path.isdir(cand):
                                ckpts = []
                                p = _find_checkpoint_in_dir(cand)
                                if p is not None:
                                    ckpts.append(p)
                                ckpts.extend(
                                    _list_ckpt_files_recursively(root_dir=cand)
                                )
                                if ckpts:
                                    ckpts = sorted(
                                        list(set(ckpts)),
                                        key=_score_ckpt_path,
                                        reverse=True,
                                    )
                                    return ckpts[0]
                except Exception:
                    pass

        for root in ["../input/models", "/kaggle/input"]:
            ckpts = _list_ckpt_files_recursively(root_dir=root)
            if ckpts:
                ckpts = sorted(ckpts, key=_score_ckpt_path, reverse=True)
                return ckpts[0]

        return None

    def _extract_state_dict(ckpt_obj):
        if isinstance(ckpt_obj, dict):
            for key in [
                "model_ema",
                "model",
                "state_dict",
                "model_state_dict",
                "net",
                "weights",
                "teacher",
                "student",
                "module",
                "base_model",
            ]:
                if (
                    key in ckpt_obj
                    and isinstance(ckpt_obj[key], dict)
                    and len(ckpt_obj[key]) > 0
                ):
                    return ckpt_obj[key]
        if isinstance(ckpt_obj, dict):
            tensor_vals = [v for v in ckpt_obj.values() if torch.is_tensor(v)]
            if len(tensor_vals) > 0:
                return ckpt_obj
        return ckpt_obj

    def _clean_state_dict_keys(state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("net."):
                nk = nk[len("net.") :]
            if nk.startswith("base_model."):
                nk = nk[len("base_model.") :]
            new_sd[nk] = v
        return new_sd

    def _remap_common_classifier_keys(state_dict):
        remapped = dict(state_dict)

        if "fc.weight" in remapped and "head.weight" not in remapped:
            remapped["head.weight"] = remapped.pop("fc.weight")
        if "fc.bias" in remapped and "head.bias" not in remapped:
            remapped["head.bias"] = remapped.pop("fc.bias")

        if "classifier.weight" in remapped and "head.weight" not in remapped:
            remapped["head.weight"] = remapped.pop("classifier.weight")
        if "classifier.bias" in remapped and "head.bias" not in remapped:
            remapped["head.bias"] = remapped.pop("classifier.bias")

        if "head_dist.weight" in remapped and "head.weight" not in remapped:
            remapped["head.weight"] = remapped["head_dist.weight"]
        if "head_dist.bias" in remapped and "head.bias" not in remapped:
            remapped["head.bias"] = remapped["head_dist.bias"]
        if "head.weight" in remapped and "head_dist.weight" not in remapped:
            remapped["head_dist.weight"] = remapped["head.weight"]
        if "head.bias" in remapped and "head_dist.bias" not in remapped:
            remapped["head_dist.bias"] = remapped["head.bias"]

        return remapped

    def _drop_incompatible_classifier_weights(model, state_dict):
        model_sd = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if k in state_dict and k in model_sd:
                if (
                    hasattr(state_dict[k], "shape")
                    and state_dict[k].shape != model_sd[k].shape
                ):
                    state_dict.pop(k, None)
        return state_dict

    def _maybe_resize_pos_embed(model, state_dict):
        if "pos_embed" not in state_dict:
            return state_dict
        if not hasattr(model, "pos_embed"):
            return state_dict

        pe_ckpt = state_dict["pos_embed"]
        pe_model = model.pos_embed
        if not (torch.is_tensor(pe_ckpt) and torch.is_tensor(pe_model)):
            return state_dict
        if pe_ckpt.shape == pe_model.shape:
            return state_dict
        if pe_ckpt.ndim != 3 or pe_model.ndim != 3:
            return state_dict
        if pe_ckpt.shape[0] != 1 or pe_model.shape[0] != 1:
            return state_dict

        n_tok_ckpt = pe_ckpt.shape[1]
        dim = pe_ckpt.shape[2]
        if dim != pe_model.shape[2]:
            return state_dict

        num_prefix = pe_model.shape[1] - (
            model.patch_embed.grid_size[0] * model.patch_embed.grid_size[1]
        )
        num_prefix = max(num_prefix, 0)

        prefix_ckpt = (
            pe_ckpt[:, :num_prefix, :]
            if n_tok_ckpt >= num_prefix
            else pe_ckpt[:, :0, :]
        )
        grid_ckpt = pe_ckpt[:, num_prefix:, :]

        gs_new = model.patch_embed.grid_size
        n_new = gs_new[0] * gs_new[1]
        if grid_ckpt.shape[1] == n_new:
            state_dict["pos_embed"] = pe_ckpt
            return state_dict

        gs_old = int(np.sqrt(grid_ckpt.shape[1]))
        if gs_old * gs_old != grid_ckpt.shape[1]:
            return state_dict

        grid_ckpt = grid_ckpt.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)
        grid_ckpt = torch.nn.functional.interpolate(
            grid_ckpt, size=(gs_new[0], gs_new[1]), mode="bicubic", align_corners=False
        )
        grid_ckpt = grid_ckpt.permute(0, 2, 3, 1).reshape(1, n_new, dim)

        if prefix_ckpt.shape[1] != num_prefix:
            prefix_model = pe_model[:, :num_prefix, :].detach().cpu()
            prefix_ckpt = prefix_model

        state_dict["pos_embed"] = torch.cat([prefix_ckpt, grid_ckpt], dim=1)
        return state_dict

    def _ckpt_suggests_distilled(state_dict):
        if not isinstance(state_dict, dict):
            return False
        return ("head_dist.weight" in state_dict) or ("head_dist.bias" in state_dict)

    def _load_timm_vit(model_dir_or_ckpt, device):
        ckpt_path = _locate_ckpt_path(model_dir_or_ckpt)

        if ckpt_path is None:
            print(
                f"WARNING: No local ViT checkpoint found at {model_dir_or_ckpt}. Falling back to timm pretrained (ImageNet) ViT weights."
            )
            model = timm.create_model(vit_arch, pretrained=True)
            model.reset_classifier(num_classes=num_classes)
            model.to(device)
            return model

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        if not isinstance(state_dict, dict) or len(state_dict) == 0:
            print(
                f"WARNING: Checkpoint at {ckpt_path} did not contain a usable state_dict; falling back to timm pretrained weights."
            )
            model = timm.create_model(vit_arch, pretrained=True)
            model.reset_classifier(num_classes=num_classes)
            model.to(device)
            return model

        state_dict = _clean_state_dict_keys(state_dict)
        state_dict = _remap_common_classifier_keys(state_dict)

        arch_to_use = vit_arch
        if _ckpt_suggests_distilled(state_dict):
            arch_to_use = "deit_base_distilled_patch16_224"

        model = timm.create_model(
            arch_to_use, pretrained=False, num_classes=num_classes
        )

        state_dict = _maybe_resize_pos_embed(model, state_dict)
        state_dict = _drop_incompatible_classifier_weights(model, state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"INFO: Loaded ViT weights from: {ckpt_path} (arch={arch_to_use}, missing={len(missing)} unexpected={len(unexpected)})."
        )

        model.to(device)
        return model

    def predict_timm(model, test_dataset, device):
        preds = []
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        model.eval()
        for test_images in tqdm(test_dataloader, total=len(test_dataloader)):
            test_images = test_images.to(device)
            with torch.inference_mode():
                output = model(test_images)
                logits = output[0] if isinstance(output, (tuple, list)) else output
            preds.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())
        return preds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_vit = pd.DataFrame(columns=["image_id"])
    predictions_vit["image_id"] = submission_df["image_id"].values

    model = _load_timm_vit(used_models_pytorch["vit"], device=device)

    data_cfg = resolve_model_data_config(model)
    transforms_val = create_transform(**data_cfg, is_training=False)

    try:
        _tmp = Image.new(
            "RGB",
            (
                data_cfg.get("input_size", (3, 224, 224))[1],
                data_cfg.get("input_size", (3, 224, 224))[2],
            ),
            color=(128, 128, 128),
        )
        _out = transforms_val(_tmp)
        if not torch.is_tensor(_out):
            raise TypeError(
                f"timm transform did not return a torch.Tensor (got {type(_out)})."
            )
    except Exception as e:
        print(
            f"WARNING: ViT transform sanity check failed ({e}); continuing with created transform."
        )

    test_dataset = LeafDataset(
        df=predictions_vit, data_path=path, mode="test", transforms=transforms_val
    )

    predictions_raw_vit = predict_timm(model, test_dataset, device=device)

    predictions_vit["vit"] = [np.squeeze(p) for p in predictions_raw_vit]
    print(predictions_vit.head())

    torch.cuda.empty_cache()
    try:
        del model
    except Exception:
        pass
    gc.collect()



## === cell 5
submission_df = submission_df[["image_id", "label"]].copy()
submission_df["label"] = 0

if (
    "resnext" in used_models_pytorch
    and "predictions_resnext" in globals()
    and "resnext" in predictions_resnext.columns
):
    submission_df = submission_df.merge(predictions_resnext, on="image_id", how="left")

if (
    "vit" in used_models_pytorch
    and "predictions_vit" in globals()
    and "vit" in predictions_vit.columns
):
    submission_df = submission_df.merge(predictions_vit, on="image_id", how="left")



## === cell 6
model_cols = list(used_models_pytorch.keys()) + list(used_models_keras.keys())
model_cols = [c for c in model_cols if c in submission_df.columns]

if len(model_cols) == 0:
    submission_df["label"] = 0
else:
    probs_list = []
    for c in model_cols:
        col = submission_df[c].values
        filled = []
        for v in col:
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 5:
                filled.append(np.asarray(v, dtype=np.float32))
            else:
                filled.append(np.full((5,), 1.0 / 5.0, dtype=np.float32))
        probs_list.append(np.stack(filled, axis=0))
    probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    submission_df["label"] = np.argmax(probs, axis=1).astype(int)

submission_df["label"] = submission_df["label"].astype(int)



## === cell 7
submission_df.head(1)



## === cell 8
out_path = "submission.csv"
submission_df[["image_id", "label"]].to_csv(out_path, index=False)

with open(out_path, "r") as f:
    for _ in range(5):
        line = f.readline()
        if not line:
            break
        print(line.rstrip())
