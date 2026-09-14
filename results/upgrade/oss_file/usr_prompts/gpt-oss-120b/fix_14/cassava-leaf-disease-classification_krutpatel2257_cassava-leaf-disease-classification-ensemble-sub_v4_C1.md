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
scipy==1.15.3
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

0.8952855847688124

# 6. Current score

0.11248

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23318) has done: 'Implemented a fix for the Albumentations RandomResizedCrop API change by supplying the required `size` argument (tuple of height and width). This resolves the validation error, allows the augmentation pipeline to be created, and consequently enables the inference loop to run and generate a proper `submission.csv` file.'
- What this solution (achieved 0.21001) has done: 'The fix replaces the random test‑time augmentation with a deterministic resize‑to‑224 pipeline (the size the pretrained models expect) and averages **probabilities** instead of raw logits. Using equal weights for the two models and a fixed augmentation reduces noise, so the predictions become more accurate and move the validation score toward the target. The rest of the logic and model definitions stay unchanged.'
- What this solution (achieved 0.10463) has done: 'I make three small but targeted updates: (1) add a robust checkpoint‑search that loads the provided fine‑tuned weights if they exist in any typical `input` or `working` folder (instead of silently falling back to ImageNet‑only weights); (2) enrich the test‑time augmentation by inserting a random horizontal flip so each TTA pass sees a slightly different view; (3) raise the TTA repeat count a bit for more stable averaging. These tweaks keep the original model architecture and inference flow unchanged while giving the pretrained checkpoints a chance to be used, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.31129) has done: 'I make two minimal but impactful fixes: (1) improve checkpoint loading by handling common keys (`model` or `state_dict`) so the fine‑tuned weights are actually used, and (2) correct the image‑to‑tensor conversion – after Albumentations’ `Normalize` we must not rescale again with `transforms.ToTensor`. I replace it with a small helper that only reorders the axes, preserving the normalized values. These changes keep the model architecture and inference logic unchanged while allowing the pretrained checkpoints to be applied correctly, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.10015) has done: 'I add a modest rotation to the test‑time augmentation pipeline, increase the TTA repeats slightly, and change the ensembling logic to average *logits* (which is mathematically more sound) before applying a single softmax. These tiny adjustments keep the model architecture, loading, and overall workflow unchanged while giving the predictions a better chance to move the validation accuracy toward the target score.'
- What this solution (achieved 0.21637) has done: 'I make three small, targeted fixes that keep the original architecture and workflow but aim to raise the validation accuracy toward the target: (1) load model checkpoints with `strict=False` so any fine‑tuned weights are applied even if the key names differ; (2) make test‑time augmentation deterministic (remove random flips/rotations) to avoid hurting predictions; (3) average **probabilities** instead of raw logits during TTA, which better matches the accuracy metric. These changes are minimal, preserve core logic, and should improve the score without over‑hauling the solution.'
- What this solution (achieved 0.11024) has done: 'I fix the Albumentations RandomResizedCrop initialization by providing the required `size` argument (a (height, width) tuple) instead of the deprecated `height` and `width` parameters. This resolves the validation error, allowing the augmentation pipeline and inference loop to run so a proper `submission.csv` is produced.'
- What this solution (achieved 0.19357) has done: 'I make the test‑time preprocessing deterministic (simple resize + normalize) to remove noisy random crops, flips and rotations that were hurting the weak ImageNet‑only models. I also reduce the TTA loop to a small fixed count (4) so the same clean view is averaged, which stabilises predictions and should raise accuracy toward the target while keeping the core model architecture and loading logic unchanged.'
- What this solution (achieved 0.11248) has done: 'I added a small helper that strips a possible “module.” prefix from checkpoint keys so the fine‑tuned weights are correctly loaded, and switched the inference to average probabilities (rather than logits) from the two models. I also introduced a lightweight test‑time augmentation (random horizontal flip) and increased the TTA count to 8 for a more stable prediction while keeping the overall architecture unchanged. These minimal tweaks should raise the validation accuracy toward the target without altering the core logic.'

# 9. Code solution

## === cell 0
import os
import glob

import albumentations
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms




## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v3/eff_epoch_11.pth"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"


def locate_checkpoint(default_path):
    """Return the first existing checkpoint matching the filename in common Kaggle folders."""
    if os.path.exists(default_path):
        return default_path
    filename = os.path.basename(default_path)
    search_patterns = [
        f"/kaggle/input/**/{filename}",
        f"/kaggle/working/**/{filename}",
        f"./input/**/{filename}",
        f"./working/**/{filename}",
    ]
    for pattern in search_patterns:
        candidates = glob.glob(pattern, recursive=True)
        for cand in candidates:
            if os.path.exists(cand):
                return cand
    basename = os.path.splitext(filename)[0]
    extra_patterns = [
        f"/kaggle/input/**/*.pth",
        f"/kaggle/working/**/*.pth",
        f"./input/**/*.pth",
        f"./working/**/*.pth",
    ]
    for pattern in extra_patterns:
        for cand in glob.glob(pattern, recursive=True):
            if basename in os.path.basename(cand):
                return cand
    return default_path  # will load ImageNet weights only


resnet_model_path = locate_checkpoint(resnet_model_path)
effnet_model_path = locate_checkpoint(effnet_model_path)




## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_finetuned_weights(model, ckpt_path):
    """Load checkpoint handling possible 'module.' prefixes."""
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return model
    try:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("module.", "")
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}")
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
    return model


resnet_model = models.resnext50_32x4d(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
resnet_model = resnet_model.to(device)
resnet_model = load_finetuned_weights(resnet_model, resnet_model_path)
resnet_model.eval()

effnet_model = models.efficientnet_b4(pretrained=True)
effnet_model.classifier[1] = nn.Linear(effnet_model.classifier[1].in_features, 5)
effnet_model = effnet_model.to(device)
effnet_model = load_finetuned_weights(effnet_model, effnet_model_path)
effnet_model.eval()




## === cell 3
base_aug = albumentations.Compose(
    [
        albumentations.Resize(height=224, width=224, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

tta_aug = albumentations.HorizontalFlip(p=0.5)

sample_sub = pd.read_csv(sample_sub_path)
tta_count = 8  # more TTA passes for stable averaging

predictions = []


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert an Albumentations‑processed numpy image to a torch tensor."""
    img = img.astype(np.float32)  # already normalized by Albumentations
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    return torch.from_numpy(img)


with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        prob_sum = np.zeros(5, dtype=np.float32)  # accumulate probabilities
        img_path = os.path.join(test_images_path, sample_row.image_id)
        raw_image = np.array(Image.open(img_path).convert("RGB"))
        for _ in range(tta_count):
            aug_img = base_aug(image=raw_image)["image"]
            aug_img = tta_aug(image=aug_img)["image"]
            tensor_image = np_to_tensor(aug_img).to(device)

            resnet_logits = resnet_model(tensor_image.unsqueeze(0)).squeeze()
            effnet_logits = effnet_model(tensor_image.unsqueeze(0)).squeeze()

            resnet_prob = F.softmax(resnet_logits, dim=0)
            effnet_prob = F.softmax(effnet_logits, dim=0)

            combined_prob = 0.5 * resnet_prob + 0.5 * effnet_prob
            prob_sum += combined_prob.cpu().numpy()

        avg_prob = prob_sum / tta_count
        pred_label = np.argmax(avg_prob)
        predictions.append(pred_label)

sub_df = pd.DataFrame(
    {"image_id": sample_sub.image_id, "label": np.array(predictions, dtype=int)}
)
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
