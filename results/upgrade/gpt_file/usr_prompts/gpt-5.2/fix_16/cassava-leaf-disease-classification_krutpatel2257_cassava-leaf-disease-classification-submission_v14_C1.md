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

0.8844061650045331

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10538) has done: 'I fix the missing model checkpoint by falling back to a standard ImageNet-pretrained ResNeXt50 (same architecture) when the external `.pth` file is unavailable, so inference can run end-to-end. I update the Albumentations augmentation to the v2 API (the `RandomResizedCrop` signature changed and `Cutout` is removed), keeping the intended TTA behavior but making it valid. I also ensure device placement and `torch.load(..., map_location=...)` are correct, and that we always write a `submission.csv` with the exact required columns. These changes are directly aimed at unblocking execution and producing a reasonably strong baseline score rather than failing with file/API errors.'
- What this solution (achieved 0.08744) has done: 'Your current score is extremely low because you’re applying heavy *training-time* augmentations (random crop/rotate/color jitter/coarse dropout) at inference, and also feeding a tensor that is effectively **double-normalized** (Albumentations `Normalize` already scales to float/mean/std, then `ToTensor()` scales by 1/255 again). I keep your model and TTA loop intact, but change the inference transform to a deterministic, evaluation-appropriate pipeline (resize/center-crop + Normalize) and replace `ToTensor()` with a correct HWC->CHW conversion that does not rescale again. This is a minimal change that should dramatically increase accuracy toward your target while preserving the same inference semantics (average logits over 10 passes). I also ensure the model weights load correctly when using `DataParallel` by stripping/adding the `module.` prefix if needed (this affects correctness, not architecture).'
- What this solution (achieved 0.47683) has done: 'Your current score is far below the target, so we should cautiously improve accuracy with minimal, metric-aligned fixes while keeping your model and 10x TTA averaging loop intact. The biggest remaining issue is that your inference preprocessing is not consistent with standard ResNeXt50 ImageNet evaluation (you’re center-cropping to 256 instead of the typical 224 after a 256 resize), which can noticeably hurt accuracy. I change the inference transform to `Resize(256) + CenterCrop(224) + Normalize` while keeping everything else (architecture, weights loading logic, and TTA loop) the same. I also add `torch.backends.cudnn.benchmark=True` for stable faster inference without changing evaluation semantics.'
- What this solution (achieved 0.49925) has done: 'Your current score (0.47683) is far below the target (0.8844), so we should make small, metric-aligned inference fixes without changing your model, loss, or the 10x TTA averaging loop. The biggest issue left is a ResNeXt50 input-size mismatch: the model’s pretrained evaluation recipe expects a 224 crop taken from a shorter-side resize to 256, but your current `Resize(256,256)` forces a square resize that can distort aspect ratio and hurt accuracy. I switch to `SmallestMaxSize(256)` + `CenterCrop(224,224)` to preserve aspect ratio (still deterministic) while keeping Normalize and your tensor conversion identical. I also ensure the input tensor is explicitly `float32` on-device (no semantic change, but avoids any dtype surprises).'
- What this solution (achieved 0.20628) has done: 'Your score is far below the target, so we should make a small, metric-aligned inference-only fix without changing your model or the 10x averaging loop. The main likely issue now is that your “TTA” loop is actually identical 10 times (fully deterministic), so you’re not getting any benefit; we can introduce a very light, evaluation-safe test-time augmentation (horizontal flip) and average predictions over original+flipped views. To keep changes minimal and stable, we keep the same ResNeXt50, same weights loading, same normalization, and still do 10 forward passes—just alternating flip/no-flip. This should move accuracy upward toward your target while preserving the solution’s core logic.'
- What this solution (achieved 0.0867) has done: 'Your score is far below the target, so we should improve accuracy with the smallest inference-only fixes. The biggest bug is that your “TTA” loop applies the exact same deterministic transform every time because `force_apply=True` forces the random flip decision to be fixed, and both branches are identical. I keep your model, weights-loading, and 10-pass averaging loop intact, but change the augmentation call so randomness is resampled per pass, and explicitly create flipped vs non-flipped views (still just horizontal flip) to make the averaging meaningful. This should move your score upward toward the target while preserving the core logic and producing the same `submission.csv` format.'
- What this solution (achieved 0.18984) has done: 'Your current score (0.0867) is far below the target (0.8844), so we should make the smallest inference-only fixes that address the most likely remaining correctness issue. Right now the horizontal-flip “TTA” is implemented by reversing the image, but the model is not told that this view is flipped (unlike proper TTA where you either average probabilities or “de-flip” spatial outputs; for classification you just need to ensure the flip is applied in the same pixel space the model expects). The main improvement is to apply the flip *inside Albumentations* (so it’s consistent with the rest of the pipeline) and to keep the 10-pass averaging loop but alternate `NoOp` and `HorizontalFlip(p=1.0)` transforms. This preserves your core model/loop while making the TTA deterministic and correctly applied per pass, which should move accuracy upward toward the target.'
- What this solution (achieved 0.09268) has done: 'Your score is far below the target, so we should make the smallest inference-only change that fixes a likely train/test preprocessing mismatch without altering your model or the 10-pass averaging loop. The biggest remaining correctness issue is that you are normalizing to ImageNet, but the saved fine-tuned checkpoint was very likely trained with a different normalization (common in Cassava kernels is mean=std=0.5); using the wrong normalization can collapse accuracy. I keep the exact same ResNeXt50, state-dict loading, and alternating no-flip/flip TTA, but switch Normalize to a safer Cassava-style normalization and (to preserve semantics) average probabilities (softmax) instead of raw logits across TTA. This should move accuracy upward toward your target while remaining minimal and producing the same valid `submission.csv`.'
- What this solution (achieved 0.10837) has done: 'Your score is far below the target, so we should make a minimal, metric-aligned inference fix rather than changing the model or loop. The biggest likely issue is the normalization: you switched to mean/std=0.5, but both ImageNet-pretrained ResNeXt and most public Cassava fine-tuned ResNeXt checkpoints use ImageNet normalization, so the current inputs are probably badly shifted. I switch `Normalize` back to ImageNet mean/std while keeping the exact same architecture, checkpoint-loading logic, and the same 10-pass (no-flip/flip alternating) probability averaging. This should move accuracy upward substantially toward your target while preserving the core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 0.13004) has done: 'Your score is far below the target, so the most likely issue is not the model but a train/test preprocessing mismatch that collapses accuracy. I keep your exact ResNeXt50 architecture and the same 10-pass (no-flip/flip alternating) averaging loop, but make the preprocessing more robust by switching from fixed CenterCrop(224) to a deterministic “resize longest-side then center-pad to square” pipeline that avoids cropping away diseased regions on rectangular images. This preserves evaluation semantics (still deterministic per view, still same normalization, still same averaging), but typically lifts accuracy substantially on Cassava. I also switch the model construction to the non-deprecated `weights=` API without changing the effective initialization.'
- What this solution (achieved 0.40695) has done: 'Your score is far below the target, so the smallest likely “real” fix is to align test preprocessing with what ResNeXt50 expects at inference: resize the shorter side to 256 and take a centered 224 crop, rather than padding to 256×256 (padding introduces large black borders and domain shift). I keep your exact model, checkpoint-loading logic, and the same 10-pass (no-flip/flip alternating) probability-averaging loop, but replace the LongestMaxSize+Pad pipeline with a deterministic SmallestMaxSize+CenterCrop pipeline. I also ensure the augmented output is contiguous float32 before tensor conversion (no semantic change, just avoids edge-case slowdowns). This should move accuracy upward toward your target while preserving the core approach and still writing a valid `submission.csv`.'
- What this solution (achieved 0.16218) has done: 'Your current score is far below the target, so we should make a minimal, inference-only change that is most likely to recover lost accuracy without altering your model or the 10-pass TTA loop. The biggest likely remaining issue is a test-time preprocessing mismatch: many strong Cassava ResNeXt50 checkpoints (and common Kaggle kernels) use a **square resize to 512 then center-crop to 512** at inference, not the ImageNet 256→224 recipe, and this mismatch can heavily degrade performance. I keep your exact architecture, checkpoint-loading logic, and 10-pass flip/no-flip averaging, but switch the transform to deterministic `Resize(512,512)+Normalize` (with the same ImageNet mean/std you already use). I also make the DataLoader-free loop slightly more robust by reading `image_id` as string explicitly and ensuring the submission order matches the sample submission exactly (no semantic change, just safety).'
- What this solution (achieved 0.08221) has done: 'Your score is far below the target, so we should make the smallest inference-only change that most plausibly fixes a train/test preprocessing mismatch while keeping your ResNeXt50, checkpoint-loading, and 10-pass flip/no-flip TTA averaging intact. The current `Resize(512,512)` can distort aspect ratio and change lesion shapes; many Cassava pipelines instead resize the shorter side then center-crop (or center-crop at 512 after a shortest-side resize), which is both deterministic and metric-aligned. I switch the test-time transform to `SmallestMaxSize(512) + CenterCrop(512,512) + Normalize` (and the same with HorizontalFlip) to preserve aspect ratio without changing your model or TTA loop. This is a minimal, correctness-focused preprocessing adjustment expected to move accuracy upward toward your target while still producing the same valid `submission.csv`.'
- What this solution (achieved 0.12481) has done: 'Your current score is far below the target, so we should make the smallest inference-only changes that are most likely to recover accuracy without changing your model or the 10-pass averaging idea. The biggest issue is that you built a ResNeXt50 that expects the standard ImageNet 224-crop regime, but you’re feeding 512×512 inputs; that mismatch alone can crater accuracy. I switch the test-time preprocessing back to the standard deterministic ResNeXt/ImageNet evaluation pipeline (shorter-side resize to 256 + center crop to 224) while keeping your same flip/no-flip 10-pass probability averaging loop intact. I also keep the submission alignment with `sample_submission.csv` unchanged so the output format stays valid.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, and the biggest remaining issue is that when the external fine-tuned checkpoint is missing you fall back to an ImageNet-pretrained ResNeXt50 whose `fc` layer is *randomly initialized* (5 classes), which makes predictions near-random and caps accuracy around chance. I keep the exact same architecture and 10-pass flip/no-flip averaging loop, but when `model_path` is unavailable I replace the random head with a deterministic, data-driven head computed from the training-label priors (a bias-only classifier), which is the smallest legitimate improvement that doesn’t change the core approach and should move accuracy strongly upward. I also add a safe fallback to locate `train.csv` in the provided input paths, and keep submission alignment exactly as `sample_submission.csv`. This preserves evaluation semantics and produces a valid `submission.csv` end-to-end within the time limit.'

# 9. Code solution

## === cell 0
import os

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/rn-tta-calr-ft-of/model(10).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_candidates = [
    "../input/cassava-leaf-disease-classification/train.csv",
    "../input/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/train.csv",
]
train_csv_path = next((p for p in train_csv_candidates if os.path.exists(p)), None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## === cell 2
use_imagenet = not os.path.exists(model_path)
weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if use_imagenet else None

model = models.resnext50_32x4d(weights=weights)
model.fc = nn.Linear(2048, 5)

use_dp = torch.cuda.is_available() and torch.cuda.device_count() > 1
if use_dp:
    model = nn.DataParallel(model)

model = model.to(device)

if os.path.exists(model_path):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model_state_keys = list(model.state_dict().keys())
    loaded_keys = list(state.keys())

    def has_module_prefix(keys):
        return len(keys) > 0 and all(k.startswith("module.") for k in keys)

    model_has_module = has_module_prefix(model_state_keys)
    loaded_has_module = has_module_prefix(loaded_keys)

    if model_has_module and not loaded_has_module:
        state = {("module." + k): v for k, v in state.items()}
    elif (not model_has_module) and loaded_has_module:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
else:
    if train_csv_path is None:
        raise FileNotFoundError(
            "No fine-tuned checkpoint found and train.csv was not located; cannot build prior head."
        )
    train_df = pd.read_csv(train_csv_path)
    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(5), fill_value=0)
        .astype(np.float64)
    )
    prior = (counts.values + 1.0) / (counts.values.sum() + 5.0)  # Laplace smoothing
    prior_logits = np.log(prior)

    fc = model.module.fc if isinstance(model, nn.DataParallel) else model.fc
    with torch.no_grad():
        fc.weight.zero_()
        fc.bias.copy_(
            torch.tensor(prior_logits, dtype=fc.bias.dtype, device=fc.bias.device)
        )

model.eval()



## === cell 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

base_aug_no_flip = albumentations.Compose(
    [
        albumentations.SmallestMaxSize(max_size=256, p=1.0),
        albumentations.CenterCrop(height=224, width=224, p=1.0),
        albumentations.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

base_aug_hflip = albumentations.Compose(
    [
        albumentations.SmallestMaxSize(max_size=256, p=1.0),
        albumentations.CenterCrop(height=224, width=224, p=1.0),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 4
def albumentations_image_to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    if img_hwc.dtype != np.float32:
        img_hwc = img_hwc.astype(np.float32)
    img_hwc = np.ascontiguousarray(img_hwc)
    return torch.from_numpy(np.transpose(img_hwc, (2, 0, 1))).contiguous()




## === cell 5
sample_sub = pd.read_csv(sample_sub_path, dtype={"image_id": str})

predictions = []

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        image_prob = None

        img_path = os.path.join(test_images_path, sample_row.image_id)
        base_image = np.array(Image.open(img_path).convert("RGB"))

        for t in range(10):
            if t % 2 == 0:
                aug_image = base_aug_no_flip(image=base_image)["image"]
            else:
                aug_image = base_aug_hflip(image=base_image)["image"]

            image_t = albumentations_image_to_tensor(aug_image).to(
                device=device, dtype=torch.float32
            )

            outputs = model(image_t.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)

            if image_prob is None:
                image_prob = probs
            else:
                image_prob = image_prob + probs

        image_prob = image_prob / 10.0
        pred_label = int(torch.argmax(image_prob, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])

sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")
sub_df["label"] = sub_df["label"].astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print(f"Wrote submission.csv with shape={sub_df.shape}")
