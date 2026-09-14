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

0.8729223330311272

# 6. Current score

0.10127

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23281) has done: 'I fix the missing pretrained weight path by falling back to a torchvision ImageNet-pretrained ResNeXt50 when the external `.pth` file isn’t available, so inference can run end-to-end. I also update the Albumentations pipeline to the v2 API (notably `RandomResizedCrop` signature changes and removal of deprecated `Cutout`) so `sub_aug` is defined without validation errors. Finally, I make inference stable and correct by using `torch.no_grad()`, consistent device handling, RGB conversion, and softmax-averaged TTA logits before argmax, then write a properly formatted `submission.csv`.'
- What this solution (achieved 0.20927) has done: 'Your current 0.23281 is far below the 0.8729 target, and the biggest reason is that you’re doing *strong training-time augmentations at test-time* (RandomResizedCrop/HSV/CoarseDropout/etc.), which destroys signal and tanks accuracy. I keep the same model and inference loop, but switch the test-time pipeline to a standard deterministic resize/center-crop + ImageNet normalization, and keep only safe TTA (horizontal flip) while removing color/dropout/geometric randomness. I also fix a subtle tensor conversion issue by replacing `transforms.ToTensor()` (which can mishandle already-float normalized arrays) with a direct NumPy→Torch CHW conversion to preserve your normalization exactly. These minimal changes should substantially increase accuracy and move the score toward your target without changing the core model logic.'
- What this solution (achieved 0.10127) has done: 'Your score is far below the target, so we should safely *increase* accuracy without changing the model or training logic. The main issue now is that if the external `.pth` is missing, you’re using an ImageNet-pretrained backbone but a randomly initialized 5-class `fc`, which predict near-random and caps accuracy. I keep the exact same architecture/inference loop, but when `model_path` is missing I replace the `fc` with the ImageNet classifier and output the Cassava 5 classes by mapping from the 1000 ImageNet logits using a fixed “nearest prototype in fc-weight space” (computed once from the pretrained head), which is a lightweight, deterministic improvement that stays within the same forward pass semantics. I also speed up and stabilize inference (no score semantics change) by batching instead of per-row model calls, so it finishes reliably under the time limit while producing the same submission format.'

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
model_path = "../input/rnwcnwcttacalr/model(7).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.exists(test_images_path):
    test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 2
using_imagenet_fallback = False

model = models.resnext50_32x4d(weights=None)
model.fc = nn.Linear(2048, 5)

if os.path.exists(model_path):
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
else:
    using_imagenet_fallback = True
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

model.to(device)
model.eval()



## === cell 3
sub_aug = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

sub_aug_hflip = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 4
def hwc_float_to_chw_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Change is correctness/score-relevant: albumentations outputs HWC float32.
    Using torchvision.transforms.ToTensor on already-float normalized arrays can
    apply unintended scaling depending on dtype/range. Convert explicitly.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()  # CHW float32


def build_imagenet_to_5_mapping_from_fc(model_imagenet: nn.Module) -> torch.Tensor:
    """
    Score-relevant minimal addition for the no-.pth fallback:
    Build a fixed mapping from 1000 ImageNet logits -> 5 cassava classes using
    nearest prototype in the pretrained fc weight space.

    This does NOT change the backbone/forward; it only post-processes logits
    when we're otherwise forced to use ImageNet weights (and would be random with a 5-class fc).
    """
    fc = model_imagenet.fc
    W = fc.weight.detach().float().cpu()  # [1000, 2048]
    b = fc.bias.detach().float().cpu()  # [1000]

    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-12)  # cosine space
    seed_idx = torch.tensor([0, 250, 500, 750, 999], dtype=torch.long)
    prot = Wn[seed_idx]  # [5, 2048]

    sims = Wn @ prot.t()  # [1000, 5]
    mapping = torch.argmax(sims, dim=1).to(torch.long)  # [1000] in {0..4}
    return mapping




## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

tta_augs = [sub_aug, sub_aug_hflip]

batch_size = 32

imagenet_to5 = None
if using_imagenet_fallback:
    imagenet_to5 = build_imagenet_to_5_mapping_from_fc(model).to(device)  # [1000]

pred_image_ids = []
pred_labels = []

with torch.no_grad():
    batch_imgs = []
    batch_ids = []

    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        base_img = Image.open(img_path).convert("RGB")
        base_img = np.array(base_img)

        batch_imgs.append(base_img)
        batch_ids.append(sample_row.image_id)

        if len(batch_imgs) >= batch_size:
            probs_sum = None

            for aug in tta_augs:
                aug_tensors = []
                for img in batch_imgs:
                    aug_img = aug(image=img)["image"]  # HWC float32 normalized
                    aug_tensors.append(hwc_float_to_chw_tensor(aug_img))
                x = torch.stack(aug_tensors, dim=0).to(device)  # [B,3,224,224]

                logits = model(x)

                if using_imagenet_fallback:
                    p1000 = torch.softmax(logits, dim=1)  # [B,1000]
                    p5 = torch.zeros(
                        (p1000.shape[0], 5), device=device, dtype=p1000.dtype
                    )
                    p5.scatter_add_(
                        1, imagenet_to5.unsqueeze(0).expand(p1000.shape[0], -1), p1000
                    )
                    probs = p5
                else:
                    probs = torch.softmax(logits, dim=1)  # [B,5]

                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            probs_avg = probs_sum / len(tta_augs)
            batch_pred = (
                torch.argmax(probs_avg, dim=1).detach().cpu().numpy().astype(int)
            )

            pred_image_ids.extend(batch_ids)
            pred_labels.extend(batch_pred.tolist())

            batch_imgs = []
            batch_ids = []

    if len(batch_imgs) > 0:
        probs_sum = None
        for aug in tta_augs:
            aug_tensors = []
            for img in batch_imgs:
                aug_img = aug(image=img)["image"]
                aug_tensors.append(hwc_float_to_chw_tensor(aug_img))
            x = torch.stack(aug_tensors, dim=0).to(device)

            logits = model(x)

            if using_imagenet_fallback:
                p1000 = torch.softmax(logits, dim=1)
                p5 = torch.zeros((p1000.shape[0], 5), device=device, dtype=p1000.dtype)
                p5.scatter_add_(
                    1, imagenet_to5.unsqueeze(0).expand(p1000.shape[0], -1), p1000
                )
                probs = p5
            else:
                probs = torch.softmax(logits, dim=1)

            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        probs_avg = probs_sum / len(tta_augs)
        batch_pred = torch.argmax(probs_avg, dim=1).detach().cpu().numpy().astype(int)

        pred_image_ids.extend(batch_ids)
        pred_labels.extend(batch_pred.tolist())

sub_df = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Fallback used (ImageNet->5 mapping):", using_imagenet_fallback)
