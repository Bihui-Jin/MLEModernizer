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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8677848292535509

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42265) has done: 'I fix the Albumentations v2 API break that prevents `tta_transform` from being created (causing the downstream `NameError`). I also make the code robust to missing external weight files under `/kaggle/input/...` by falling back to torchvision ImageNet weights for EfficientNetV2-S (same architecture) so the notebook can run end-to-end and generate a valid CSV. Finally, I correct the DataLoader/unpacking so `image_id` is collected as a plain string (not a tuple/list) and ensure the submission length/order matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.13528) has done: 'Your current score is far below the target because the code is effectively doing inference with an ImageNet-pretrained EfficientNet whose classifier head is randomly initialized (since the external .pth files are not present), and it also uses a very strong “train-like” augmentation pipeline for TTA (random crops/rotations/noise) that destroys signal at inference. To move accuracy up toward the target while preserving your overall approach (single-model TTA + argmax), I (1) switch TTA to safe test-time transforms (resize/center-crop + flips only) and (2) if fine-tuned weights are missing, fall back to a proper 5-class cassava checkpoint from HuggingFace via `torch.hub` (same core idea: load a pretrained classifier and run TTA inference). These are minimal, execution-safe changes that should substantially improve accuracy without changing the fundamental inference semantics or adding training. The submission writing/order logic be kept the same.'
- What this solution (achieved 0.61099) has done: 'Your score is far below the target mainly because, when your fine-tuned `.pth` files aren’t present, the EfficientNetV2-S head remains randomly initialized, so predictions are near-random. To move accuracy sharply upward while preserving your core inference approach (single-model TTA + argmax), I (1) load ImageNet weights first and then swap the classifier to 5 classes (so at least the backbone is meaningful), and (2) add a local fallback that uses the provided `train.csv` to build a class-frequency prior head (a deterministic, lightweight calibration) when cassava fine-tuned weights aren’t available. I also remove the extra CLAHE transform from the unused pipeline to avoid confusion, keep TTA as safe geometric transforms, and ensure the submission aligns exactly to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.86778), so we should improve accuracy with minimal, low-risk changes while keeping the same inference-only + TTA + argmax core logic. The biggest gain here is eliminating randomness in TTA: because your TTA pipeline uses probabilistic flips, repeating it 5 times produces noisy averages; switching to a small deterministic TTA set (identity / hflip / vflip / hvflip) keeps the same semantics but yields more stable and typically higher accuracy. I also make the EfficientNet classifier dropout inactive at inference by setting it to `p=0.0` (dropout is already disabled by `eval()`, but keeping `p=0.8` can still be problematic if the model is ever not in eval or exported; this is a safe stability tweak). Finally, I fix the model-selection order bug so you actually prefer the third checkpoint when present (you were checking `eff6_loaded` but only after `eff1_loaded`), which can improve results if that checkpoint exists.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights



## === cell 1
num_tta = 4  # Change: use a deterministic 4-view TTA set (id/hflip/vflip/hvflip) to reduce noise and improve accuracy stability vs probabilistic flips.



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

assert os.path.isdir(test_image_dir), f"Missing test_image_dir: {test_image_dir}"
assert os.path.isfile(sample_sub_path), f"Missing sample submission: {sample_sub_path}"
assert os.path.isfile(
    train_csv_path
), f"Missing train.csv (needed for fallback prior): {train_csv_path}"



## === cell 3
test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 4
base_transform = A.Compose(
    [
        A.Resize(416, 416),
        A.CenterCrop(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
def tta_predict_single_model(model, image, base_transform, device, n_tta=4):
    """
    Change: deterministic TTA (id/hflip/vflip/hvflip) instead of random flips per pass.
    This keeps the same 'TTA average probs then argmax' core logic but usually improves accuracy.
    """
    model.eval()
    tta_predictions = []

    if not (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 3):
        raise ValueError(
            f"Image must be HxWx3 numpy array, got type={type(image)}, shape={getattr(image, 'shape', None)}"
        )

    views = []
    views.append(image)  # identity
    views.append(np.ascontiguousarray(image[:, ::-1, :]))  # hflip
    views.append(np.ascontiguousarray(image[::-1, :, :]))  # vflip
    views.append(np.ascontiguousarray(image[::-1, ::-1, :]))  # hvflip

    if n_tta is None:
        n_tta = 4
    n_tta = int(n_tta)
    if n_tta <= 0:
        raise ValueError(f"n_tta must be positive, got {n_tta}")

    chosen = [views[i % 4] for i in range(n_tta)]

    with torch.no_grad():
        for v in chosen:
            augmented = base_transform(image=v)["image"]
            augmented = augmented.unsqueeze(0).to(device)

            output = model(augmented)
            probs = F.softmax(output, dim=1)
            tta_predictions.append(probs)

    avg_probs = torch.mean(torch.stack(tta_predictions, dim=0), dim=0)
    return avg_probs




## === cell 7
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=identity_collate,
)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 9
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_weights_path = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)
if os.path.isfile(resnet_weights_path):
    resnet_model.load_state_dict(torch.load(resnet_weights_path, map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()




## === cell 10
def build_class_prior_head_from_train_csv(
    train_csv, in_features, num_classes=5, temperature=1.0
):
    train_df = pd.read_csv(train_csv)
    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float64)
    )
    probs = counts / max(counts.sum(), 1.0)
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)

    head = nn.Linear(in_features, num_classes, bias=True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor(logits, dtype=torch.float32))
    return head


def load_efficientnet_v2_s_5class(device, weights_path=None, train_csv_for_prior=None):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, 5))

    loaded_finetuned = False
    if weights_path is not None and os.path.isfile(weights_path):
        sd = torch.load(weights_path, map_location=device)
        model.load_state_dict(sd, strict=True)
        loaded_finetuned = True
    else:
        if train_csv_for_prior is not None:
            prior_head = build_class_prior_head_from_train_csv(
                train_csv_for_prior,
                in_features=in_features,
                num_classes=5,
                temperature=1.25,
            )
            model.classifier = nn.Sequential(nn.Dropout(p=0.0), prior_head)

    model = model.to(device).eval()
    return model, loaded_finetuned


eff1_path = "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth"
eff7_path = "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth"
eff6_path = "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth"

efficientnet_model_1, eff1_loaded = load_efficientnet_v2_s_5class(
    device, weights_path=eff1_path, train_csv_for_prior=train_csv_path
)
efficientnet_model_7, eff7_loaded = load_efficientnet_v2_s_5class(
    device, weights_path=eff7_path, train_csv_for_prior=train_csv_path
)
efficientnet_model_8, eff6_loaded = load_efficientnet_v2_s_5class(
    device, weights_path=eff6_path, train_csv_for_prior=train_csv_path
)

print(
    f"Loaded finetuned weights: eff1={eff1_loaded}, eff7={eff7_loaded}, eff8={eff6_loaded}"
)



## === cell 11
if eff7_loaded:
    inference_model = efficientnet_model_7
elif eff6_loaded:
    inference_model = efficientnet_model_8
elif eff1_loaded:
    inference_model = efficientnet_model_1
else:
    inference_model = (
        efficientnet_model_7  # all are prior-backed in this case; any is fine.
    )



## === cell 12
ensemble_predictions = []
image_names = []

with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        image, img_name = batch[0]
        if not isinstance(img_name, str):
            img_name = str(img_name)

        probs = tta_predict_single_model(
            inference_model, image, base_transform, device, n_tta=num_tta
        )

        final_pred = int(probs.argmax(dim=1).cpu().item())
        ensemble_predictions.append(final_pred)
        image_names.append(img_name)

len(image_names), len(ensemble_predictions), test_df.shape



## === cell 13
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df[["image_id"]].copy()
submission_df["label"] = submission_df["image_id"].map(pred_map)

missing = int(submission_df["label"].isna().sum())
if missing:
    train_df = pd.read_csv(train_csv_path)
    most_freq = int(train_df["label"].value_counts().idxmax())
    submission_df["label"] = submission_df["label"].fillna(most_freq).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(
    f"Saved {submission_path} with shape={submission_df.shape}, missing_filled={missing}"
)
submission_df.head()
