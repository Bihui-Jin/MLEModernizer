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

0.8785131459655485

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Your pipeline doesn’t yield a Kaggle score primarily because it can fail before writing `submission.csv` (missing external weight files / wrong paths) and because the saved predictions may not be aligned to the sample submission order. I make minimal, execution-unblocking changes: robustly resolve input paths, load checkpoints safely (with a clear fallback that still produces a valid CSV), and reorder the final submission to exactly match `sample_submission.csv`’s `image_id` order. I also enable deterministic settings to avoid run-to-run drift without changing the modeling logic. These changes preserve your ensemble/inference logic and only touch reliability + submission correctness.'
- What this solution (achieved 0.39499) has done: 'Your current score (0.61099) is far below the target (0.8785), and the main reason is that the models are being instantiated with `weights=None` and your external checkpoints are not available in the provided file tree—so inference is effectively using random weights. The smallest legitimate change that preserves your ensemble/inference core logic is to load ImageNet pretrained weights when custom checkpoints are missing, so the models produce meaningful features. I keep the exact same architecture heads, transforms, and ensemble averaging, only adding a safe fallback to torchvision pretrained weights and ensuring the submission order matches `sample_submission.csv`. This should move accuracy up substantially toward the target without changing your overall approach.'
- What this solution (achieved 0.09791) has done: 'Your score is far below target because, when custom checkpoints are missing, you currently keep randomly initialized 5-class classifier heads on top of ImageNet backbones, which yields near-random predictions. To move accuracy toward the target while preserving your ensemble/inference core logic, I (1) add a tiny fallback “head calibration” step that initializes the 5-class heads from the corresponding ImageNet 1000-class heads (so the model produces meaningful logits even without cassava fine-tuning), and (2) use the correct ImageNet preprocessing for ResNet50 (your current pipeline uses EfficientNet normalization for all models). These are minimal, inference-only changes that keep your architecture/loops intact and should raise the score substantially toward the target. I also keep the submission alignment logic but make it stricter by forcing the output order to exactly match `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score is near-random because the EfficientNet models are being fed 384×384 images, but EfficientNetV2-S is pretrained on 384×384 only for the *L* variant; V2-S pretrained expects 224×224 and its ImageNet preprocessing (including resize/crop behavior) is different from the simple resize you’re using. To move the score sharply upward toward the target while keeping the same ensemble/models/inference loop, I only change the EfficientNet test transform to match `EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()` (correct size + normalization), and keep ResNet as-is. I also make the DataLoader use a small number of workers for faster, more stable throughput under Kaggle, without changing predictions. Submission alignment logic is kept identical and still forces the exact `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 3
num_tta = 5  # kept as-is (not used in current core logic)



## === cell 4
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
test_df = pd.read_csv(sample_path)
test_df.head()



## === cell 6
_eff_w = EfficientNet_V2_S_Weights.IMAGENET1K_V1
_eff_tfm = _eff_w.transforms()

eff_h, eff_wid = _eff_tfm.crop_size  # typically (224, 224) for EfficientNetV2-S
eff_mean = tuple(float(x) for x in _eff_tfm.mean)
eff_std = tuple(float(x) for x in _eff_tfm.std)

efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(eff_h, eff_wid),
        A.Normalize(mean=eff_mean, std=eff_std),
        ToTensorV2(),
    ]
)

resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/10060321.py in <cell line: 0>()
     10 # - normalize with the weights' mean/std
     11 # (we keep CLAHE since it is part of your core pipeline)
---> 12 eff_h, eff_wid = _eff_tfm.crop_size  # typically (224, 224) for EfficientNetV2-S
     13 eff_mean = tuple(float(x) for x in _eff_tfm.mean)
     14 eff_std = tuple(float(x) for x in _eff_tfm.std)

ValueError: not enough values to unpack (expected 2, got 1)

## === cell 7
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # Image ID
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 8
test_dataset = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/963297583.py in <cell line: 0>()
      1 test_dataset = CassavaTestDataset(
----> 2     test_df, test_image_dir, transform=efficientnet_transforms
      3 )
      4 
      5 # CHANGE (runtime stability): a couple workers is usually safe and faster on Kaggle,

NameError: name 'efficientnet_transforms' is not defined

## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 10
def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        print(f"[WARN] Checkpoint not found, skipping load: {ckpt_path}")
        return False
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"[INFO] Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
    return True


def init_head_from_imagenet_effv2s(
    model, class_groups, weights_enum=EfficientNet_V2_S_Weights.IMAGENET1K_V1
):
    """
    model: efficientnet_v2_s with classifier[1] = Linear(in_features, 5) or Sequential(..., Linear(...,5))
    class_groups: list of lists of imagenet class indices, length=5
    """
    pretrained = models.efficientnet_v2_s(weights=weights_enum)
    w1000 = pretrained.classifier[1].weight.detach().cpu()  # [1000, in_features]
    b1000 = pretrained.classifier[1].bias.detach().cpu()  # [1000]

    head = None
    if isinstance(model.classifier, nn.Sequential):
        for m in reversed(model.classifier):
            if isinstance(m, nn.Linear):
                head = m
                break
    else:
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
            try:
                if isinstance(model.classifier[1], nn.Linear):
                    head = model.classifier[1]
            except Exception:
                head = None
    if head is None or not isinstance(head, nn.Linear) or head.out_features != 5:
        print(
            "[WARN] Could not locate 5-class EfficientNet head to initialize from ImageNet."
        )
        return

    W = torch.zeros((5, w1000.shape[1]), dtype=w1000.dtype)
    B = torch.zeros((5,), dtype=b1000.dtype)
    for i, idxs in enumerate(class_groups):
        idxs_t = torch.tensor(idxs, dtype=torch.long)
        W[i] = w1000.index_select(0, idxs_t).mean(0)
        B[i] = b1000.index_select(0, idxs_t).mean(0)

    head.weight.data.copy_(W.to(head.weight.device))
    head.bias.data.copy_(B.to(head.bias.device))
    print(
        "[INFO] Initialized EfficientNetV2-S 5-class head from ImageNet classifier (fallback)."
    )


def init_head_from_imagenet_resnet50(
    model, class_groups, weights_enum=ResNet50_Weights.IMAGENET1K_V2
):
    """
    model: resnet50 with fc = Linear(in_features, 5)
    """
    pretrained = models.resnet50(weights=weights_enum)
    w1000 = pretrained.fc.weight.detach().cpu()  # [1000, in_features]
    b1000 = pretrained.fc.bias.detach().cpu()  # [1000]

    if (
        not hasattr(model, "fc")
        or not isinstance(model.fc, nn.Linear)
        or model.fc.out_features != 5
    ):
        print("[WARN] Could not locate 5-class ResNet fc to initialize from ImageNet.")
        return

    W = torch.zeros((5, w1000.shape[1]), dtype=w1000.dtype)
    B = torch.zeros((5,), dtype=b1000.dtype)
    for i, idxs in enumerate(class_groups):
        idxs_t = torch.tensor(idxs, dtype=torch.long)
        W[i] = w1000.index_select(0, idxs_t).mean(0)
        B[i] = b1000.index_select(0, idxs_t).mean(0)

    model.fc.weight.data.copy_(W.to(model.fc.weight.device))
    model.fc.bias.data.copy_(B.to(model.fc.bias.device))
    print(
        "[INFO] Initialized ResNet50 5-class head from ImageNet classifier (fallback)."
    )




## === cell 11
imagenet_groups = [
    list(range(0, 200)),
    list(range(200, 400)),
    list(range(400, 600)),
    list(range(600, 800)),
    list(range(800, 1000)),
]



## === cell 12
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpt = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)
loaded_resnet = safe_load_state_dict(resnet_model, resnet_ckpt, device)
if not loaded_resnet:
    print(
        "[INFO] Using torchvision pretrained ResNet50 backbone weights (no custom ckpt loaded)."
    )
    init_head_from_imagenet_resnet50(
        resnet_model, imagenet_groups, ResNet50_Weights.IMAGENET1K_V2
    )

resnet_model = resnet_model.to(device)
resnet_model.eval()



## === cell 13
efficientnet_model_1 = models.efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)

eff1_ckpt = "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth"
loaded_eff1 = safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)
if not loaded_eff1:
    print(
        "[INFO] Using torchvision pretrained EfficientNetV2-S backbone weights for model_1 (no custom ckpt loaded)."
    )
    init_head_from_imagenet_effv2s(
        efficientnet_model_1, imagenet_groups, EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## === cell 14
efficientnet_model_7 = models.efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_ckpt = "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth"
loaded_eff7 = safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)
if not loaded_eff7:
    print(
        "[INFO] Using torchvision pretrained EfficientNetV2-S backbone weights for model_7 (no custom ckpt loaded)."
    )
    init_head_from_imagenet_effv2s(
        efficientnet_model_7, imagenet_groups, EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## === cell 15
efficientnet_model_8 = models.efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff8_ckpt = "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth"
loaded_eff8 = safe_load_state_dict(efficientnet_model_8, eff8_ckpt, device)
if not loaded_eff8:
    print(
        "[INFO] Using torchvision pretrained EfficientNetV2-S backbone weights for model_8 (no custom ckpt loaded)."
    )
    init_head_from_imagenet_effv2s(
        efficientnet_model_8, imagenet_groups, EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()



## === cell 16
weight_efficientnet = 0.7  # kept (not used in current core logic)
weight_resnet = 0.3  # kept (not used in current core logic)

ensemble_predictions = []
image_names = []

with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device, non_blocking=True)

        outputs_efficientnet1 = efficientnet_model_1(images)
        probs_efficientnet1 = F.softmax(outputs_efficientnet1, dim=1)

        outputs_efficientnet7 = efficientnet_model_7(images)
        probs_efficientnet7 = F.softmax(outputs_efficientnet7, dim=1)

        outputs_efficientnet8 = efficientnet_model_8(images)
        probs_efficientnet8 = F.softmax(outputs_efficientnet8, dim=1)

        combined_probs = (
            probs_efficientnet1 + probs_efficientnet7 + probs_efficientnet8
        ) / 3.0

        preds = combined_probs.argmax(dim=1).cpu().numpy()

        ensemble_predictions.extend(preds.tolist())
        image_names.extend(list(img_names))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/621815685.py in <cell line: 0>()
      6 
      7 with torch.no_grad():
----> 8     for images, img_names in test_loader:
      9         images = images.to(device, non_blocking=True)
     10 

NameError: name 'test_loader' is not defined

## === cell 17
pred_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})

submission_df = test_df[["image_id"]].merge(
    pred_df, on="image_id", how="left", validate="one_to_one"
)

if submission_df["label"].isna().any():
    n_missing = int(submission_df["label"].isna().sum())
    print(
        f"[WARN] {n_missing} test images missing predictions; filling with 0 to keep submission valid."
    )
    submission_df["label"] = submission_df["label"].fillna(0)

submission_df["label"] = submission_df["label"].astype(int)

assert len(submission_df) == len(
    test_df
), "Submission row count mismatch vs sample_submission."

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
