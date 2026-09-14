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

0.8771532184950136

# 6. Current score

0.46824

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23281) has done: 'Your notebook currently can’t yield a Kaggle score because it fail at runtime due to missing external weight files (`/kaggle/input/casava-aug/...` and `/kaggle/input/eff-t/...` are not in the provided data paths). To make it run end-to-end and generate a valid `submission.csv`, I keep your exact inference/ensemble logic but add a safe fallback: load ImageNet pretrained weights when the custom `.pth` files aren’t available. I also ensure transforms match each model’s expected input size by running EfficientNet models on 384px inputs (as you already do) and running ResNet on a separate 224px loader so ResNet inference doesn’t crash on shape mismatch. These are minimal changes focused on producing a valid submission and improving accuracy versus random initialization, moving you toward the target score.'
- What this solution (achieved 0.18535) has done: 'Your current ensemble votes on hard class labels from three EfficientNet heads, then only uses ResNet as a tie-breaker; when you fall back to ImageNet weights (no cassava finetuned checkpoints), those randomly-initialized 5-class heads produce near-random argmaxes and dominate the vote, explaining the very low score. To move accuracy up toward the target with minimal logic change, I keep your exact models/transforms and still ensemble the same models, but switch the vote from “argmax majority” to averaging softmax probabilities across all available models (including ResNet) and then taking argmax once. I also make the checkpoint loader tolerant to common checkpoint formats (`state_dict` key, `module.` prefix) so you actually benefit if the provided `.pth` files exist in some environments. This stays within the same inference-only approach, produces the same submission format, and should improve score substantially toward your target.'
- What this solution (achieved 0.14537) has done: 'Your score is far below target because, in the fallback (no finetuned checkpoints available), you replace the final classifier layers with new random 5-class heads, so the ensemble predictions are essentially random even though the backbones are ImageNet-pretrained. To move accuracy strongly upward with minimal core-logic change (still inference-only, same models, same loaders/transforms, same softmax-averaging ensemble), I keep your exact architectures but change the fallback behavior: instead of random heads, use the ImageNet logits and map the ImageNet top-1 class name to one of the 5 cassava labels via a small keyword-based mapping. This yields a deterministic, non-random prediction rule that should substantially improve accuracy versus ~0.18 while remaining within the same evaluation semantics (still predicting labels 0–4 per image). I also keep the checkpoint loader tolerant and only apply the mapping fallback when the cassava 5-class checkpoint isn’t found/loaded, so if you later provide proper weights, the original intended 5-class inference path is used.'
- What this solution (achieved 0.61099) has done: 'Your current score is low because in this environment the finetuned `.pth` checkpoints are missing, so you fall back to ImageNet models and then use a brittle keyword-based mapping from ImageNet class names to cassava labels (which is effectively random for this task). To move strongly toward the target with minimal core-logic change, I keep your exact models/loaders/transforms and ensemble flow, but change the *fallback prediction rule* to a deterministic, data-driven prior: use the training-set label distribution (majority-class prior) when no 5-class checkpoint is loaded. I also fix the dataset path to the one that actually exists in your file tree (`/kaggle/input/...`), so the code reliably runs end-to-end and writes `submission.csv`. If any 5-class checkpoint is available, your original softmax-averaging 5-class ensemble path is unchanged.'
- What this solution (achieved 0.59305) has done: 'Your current 0.61099 score comes from always predicting the global majority class when no 5-class finetuned checkpoints are found, which is a decent but limited prior. To move toward the 0.877 target without changing your core inference/ensemble structure, I upgrade only the *fallback* to a stronger, still data-driven prior: predict the majority label **conditioned on the test image’s mean color**, using binning learned from train images. This keeps everything inference-only and deterministic, uses only the provided train/test images, and should improve accuracy over the single-class prior while preserving your existing “use 5-class checkpoints if available, otherwise fallback” logic. I also speed up image reading slightly and keep output formatting identical, still writing a valid `submission.csv`.'
- What this solution (achieved 0.42862) has done: 'To move your score up toward 0.877 with minimal core changes, I keep your exact model definitions and ensemble structure, but strengthen the *fallback* (when no 5-class checkpoints load) so it’s no longer limited to “mean-color bin → majority label”. Concretely, I replace the fallback prior with a lightweight, deterministic kNN classifier over downsampled RGB thumbnails: build features for a subset of train images, then predict each test image by majority vote of its nearest neighbors; this is still inference-only and uses only provided train/test images. I also make the train subset stratified and use all available CPU workers to fit within the 600s budget while improving the fallback’s correlation with true labels. Submission formatting, paths, and the “use checkpoints if present, else fallback” semantics remain unchanged.'
- What this solution (achieved 0.43759) has done: 'Your current gap to the target is large (0.42862 vs 0.87715), and the main limiter is that the finetuned checkpoints aren’t available, so you’re relying on the fallback kNN thumbnail classifier. To move accuracy upward without changing your core model/ensemble logic, I strengthen only the fallback by (1) using a more discriminative thumbnail feature (RGB + Lab, slightly larger) and (2) switching from unweighted kNN to distance-weighted voting, which usually improves nearest-neighbor accuracy with minimal semantic change. I also make the fallback train subset slightly larger but keep it bounded and deterministic so runtime stays under the 600s limit. Everything else (paths, loaders, models, “use 5-class checkpoints if available, else fallback”, and submission writing) remains the same.'
- What this solution (achieved 0.32586) has done: 'Your current score (0.43759) is far below the target (0.87715), and since the finetuned checkpoints are missing, the only lever we have (without changing your model/ensemble core logic) is to strengthen the *fallback* kNN thumbnail classifier. I keep the exact same fallback approach (thumbnail feature → kNN distance-weighted vote) but make two minimal, high-impact, deterministic upgrades: add simple per-feature standardization (fit on the fallback train thumbnails) so distance is better-behaved, and switch to cosine-distance kNN (implemented with normalized vectors) which is typically more robust for concatenated color features. I also mildly increase the thumbnail size to 48 (still fast) and slightly increase K to 11 to reduce variance; the rest of the pipeline, including “use 5-class checkpoints if available else fallback”, stays unchanged and it still write `submission.csv`.'
- What this solution (achieved 0.49925) has done: 'Your current score is far below target because no finetuned checkpoints are available, so performance depends entirely on the fallback kNN thumbnail classifier. To move accuracy up with minimal change (keeping the same fallback approach and ensemble semantics), I only strengthen the fallback features: apply a small per-image normalization (mean/std) to reduce lighting/background bias, and add compact texture cues via simple gradient-magnitude downsampled thumbnails. I also increase the fallback prior/kNN train subset sizes slightly (still bounded) to reduce variance, and keep the same cosine-similarity + distance-weighted voting and the same submission writing. These changes are deterministic, inference-only, and should improve the kNN’s correlation with true cassava labels, nudging the score closer to the 0.877 target.'
- What this solution (achieved 0.46824) has done: 'Your score is far below the target, and since no finetuned checkpoints are available in this environment the entire performance is coming from the fallback kNN thumbnail classifier. To move accuracy upward with minimal change while preserving the same overall “use checkpoints if available else fallback” semantics, I only strengthen the fallback representation in a deterministic way: add a compact HSV thumbnail (good for discoloration patterns) and a simple local-contrast (blur-residual) texture thumbnail. I also make the kNN voting slightly more robust by using a softmax over similarities (temperature) instead of 1/(d+eps), which reduces sensitivity to very small distances without changing the kNN core logic. Everything else (data paths, loaders, model definitions, ensemble averaging when checkpoints exist, and writing `submission.csv`) remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava dataset root. Tried:\n"
        + "\n".join(DATA_ROOT_CANDIDATES)
    )

test_image_dir = f"{DATA_ROOT}/test_images"
train_image_dir = f"{DATA_ROOT}/train_images"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
train_csv_path = f"{DATA_ROOT}/train.csv"

test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 3
resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 5
test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_loader_eff = DataLoader(
    test_dataset_eff,
    batch_size=32,
    shuffle=False,
    num_workers=min(4, (os.cpu_count() or 1)),
    pin_memory=torch.cuda.is_available(),
)

test_dataset_res = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms
)
test_loader_res = DataLoader(
    test_dataset_res,
    batch_size=32,
    shuffle=False,
    num_workers=min(4, (os.cpu_count() or 1)),
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 7
def safe_load_state_dict(model, path, map_location):
    if path is None or (not os.path.exists(path)):
        return False
    ckpt = torch.load(path, map_location=map_location)

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    else:
        return False

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        try:
            model.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False




## === cell 8
resnet_ckpt_path = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)

resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
loaded_resnet = safe_load_state_dict(
    resnet_model, resnet_ckpt_path, map_location=device
)

use_resnet_imagenet_fallback = False
if not loaded_resnet:
    resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    use_resnet_imagenet_fallback = True

resnet_model = resnet_model.to(device)
resnet_model.eval()

loaded_resnet, use_resnet_imagenet_fallback



## === cell 9
eff_ckpt_path = "/kaggle/input/eff-t/pytorch/default/1/Eff.pth"

efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)
loaded_eff1 = safe_load_state_dict(
    efficientnet_model_1, eff_ckpt_path, map_location=device
)

use_eff1_imagenet_fallback = False
if not loaded_eff1:
    efficientnet_model_1 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    use_eff1_imagenet_fallback = True

efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()

loaded_eff1, use_eff1_imagenet_fallback



## === cell 10
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8),
    nn.Linear(num_features_efficientnet, 5),
)
loaded_eff7 = safe_load_state_dict(
    efficientnet_model_7, eff_ckpt_path, map_location=device
)

use_eff7_imagenet_fallback = False
if not loaded_eff7:
    efficientnet_model_7 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    use_eff7_imagenet_fallback = True

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()

loaded_eff7, use_eff7_imagenet_fallback



## === cell 11
efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8),
    nn.Linear(num_features_efficientnet, 5),
)
loaded_eff8 = safe_load_state_dict(
    efficientnet_model_8, eff_ckpt_path, map_location=device
)

use_eff8_imagenet_fallback = False
if not loaded_eff8:
    efficientnet_model_8 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    use_eff8_imagenet_fallback = True

efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()

loaded_eff8, use_eff8_imagenet_fallback



## === cell 12
train_df = pd.read_csv(train_csv_path)
label_counts = train_df["label"].value_counts().sort_index()
majority_label = int(label_counts.idxmax())

eff_imagenet_cats = EfficientNet_V2_S_Weights.IMAGENET1K_V1.meta.get("categories", None)
res_imagenet_cats = ResNet50_Weights.IMAGENET1K_V2.meta.get("categories", None)

majority_label, label_counts.to_dict()




## === cell 13
def _mean_rgb_16bin_from_path(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m = img.reshape(-1, 3).mean(axis=0)  # float64
    b = np.clip((m / 16.0).astype(np.int32), 0, 15)
    return int(b[0] * 256 + b[1] * 16 + b[2])


bin_to_counts = {}

MAX_TRAIN_FOR_PRIOR = 9000

train_slice = train_df.iloc[: min(len(train_df), MAX_TRAIN_FOR_PRIOR)]

for img_id, lbl in zip(train_slice["image_id"].values, train_slice["label"].values):
    p = os.path.join(train_image_dir, img_id)
    b = _mean_rgb_16bin_from_path(p)
    if b is None:
        continue
    if b not in bin_to_counts:
        bin_to_counts[b] = np.zeros(5, dtype=np.int32)
    bin_to_counts[b][int(lbl)] += 1

bin_to_majority = {
    b: int(np.argmax(c)) for b, c in bin_to_counts.items() if c.sum() > 0
}

len(bin_to_majority), majority_label




## === cell 14
def _thumb_feature_from_path(img_path: str, size: int = 48):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rgb = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    img_rgb = img_rgb / 255.0

    mu = img_rgb.reshape(-1, 3).mean(axis=0, dtype=np.float32)
    sd = img_rgb.reshape(-1, 3).std(axis=0, dtype=np.float32)
    sd = np.maximum(sd, 1e-3).astype(np.float32)
    img_rgb_n = (img_rgb - mu) / sd

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_lab = cv2.resize(img_lab, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    img_lab[..., 0] = img_lab[..., 0] / 255.0
    img_lab[..., 1] = img_lab[..., 1] / 255.0
    img_lab[..., 2] = img_lab[..., 2] / 255.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    gray = gray / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    gmag = np.clip(gmag, 0.0, 1.0)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv = cv2.resize(img_hsv, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    img_hsv[..., 0] = img_hsv[..., 0] / 179.0  # OpenCV H range
    img_hsv[..., 1] = img_hsv[..., 1] / 255.0
    img_hsv[..., 2] = img_hsv[..., 2] / 255.0

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    resid = gray - blur
    resid = np.clip(resid, -1.0, 1.0).astype(np.float32)

    feat = np.concatenate(
        [
            img_rgb_n.reshape(-1),
            img_lab.reshape(-1),
            gmag.reshape(-1),
            img_hsv.reshape(-1),
            resid.reshape(-1),
        ],
        axis=0,
    )
    return feat.astype(np.float32, copy=False)


RNG_SEED = 1337
MAX_TRAIN_KNN = 16000
KNN_K = 11

by_label = train_df.groupby("label", sort=True)
per_class = max(1, MAX_TRAIN_KNN // 5)
train_knn_idx = []
rng = np.random.default_rng(RNG_SEED)
for lbl, g in by_label:
    idx = g.index.to_numpy()
    if len(idx) <= per_class:
        chosen = idx
    else:
        chosen = rng.choice(idx, size=per_class, replace=False)
    train_knn_idx.append(chosen)
train_knn_idx = np.concatenate(train_knn_idx)
if len(train_knn_idx) < min(MAX_TRAIN_KNN, len(train_df)):
    remaining = np.setdiff1d(
        train_df.index.to_numpy(), train_knn_idx, assume_unique=False
    )
    need = min(MAX_TRAIN_KNN, len(train_df)) - len(train_knn_idx)
    if need > 0 and len(remaining) > 0:
        add = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        train_knn_idx = np.concatenate([train_knn_idx, add])

train_knn_df = train_df.loc[train_knn_idx].reset_index(drop=True)

X_train = []
y_train = []
for img_id, lbl in zip(train_knn_df["image_id"].values, train_knn_df["label"].values):
    p = os.path.join(train_image_dir, img_id)
    f = _thumb_feature_from_path(p, size=48)
    if f is None:
        continue
    X_train.append(f)
    y_train.append(int(lbl))

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)

fallback_label_if_no_knn = majority_label

if X_train.size > 0:
    feat_mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    feat_std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6).astype(np.float32)
    X_train_std = (X_train - feat_mean) / feat_std

    X_train_norm = X_train_std / (
        np.linalg.norm(X_train_std, axis=1, keepdims=True) + 1e-12
    )
else:
    feat_mean = None
    feat_std = None
    X_train_std = X_train
    X_train_norm = X_train

X_train.shape, int((y_train == majority_label).mean() * 100)



## === cell 15
ensemble_predictions = []
image_names = []

softmax = nn.Softmax(dim=1)

KNN_TAU = 0.07  # temperature; smaller -> sharper, larger -> smoother


def _knn_predict_one(test_feat: np.ndarray):
    if X_train.size == 0:
        return fallback_label_if_no_knn

    tf = test_feat
    tf = (tf - feat_mean) / feat_std
    tf = tf / (np.linalg.norm(tf) + 1e-12)

    sims = (X_train_norm @ tf.astype(np.float32)).astype(np.float32, copy=False)

    k = min(KNN_K, len(sims))
    nn_idx = np.argpartition(-sims, kth=k - 1)[:k]
    nn_s = sims[nn_idx].astype(np.float64, copy=False)
    nn_y = y_train[nn_idx]

    z = (nn_s / float(KNN_TAU)).astype(np.float64, copy=False)
    z = z - z.max()
    w = np.exp(z)
    w = w / (w.sum() + 1e-12)

    scores = np.zeros(5, dtype=np.float64)
    for yi, wi in zip(nn_y, w):
        scores[int(yi)] += float(wi)

    top = np.where(scores == scores.max())[0]
    if len(top) == 1:
        return int(top[0])
    if majority_label in top:
        return int(majority_label)
    return int(top.min())


with torch.no_grad():
    for (images_eff, img_names_eff), (images_res, img_names_res) in zip(
        test_loader_eff, test_loader_res
    ):
        assert list(img_names_eff) == list(
            img_names_res
        ), "Loader order mismatch; cannot ensemble safely."

        images_eff = images_eff.to(device, non_blocking=True)
        images_res = images_res.to(device, non_blocking=True)

        have_any_5class = loaded_eff1 or loaded_eff7 or loaded_eff8 or loaded_resnet

        if have_any_5class:
            probs_sum = 0.0
            n_models = 0

            out1 = efficientnet_model_1(images_eff)
            if loaded_eff1:
                probs_sum = probs_sum + softmax(out1)
                n_models += 1

            out7 = efficientnet_model_7(images_eff)
            if loaded_eff7:
                probs_sum = probs_sum + softmax(out7)
                n_models += 1

            out8 = efficientnet_model_8(images_eff)
            if loaded_eff8:
                probs_sum = probs_sum + softmax(out8)
                n_models += 1

            outr = resnet_model(images_res)
            if loaded_resnet:
                probs_sum = probs_sum + softmax(outr)
                n_models += 1

            if n_models > 0:
                probs_avg = probs_sum / float(n_models)
                preds = (
                    probs_avg.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
                )
            else:
                have_any_5class = False  # fall through

        if not have_any_5class:
            preds = []
            for nm in img_names_eff:
                tf = _thumb_feature_from_path(os.path.join(test_image_dir, nm), size=48)
                if tf is not None:
                    preds.append(_knn_predict_one(tf))
                    continue

                b = _mean_rgb_16bin_from_path(os.path.join(test_image_dir, nm))
                if b is not None and b in bin_to_majority:
                    preds.append(bin_to_majority[b])
                else:
                    preds.append(majority_label)

        ensemble_predictions.extend(preds)
        image_names.extend(img_names_eff)



## === cell 16
sub = test_df.copy()
pred_map = dict(zip(image_names, ensemble_predictions))
sub["label"] = sub["image_id"].map(pred_map).astype(int)

assert sub.shape[0] == test_df.shape[0]
assert sub["label"].between(0, 4).all()

sub.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(sub.head())
