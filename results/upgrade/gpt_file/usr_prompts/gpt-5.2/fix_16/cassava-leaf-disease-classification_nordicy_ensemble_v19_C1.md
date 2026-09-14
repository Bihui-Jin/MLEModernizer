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

0.8756421879721971

# 6. Current score

0.62033

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.36958) has done: 'I fix the Albumentations `RandomResizedCrop` API error by using the v2 signature (`size=(h,w)`), which unblocks definition of `tta_transform`. Then I remove hard dependencies on missing external weight files by falling back to a standard ImageNet-pretrained EfficientNetV2-S when those files are not present, so the notebook runs end-to-end and produces predictions. Finally, I fix the submission length/format issue by ensuring `image_names` are plain strings (not tuples) and that the output `submission.csv` matches exactly the `sample_submission.csv` order and length.'
- What this solution (achieved 0.61846) has done: 'Your score is far below the target, and the main cause is that you are effectively submitting predictions from a randomly-initialized 5-class head (because the external fine-tuned checkpoints usually aren’t available), so accuracy collapses. The smallest change that preserves your core approach (same model family, same TTA loop, same argmax) is to keep EfficientNetV2-S ImageNet weights but replace the random 5-class classifier with a deterministic “ImageNet → Cassava” label mapping derived from the training set (majority cassava label per ImageNet top-1 class). I also fix the subtle but important bug where `img_name` becomes `"('xxx.jpg',)"` due to the custom collate, which breaks the mapping and forces many fillna(0) labels. These changes should move accuracy substantially upward toward your target while keeping the inference-only pipeline and TTA logic intact and still producing a valid `submission.csv`.'
- What this solution (achieved 0.61061) has done: 'We keep your exact inference-only + TTA + “ImageNet→Cassava majority map” core logic, but strengthen the mapping step so it generalizes better and reduces the gap to your target. The smallest high-impact change is to build the mapping from the full training set (not a 4k sample) and to do it efficiently on GPU by accumulating counts with `torch.bincount`, which fits within the time budget and avoids the slow Python loop. We also align preprocessing between mapping and prediction by using the same CLAHE+Resize pipeline (without random ops) for mapping so the ImageNet top-1 classes are more consistent with what you feed at test time. These changes should increase accuracy toward ~0.875 without changing model architectures, losses, or training loops (none exist here).'
- What this solution (achieved 0.61734) has done: 'Your current score is far below the target, so we should improve accuracy with the smallest possible change that keeps your inference-only + TTA + “ImageNet→Cassava majority map” approach intact. The main weakness is that the mapping uses only ImageNet top-1; we can make it more robust by accumulating Cassava label counts from ImageNet top-k (k=5) predictions per training image, then mapping each ImageNet class to the Cassava label it most often co-occurs with. This preserves your exact pipeline (no cassava training, same EfficientNetV2-S ImageNet backbone, same argmax-at-the-end semantics) but reduces mapping noise and typically boosts accuracy. I also keep submission alignment exactly to `sample_submission.csv` and keep all file paths unchanged.'
- What this solution (achieved 0.61248) has done: 'Your current score (0.61734) is well below the target (0.87564), and the biggest limiter is that your mapping uses only top-1 at inference even though you already built a top-k co-occurrence mapping. The smallest change that preserves your exact “ImageNet EfficientNetV2-S + TTA + mapping to cassava” core logic is to also use top-k at inference and aggregate the mapped cassava votes across (TTA × top-k) before taking the argmax. To keep behavior stable and avoid submission mismatches, I also make `img_name` unwrapping more robust and enforce that predictions are written in exactly the `sample_submission.csv` row order/length (same as you already do). This should move accuracy upward toward the target without changing model architecture, training, or loss (still inference-only).'
- What this solution (achieved 0.61248) has done: 'Your current gap to the target is large (0.61248 → 0.87564), so we should make a small, metric-aligned improvement without changing your core “ImageNet EfficientNetV2-S + TTA + ImageNet→Cassava mapping + argmax votes” logic. The biggest low-risk win is to stop mixing multiple random augmentations during TTA, because that adds distribution shift (especially `RandomResizedCrop` + heavy color/noise) and makes ImageNet top-k unstable; instead we use a deterministic, test-like TTA set (identity + horizontal flip) and increase the number of votes via more stable views. We also make the mapping step consistent with inference by using the exact same (deterministic) preprocessing used at inference, which typically improves the mapping quality. Finally, we keep submission ordering identical to `sample_submission.csv` and still output `submission.csv`.'
- What this solution (achieved 0.61248) has done: 'Your current score is far below the target, so we should nudge accuracy upward with the smallest possible change that preserves your inference-only + TTA + ImageNet→Cassava mapping voting logic. The biggest low-risk issue is that the mapping is built using only the non-flipped view, while inference uses both identity and horizontal flip; this mismatch makes the ImageNet top-k distribution less consistent and degrades the mapping quality. I build the mapping using the same two deterministic views (identity + flip) and aggregate their top-k co-occurrence counts, keeping everything else (model, transforms, voting/argmax, submission alignment) the same. This should improve mapping robustness and move the leaderboard score closer to your target without changing architecture/training/loss.'
- What this solution (achieved 0.61435) has done: 'Your score gap to the target is large (0.61248 → 0.87564), so we should improve accuracy with the smallest change that preserves your current inference-only + ImageNet→Cassava mapping + TTA voting approach. The biggest issue is that your mapping and inference currently use hard votes over top-k indices, which discards confidence information; we can keep the same top-k and voting semantics but weight votes by the ImageNet probabilities so stronger predictions contribute more. To keep behavior stable and avoid distribution shift, we keep your deterministic two-view TTA (identity + horizontal flip) and the same EfficientNetV2-S ImageNet backbone. Finally, we keep the submission alignment exactly to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.62519) has done: 'Your current score is far below the target, so we should improve accuracy with the smallest change that keeps your exact inference-only EfficientNetV2-S + deterministic (id+flip) TTA + ImageNet→Cassava mapping approach. The main weakness is that the mapping/inference currently collapses ImageNet outputs using only top-k class IDs, which can be noisy; we can keep the same top-k voting semantics but weight the mapping itself by ImageNet probabilities computed on the training set (same two deterministic views), making the mapping more confidence-aware. This doesn’t change any model architecture/training/loss (still none), and it keeps the same final argmax over 5 cassava classes. We also reduce submission risk by ensuring all filenames are unwrapped to plain strings and enforcing alignment to `sample_submission.csv`.'
- What this solution (achieved 0.62257) has done: 'Your current score (0.62519) is far below the target (0.87564), so we should improve accuracy with the smallest change that preserves your exact inference-only “ImageNet EfficientNetV2-S + deterministic id/flip TTA + ImageNet→Cassava mapping + weighted top-k votes + argmax” core logic. The main issue is that your mapping counts and inference votes use raw ImageNet probabilities, which are often miscalibrated; applying a simple temperature to sharpen probabilities typically makes the top-k signal cleaner and improves the mapping consistency without changing the approach. I add a single `TEMPERATURE` used in both mapping construction and inference (so they stay aligned), and keep everything else (transforms, TTA structure, submission alignment) unchanged. This is a minimal, metric-aligned tweak that should move your score upward toward the target band while staying within runtime and Kaggle constraints.'
- What this solution (achieved 0.62257) has done: 'Main bottlenecks are (1) building the ImageNet→Cassava mapping by running EfficientNet over all 18k train images twice (id+flip) and (2) per-test-image TTA running the same model 8 times with Python-loop overhead and repeated CPU↔GPU transfers. To fit 600s without changing the algorithm, the optimized version keeps identical transforms/model/mapping logic but makes it batch-oriented: compute TTA in a single forward per batch (stack N_TTA augmentations), and run test inference in real batches with multi-worker loading/pinned memory. It also removes expensive per-iteration `.cpu().numpy()` conversions by keeping top-k indices on GPU and using `mapping` as a GPU tensor for a provably equivalent gather+scatter_add implementation. Finally, it enables `torch.inference_mode()` and channels_last/cudnn/TF32 fastpaths (numerically negligible) to reduce overhead while preserving evaluation semantics.'
- What this solution (achieved 0.3651) has done: 'We keep your exact inference-only EfficientNetV2-S + (id/flip) deterministic TTA + ImageNet→Cassava mapping approach, but fix the biggest accuracy limiter: the mapping and inference currently treat all training images equally even though the Cassava train set is class-imbalanced, so the mapping tends to over-predict the majority class. The minimal metric-aligned improvement is to build the mapping with per-sample class-balancing weights (inverse frequency) while keeping the same top-k, temperature, probability smoothing, and id+flip views; this usually improves minority-class accuracy and should move your score upward toward the target. We also keep the submission alignment identical to `sample_submission.csv` and preserve your batching/throughput optimizations. No model architecture, training, loss, or TTA structure is changed—only the mapping accumulation weights.'
- What this solution (achieved 0.36659) has done: 'Your current score (0.3651) is far below the target (0.8756), and the largest accuracy killer in your current script is that you set `num_tta=8` while only having two deterministic views (id/flip), which makes you repeatedly re-run the *same* two views and effectively overweight them in a redundant way; this doesn’t add signal but increases instability and runtime, and can interact badly with probability smoothing/temperature. I make the TTA count consistent with the actual deterministic TTA set (2 views) while keeping the exact same inference-only EfficientNetV2-S ImageNet backbone, the same ImageNet→Cassava mapping construction, the same top-k weighted vote aggregation, and the same submission alignment. I also remove the unused extra cassava-head models from execution (ResNet/Eff heads) to avoid any accidental interference and wasted load time (they were never used for prediction anyway), keeping all core prediction logic unchanged. These minimal, directly-relevant changes should increase accuracy toward your target by making the mapping/inference more consistent and less noisy, without changing architecture/training/loss semantics.'
- What this solution (achieved 0.62033) has done: 'Your current score is far below the target, and the most direct minimal change is to remove the class-balanced weighting you added into the ImageNet→Cassava mapping step, because it heavily distorts the co-occurrence counts and in practice collapses accuracy (as seen by the drop to ~0.36). I keep your exact core approach (ImageNet EfficientNetV2-S, deterministic id+flip TTA, top-k probability-weighted mapping/votes, same temperature/smoothing, same argmax) and only change the mapping accumulation to be unweighted by class frequency. I also keep the submission alignment exactly to `sample_submission.csv` and still write `submission.csv`. This should move the score back upward toward your previous ~0.62+ band (closer to the 0.8756 target) without changing architecture/training/loss semantics.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
num_tta = 2

TEMPERATURE = 0.7
PROB_SMOOTH = 0.02

USE_CLASS_BALANCED_MAPPING = False




## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"




## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()




## === cell 4
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
tta_transform = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

tta_transform_flip = A.Compose(
    [
        A.HorizontalFlip(p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
def cassava_collate(batch):
    images, names = zip(*batch)
    return list(images), list(names)


test_dataset = CassavaTestDataset(test_df, test_image_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,  # batching only changes throughput, not predictions
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if (os.cpu_count() or 1) > 1 else False,
    collate_fn=cassava_collate,
)




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 9
train_df = pd.read_csv(train_csv_path)

map_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
map_df = train_df.reset_index(drop=True)

mapping_transform = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

mapping_transform_flip = A.Compose(
    [
        A.HorizontalFlip(p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class CassavaMapDataset(Dataset):
    def __init__(self, dataframe, image_dir, class_weight_map=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.class_weight_map = class_weight_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = mapping_transform(image=image)["image"]
        x_flip = mapping_transform_flip(image=image)["image"]

        if self.class_weight_map is None:
            w = 1.0
        else:
            w = float(self.class_weight_map[y])

        return x, x_flip, y, w


label_counts = train_df["label"].value_counts().sort_index()
inv_freq = 1.0 / label_counts.values.astype(np.float32)
inv_freq = inv_freq / inv_freq.mean()
class_weight_map = (
    {int(i): float(inv_freq[i]) for i in range(len(inv_freq))}
    if USE_CLASS_BALANCED_MAPPING
    else None
)

map_loader = DataLoader(
    CassavaMapDataset(map_df, map_image_dir, class_weight_map=class_weight_map),
    batch_size=64,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if (os.cpu_count() or 1) > 1 else False,
)

imagenet_effnet = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
).to(device)
imagenet_effnet.eval()

if torch.cuda.is_available():
    imagenet_effnet = imagenet_effnet.to(memory_format=torch.channels_last)

topk_for_mapping = 5
counts = torch.zeros((1000, 5), dtype=torch.float32, device=device)

with torch.inference_mode():
    for xb, xb_flip, yb, wb in tqdm(
        map_loader,
        total=len(map_loader),
        desc="Building ImageNet->Cassava map (top-k, prob-weighted, id+flip)",
    ):
        xb = xb.to(device, non_blocking=True)
        xb_flip = xb_flip.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()  # [B]

        wb = torch.ones_like(yb, dtype=torch.float32, device=device)

        if torch.cuda.is_available():
            xb = xb.to(memory_format=torch.channels_last)
            xb_flip = xb_flip.to(memory_format=torch.channels_last)

        logits = imagenet_effnet(xb)  # [B,1000]
        probs = F.softmax(logits / TEMPERATURE, dim=1)  # [B,1000]
        topk = probs.topk(k=topk_for_mapping, dim=1)  # values+indices
        topk_idx = topk.indices.long()  # [B,k]
        topk_val = topk.values  # [B,k] float

        if PROB_SMOOTH > 0:
            topk_val = topk_val * (1.0 - PROB_SMOOTH) + (PROB_SMOOTH / topk_for_mapping)

        topk_val = topk_val * wb[:, None]

        yb_rep = yb[:, None].expand(-1, topk_for_mapping)  # [B,k]
        idx = (topk_idx.reshape(-1) * 5 + yb_rep.reshape(-1)).long()  # [B*k]
        counts += torch.bincount(
            idx, weights=topk_val.reshape(-1), minlength=1000 * 5
        ).view(1000, 5)

        logits_f = imagenet_effnet(xb_flip)
        probs_f = F.softmax(logits_f / TEMPERATURE, dim=1)
        topk_f = probs_f.topk(k=topk_for_mapping, dim=1)
        topk_f_idx = topk_f.indices.long()
        topk_f_val = topk_f.values

        if PROB_SMOOTH > 0:
            topk_f_val = topk_f_val * (1.0 - PROB_SMOOTH) + (
                PROB_SMOOTH / topk_for_mapping
            )

        topk_f_val = topk_f_val * wb[:, None]

        idx_f = (topk_f_idx.reshape(-1) * 5 + yb_rep.reshape(-1)).long()
        counts += torch.bincount(
            idx_f, weights=topk_f_val.reshape(-1), minlength=1000 * 5
        ).view(1000, 5)

global_majority = int(train_df["label"].value_counts().idxmax())

mapping = counts.argmax(dim=1).detach().cpu().numpy().astype(int)
seen = counts.sum(dim=1).detach().cpu().numpy()
mapping[seen == 0] = global_majority

mapping[:10], global_majority




## === cell 10
pass




## === cell 11
pass




## === cell 12
pass




## === cell 13
pass




## === cell 14
def _unwrap_img_name(x):
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return str(x)


def tta_predict_imagenet_to_cassava(
    imagenet_model,
    image,
    tta_transform_a,
    tta_transform_b,
    device,
    mapping_arr_or_tensor,
    n_tta=2,
    topk_infer=5,
):
    imagenet_model.eval()

    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (H, W, 3)")

    if isinstance(mapping_arr_or_tensor, np.ndarray):
        mapping_t = torch.from_numpy(mapping_arr_or_tensor.astype(np.int64)).to(device)
    else:
        mapping_t = mapping_arr_or_tensor.to(device=device)

    votes = torch.zeros((5,), dtype=torch.float32, device=device)

    aug_tensors = []
    for i in range(n_tta):
        tfm = tta_transform_a if (i % 2 == 0) else tta_transform_b
        aug_tensors.append(tfm(image=image)["image"])
    xb = torch.stack(aug_tensors, dim=0).to(device, non_blocking=True)  # [T,3,H,W]

    if torch.cuda.is_available():
        xb = xb.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        logits = imagenet_model(xb)  # [T,1000]
        probs = F.softmax(logits / TEMPERATURE, dim=1)  # [T,1000]
        topk = probs.topk(k=topk_infer, dim=1)  # values [T,k], indices [T,k]

        topk_idx = topk.indices.long()  # [T,k]
        topk_prob = topk.values  # [T,k]

        if PROB_SMOOTH > 0:
            topk_prob = topk_prob * (1.0 - PROB_SMOOTH) + (PROB_SMOOTH / topk_infer)

        cassava_idx = (
            mapping_t.gather(0, topk_idx.reshape(-1)).reshape_as(topk_idx).long()
        )

        votes.scatter_add_(0, cassava_idx.reshape(-1), topk_prob.reshape(-1))

    return int(votes.argmax().item())


mapping_t_device = torch.from_numpy(mapping.astype(np.int64)).to(device)

ensemble_predictions = []
image_names = []

for images, img_names in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
    for image, img_name in zip(images, img_names):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        img_name = _unwrap_img_name(img_name)

        pred = tta_predict_imagenet_to_cassava(
            imagenet_effnet,
            image,
            tta_transform,
            tta_transform_flip,
            device,
            mapping_t_device,
            n_tta=num_tta,
            topk_infer=5,
        )
        ensemble_predictions.append(int(pred))
        image_names.append(img_name)

len(image_names), len(ensemble_predictions), len(test_df)




## === cell 15
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map)

submission_df["label"] = submission_df["label"].fillna(global_majority).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Saved: {submission_path}")
print(submission_df.head())
print(
    "Rows:", len(submission_df), "Missing labels:", submission_df["label"].isna().sum()
)
