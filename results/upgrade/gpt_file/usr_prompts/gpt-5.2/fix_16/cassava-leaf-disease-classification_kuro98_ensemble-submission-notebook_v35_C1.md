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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8939256572982774

# 6. Current score

0.75673

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it runs end-to-end in the Kaggle filesystem by removing the hard dependency on missing external `.pt` model files and replacing them with a lightweight, local torchvision model that can be loaded without extra inputs. I also fix submission-length/alignment issues by iterating the test set in a deterministic order (no shuffle) and writing predictions in exactly the same order as `sample_submission.csv` expects. To keep runtime safe and produce a valid CSV reliably, the model run in `eval()` with `no_grad()` and map ImageNet logits to 5 classes via a fixed linear head (score be modest but valid, and the main goal here is correctness + submission creation). Finally, I add robust path handling so it works whether the dataset is under `/kaggle/input/...` or mirrored under `/kaggle/data/...`.'
- What this solution (achieved 0.66069) has done: 'Your current 0.05531 score is mainly because the model head is initialized to all-zeros, so it predicts the same class for every image. To move the score toward the 0.8939 target while keeping your core inference-only pipeline intact, I train only the existing linear projection head (`proj`) on `train.csv` using frozen EfficientNet-B0 features, then reuse the same preprocessing and ordered test inference to write `submission.csv`. This is a minimal change: same backbone, same head type, same transforms, and no architectural changes—just fitting the already-present head on the provided training labels. I also ensure label types and file alignment remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.71824) has done: 'Main bottlenecks are (1) repeatedly decoding/resizing images on CPU for every epoch, (2) running the frozen EfficientNet backbone inside the training loop for all 18.7k images × 6 epochs, and (3) DataLoader overhead without persistent workers/prefetch. To preserve identical training semantics while cutting runtime drastically, the optimized script precomputes the frozen backbone’s 1000-dim features for the entire training set once (same transforms), then trains the linear projection on those cached features for the same number of epochs/batches. It also enables DataLoader performance flags (persistent_workers, prefetch_factor) and avoids redundant model_b image work since model_b inputs are never used. Inference logic and submission formatting remain unchanged.'
- What this solution (achieved 0.76756) has done: 'Your current gap to the target is large (0.71824 → 0.89393), so the smallest score-moving change is to fix the training mismatch: EfficientNet’s forward output is 1000-class logits, not stable “features”, so training `proj` on those logits is weak. We keep the same backbone (EfficientNet-B0) and the same linear head + loss/training loop, but we extract proper penultimate features (1280-d) from the model’s `features + avgpool` and update `proj` accordingly. This preserves the approach (frozen backbone + train only a linear head) while materially improving separability and should move accuracy substantially toward your target. Submission ordering/format and the rest of the pipeline remain unchanged.'
- What this solution (achieved 0.76046) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end: (1) `weights.meta["mean"/"std"]` is not present in this torchvision version, so I use the standard ImageNet mean/std (matching the pretrained weights) to build `train_transforms`. Because cell 2 currently crashes, downstream variables like `train_transforms`, `sample_sub`, and `test_loader` never get defined; fixing cell 2 resolves the cascade of `NameError`s in later cells. I keep the core approach unchanged (frozen EfficientNet-B0 feature extractor + trained linear head) and preserve submission ordering by iterating exactly in `sample_submission.csv` order. Finally, I add a tiny safety check to ensure all mapped labels are present before writing `submission.csv`.'
- What this solution (achieved 0.76495) has done: 'Your current score (0.76046) is well below the target (0.89393), so we should improve accuracy with the smallest change that keeps your frozen EfficientNet-B0 + linear head approach intact. The biggest low-risk gain here is to remove the train/test preprocessing mismatch: you currently train with strong random crops but infer with EfficientNet’s center-crop preprocessing, which can hurt generalization for a linear head. I switch training to use the exact same `weights.transforms()` preprocessing as test (no augmentation), keeping the same backbone, head, loss, optimizer, and epoch count. This typically increases stability and accuracy for this “precompute features + train linear head” pipeline without altering core logic.'
- What this solution (achieved 0.71749) has done: 'Your current score (0.76495) is well below the target (0.89393), so we should improve accuracy with the smallest change that keeps your frozen EfficientNet-B0 + linear head approach intact. The lowest-risk gain is to fix class-imbalance bias in the linear-head training by using weighted cross-entropy computed from `train.csv` label frequencies; this does not change the model, features, or loop structure, only the loss weighting. I also make the run fully deterministic (same semantics, just reduced randomness) so the score is more stable and easier to move toward the target. Everything else (precompute frozen features once, train only `proj`, same preprocessing for train/test, and submission ordering) remains unchanged.'
- What this solution (achieved 0.72272) has done: 'Your current score (0.71749) is far below the target (0.89393), so we should improve accuracy with the smallest change that preserves your core approach (frozen EfficientNet-B0 feature extractor + trained linear head on cached features). The most direct, low-risk gain is to train the linear head with a simple validation split and pick the best-epoch weights (still same model, same loss, same optimizer, same number of epochs; we just keep the best checkpoint instead of the last one). This typically avoids ending on an overfit epoch and moves accuracy upward without changing inference semantics or submission formatting. I also keep your deterministic setup and ensure feature extraction is still done once (fast) and the final submission ordering remains identical to `sample_submission.csv`.'
- What this solution (achieved 0.73879) has done: 'Your current score (0.72272) is far below the target (0.89393), so we should increase accuracy with the smallest safe change that preserves your frozen EfficientNet-B0 feature extractor + linear head training/inference pipeline. The biggest issue is that you select the “best epoch” using a single random 90/10 split, which is noisy and can pick a suboptimal head; switching to stratified K-fold validation to choose the best head (still training the same `proj` with the same loss/optimizer/epochs) usually gives a sizable, stable boost. To keep runtime within limits, we still precompute frozen features once for the whole train set, then reuse them across folds (no repeated image decoding/backbone forward per fold). Finally, we average test logits across the K fold-best heads (same semantics: argmax over logits) to improve generalization without changing the model architecture.'
- What this solution (achieved 0.77541) has done: 'Your current score (0.73879) is well below the target (0.89393), so we should increase accuracy with the smallest change that preserves your frozen EfficientNet-B0 feature extractor + linear head training/inference pipeline. The most direct improvement without changing architecture or training semantics is to reduce the bias from fold-specific class-weighted losses by switching to an “effective number of samples” weighting (smoother than pure inverse-frequency) and then normalizing weights; this usually improves macro performance on imbalanced Cassava labels and tends to lift public accuracy. Additionally, we remove an inefficiency in test-time ensembling that re-creates and re-loads the fold head inside every batch; we pre-build the fold heads once (same weights, same logits), which doesn’t change predictions but makes runtime more stable within the 600s limit. Finally, we keep everything else identical: same preprocessing, cached frozen features, same K-fold selection of best epoch, and the same submission ordering aligned to `sample_submission.csv`.'
- What this solution (achieved 0.7287) has done: 'Your current score (0.77541) is still far below the target (0.89393), so we should make a small, low-risk change that improves generalization without changing the core approach (frozen EfficientNet-B0 feature extractor + linear head training on cached features + K-fold selection + averaged logits). The smallest meaningful improvement is to train the linear heads on L2-normalized features and also apply the same normalization at inference; this keeps the backbone/heads/loss/training loop intact, but makes optimization better-conditioned and typically improves linear-probe accuracy. I also (minimally) adjust the learning rate slightly downward to match the new feature scaling and reduce fold-to-fold instability, keeping epochs/optimizer/loss/architecture the same. Submission ordering/format and all file paths remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.75897) has done: 'Your current score (0.7287) is far below the target (0.8939), so we need a small change that improves generalization without changing the core “frozen EfficientNet-B0 + linear head(s) on cached features + K-fold + averaged logits” approach. The highest-impact minimal fix is to use a stronger pretrained backbone within the same architecture family: switch from EfficientNet-B0 to EfficientNet-B3 while keeping identical feature extraction (features+avgpool), the same linear head training loop, the same K-fold procedure, and the same submission formatting. This typically boosts linear-probe accuracy substantially on Cassava while remaining lightweight enough for the 600s limit because we still precompute features once. I’m also keeping preprocessing consistent by using the B3 weights’ own transforms for both train and test to avoid train/test mismatch.'
- What this solution (achieved 0.75673) has done: 'Your current score is far below the target, so we should improve accuracy without changing the core “frozen EfficientNet-B3 feature extractor + cached features + K-fold trained linear heads + averaged logits” pipeline. The smallest high-impact fix is to correct the EfficientNet-B3 preprocessing: `weights.transforms()` for B3 uses a larger `crop_size` (320), but your `model_input_size` is set to 300, creating a subtle train/test mismatch in resizing/cropping that harms a linear probe. I (1) set `model_input_size` directly from the weights transform’s `crop_size`, and (2) rebuild `preprocess` explicitly with the correct `resize_size`/`crop_size` and ImageNet mean/std, so both training feature caching and test inference see identical, correct inputs. Everything else (architecture, feature extraction, loss, optimizer, epochs, K-fold, ensembling, submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
import torchvision

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(f"Could not find dataset root in: {DATA_ROOT_CANDIDATES}")

test_dir = os.path.join(DATA_ROOT, "test_images")
train_dir = os.path.join(DATA_ROOT, "train_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_csv_path = os.path.join(DATA_ROOT, "train.csv")

weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT

_wt = weights.transforms()
_crop = (
    _wt.crop_size[0] if isinstance(_wt.crop_size, (tuple, list)) else int(_wt.crop_size)
)
_resize = (
    _wt.resize_size[0]
    if isinstance(_wt.resize_size, (tuple, list))
    else int(_wt.resize_size)
)

preprocess = v2.Compose(
    [
        v2.Resize(_resize, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.CenterCrop(_crop),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model_input_size = _crop  # B3 default crop size used by weights.transforms()

model_b_img_size = model_input_size
model_a_img_size = model_input_size
batch_size = 64
num_workers = 2 if os.cpu_count() is None else min(4, os.cpu_count())
num_classes = 5
tta = False

backbone = torchvision.models.efficientnet_b3(weights=weights).to(device).eval()
feat_dim = backbone.classifier[1].in_features  # 1536 for B3

proj = torch.nn.Linear(feat_dim, num_classes, bias=True).to(device)
torch.nn.init.normal_(proj.weight, mean=0.0, std=0.01)
torch.nn.init.zeros_(proj.bias)

pin_memory = torch.cuda.is_available()
loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=pin_memory,
)
if num_workers > 0:
    loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava images (test/inference-style)."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        img_size=384,
        images=None,
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas

        if images is None:
            self.images = sorted(
                [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
            )
        else:
            self.images = list(images)

        self.cc = None
        self.resize_model_a = None
        self.resize_model_b = None

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        model_a_img = img
        model_b_img = img

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Train dataset returning image tensor + label, using same preprocessing as test."""

    def __init__(self, data_dir, df, model_a_size, model_b_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cc = None
        self.resize_model_a = None
        self.resize_model_b = None

    def __getitem__(self, idx):
        filename = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        model_a_img = img
        model_b_img = img

        if self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.df)


class CassavaTrainDatasetAOnly(VisionDataset):
    def __init__(self, data_dir, df, model_a_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cc = None
        self.resize_model_a = None

    def __getitem__(self, idx):
        filename = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        model_a_img = img
        if self.transform:
            model_a_img = self.transform(model_a_img)
        return model_a_img, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.df)




## === cell 2
test_transforms = preprocess
train_transforms = preprocess

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_images_order = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    images=test_images_order,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)




## === cell 3
def extract_effnet_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)  # [B, feat_dim]
    return x


train_df = pd.read_csv(train_csv_path)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise RuntimeError("train.csv must contain image_id and label columns")

train_img_dataset_all = CassavaTrainDatasetAOnly(
    train_dir,
    train_df,
    model_a_img_size,
    transform=train_transforms,
)

feat_loader_all = DataLoader(
    train_img_dataset_all,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)

for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()


def precompute_feats(loader):
    all_feats = []
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            feats = extract_effnet_features(backbone, x)  # [B, feat_dim]
            all_feats.append(feats.detach().cpu())
            all_y.append(y.cpu())
    return torch.cat(all_feats, dim=0), torch.cat(all_y, dim=0)


train_feats_all, train_y_all = precompute_feats(feat_loader_all)

train_feats_all = torch.nn.functional.normalize(train_feats_all, p=2, dim=1)

y_all = train_y_all.numpy()


def stratified_kfold_indices(y, n_splits=5, seed=3407):
    rng = random.Random(seed)
    cls_to_idx = {}
    for i, yi in enumerate(y):
        cls_to_idx.setdefault(int(yi), []).append(i)
    for c in cls_to_idx:
        rng.shuffle(cls_to_idx[c])
    folds = [[] for _ in range(n_splits)]
    for c, idxs in cls_to_idx.items():
        for j, ix in enumerate(idxs):
            folds[j % n_splits].append(ix)
    all_idx = set(range(len(y)))
    out = []
    for k in range(n_splits):
        va = sorted(folds[k])
        tr = sorted(list(all_idx - set(va)))
        out.append((tr, va))
    return out


n_splits = 5
splits = stratified_kfold_indices(y_all, n_splits=n_splits, seed=3407)

epochs = 6
lr = 1.5e-3
wd = 1e-2

fold_best_states = []
fold_best_accs = []

for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
    tr_idx_t = torch.tensor(tr_idx, dtype=torch.long)
    va_idx_t = torch.tensor(va_idx, dtype=torch.long)

    feats_tr = train_feats_all.index_select(0, tr_idx_t)
    y_tr = train_y_all.index_select(0, tr_idx_t)
    feats_va = train_feats_all.index_select(0, va_idx_t)
    y_va = train_y_all.index_select(0, va_idx_t)

    label_counts = pd.Series(y_tr.numpy()).value_counts().sort_index()
    for k in range(num_classes):
        if k not in label_counts.index:
            raise RuntimeError(f"Missing class {k} in fold {fold} train split.")
    counts = label_counts.values.astype("float64")

    beta = 0.999  # smooth imbalance weighting; keep as-is
    effective_num = 1.0 - (beta**counts)
    class_weights = (1.0 - beta) / (effective_num + 1e-12)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_tensor_ds_tr = TensorDataset(feats_tr, y_tr)
    train_loader = DataLoader(
        train_tensor_ds_tr,
        batch_size=batch_size,
        shuffle=True,  # preserve training semantics
        **loader_kwargs,
    )

    proj_fold = torch.nn.Linear(feat_dim, num_classes, bias=True).to(device)
    torch.nn.init.normal_(proj_fold.weight, mean=0.0, std=0.01)
    torch.nn.init.zeros_(proj_fold.bias)

    optimizer = torch.optim.AdamW(proj_fold.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    best_state = None
    best_val_acc = -1.0

    for ep in range(epochs):
        proj_fold.train()
        running_loss = 0.0
        n = 0
        for feats_cpu, y_cpu in train_loader:
            feats = feats_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)

            logits = proj_fold(feats)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * feats.size(0)
            n += feats.size(0)

        proj_fold.eval()
        with torch.no_grad():
            feats = feats_va.to(device, non_blocking=True)
            y = y_va.to(device, non_blocking=True)
            logits = proj_fold(feats)
            pred = torch.argmax(logits, dim=1)
            val_acc = (pred == y).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in proj_fold.state_dict().items()
            }

        print(
            f"fold {fold}/{n_splits} - epoch {ep+1}/{epochs} - train_loss: {running_loss/max(n,1):.5f} - val_acc: {val_acc:.5f} - best: {best_val_acc:.5f}"
        )

    if best_state is None:
        raise RuntimeError("Failed to select best_state for fold training.")

    fold_best_states.append(best_state)
    fold_best_accs.append(best_val_acc)

print(
    "fold best accs:",
    fold_best_accs,
    "mean:",
    sum(fold_best_accs) / len(fold_best_accs),
)

proj.eval()



## === cell 4
fold_heads = []
for st in fold_best_states:
    m = torch.nn.Linear(feat_dim, num_classes, bias=True).to(device).eval()
    m.load_state_dict({k: v.to(device) for k, v in st.items()})
    fold_heads.append(m)

all_names = []
all_preds = []

backbone.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        if tta:
            bsz = len(filenames)
            x = torch.cat(model_a_inputs, dim=0).to(device, non_blocking=True)
            feats = extract_effnet_features(backbone, x)  # [B*T, feat_dim]
        else:
            x = model_a_inputs.to(device, non_blocking=True)
            feats = extract_effnet_features(backbone, x)  # [B, feat_dim]

        feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        ens_logits = None
        for proj_tmp in fold_heads:
            logits_5 = proj_tmp(feats)
            ens_logits = logits_5 if ens_logits is None else (ens_logits + logits_5)
        ens_logits = ens_logits / float(len(fold_heads))

        if tta:
            batch_logits = torch.stack(torch.split(ens_logits, bsz), dim=0)
            mean_logits = torch.mean(batch_logits, dim=0)
            pred_labels = torch.argmax(mean_logits, 1).tolist()
        else:
            pred_labels = torch.argmax(ens_logits, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

if len(all_names) != len(sample_sub):
    raise RuntimeError(
        f"Prediction count mismatch: got {len(all_names)} preds, expected {len(sample_sub)}"
    )

pred_map = dict(zip(all_names, all_preds))
submission = sample_sub.copy()
submission["label"] = submission["image_id"].map(pred_map)

missing = submission["label"].isna().sum()
if missing:
    raise RuntimeError(
        f"Found {missing} test images without predictions (mapping failed)."
    )

submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("rows:", len(submission), "cols:", submission.shape[1])
