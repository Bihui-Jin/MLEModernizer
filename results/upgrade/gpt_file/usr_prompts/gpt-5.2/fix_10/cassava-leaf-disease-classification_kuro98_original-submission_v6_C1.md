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

0.8930190389845875

# 6. Current score

0.58894

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by replacing the missing `/kaggle/input/vit-v1/vit.pt` dependency with an equivalent in-notebook ViT from `torchvision` (same overall “ViT inference” core logic), so the pipeline can run end-to-end in this environment. I also fix the invalid submission length by (1) disabling `shuffle` for the test loader and (2) ensuring predictions are aligned exactly to `sample_submission.csv` image order, which is the safest way to guarantee correct length and IDs. Finally, I make TTA deterministic and correct by applying TTAs on already-resized tensors (so they can be stacked), and I use `eval()` + `no_grad()` exactly as before to keep evaluation semantics stable while producing a valid `submission.csv`.'
- What this solution (achieved 0.22795) has done: 'I fix the runtime assertion by aligning `img_size` (and corresponding transforms) with the `vit_b_16` model’s expected `image_size=224`. I also load the pretrained weights but keep the existing “replace head to 5 classes” core logic; since no finetuning is present, I make predictions by using the ImageNet-pretrained head and mapping its logits to 5 classes via a deterministic modulo to lift accuracy from near-random toward the target band without changing the overall inference approach. Finally, I keep your submission alignment logic (using `sample_submission.csv` order) and ensure the script runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current 0.22795 score is far below the 0.8930 target, so we need a real (but still minimal) improvement instead of the current “ImageNet top1 % 5” heuristic. I keep your core ViT inference setup and data pipeline, but change the prediction head to use a simple, legitimate nearest-prototype classifier built from the provided `train.csv` + `train_images` using the same ViT backbone features (no training loops, no new model architecture). This uses class prototypes (mean embedding per class) and predicts the closest prototype for each test image, which typically jumps accuracy substantially on this dataset while staying within runtime. I also ensure transforms are consistent between train/test feature extraction and keep submission alignment exactly as you already do.'
- What this solution (achieved 0.59193) has done: 'I fix the runtime error by replacing the nonexistent `forward_features()` call with the correct way to extract embeddings from `torchvision`’s ViT (`model._process_input` + encoder + class token), keeping the same “ViT backbone → normalized embeddings → class prototypes → cosine similarity” core logic. I also make the embedding dimension dynamic (instead of hardcoding 768) so it stays correct across weight variants and avoids shape bugs. Finally, I keep your deterministic setup and submission alignment unchanged so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current prototype classifier is solid but likely underperforming because the ViT embeddings are taken before the final LayerNorm and without the model’s pre-logits/head normalization that the pretrained weights expect. I keep the same “ViT backbone → L2-normalized CLS embedding → per-class mean prototype → cosine similarity” core logic, but change `vit_embeddings()` to use the model’s own `forward_features()` (which includes the correct normalization) and then L2-normalize as before. This is a minimal, inference-only change that typically improves separability of classes and should move accuracy upward toward your 0.893 target. Everything else (data order/alignment, TTA, prototype building, submission writing) stays the same.'
- What this solution (achieved 0.59193) has done: 'I fix the runtime error by replacing the nonexistent `forward_features()` call with a version-tolerant ViT feature extractor that uses torchvision’s internal `_process_input` + `encoder` + final `ln` to produce the normalized CLS embedding (same backbone→embedding→prototype cosine logic as before). I keep the rest of your pipeline (transforms, prototype building, TTA, submission alignment) unchanged to preserve evaluation semantics and move accuracy back toward your previous ~0.59+ behavior. I also make the embedding dimension detection robust so it doesn’t break across torchvision versions. The script run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.58894) has done: 'Your current prototype approach is already valid but underpowered because it builds class prototypes from raw training images without any label-preserving augmentation, while your test-time TTA increases invariance only on the test side. I make a minimal, symmetric change: build prototypes using a small, deterministic multi-view augmentation (same kinds of flips/rotation you already use for TTA), averaging embeddings per image before accumulating into class prototypes. This keeps the exact same ViT backbone → normalized CLS embedding → per-class mean prototype → cosine similarity core logic, but improves prototype quality and should raise accuracy toward your 0.893 target without adding training loops or changing the model. I also switch the train loader to `drop_last=False` explicitly and keep test alignment/submission writing unchanged.'
- What this solution (achieved 0.59342) has done: 'I make the smallest changes that improve class-prototype quality without altering your core “ViT embeddings → L2 normalize → per-class mean prototype → cosine similarity” approach. Specifically, I (1) compute prototypes as an unbiased mean by accumulating per-image averaged embeddings (so multi-view augmentation doesn’t overweight any class via batch composition), and (2) add a tiny, deterministic “shrinkage” toward the global mean prototype to reduce noise in minority classes—this usually improves accuracy for nearest-prototype classifiers while keeping inference-only semantics. I also keep test-time prediction and submission alignment exactly as you already have it. These changes are directly aimed at lifting accuracy from ~0.589 toward your 0.893 target with minimal risk and within the time limit.'
- What this solution (achieved 0.58894) has done: 'Your current score (0.59342) is far below the target (0.8930), so we should improve accuracy while keeping the same “ViT embeddings → L2 normalize → per-class mean prototype → cosine similarity” core logic. The biggest likely blocker is using random-parameter TTAs (`RandomRotation`) during both prototype building and test inference, which makes features non-deterministic and can easily hurt nearest-prototype performance; I replace those with deterministic TTAs (fixed rotations) while preserving the same augmentation idea. I also apply the same small prototype “shrinkage” you already use, but compute it in a class-count-aware way (stronger shrinkage for low-count classes) to reduce noise without changing the classifier type. Finally, I keep submission alignment exactly as-is to ensure a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
from torchvision import models



## === cell 1
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = f"{DATA_DIR}/train_images/"
test_dir = f"{DATA_DIR}/test_images/"
train_csv_path = f"{DATA_DIR}/train.csv"
sample_path = f"{DATA_DIR}/sample_submission.csv"

img_size = 224

batch_size = 32
num_workers = 4
num_classes = 5
tta = True

vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
vit_imagenet = models.vit_b_16(weights=vit_weights).to(device)
vit_imagenet.eval()




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset returning (image_tensor, filename)."""

    def __init__(self, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Train dataset returning (image_tensor, label). Only used to build class prototypes."""

    def __init__(self, data_dir, df, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.df = df.reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        y = int(row["label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.df)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(15, 15)),
        v2.RandomRotation(degrees=(-15, -15)),
    ]
else:
    ttas = None

if tta:
    proto_views = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(15, 15)),
        v2.RandomRotation(degrees=(-15, -15)),
    ]
else:
    proto_views = [v2.Identity()]

train_df = pd.read_csv(train_csv_path)
train_dataset = CassavaTrainDataset(train_dir, df=train_df, transform=test_transforms)
test_dataset = CassavaDataset(test_dir, transform=test_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)




## === cell 4
@torch.no_grad()
def vit_embeddings(model, x: torch.Tensor) -> torch.Tensor:
    """
    Extract the normalized CLS embedding using torchvision ViT internals:
      _process_input -> prepend class token -> encoder -> take CLS -> final LayerNorm.
    """
    x = model._process_input(x)  # (B, num_patches, hidden_dim)
    n = x.shape[0]

    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)  # (B, 1+num_patches, hidden_dim)

    x = model.encoder(x)  # (B, 1+num_patches, hidden_dim)
    x = x[:, 0]

    if hasattr(model, "ln"):
        x = model.ln(x)

    x = torch.nn.functional.normalize(x, p=2, dim=1)
    return x


embed_dim = int(
    getattr(vit_imagenet, "hidden_dim", None)
    or getattr(vit_imagenet, "hidden_size", None)
    or vit_imagenet.heads.head.in_features
)

prototypes_sum = torch.zeros(
    (num_classes, embed_dim), device=device, dtype=torch.float32
)
prototypes_count = torch.zeros((num_classes,), device=device, dtype=torch.float32)

vit_imagenet.eval()
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if proto_views is not None and len(proto_views) > 1:
            view_batch = [t(inputs) for t in proto_views]
            view_inputs = torch.cat(view_batch, dim=0)  # (V*B, C, H, W)
            feats = vit_embeddings(vit_imagenet, view_inputs)  # (V*B, D)
            V = len(proto_views)
            B = inputs.shape[0]
            feats = feats.view(V, B, -1).mean(dim=0)  # (B, D)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        else:
            feats = vit_embeddings(vit_imagenet, inputs)  # (B, D)

        prototypes_sum.index_add_(0, labels, feats)
        ones = torch.ones_like(labels, dtype=torch.float32, device=device)
        prototypes_count.index_add_(0, labels, ones)

prototypes = prototypes_sum / prototypes_count.clamp_min(1.0).unsqueeze(1)

global_proto = prototypes.mean(dim=0, keepdim=True)  # (1, D)
base_alpha = (
    0.08  # slightly stronger than 0.05 to move score upward, still a small adjustment
)
max_count = prototypes_count.max().clamp_min(1.0)
alpha_per_class = base_alpha * (1.0 - (prototypes_count / max_count))  # (C,)
alpha_per_class = alpha_per_class.clamp(0.0, base_alpha).unsqueeze(1)  # (C, 1)
prototypes = (1.0 - alpha_per_class) * prototypes + alpha_per_class * global_proto

prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)

print(
    "Built prototypes with counts:",
    prototypes_count.detach().cpu().numpy().astype(int).tolist(),
)



## === cell 5
all_names = []
all_preds = []

vit_imagenet.eval()
with torch.no_grad():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device, non_blocking=True)

        if ttas is not None:
            tta_batch = [t(inputs) for t in ttas]
            tta_inputs = torch.cat(tta_batch, dim=0)  # (T*B, C, H, W)

            feats = vit_embeddings(vit_imagenet, tta_inputs)  # (T*B, D)

            T = len(ttas)
            B = inputs.shape[0]
            feats = feats.view(T, B, -1).mean(dim=0)  # (B, D)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        else:
            feats = vit_embeddings(vit_imagenet, inputs)  # (B, D)

        sims = feats @ prototypes.T  # (B, 5)
        pred_labels = torch.argmax(sims, dim=1).to(torch.int64).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)



## === cell 6
sample_sub = pd.read_csv(sample_path)
pred_map = dict(zip(all_names, all_preds))

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    for m in missing:
        pred_map[m] = 0

my_submission = sample_sub.copy()
my_submission["label"] = my_submission["image_id"].map(pred_map).astype(int)

assert len(my_submission) == len(sample_sub)
assert my_submission["image_id"].isna().sum() == 0
assert my_submission["label"].isna().sum() == 0

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 7
my_submission.head(10)
