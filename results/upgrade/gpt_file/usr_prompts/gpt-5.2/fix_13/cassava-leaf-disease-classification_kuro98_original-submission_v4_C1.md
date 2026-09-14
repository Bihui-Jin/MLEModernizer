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

0.8886370504684195

# 6. Current score

0.76644

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the runtime blocker by removing the hard dependency on a missing `/kaggle/input/vit-update/vit.pt` and instead instantiate a torchvision ViT with the same forward semantics so inference can run end-to-end. I also fix the invalid submission length by ensuring we predict exactly once per `image_id` and then reorder/align predictions to `sample_submission.csv` (and avoid duplicates caused by `shuffle=True`). Finally, I make image loading robust (`RGB` conversion) and keep the rest of your logic (transforms, softmax+argmax) unchanged so it produces a valid `submission.csv`.'
- What this solution (achieved 0.22048) has done: 'I fix the runtime error by making the model input size consistent with the torchvision ViT-B/16 expectation (224×224), since your transforms currently produce 384×384 and ViT asserts on the configured `image_size`. This is a minimal change that preserves the same inference-only core logic (same model family, same softmax+argmax, same dataloader loop) while unblocking end-to-end execution and producing a valid `submission.csv`. This should also substantially improve your score versus the current broken/degenerate output, moving it toward the target accuracy. I also ensure the resize/crop pipeline truly outputs the expected spatial size.'
- What this solution (achieved 0.05531) has done: 'I fix the `IsADirectoryError` by filtering `os.listdir()` to include only actual image files (and ignore nested `test_images/` directories that exist inside the folder). I also make the test image directory resolution robust by automatically switching into a nested `test_images` subfolder if the provided path contains one, while keeping your inference logic (model, transforms, softmax+argmax, dataloader loop) unchanged. Finally, I keep submission alignment to `sample_submission.csv` as-is so the output has the correct ordering and row count and writes a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score is extremely low because the fallback ViT is running with a randomly initialized 5-class head, so predictions are effectively random. The smallest change that legitimately moves accuracy toward the target is to train only the classification head on the provided `train.csv` + `train_images` using the same ViT backbone, then run inference on test. To keep core logic intact, I preserve the model family (torchvision ViT-B/16), the softmax+argmax prediction semantics, and the overall dataloader-based loop; I only add a short head-training step and switch the test transform to match the pretrained weights’ expected preprocessing. This should move the score sharply upward toward your target while still finishing within the time limit.'
- What this solution (achieved 0.05531) has done: 'I fix the `KeyError: 'image_id'` by making the training dataset accept either a DataFrame with an `image_id` column or one already indexed by `image_id` (as your code does after `set_index`). I also make the label lookup robust/fast by precomputing a filename→label dict, avoiding `.loc` edge cases during batching. These changes are execution-stability fixes and keep your core model/training/inference logic identical, so they should improve score substantially versus the current broken training run. The rest of the pipeline (ViT-B/16 pretrained backbone + trained 5-class head, softmax+argmax, submission alignment) is left unchanged.'
- What this solution (achieved 0.75075) has done: 'I fix the dataset path resolution that currently points to `train_images/train_images` and `test_images/test_images` (a directory that exists but contains no images), which is why both datasets end up empty and later variables are undefined. The minimal change is to make `_resolve_image_dir` prefer the nested directory only if it actually contains image files; otherwise it fall back to the base directory. With that fixed, training and inference run end-to-end and the script always write a valid `submission.csv` with the correct row count and ordering aligned to `sample_submission.csv`. I keep the model, transforms (224), training loop (2 epochs head-only), and softmax+argmax semantics unchanged.'
- What this solution (achieved 0.67302) has done: 'Your current score (0.75075) is below the target (0.888637…), so we should improve accuracy with the smallest safe changes that keep your ViT-B/16 + head-only training and inference semantics intact. The main issue is that you train the head on the full training set without any class imbalance handling; Cassava labels are imbalanced, so a minimal and very effective tweak is to use class-weighted cross-entropy (same loss family) computed from `train.csv`. Additionally, keeping the backbone in `eval()` during head training stabilizes feature statistics (Dropout/LayerNorm behavior) while still training the head, which is consistent with “head-only” training. Finally, we add a minimal train-time augmentation (`RandomResizedCrop`) instead of a fixed resize to improve generalization without changing model/loop structure.'
- What this solution (achieved 0.7059) has done: 'Your current score (0.67302) is far below the target (0.8886), so we should improve accuracy with the smallest safe changes that preserve your ViT-B/16 + head-only training and inference semantics. The biggest low-risk gain is to make the head training match the inference preprocessing: right now you train with strong random crops, but you infer with center-crop; switching train to the pretrained weights’ recommended `weights.transforms()` (train=False) keeps the model/loop identical while reducing distribution shift. Next, we use a stratified train/validation split to monitor accuracy and to calibrate a single scalar temperature for logits (post-hoc, no architecture change) which often improves argmax accuracy slightly by stabilizing confidence; this is applied only at inference. Finally, we enable CUDA AMP for speed (no approximations to convergence—same epochs/steps) so the pipeline stays within the time budget while training on the full data.'
- What this solution (achieved 0.70067) has done: 'Your current score (0.7059) is well below the target (0.8886), so we should improve accuracy with the smallest safe changes that keep your ViT-B/16 + head-only training and softmax+argmax inference semantics intact. The biggest bottleneck is that training uses only 2 epochs and a relatively high LR; a minimal, stable improvement is to keep 2 epochs but (1) use a slightly lower LR and (2) add a standard cosine LR schedule across the same steps (no early stopping, same loop). Next, we avoid the current “temperature scaling” step because temperature does not change argmax in exact math and can only add numerical noise; removing it keeps evaluation semantics (argmax of logits) but is more stable. Finally, we enable simple mixup-style augmentation is NOT allowed (core logic), so instead we add a light train-time RandomHorizontalFlip while keeping the same pretrained normalization and 224 pipeline, which typically improves generalization with negligible logic change.'
- What this solution (achieved 0.72422) has done: 'Your score gap to the target is large (0.70067 vs 0.88864), so we should increase accuracy with minimal, low-risk changes that keep your ViT-B/16 + head-only training and argmax inference intact. The biggest issue is under-training: with only 2 epochs, the head often doesn’t converge; increasing to a small number of epochs (while keeping the same loop and loss) typically yields a sizable gain. To preserve stability and avoid changing evaluation semantics, I also ensure the backbone is frozen in `eval()` during training while the head stays in `train()`, and I compute `steps_per_epoch` after building the loader (unchanged) so the cosine schedule matches the true step count. Everything else (paths, preprocessing, model, loss family, submission alignment) remains the same and it still writes `submission.csv`.'
- What this solution (achieved 0.76644) has done: 'Your current score (0.72422) is well below the target (0.88864), so we should improve accuracy with the smallest safe changes that keep your ViT-B/16 + head-only training and argmax inference semantics intact. The main issue is underfitting from freezing the entire backbone; a minimal, still “same model/loop” improvement is to unfreeze and fine-tune only the last ViT encoder block (plus the head) while keeping the rest frozen, which typically yields a solid accuracy jump without changing architecture or training approach. To keep training stable and within time, we use a smaller LR for the unfrozen block, keep your existing cosine scheduler structure, and keep all preprocessing and submission alignment identical. Everything still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
random.seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def _resolve_existing_path(preferred_path: str) -> str:
    if os.path.exists(preferred_path):
        return preferred_path
    if preferred_path.startswith("/kaggle/input/"):
        alt = preferred_path.replace("/kaggle/input/", "/kaggle/data/")
        if os.path.exists(alt):
            return alt
    if preferred_path.startswith("/kaggle/data/"):
        alt = preferred_path.replace("/kaggle/data/", "/kaggle/input/")
        if os.path.exists(alt):
            return alt
    return preferred_path  # will error later with a clear message


def _dir_has_images(path: str) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    try:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if not os.path.isfile(full):
                continue
            _, ext = os.path.splitext(name.lower())
            if ext in exts:
                return True
    except FileNotFoundError:
        return False
    return False


def _resolve_image_dir(base_dir: str, leaf_dir_name: str) -> str:
    """
    Prefer nested dir only if it contains image files; otherwise fall back to base_dir.
    """
    base_dir = _resolve_existing_path(base_dir)
    if not os.path.isdir(base_dir):
        return base_dir

    nested = os.path.join(base_dir, leaf_dir_name)
    if os.path.isdir(nested) and _dir_has_images(nested):
        return nested
    if _dir_has_images(base_dir):
        return base_dir
    return nested if os.path.isdir(nested) else base_dir


test_dir = _resolve_image_dir(
    "/kaggle/input/cassava-leaf-disease-classification/test_images/", "test_images"
)
train_dir = _resolve_image_dir(
    "/kaggle/input/cassava-leaf-disease-classification/train_images/", "train_images"
)

train_csv_path = _resolve_existing_path(
    "/kaggle/input/cassava-leaf-disease-classification/train.csv"
)

img_size = 224
batch_size = 32
num_workers = 4
num_classes = 5
tta = False

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_model = vit_b_16(weights=weights)
in_features = vit_model.heads.head.in_features
vit_model.heads.head = torch.nn.Linear(in_features, num_classes)

vit_model.to(device)

print("Resolved paths:")
print(
    "  train_dir:",
    train_dir,
    "exists:",
    os.path.isdir(train_dir),
    "has_images:",
    _dir_has_images(train_dir),
)
print(
    "  test_dir :",
    test_dir,
    "exists:",
    os.path.isdir(test_dir),
    "has_images:",
    _dir_has_images(test_dir),
)
print("  train_csv:", train_csv_path, "exists:", os.path.isfile(train_csv_path))




## === cell 2
class CassavaDataset(VisionDataset):
    """Dataset for Cassava images.

    - If `labels_df` is provided, returns (image_tensor, label_int).
    - Otherwise, returns (image_tensor, filename) for test.
    """

    def __init__(self, data_dir, transform=None, labels_df=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.labels_df = labels_df

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if labels_df is None:
            entries = []
            for name in os.listdir(data_dir):
                full = os.path.join(data_dir, name)
                if not os.path.isfile(full):
                    continue
                _, ext = os.path.splitext(name.lower())
                if ext in exts:
                    entries.append(name)
            self.images = sorted(entries)
            self.label_map = None
        else:
            if "image_id" in labels_df.columns:
                img_ids = labels_df["image_id"].astype(str).tolist()
                tmp = labels_df.set_index("image_id")
            else:
                img_ids = labels_df.index.astype(str).tolist()
                tmp = labels_df

            self.images = [
                x for x in img_ids if os.path.isfile(os.path.join(data_dir, x))
            ]
            self.label_map = tmp["label"].to_dict()

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        if self.labels_df is None:
            return img, filename
        else:
            y = int(self.label_map[filename])
            return img, y

    def __len__(self):
        return len(self.images)




## === cell 3
base_preprocess = weights.transforms()

test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=base_preprocess.mean, std=base_preprocess.std),
    ]
)

train_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.RandomHorizontalFlip(p=0.5),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=base_preprocess.mean, std=base_preprocess.std),
    ]
)

if tta:
    ttas = [
        v2.RandomResizedCrop((img_size, img_size), (0.5, 1)),
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None



## === cell 4
train_df = pd.read_csv(train_csv_path)
train_df = train_df[["image_id", "label"]].copy()


def _stratified_split(df: pd.DataFrame, label_col: str, val_frac: float, seed: int):
    rng = random.Random(seed)
    val_idx = []
    for lab, grp in df.groupby(label_col):
        idxs = grp.index.tolist()
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac)))
        val_idx.extend(idxs[:n_val])
    val_mask = df.index.isin(val_idx)
    return df.loc[~val_mask].copy(), df.loc[val_mask].copy()


train_df_train, train_df_val = _stratified_split(
    train_df, "label", val_frac=0.08, seed=3407
)

train_df_train = train_df_train.set_index("image_id")
train_df_val = train_df_val.set_index("image_id")

train_dataset = CassavaDataset(
    train_dir, transform=train_transforms, labels_df=train_df_train, ttas=None
)
val_dataset = CassavaDataset(
    train_dir, transform=test_transforms, labels_df=train_df_val, ttas=None
)

if len(train_dataset) == 0:
    raise RuntimeError(
        "Training dataset has 0 samples. Check that train_dir points to the extracted images folder. "
        f"Resolved train_dir={train_dir!r} exists={os.path.isdir(train_dir)}"
    )
if len(val_dataset) == 0:
    raise RuntimeError("Validation dataset has 0 samples after split; reduce val_frac.")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

for p in vit_model.parameters():
    p.requires_grad = False

for p in vit_model.heads.head.parameters():
    p.requires_grad = True

if (
    hasattr(vit_model, "encoder")
    and hasattr(vit_model.encoder, "layers")
    and len(vit_model.encoder.layers) > 0
):
    for p in vit_model.encoder.layers[-1].parameters():
        p.requires_grad = True

label_counts = (
    train_df_train["label"]
    .value_counts()
    .reindex(range(num_classes), fill_value=0)
    .sort_index()
)
counts = torch.tensor(label_counts.values, dtype=torch.float32)
class_weights = (counts.sum() / (num_classes * counts.clamp_min(1.0))).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

param_groups = [
    {"params": vit_model.heads.head.parameters(), "lr": 1.5e-3, "weight_decay": 0.0},
]
if (
    hasattr(vit_model, "encoder")
    and hasattr(vit_model.encoder, "layers")
    and len(vit_model.encoder.layers) > 0
):
    param_groups.append(
        {
            "params": vit_model.encoder.layers[-1].parameters(),
            "lr": 3.0e-5,
            "weight_decay": 0.0,
        }
    )

optimizer = torch.optim.AdamW(param_groups)

epochs = 6

steps_per_epoch = len(train_loader)
total_steps = max(1, epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler(enabled=use_amp)

vit_model.train()
global_step = 0
for ep in range(epochs):
    vit_model.train()

    running_loss = 0.0
    correct = 0
    seen = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = vit_model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1

        running_loss += float(loss.item()) * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == targets).sum().item())
        seen += int(inputs.size(0))

    vit_model.eval()
    v_correct = 0
    v_seen = 0
    with torch.no_grad():
        for v_inputs, v_targets in val_loader:
            v_inputs = v_inputs.to(device, non_blocking=True)
            v_targets = v_targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                v_logits = vit_model(v_inputs)
            v_preds = torch.argmax(v_logits, dim=1)
            v_correct += int((v_preds == v_targets).sum().item())
            v_seen += int(v_inputs.size(0))

    print(
        f"epoch {ep+1}/{epochs} loss={running_loss/seen:.4f} acc={correct/seen:.4f} "
        f"val_acc={v_correct/max(1,v_seen):.4f} lr_head={optimizer.param_groups[0]['lr']:.6f}"
    )



## === cell 5
T_value = 1.0
print("Using temperature T =", T_value)



## === cell 6
test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

if len(test_dataset) == 0:
    raise RuntimeError(
        "Test dataset has 0 samples. Check that test_dir points to the extracted images folder. "
        f"Resolved test_dir={test_dir!r} exists={os.path.isdir(test_dir)}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)

all_names = []
all_preds = []

vit_model.eval()
with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            inputs, filenames = torch.cat(inputs, dim=0).to(device), list(filenames)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = vit_model(inputs) / T_value
            preds = normalizer(logits)

            batch_preds = torch.stack(torch.split(preds, len(filenames)), dim=0)
            mean_preds = torch.mean(batch_preds, dim=0)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            inputs, filenames = inputs.to(device), list(filenames)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = vit_model(inputs) / T_value
            preds = normalizer(logits)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("Predicted:", len(all_preds), "images")



## === cell 7
sample_path = _resolve_existing_path(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
labels_aligned = [
    int(pred_map.get(img_id, 0)) for img_id in sample_sub["image_id"].tolist()
]

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": labels_aligned}
)
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
