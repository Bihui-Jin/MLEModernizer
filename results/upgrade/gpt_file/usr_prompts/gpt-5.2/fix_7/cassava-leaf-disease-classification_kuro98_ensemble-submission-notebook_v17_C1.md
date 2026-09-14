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

0.8677848292535509

# 6. Current score

0.71375

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14761) has done: 'I fix the ViT input size mismatch by resizing images to the model’s expected `image_size` (224 for `vit_b_16`), which resolves the runtime `AssertionError`. I also make TTA deterministic-at-inference and properly structured by including an identity transform and using non-random flip/rotate variants, ensuring the DataLoader collation and concatenation work reliably. Finally, I make the test image directory robust (handles nested `test_images/test_images`) so `all_names` aligns with `sample_submission.csv` and a valid `submission.csv` is always written.'
- What this solution (achieved 0.75747) has done: 'I fix the `weights_a.meta["mean"]/["std"]` KeyError by using the official ViT weight transforms to get normalization values safely, which also restores `train_transforms` and `test_loader` creation. I also make the test/train image directory detection robust against the common nested `*_images/*_images` structure so the dataset actually finds JPGs and produces predictions for all sample submission rows. Finally, I keep the model/training logic the same (two ViT-B/16 heads trained for 1 epoch, averaged logits, same TTA structure) and only adjust preprocessing to match the pretrained weights so the score moves upward from “no submission” toward the target.'
- What this solution (achieved 0.76009) has done: 'Your current pipeline is underperforming mainly because the training preprocessing is too weak/mismatched for fine-tuning (pure center-crop + normalize) and the TTA uses stochastic transforms (the `Random*` ops) in a way that can be inconsistent and less effective. I keep the same core setup (two ViT-B/16 models, only heads trainable, 1 epoch, averaged logits, same TTA count) but make two minimal, score-relevant changes: (1) use the pretrained weights’ full training transform pipeline for training (so input scaling/cropping matches what ViT expects), and (2) make TTA truly deterministic by replacing `Random*` transforms with deterministic functional transforms. These changes typically improve generalization accuracy without altering the architecture or training loop, and should move your score upward toward the 0.8678 target.'
- What this solution (achieved 0.71375) has done: 'Your score gap to the target is about 0.108 (0.76009 → 0.86778), so we should improve generalization while keeping your exact modeling/training setup (two ViT-B/16 models, only heads trainable, 1 epoch, average logits, same TTA count). The most impactful minimal fix is to stop using the *training* augmentation pipeline (random crops, etc.) for the *evaluation* on training images: add a small stratified validation split, evaluate accuracy in `eval()` mode with the deterministic pretrained `weights.transforms()` pipeline, and then train on the full data exactly as before to generate the submission. This doesn’t change your architecture or training loop semantics; it corrects preprocessing/eval mismatch and gives you a reliable signal, while also making a small score-relevant tweak: using the same deterministic preprocessing for validation and test generally improves the head fine-tuning stability and nudges leaderboard accuracy upward. I also add a tiny weighted sampler (still 1 epoch, same optimizer/loss) to reduce class-imbalance bias, which is a minimal change that often improves accuracy on Cassava without altering the core approach.'

# 9. Code solution

## === cell 0
import os

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

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def _resolve_images_dir(base_dir: str, nested_name: str) -> str:
    """
    Cassava dataset sometimes appears as:
      base_dir/test_images/*.jpg
    or
      base_dir/test_images/test_images/*.jpg
    This helper returns a directory that actually contains jpgs if possible.
    """
    base_dir = base_dir.rstrip("/") + "/"
    if os.path.isdir(base_dir):
        jpgs = [f for f in os.listdir(base_dir) if f.lower().endswith(".jpg")]
        if len(jpgs) > 0:
            return base_dir

        nested = os.path.join(base_dir, nested_name)
        if os.path.isdir(nested):
            jpgs2 = [f for f in os.listdir(nested) if f.lower().endswith(".jpg")]
            if len(jpgs2) > 0:
                return nested.rstrip("/") + "/"

    return base_dir  # fall back; later checks will reveal if empty


test_dir = _resolve_images_dir(
    "/kaggle/input/cassava-leaf-disease-classification/test_images", "test_images"
)
train_images_dir = _resolve_images_dir(
    "/kaggle/input/cassava-leaf-disease-classification/train_images", "train_images"
)

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights_a = ViT_B_16_Weights.IMAGENET1K_V1
weights_b = ViT_B_16_Weights.IMAGENET1K_V1

model_a = vit_b_16(weights=weights_a)
model_b = vit_b_16(weights=weights_b)

model_a_img_size = int(model_a.image_size)
model_b_img_size = int(model_b.image_size)

model_a.heads.head = torch.nn.Linear(model_a.heads.head.in_features, num_classes)
model_b.heads.head = torch.nn.Linear(model_b.heads.head.in_features, num_classes)

model_a.to(device)
model_b.to(device)

linear_head = None

print("Resolved train_images_dir:", train_images_dir)
print("Resolved test_dir:", test_dir)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data.

    Args:
        data_dir: base directory to the images.
        transforms: set of transforms to be used.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )
        self.ttas = ttas

        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
class CassavaTrainDataset(VisionDataset):
    def __init__(self, df, data_dir, model_a_size, model_b_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)
        if self.transform is not None:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)
        return model_a_img, model_b_img, torch.tensor(label, dtype=torch.long)




## === cell 4
_preprocess = weights_a.transforms()

_norm = None
for t in getattr(_preprocess, "transforms", []):
    if isinstance(t, v2.Normalize):
        _norm = t
        break
if _norm is None:
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
else:
    mean, std = _norm.mean, _norm.std

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
)

train_transforms = _preprocess

val_transforms = _preprocess

if tta:
    ttas = [
        lambda x: x,
        lambda x: v2.functional.horizontal_flip(x),
        lambda x: v2.functional.vertical_flip(x),
        lambda x: v2.functional.rotate(
            x, angle=90, interpolation=InterpolationMode.BILINEAR
        ),
    ]
else:
    ttas = None

if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"test_dir not found: {test_dir}")
if not os.path.isdir(train_images_dir):
    raise FileNotFoundError(f"train_images_dir not found: {train_images_dir}")

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
)
if len(test_dataset) == 0:
    raise RuntimeError(
        f"No .jpg files found in resolved test_dir={test_dir}. "
        f"Check dataset path nesting."
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 5
train_df = pd.read_csv(train_csv_path)

g = torch.Generator().manual_seed(3407)
val_frac = 0.10

train_indices = []
val_indices = []
for lbl in sorted(train_df["label"].unique().tolist()):
    idxs = train_df.index[train_df["label"] == lbl].to_numpy()
    idxs_t = torch.as_tensor(idxs, dtype=torch.long)
    perm = idxs_t[torch.randperm(len(idxs_t), generator=g)].tolist()
    n_val = max(1, int(round(len(perm) * val_frac)))
    val_indices.extend(perm[:n_val])
    train_indices.extend(perm[n_val:])

train_df_tr = train_df.loc[train_indices].reset_index(drop=True)
train_df_va = train_df.loc[val_indices].reset_index(drop=True)
print("Train split:", len(train_df_tr), "Val split:", len(train_df_va))

train_dataset = CassavaTrainDataset(
    train_df_tr,
    train_images_dir,
    model_a_img_size,
    model_b_img_size,
    transform=train_transforms,
)
val_dataset = CassavaTrainDataset(
    train_df_va,
    train_images_dir,
    model_a_img_size,
    model_b_img_size,
    transform=val_transforms,
)

class_counts = train_df_tr["label"].value_counts().to_dict()
weights_per_class = {c: 1.0 / max(1, n) for c, n in class_counts.items()}
sample_weights = torch.as_tensor(
    [weights_per_class[int(y)] for y in train_df_tr["label"].tolist()],
    dtype=torch.double,
)
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
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

for p in model_a.parameters():
    p.requires_grad = False
for p in model_b.parameters():
    p.requires_grad = False
for p in model_a.heads.head.parameters():
    p.requires_grad = True
for p in model_b.heads.head.parameters():
    p.requires_grad = True

model_a.train()
model_b.train()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    list(model_a.heads.head.parameters()) + list(model_b.heads.head.parameters()),
    lr=3e-4,
    weight_decay=0.01,
)

epochs = 1
for epoch in range(epochs):
    running_loss = 0.0
    seen = 0
    correct = 0
    for model_a_inputs, model_b_inputs, labels in train_loader:
        model_a_inputs = model_a_inputs.to(device, non_blocking=True)
        model_b_inputs = model_b_inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out_a = model_a(model_a_inputs)
        out_b = model_b(model_b_inputs)
        outputs = (out_a + out_b) / 2.0

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    print(
        f"Epoch {epoch+1}/{epochs} - loss: {running_loss/max(seen,1):.4f} - train_acc: {correct/max(seen,1):.4f}"
    )

    model_a.eval()
    model_b.eval()
    v_seen = 0
    v_correct = 0
    with torch.no_grad():
        for model_a_inputs, model_b_inputs, labels in val_loader:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out_a = model_a(model_a_inputs)
            out_b = model_b(model_b_inputs)
            outputs = (out_a + out_b) / 2.0
            preds = outputs.argmax(dim=1)
            v_seen += labels.size(0)
            v_correct += (preds == labels).sum().item()
    print(f"Val_acc: {v_correct/max(v_seen,1):.4f} ({v_correct}/{v_seen})")
    model_a.train()
    model_b.train()



## === cell 6
full_train_dataset = CassavaTrainDataset(
    train_df,
    train_images_dir,
    model_a_img_size,
    model_b_img_size,
    transform=train_transforms,
)

full_class_counts = train_df["label"].value_counts().to_dict()
full_weights_per_class = {c: 1.0 / max(1, n) for c, n in full_class_counts.items()}
full_sample_weights = torch.as_tensor(
    [full_weights_per_class[int(y)] for y in train_df["label"].tolist()],
    dtype=torch.double,
)
full_sampler = torch.utils.data.WeightedRandomSampler(
    weights=full_sample_weights,
    num_samples=len(full_sample_weights),
    replacement=True,
)

full_train_loader = DataLoader(
    full_train_dataset,
    batch_size=batch_size,
    sampler=full_sampler,
    num_workers=num_workers,
    pin_memory=True,
)

model_a.train()
model_b.train()
for epoch in range(1):  # keep exactly 1 epoch as in your core logic
    running_loss = 0.0
    seen = 0
    correct = 0
    for model_a_inputs, model_b_inputs, labels in full_train_loader:
        model_a_inputs = model_a_inputs.to(device, non_blocking=True)
        model_b_inputs = model_b_inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out_a = model_a(model_a_inputs)
        out_b = model_b(model_b_inputs)
        outputs = (out_a + out_b) / 2.0

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    print(
        f"Full-train Epoch {epoch+1}/1 - loss: {running_loss/max(seen,1):.4f} - train_acc: {correct/max(seen,1):.4f}"
    )



## === cell 7
all_names = []
all_preds = []

model_a.eval()
model_b.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        cur_bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(
                device, non_blocking=True
            )
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(
                device, non_blocking=True
            )

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            model_a_batch_logits = torch.stack(
                torch.split(model_a_outputs, cur_bs), dim=0
            )
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(
                torch.split(model_b_outputs, cur_bs), dim=0
            )
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = (model_a_mean_logits + model_b_mean_logits) / 2
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            outputs = (model_a_outputs + model_b_outputs) / 2
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("Predicted:", len(all_preds), "images from test_dir:", test_dir)



## === cell 8
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    raise RuntimeError(
        f"Missing predictions for {len(missing)} test images. Example missing: {missing[:5]}. "
        f"Found {len(pred_map)} preds from dir: {test_dir}"
    )

my_submission = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"],
        "label": sample_sub["image_id"].map(pred_map).astype(int),
    }
)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
