# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"test_dir not found: {test_dir}")

top_level_jpgs = [
    f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
nested_dir = os.path.join(test_dir, "test_images")
if len(top_level_jpgs) == 0 and os.path.isdir(nested_dir):
    test_dir = nested_dir

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
if not os.path.isdir(train_img_dir):
    raise FileNotFoundError(f"train_img_dir not found: {train_img_dir}")

img_size = 384
batch_size = 16
num_workers = min(8, (os.cpu_count() or 8))
num_classes = 5
tta = False

train_epochs = 2
train_lr = 3e-4
weight_decay = 0.01
freeze_backbone = True  # keep core model; train only head for speed/stability

torch.set_num_threads(min(8, (os.cpu_count() or 8)))
torch.set_num_interop_threads(1)

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
vit_model = vit_b_16(weights=weights)
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)

vit_model.to(device)
vit_model = vit_model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        vit_model = torch.compile(vit_model, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print("torch.compile unavailable; continuing in eager mode:", repr(e))




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory to the images.
        transform: set of transforms to be used.
        ttas: list of transforms for test-time augmentation.
    """

    def __init__(self, data_dir, transform=None, ttas=None, img_size=384):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        files = []
        with os.scandir(data_dir) as it:
            for entry in it:
                if entry.is_file():
                    n = entry.name
                    if n.lower().endswith(exts):
                        files.append(n)
        self.images = sorted(files)

        if len(self.images) == 0:
            raise RuntimeError(f"No image files found in: {data_dir}")

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    def __init__(self, df, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._image_ids = self.df["image_id"].to_numpy()
        self._labels = self.df["label"].to_numpy()

    def __getitem__(self, idx):
        filename = self._image_ids[idx]
        label = int(self._labels[idx])
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = test_transforms

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

test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

g = torch.Generator()
g.manual_seed(3407)


def _seed_worker(worker_id: int):
    seed = 3407 + worker_id
    torch.manual_seed(seed)


_loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=8 if num_workers > 0 else None,
)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,  # keep deterministic ordering
    drop_last=False,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

normalizer = torch.nn.Softmax(dim=1)

train_df = pd.read_csv(train_csv_path)
train_dataset = CassavaTrainDataset(train_df, train_img_dir, transform=train_transforms)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    drop_last=False,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)




## === cell 3
vit_model.train()

if freeze_backbone:
    for p in vit_model.parameters():
        p.requires_grad = False
    for p in vit_model.heads.parameters():
        p.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    (p for p in vit_model.parameters() if p.requires_grad),
    lr=train_lr,
    weight_decay=weight_decay,
)

for epoch in range(train_epochs):
    running_loss = 0.0
    n_seen = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = vit_model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n_seen += bs

    print(
        f"epoch {epoch+1}/{train_epochs} - train_loss: {running_loss / max(1, n_seen):.4f}"
    )




## === cell 4
all_names = []
all_preds = []

vit_model.eval()

with torch.inference_mode():
    for inputs, filenames in test_loader:
        if tta:
            inputs = (
                torch.cat(inputs, dim=0)
                .to(device, non_blocking=True)
                .contiguous(memory_format=torch.channels_last)
            )
            filenames = list(filenames)

            preds = normalizer(vit_model(inputs))
            batch_preds = torch.stack(torch.split(preds, len(filenames)), dim=0)
            mean_preds = torch.mean(batch_preds, dim=0)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            filenames = list(filenames)

            preds = normalizer(vit_model(inputs))
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_submission = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
missing = [
    img_id
    for img_id in sample_submission["image_id"].tolist()
    if img_id not in pred_map
]
if missing:
    for img_id in missing:
        pred_map[img_id] = 0

my_submission = sample_submission.copy()
my_submission["label"] = my_submission["image_id"].map(pred_map).astype(int)

assert len(my_submission) == len(sample_submission)
assert list(my_submission.columns) == ["image_id", "label"]

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)




## === cell 6
my_submission.head()
