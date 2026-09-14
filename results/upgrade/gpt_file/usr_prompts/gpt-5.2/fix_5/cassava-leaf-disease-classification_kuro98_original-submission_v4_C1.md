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

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the runtime blocker by removing the hard dependency on a missing `/kaggle/input/vit-update/vit.pt` and instead instantiate a torchvision ViT with the same forward semantics so inference can run end-to-end. I also fix the invalid submission length by ensuring we predict exactly once per `image_id` and then reorder/align predictions to `sample_submission.csv` (and avoid duplicates caused by `shuffle=True`). Finally, I make image loading robust (`RGB` conversion) and keep the rest of your logic (transforms, softmax+argmax) unchanged so it produces a valid `submission.csv`.'
- What this solution (achieved 0.22048) has done: 'I fix the runtime error by making the model input size consistent with the torchvision ViT-B/16 expectation (224×224), since your transforms currently produce 384×384 and ViT asserts on the configured `image_size`. This is a minimal change that preserves the same inference-only core logic (same model family, same softmax+argmax, same dataloader loop) while unblocking end-to-end execution and producing a valid `submission.csv`. This should also substantially improve your score versus the current broken/degenerate output, moving it toward the target accuracy. I also ensure the resize/crop pipeline truly outputs the expected spatial size.'
- What this solution (achieved 0.05531) has done: 'I fix the `IsADirectoryError` by filtering `os.listdir()` to include only actual image files (and ignore nested `test_images/` directories that exist inside the folder). I also make the test image directory resolution robust by automatically switching into a nested `test_images` subfolder if the provided path contains one, while keeping your inference logic (model, transforms, softmax+argmax, dataloader loop) unchanged. Finally, I keep submission alignment to `sample_submission.csv` as-is so the output has the correct ordering and row count and writes a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score is extremely low because the fallback ViT is running with a randomly initialized 5-class head, so predictions are effectively random. The smallest change that legitimately moves accuracy toward the target is to train only the classification head on the provided `train.csv` + `train_images` using the same ViT backbone, then run inference on test. To keep core logic intact, I preserve the model family (torchvision ViT-B/16), the softmax+argmax prediction semantics, and the overall dataloader-based loop; I only add a short head-training step and switch the test transform to match the pretrained weights’ expected preprocessing. This should move the score sharply upward toward your target while still finishing within the time limit.'

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

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
nested_test_dir = os.path.join(test_dir, "test_images")
if os.path.isdir(nested_test_dir):
    test_dir = nested_test_dir

train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
nested_train_dir = os.path.join(train_dir, "train_images")
if os.path.isdir(nested_train_dir):
    train_dir = nested_train_dir

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

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
        else:
            imgs = labels_df["image_id"].tolist()
            self.images = [x for x in imgs if os.path.isfile(os.path.join(data_dir, x))]

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
            y = int(self.labels_df.loc[filename, "label"])
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
train_df = train_df.set_index("image_id")

train_dataset = CassavaDataset(
    train_dir, transform=train_transforms, labels_df=train_df, ttas=None
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

for p in vit_model.parameters():
    p.requires_grad = False
for p in vit_model.heads.head.parameters():
    p.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    vit_model.heads.head.parameters(), lr=3e-3, weight_decay=0.0
)

vit_model.train()
epochs = 2  # small, to stay within time; enough to move far from random toward target
for ep in range(epochs):
    running_loss = 0.0
    correct = 0
    seen = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = vit_model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == targets).sum().item())
        seen += int(inputs.size(0))

    print(f"epoch {ep+1}/{epochs} loss={running_loss/seen:.4f} acc={correct/seen:.4f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'image_id'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/488301797.py in <cell line: 0>()
      5 train_df = train_df.set_index("image_id")
      6 
----> 7 train_dataset = CassavaDataset(
      8     train_dir, transform=train_transforms, labels_df=train_df, ttas=None
      9 )

/tmp/ipykernel_55/1687977430.py in __init__(self, data_dir, transform, labels_df, ttas)
     27         else:
     28             # Train mode: follow train.csv order (filter to files that exist)
---> 29             imgs = labels_df["image_id"].tolist()
     30             self.images = [x for x in imgs if os.path.isfile(os.path.join(data_dir, x))]
     31 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'image_id'

## === cell 5
test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

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
            preds = normalizer(vit_model(inputs))

            batch_preds = torch.stack(torch.split(preds, len(filenames)), dim=0)
            mean_preds = torch.mean(batch_preds, dim=0)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            inputs, filenames = inputs.to(device), list(filenames)
            preds = normalizer(vit_model(inputs))
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("Predicted:", len(all_preds), "images")



## === cell 6
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
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



## === cell 7
my_submission
