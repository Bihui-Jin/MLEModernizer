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

0.8677848292535509

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14761) has done: 'I fix the ViT input size mismatch by resizing images to the model’s expected `image_size` (224 for `vit_b_16`), which resolves the runtime `AssertionError`. I also make TTA deterministic-at-inference and properly structured by including an identity transform and using non-random flip/rotate variants, ensuring the DataLoader collation and concatenation work reliably. Finally, I make the test image directory robust (handles nested `test_images/test_images`) so `all_names` aligns with `sample_submission.csv` and a valid `submission.csv` is always written.'

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

_base_test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_dir = _base_test_dir
if (
    not os.path.isdir(test_dir)
    or len([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]) == 0
):
    nested = os.path.join(_base_test_dir, "test_images")
    if os.path.isdir(nested):
        test_dir = nested

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
if (
    not os.path.isdir(train_images_dir)
    or len([f for f in os.listdir(train_images_dir) if f.lower().endswith(".jpg")]) == 0
):
    nested = os.path.join(train_images_dir, "train_images")
    if os.path.isdir(nested):
        train_images_dir = nested

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




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

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
vit_preprocess = weights_a.transforms()

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=weights_a.meta["mean"], std=weights_a.meta["std"]),
    ]
)

train_transforms = test_transforms

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/149699917.py in <cell line: 0>()
     10         v2.ToImage(),
     11         v2.ToDtype(torch.float32, scale=True),
---> 12         v2.Normalize(mean=weights_a.meta["mean"], std=weights_a.meta["std"]),
     13     ]
     14 )

KeyError: 'mean'

## === cell 5
train_df = pd.read_csv(train_csv_path)
train_dataset = CassavaTrainDataset(
    train_df,
    train_images_dir,
    model_a_img_size,
    model_b_img_size,
    transform=train_transforms,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2729045468.py in <cell line: 0>()
      7     model_a_img_size,
      8     model_b_img_size,
----> 9     transform=train_transforms,
     10 )
     11 train_loader = DataLoader(

NameError: name 'train_transforms' is not defined

## === cell 6
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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4062944189.py in <cell line: 0>()
      7 with torch.no_grad():
      8     for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
----> 9         test_loader
     10     ):
     11         cur_bs = len(filenames)

NameError: name 'test_loader' is not defined

## === cell 7
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

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3186050056.py in <cell line: 0>()
      6 ]
      7 if missing:
----> 8     raise RuntimeError(
      9         f"Missing predictions for {len(missing)} test images. Example missing: {missing[:5]}. "
     10         f"Found {len(pred_map)} preds from dir: {test_dir}"

RuntimeError: Missing predictions for 2676 test images. Example missing: ['1234294272.jpg', '1234332763.jpg', '1234375577.jpg', '1234555380.jpg', '1234571117.jpg']. Found 0 preds from dir: /kaggle/input/cassava-leaf-disease-classification/test_images/
