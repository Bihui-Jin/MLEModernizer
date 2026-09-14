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

0.8937745542459957

# 6. Current score

0.12033

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.7855) has done: 'I fix the missing `ToDtype` transform (it no longer exists in the current torchvision version) and ensure the test loader is created correctly. I also improve the fallback ResNet‑18 training a bit (more epochs) and make the inference weighting ignore a dummy `model_b` so predictions rely only on the trained model when the second pretrained model is unavailable.'
- What this solution (achieved 0.12033) has done: 'The changes bypass the costly 20‑epoch ResNet18 fine‑tuning when no external pretrained models are available. Instead of training, we directly use the ImageNet‑pretrained ResNet18 (the same architecture) and save its weights as a checkpoint so future runs load instantly. This eliminates the long training phase while preserving the model structure and inference pipeline, keeping all downstream logic unchanged and ensuring the script completes well within the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision import models, transforms as v2
from torchvision.transforms import InterpolationMode
from PIL import Image
import pandas as pd

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

base_test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
candidate_dirs = [
    base_test_dir,
    os.path.join(base_test_dir, "test_images"),
]
test_dir = next(
    (d for d in candidate_dirs if os.path.isdir(d) and len(os.listdir(d)) > 0),
    base_test_dir,
)
print("Test dir resolved to:", test_dir)

model_a_img_size = 384
model_b_img_size = 384
batch_size = 16
num_workers = 8  # increase workers for faster I/O
num_classes = 5
tta = False


def load_or_dummy(path, name):
    if os.path.exists(path):
        print(f"Loading {name} from", path)
        return torch.load(path, map_location=device)
    else:
        print(f"{name} not found at {path}; using dummy model.")

        class DummyModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()

            def forward(self, x):
                return torch.zeros(x.size(0), num_classes, device=x.device)

        return DummyModel(num_classes).to(device)


model_a = load_or_dummy("/kaggle/input/vit-v1/vit_v1.pt", "model_a")
model_b = load_or_dummy("/kaggle/input/vit-boosted/vit_boosted.pt", "model_b")

use_model_b = model_b.__class__.__name__ != "DummyModel"

linear_head = nn.Identity().to(device)

if (
    model_a.__class__.__name__ == "DummyModel"
    and model_b.__class__.__name__ == "DummyModel"
):
    print(
        "Both pretrained models missing – using pretrained ResNet18 without fine‑tuning."
    )

    train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

    backbone = models.resnet18(pretrained=True)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    backbone = backbone.to(device)

    resnet18_checkpoint = "/kaggle/working/resnet18_trained.pt"
    if os.path.exists(resnet18_checkpoint):
        print(f"Loading cached ResNet18 weights from {resnet18_checkpoint}")
        backbone.load_state_dict(torch.load(resnet18_checkpoint, map_location=device))
    else:
        torch.save(backbone.state_dict(), resnet18_checkpoint)
        print(f"Saved initial ResNet18 checkpoint to {resnet18_checkpoint}")

    backbone.eval()
    model_a = backbone
    print("ResNet18 ready – using as model_a.")
else:
    print("At least one pretrained model found – using existing models.")




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data."""

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
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f))
                and f.lower().endswith(valid_exts)
            ]
        )
        self.ttas = ttas
        self.resize = v2.Resize(
            (model_a_size, model_a_size),
            interpolation=InterpolationMode.BICUBIC,
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")
        resized_img = self.resize(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(resized_img)) for t in self.ttas]
            model_b_img = [self.transform(t(resized_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(resized_img)
            model_b_img = self.transform(resized_img)
        else:
            model_a_img = resized_img
            model_b_img = resized_img

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ttas = None
if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # keep order to match submission length
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()

weight_a = 0.96
weight_b = 0.04 if use_model_b else 0.0

with torch.no_grad():
    for model_a_inputs, model_b_inputs, filenames in test_loader:
        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(
                device, non_blocking=True
            )
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(
                device, non_blocking=True
            )

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = (
                model_b(model_b_inputs)
                if use_model_b
                else torch.zeros_like(model_a_outputs)
            )

            batch_sz = len(filenames)
            model_a_logits = torch.stack(torch.split(model_a_outputs, batch_sz), dim=0)
            model_b_logits = torch.stack(torch.split(model_b_outputs, batch_sz), dim=0)
            model_a_mean = model_a_logits.mean(dim=0)
            model_b_mean = model_b_logits.mean(dim=0)

            outputs = (weight_a * model_a_mean + weight_b * model_b_mean) / 2
        else:
            inputs = model_a_inputs.to(device, non_blocking=True)
            model_a_outputs = model_a(inputs)
            if use_model_b:
                model_b_outputs = model_b(inputs)
            else:
                model_b_outputs = torch.zeros_like(model_a_outputs)

            outputs = (weight_a * model_a_outputs + weight_b * model_b_outputs) / 2

        probs = normalizer(outputs)
        pred_labels = torch.argmax(probs, dim=1).cpu().tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

submission_path = "/kaggle/working/submission.csv"
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, rows:", len(my_submission))
my_submission.head()
