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

- What this solution (achieved 0.61099) has done: 'I guard the model loading with a try‑except so missing checkpoint files no longer raise an error, and add a simple fallback that predicts the most frequent class from the training labels for every test image. This guarantees a correctly‑sized `submission.csv` and yields a reasonable baseline score without altering the core model‑inference logic when the models are available.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

model_a_img_size = 384
model_b_img_size = 384
batch_size = 16
num_workers = 4
tta = True




## === cell 1
try:
    model_a = torch.load("/kaggle/input/vit-v1/vit_v1.pt", map_location=device)
    model_b = torch.load(
        "/kaggle/input/vit-boosted/vit_boosted.pt", map_location=device
    )
    linear_head = torch.load(
        "/kaggle/input/linear-head/linear_cls.pt", map_location=device
    )
    models_available = True
    print("Pretrained models loaded.")
except Exception as e:
    print(
        f"Model files not found or could not be loaded ({e}); using fallback predictor."
    )
    model_a = model_b = linear_head = None
    models_available = False




## === cell 2
if not models_available:

    class SimpleCassavaDataset(Dataset):
        def __init__(self, img_dir, csv_path, transform=None):
            self.img_dir = img_dir
            self.df = pd.read_csv(csv_path)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = int(row["label"])
            return img, label

    resnet_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    backbone = models.resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])  # remove fc
    backbone.to(device)
    backbone.eval()

    train_dataset = SimpleCassavaDataset(
        train_dir, train_csv_path, transform=resnet_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_sums = {}
    class_counts = {}

    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)  # shape: (B, 512, 1, 1)
            feats = feats.view(feats.size(0), -1)  # (B, 512)
            for feat, lbl in zip(feats, labels):
                lbl = int(lbl.item())
                if lbl not in class_sums:
                    class_sums[lbl] = torch.zeros_like(feat)
                    class_counts[lbl] = 0
                class_sums[lbl] += feat.cpu()
                class_counts[lbl] += 1

    class_centroids = {
        lbl: (class_sums[lbl] / class_counts[lbl]).to(device) for lbl in class_sums
    }

    class TestDataset(Dataset):
        def __init__(self, img_dir, filenames, transform=None):
            self.img_dir = img_dir
            self.filenames = filenames
            self.transform = transform

        def __len__(self):
            return len(self.filenames)

        def __getitem__(self, idx):
            fname = self.filenames[idx]
            img_path = os.path.join(self.img_dir, fname)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, fname

    test_filenames = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    test_filenames.sort()  # deterministic order

    test_dataset = TestDataset(test_dir, test_filenames, transform=resnet_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_names = []
    all_preds = []
    with torch.no_grad():
        for imgs, fnames in test_loader:
            imgs = imgs.to(device)
            feats = backbone(imgs).view(imgs.size(0), -1)  # (B, 512)

            centroid_tensor = torch.stack(
                [class_centroids[c] for c in sorted(class_centroids.keys())]
            )  # (C, 512)
            diff = feats.unsqueeze(1) - centroid_tensor.unsqueeze(0)
            dists = torch.norm(diff, dim=2)  # (B, C)
            pred_idxs = torch.argmin(dists, dim=1)  # (B,)

            class_labels = sorted(class_centroids.keys())
            preds = [class_labels[idx] for idx in pred_idxs.cpu().tolist()]

            all_names.extend(fnames)
            all_preds.extend(preds)

else:
    test_filenames = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    test_filenames.sort()  # deterministic order




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_54/2659973884.py in <cell line: 0>()
     64                     class_sums[lbl] = torch.zeros_like(feat)
     65                     class_counts[lbl] = 0
---> 66                 class_sums[lbl] += feat.cpu()
     67                 class_counts[lbl] += 1
     68 

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

## === cell 3
all_names = [] if models_available else all_names
all_preds = [] if models_available else all_preds

if models_available:

    class CassavaDataset(VisionDataset):
        """Custom dataset for the Cassava data."""

        def __init__(
            self, data_dir, model_a_size, model_b_size, transform=None, ttas=None
        ):
            super().__init__(root=data_dir)
            self.transform = transform
            self.images = os.listdir(data_dir)
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
            img = Image.open(os.path.join(self.root, filename))
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

    test_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ttas = (
        [
            v2.RandomRotation(180),
            v2.RandomVerticalFlip(1),
            v2.RandomAffine(180),
            v2.RandomPerspective(p=1),
        ]
        if tta
        else None
    )

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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    normalizer = torch.nn.Softmax(dim=1)

    model_a.eval()
    model_b.eval()
    linear_head.eval()

    with torch.no_grad():
        for model_a_inputs, model_b_inputs, filenames in test_loader:
            batch_sz = len(filenames)

            if tta:
                model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
                model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
                model_a_out = model_a(model_a_inputs)
                model_b_out = model_b(model_b_inputs)

                a_logits = torch.stack(torch.split(model_a_out, batch_sz), dim=0)
                b_logits = torch.stack(torch.split(model_b_out, batch_sz), dim=0)
                a_mean = torch.mean(a_logits, dim=0)
                b_mean = torch.mean(b_logits, dim=0)
                outputs = (a_mean + b_mean) / 2
                probs = normalizer(outputs)
                preds = torch.argmax(probs, dim=1).cpu().tolist()
            else:
                model_a_inputs = model_a_inputs.to(device)
                model_b_inputs = model_b_inputs.to(device)
                a_out = model_a(model_a_inputs)
                b_out = model_b(model_b_inputs)
                outputs = (a_out + b_out) / 2
                probs = normalizer(outputs)
                preds = torch.argmax(probs, dim=1).cpu().tolist()

            all_names.extend(filenames)
            all_preds.extend(preds)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3094833596.py in <cell line: 0>()
----> 1 all_names = [] if models_available else all_names
      2 all_preds = [] if models_available else all_preds
      3 
      4 if models_available:
      5 

NameError: name 'all_names' is not defined

## === cell 4
submission_path = "submission.csv"
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
my_submission.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/481125661.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
      3 my_submission.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}")
      5 my_submission.head()

NameError: name 'all_names' is not defined
