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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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

0.8844061650045331

# 6. Current score

0.71973

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12519) has done: 'The script fails because (1) the pretrained model file is missing, (2) the Albumentations augmentation call uses an outdated signature, and (3) the augmentation object isn’t created before inference. We add a safe fallback that loads a ResNeXt‑50 model with ImageNet weights when the custom checkpoint is absent, replace the failing `RandomResizedCrop` with a compatible `Resize` + `Normalize` pipeline, and ensure the augmentation is defined before the prediction loop. The updated cells run end‑to‑end and produce a correctly‑named `submission.csv` file.'
- What this solution (achieved 0.11996) has done: 'I fix the preprocessing so the model receives correctly‑scaled tensors (avoid double‑normalisation) and add a small centre‑crop plus optional horizontal flip for lightweight test‑time augmentation. This keeps the core model unchanged while providing a more realistic input, which should raise the validation accuracy much closer to the target.'
- What this solution (achieved 0.71973) has done: 'I fixed the Albumentations RandomResizedCrop call (which caused a validation error) by replacing it with a simple Resize transform, and I made sure the model is explicitly moved to the same device as the test tensors before inference to avoid the CUDA vs CPU tensor type mismatch. These small, targeted changes let the notebook run end‑to‑end and produce a proper submission.csv file while keeping the original training logic intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

import albumentations as A
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split




## === cell 1
model_path = "../input/rn-tta-calr-ft-of/model(10).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 2
if os.path.isfile(model_path):
    print("Loading custom checkpoint")
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(2048, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Custom checkpoint not found – loading ImageNet‑pretrained ResNeXt‑50")
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 5)

    class CassavaDataset(Dataset):
        def __init__(self, df, img_dir, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            img = np.array(Image.open(img_path).convert("RGB"))
            augmented = self.transform(image=img)
            img = augmented["image"]
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            label = torch.tensor(row["label"]).long()
            return img, label

    train_aug = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ]
    )
    val_aug = A.Compose(
        [
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ]
    )

    train_csv = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
    train_df, val_df = train_test_split(
        train_csv, test_size=0.1, stratify=train_csv["label"], random_state=42
    )

    train_ds = CassavaDataset(train_df, train_images_path, train_aug)
    val_ds = CassavaDataset(val_df, train_images_path, val_aug)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    epochs = 5  # increased epochs for higher accuracy
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        val_acc = correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), model_path)
    print(f"Fine‑tuned model saved to {model_path}")

model = model.to(device)
model.eval()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3013586000.py in <cell line: 0>()
    111         )
    112 
--> 113     torch.save(model.state_dict(), model_path)
    114     print(f"Fine‑tuned model saved to {model_path}")
    115 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in save(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization, _disable_byteorder_record)
    941 
    942     if _use_new_zipfile_serialization:
--> 943         with _open_zipfile_writer(f) as opened_zipfile:
    944             _save(
    945                 obj,

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_zipfile_writer(name_or_buffer)
    808     else:
    809         container = _open_zipfile_writer_buffer
--> 810     return container(name_or_buffer)
    811 
    812 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name)
    779             )
    780         else:
--> 781             super().__init__(torch._C.PyTorchFileWriter(self.name, _compute_crc32))
    782 
    783     def __exit__(self, *args) -> None:

RuntimeError: Parent directory ../input/rn-tta-calr-ft-of does not exist.

## === cell 3
sub_aug = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
    ]
)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return tensor




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
model = model.to(device)

with torch.no_grad():
    for _, row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, row.image_id)
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue

        img_np = np.array(pil_img)
        agg_output = torch.zeros(5, device=device)

        for _ in range(10):  # TTA loop
            aug_img = sub_aug(image=img_np)["image"]
            tensor_img = to_tensor(aug_img).to(device)
            out = model(tensor_img.unsqueeze(0)).squeeze(0)  # shape (5,)
            agg_output += out

        agg_output /= 10.0
        pred_label = torch.argmax(agg_output).item()
        predictions.append([row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print("Submission file created:")
print(sub_df.head())
