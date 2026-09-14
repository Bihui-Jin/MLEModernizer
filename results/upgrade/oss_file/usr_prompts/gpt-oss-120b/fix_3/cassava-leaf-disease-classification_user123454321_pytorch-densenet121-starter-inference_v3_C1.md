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

3.9

# 3. Installed packages

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.838773043215473

# 6. Current score

0.74439

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I add safe handling for the missing model file, fall back to a simple baseline that predicts the most frequent class from the training data, and fix the CUDA‑only code to work on CPU. These changes prevent the crashes, guarantee that the prediction array matches the submission length, and give a reasonable accuracy boost over the empty‑prediction case while preserving the original structure.'
- What this solution (achieved 0.74439) has done: 'I enable a pretrained DenseNet‑121 backbone and, when the checkpoint is missing, quickly fine‑tune the model on the provided training images for a couple of epochs. This adds only a small training loop, keeps the original architecture, and is expected to raise the validation accuracy substantially, moving the score nearer to the target while still producing a correct submission.csv.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import torch.utils.data as data
import sklearn.utils as utils
from PIL import Image
import torchvision.transforms as T
import sklearn.metrics as metrics



## === cell 1
input_path = "/kaggle/input/cassava-leaf-disease-classification/"



## === cell 2
num_classes = 5




## === cell 3
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base = models.densenet121(pretrained=True)
        self.dense = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.dense(self.base(x))




## === cell 4
device = "cuda:0" if torch.cuda.is_available() else "cpu"



## === cell 5
model = Model().to(device)



## === cell 6
try:
    model.load_state_dict(
        torch.load("/kaggle/input/cassavamodel/model.pth", map_location=device)
    )
    model_loaded = True
    print("Model checkpoint loaded successfully.")
except FileNotFoundError:
    model_loaded = False
    print("Checkpoint not found – will train a lightweight model.")



## === cell 7
train_df = pd.read_csv(os.path.join(input_path, "train.csv"))
most_common_label = train_df["label"].mode()[0]
print(f"Most common label in training data: {most_common_label}")

if not model_loaded:
    class TrainDataset(data.Dataset):
        def __init__(self, df, transforms):
            super(TrainDataset, self).__init__()
            self.df = df.reset_index(drop=True)
            self.transforms = transforms

        def __getitem__(self, index):
            row = self.df.iloc[index]
            img_path = f"{input_path}/train_images/{row.image_id}"
            img = Image.open(img_path).convert("RGB")
            img = self.transforms(img)
            label = int(row.label)
            return img, label

        def __len__(self):
            return len(self.df)

    train_transforms = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_train_dataset = TrainDataset(train_df, train_transforms)

    train_len = int(0.9 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2  # short fine‑tuning to stay within time limits
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} – validation accuracy: {val_acc:.4f}")

    model_loaded = True  # now we have a usable model



## === cell 8
submission_df = pd.read_csv(os.path.join(input_path, "sample_submission.csv"))




## === cell 9
class TestDataset(data.Dataset):
    def __init__(self, df, transforms):
        super(TestDataset, self).__init__()
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = f"{input_path}/test_images/{row.image_id}"
        img = Image.open(img_path).convert("RGB")
        return self.transforms(img)

    def __len__(self):
        return len(self.df)




## === cell 10
test_transforms = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)



## === cell 11
test_dataset = TestDataset(df=submission_df, transforms=test_transforms)
test_dataloader = data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=32)



## === cell 12
model.eval()
test_preds = []
if model_loaded:
    for x in test_dataloader:
        x = x.to(device)
        with torch.no_grad():
            y_pred = model(x)
        test_preds += y_pred.cpu().numpy().argmax(1).tolist()
    test_preds = np.array(test_preds, dtype=int)
else:
    test_preds = np.full(len(submission_df), most_common_label, dtype=int)



## === cell 13
submission_df["label"] = test_preds



## === cell 14
submission_df.to_csv("submission.csv", index=False)
