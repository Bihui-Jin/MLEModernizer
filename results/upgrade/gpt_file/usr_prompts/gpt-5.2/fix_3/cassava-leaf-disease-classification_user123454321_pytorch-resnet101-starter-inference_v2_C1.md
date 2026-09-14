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

0.8732245391356905

# 6. Current score

0.72422

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12444) has done: 'I fix the two runtime blockers: the missing external weight file (so inference never runs) and the hard-coded `.cuda()` call that crashes on CPU-only environments. To move accuracy toward the target (current is far below target), I keep your ResNet101-based model but load ImageNet pretrained weights when the custom checkpoint isn’t available, which is a minimal change that should greatly improve predictions versus random initialization. I also ensure inference uses `device` consistently and runs under `torch.inference_mode()` to prevent empty predictions and speed up safely. Finally, I write a properly aligned `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.72422) has done: 'Your current score (0.12444) is far below the target (0.8732), and the main remaining issue is that you are effectively submitting an ImageNet model with a randomly initialized 5-class head, which yields near-random predictions. To move accuracy sharply toward the target while keeping your core ResNet101 logic, the smallest legitimate improvement is to train only the final linear layer on the provided `train.csv` images (freezing the backbone) for a short, fixed number of epochs. This preserves the architecture and loss semantics while producing a head aligned to cassava labels, which should dramatically reduce the gap without introducing early stopping or approximations. I also switch to the official ResNet101 weight transforms (mean/std + crop/resize) to better match pretrained expectations, and keep the submission writing/ordering identical.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import torchvision.models as models
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T



## === cell 1
input_path = "/kaggle/input/cassava-leaf-disease-classification/"



## === cell 2
df = pd.read_csv(os.path.join(input_path, "train.csv"))



## === cell 3
num_classes = len(df.label.unique())
num_classes




## === cell 4
class Model(nn.Module):
    def __init__(self, use_imagenet_pretrained=True):
        super(Model, self).__init__()
        if use_imagenet_pretrained:
            weights = models.ResNet101_Weights.IMAGENET1K_V2
        else:
            weights = None
        backbone = models.resnet101(weights=weights)
        self.base = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.dense(x)




## === cell 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 6
model = Model(use_imagenet_pretrained=True).to(device)



## === cell 7
ckpt_path = "/kaggle/input/resnet-cassava-model/model_101_20.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    print(
        f"Checkpoint not found: {ckpt_path}. Will train a linear head on top of the ImageNet-pretrained backbone."
    )



## === cell 8
submission_df = pd.read_csv(os.path.join(input_path, "sample_submission.csv"))



## === cell 9
submission_df.head()




## === cell 10
class ImageDataset(data.Dataset):
    def __init__(self, df, image_dir, transforms, return_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.return_label = return_label

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.image_dir, row.image_id)
        img = Image.open(img_path).convert("RGB")
        x = self.transforms(img)
        if self.return_label:
            y = int(row.label)
            return x, y
        return x

    def __len__(self):
        return len(self.df)




## === cell 11
imagenet_weights = models.ResNet101_Weights.IMAGENET1K_V2
train_transforms = imagenet_weights.transforms()  # includes resize/crop + normalization
test_transforms = imagenet_weights.transforms()



## === cell 12
if not os.path.exists(ckpt_path):
    for p in model.base.parameters():
        p.requires_grad = False

    train_dataset = ImageDataset(
        df=df,
        image_dir=os.path.join(input_path, "train_images"),
        transforms=train_transforms,
        return_label=True,
    )

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.dense.parameters(), lr=3e-3, weight_decay=1e-2)

    model.train()
    epochs = 3  # fixed; not early stopping. Small to fit time limit while materially improving over random.
    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        print(
            f"epoch {ep+1}/{epochs} | loss {running_loss/max(total,1):.4f} | train_acc {correct/max(total,1):.4f}"
        )



## === cell 13
test_dataset = ImageDataset(
    df=submission_df,
    image_dir=os.path.join(input_path, "test_images"),
    transforms=test_transforms,
    return_label=False,
)
test_dataloader = data.DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=32,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 14
model.eval()
test_preds = []

with torch.inference_mode():
    for x in test_dataloader:
        x = x.to(device, non_blocking=True)
        y_pred = model(x)
        test_preds.extend(y_pred.argmax(1).cpu().numpy().tolist())

test_preds = np.asarray(test_preds, dtype=np.int64)
len(test_preds), submission_df.shape



## === cell 15
if len(test_preds) != len(submission_df):
    raise RuntimeError(
        f"Prediction length {len(test_preds)} != submission length {len(submission_df)}"
    )

submission_df["label"] = test_preds



## === cell 16
submission_df[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df[["image_id", "label"]].shape)
