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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.995

# 6. Current score

0.00208

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00208) has done: 'I fix the broken filesystem paths that prevent `ImageFolder` from finding any images (your code was pointing to `../input/.../train/train` and `test/test`, and also writing to a non-writable/odd `root_dir=''`). I keep your same approach (copy a subset into `train/` and `val/`, train MobileNetV2, then copy test images into `test/unknown/`) but make it write into `/kaggle/working/` and read images from the correct `/kaggle/input/aerial-cactus-identification/{train,test}` folders. I also fix label lookup to reliably return a scalar (avoids Series→int issues), ensure the test dataset is non-empty before inference, and remove the hard 0.5 thresholding so the submission contains probabilities as required by AUC. Finally, I build the submission using `sample_submission.csv` ordering to guarantee the row count and id alignment matches Kaggle’s expectations.'

# 9. Code solution

## === cell 0
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from os.path import join
import shutil

from tqdm import tqdm  # Progress bar

import torch
import torchvision
import torch.nn.functional as T
from torchvision import transforms, models
from torch.utils.data import DataLoader

random.seed(6)
np.random.seed(6)
torch.manual_seed(6)
if torch.cuda.is_available():
    torch.cuda.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

root_dir = "/kaggle/working"
input_dir = "/kaggle/input/aerial-cactus-identification"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
train_val_labels = pd.read_csv(join(input_dir, "train.csv"))
train_val_labels.head()



## === cell 2
plt.figure(figsize=(3, 3))
plt.title("Labels distribution")
sns.countplot(x=train_val_labels["has_cactus"])



## === cell 3
labels = ["no_cactus", "has_cactus"]

train_dir = join(root_dir, "train")
val_dir = join(root_dir, "val")
test_dir = join(root_dir, "test")

for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)

for label in labels:
    os.makedirs(join(train_dir, label), exist_ok=True)
    os.makedirs(join(val_dir, label), exist_ok=True)

id_to_label = dict(
    zip(train_val_labels["id"].values, train_val_labels["has_cactus"].values)
)



## === cell 4
source_dir = join(input_dir, "train")

for i, filename in enumerate(tqdm(sorted(os.listdir(source_dir)))):
    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    ):
        continue

    if i % 10 != 0:  # Пропускаем 9/10 всех фото
        continue

    if filename not in id_to_label:
        continue
    is_cactus = int(id_to_label[filename])

    if i % 70 == 0:
        shutil.copy(
            join(source_dir, filename), join(val_dir, labels[is_cactus], filename)
        )
    else:
        shutil.copy(
            join(source_dir, filename), join(train_dir, labels[is_cactus], filename)
        )

print("Train images:", sum(len(files) for _, _, files in os.walk(train_dir)))
print("Val images:", sum(len(files) for _, _, files in os.walk(val_dir)))




## === cell 5
def show_sample_images(
    dataloader, batch_size, images_from_batch=0, denormalize=False, classes=None
):
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([1.0, 1.0, 1.0])

    if images_from_batch == 0 or images_from_batch > batch_size:
        images_from_batch = batch_size

    for images, labels in dataloader:
        plt.figure(figsize=(20, (batch_size // 20 + 1) * 3))

        cols = 12
        rows = batch_size // cols + 1
        for i in range(min(images_from_batch, images.shape[0])):
            image = images[i].permute(1, 2, 0).cpu().numpy() * std + mean
            plt.subplot(rows, cols, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image.clip(0, 1))
            if classes is not None:
                plt.xlabel(classes[int(labels[i].cpu().numpy())])
        plt.show()
        break




## === cell 6
batch_size = 250

classes = ["No", "Cactus"]

train_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
)

for images, labels_ in train_dataloader:
    print(images.size())
    print(labels_.size())
    break



## === cell 7
show_sample_images(train_dataloader, batch_size, 72, denormalize=True, classes=classes)



## === cell 8
print(f"Batch size: {batch_size}")
print(f"Train batches: {len(train_dataloader)}, Train samples: {len(train_dataset)}")
print(f"Val batches:   {len(val_dataloader)}, Val samples:    {len(val_dataset)}")



## === cell 9
train_batch_loss_history = []
train_batch_accuracy_history = []

train_loss_history = []
train_accuracy_history = []

val_loss_history = []
val_accuracy_history = []


def validate(model, loss, optimizer):
    dataloader = val_dataloader
    model.eval()

    sum_loss = 0.0
    sum_accuracy = 0.0

    for inputs, labels_ in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels_ = labels_.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(False):
            preds = model(inputs)
            loss_value = loss(preds, labels_)
            preds_class = preds.argmax(dim=1)

        sum_loss += loss_value.item()
        sum_accuracy += (
            (preds_class == labels_.data).float().mean().detach().cpu().numpy().item()
        )

    val_loss = sum_loss / max(1, len(dataloader))
    val_accuracy = sum_accuracy / max(1, len(dataloader))

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    print(f"Validation accuracy {val_accuracy * 100:.2f} %, loss {val_loss:.4f}")

    model.train()


def train_model(model, loss, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs-1}: \n", end="")

        dataloader = train_dataloader
        model.train()

        sum_loss = 0.0
        sum_accuracy = 0.0

        for inputs, labels_ in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels_ = labels_.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):
                preds = model(inputs)
                loss_value = loss(preds, labels_)
                preds_class = preds.argmax(dim=1)

                loss_value.backward()
                optimizer.step()

            batch_loss = loss_value.item()
            batch_accuracy = (
                (preds_class == labels_.data)
                .float()
                .mean()
                .detach()
                .cpu()
                .numpy()
                .item()
            )

            sum_loss += batch_loss
            sum_accuracy += batch_accuracy

            train_batch_loss_history.append(batch_loss)
            train_batch_accuracy_history.append(batch_accuracy)

        epoch_loss = sum_loss / max(1, len(dataloader))
        epoch_acc = sum_accuracy / max(1, len(dataloader))

        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_acc)
        scheduler.step()

        print("\n End epoch: ", end="")
        validate(model, loss, optimizer)

    return model




## === cell 10
try:
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
except Exception:
    model = models.mobilenet_v2(pretrained=True)

model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40], gamma=0.1
)



## === cell 11
print(
    f"Batch size: {batch_size}\nBatches: {len(train_dataloader)}\nAll elements: {len(train_dataset)}"
)



## === cell 12
epochs = int(np.sqrt(batch_size)) + 5  # Automatically choose epoch number
train_model(model, loss, optimizer, scheduler, num_epochs=epochs)



## === cell 13
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.plot(train_batch_loss_history, label="Train Batch Loss")
plt.plot(train_batch_accuracy_history, label="Train Batch Accuracy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracy_history, label="Train accuracy")
plt.plot(val_accuracy_history, label="Val accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.legend()



## === cell 14
os.makedirs(test_dir, exist_ok=True)
os.makedirs(join(test_dir, "unknown"), exist_ok=True)

source_dir = join(input_dir, "test")

for f in os.listdir(join(test_dir, "unknown")):
    try:
        os.remove(join(test_dir, "unknown", f))
    except OSError:
        pass

for i, filename in enumerate(tqdm(sorted(os.listdir(source_dir)))):
    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    ):
        continue
    shutil.copy(join(source_dir, filename), join(test_dir, "unknown", filename))
    if i < 10:
        print(filename)

print("Test images copied:", len(os.listdir(join(test_dir, "unknown"))))



## === cell 15
test_transforms = val_transforms
test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
)

for images, labels_ in test_dataloader:
    print(images.size())
    print(labels_.size())
    break



## === cell 16
show_sample_images(
    test_dataloader, batch_size, 12, denormalize=True, classes=["Unknown"]
)



## === cell 17
model.eval()

test_predictions = []
for images, _ in test_dataloader:
    images = images.to(device, non_blocking=True)
    with torch.set_grad_enabled(False):
        preds = model(images)
        probs = T.softmax(preds, dim=1)[:, 1]
    test_predictions.append(probs.detach().cpu().numpy())

test_predictions = np.concatenate(test_predictions)
print(
    "Preds:",
    test_predictions.shape,
    "min/max:",
    float(test_predictions.min()),
    float(test_predictions.max()),
)



## === cell 18
sample_sub = pd.read_csv(join(input_dir, "sample_submission.csv"))
pred_map = dict(zip(sorted(os.listdir(join(test_dir, "unknown"))), test_predictions))

sample_sub["has_cactus"] = sample_sub["id"].map(pred_map).fillna(0.5).astype(float)
sample_sub.head(25)



## === cell 19
out_path = "submission.csv"
sample_sub.to_csv(out_path, index=False)
print("Wrote", out_path, "rows:", len(sample_sub))



## === cell 20
for d in ["train", "val", "test"]:
    p = join(root_dir, d)
    if os.path.exists(p):
        shutil.rmtree(p)
print("Cleaned up working directories.")
