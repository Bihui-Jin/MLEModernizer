# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

2.7

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
scikit-image==0.25.2
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

# 5. Code solution

## === cell 0
from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 1
class CactusImageDataset(Dataset):
    "Aerial Cactus Classification Dataset"

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.cactus_annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.cactus_annotations.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.cactus_annotations.iloc[idx, 0])
        image = io.imread(img_name)
        has_cactus = self.cactus_annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label": has_cactus}
        return sample




## === cell 2
image_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}



## === cell 3
dataset = CactusImageDataset(
    "../input/aerial-cactus-identification/train.csv",
    "../input/aerial-cactus-identification/train/train/",
    image_transforms["train"],
)



## === cell 4
get_ipython().run_line_magic("matplotlib", "inline")
plt.figure(dpi=128, figsize=(3, 3))
plt.title("Distribution of Cacti Images")
plt.xticks([0, 1])
plt.xlabel("Has Cactus")
plt.ylabel("Number of Images")
dataset.cactus_annotations["has_cactus"].hist(bins=2)



## === cell 5
train_size = int((0.80 * dataset.__len__()))
dev_size = dataset.__len__() - train_size
g = torch.Generator().manual_seed(42)
train_set, dev_set = random_split(dataset, [train_size, dev_size], generator=g)



## === cell 6
BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)




## === cell 7
class CactusIdentifier(nn.Module):
    """
    Aerial Cactus Identifier Model
    """

    def __init__(self):
        super(CactusIdentifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()

    def forward(self, inp_image):
        out = self.pool(self.act(self.conv1(inp_image)))
        out = self.pool(self.act(self.conv2(out)))
        out = self.pool(self.act(self.conv3(out)))
        out = self.pool(self.act(self.conv4(out)))

        out = out.view(-1, 128 * 2 * 2)
        out = self.drop(out)
        out = self.act(self.fc1(out))
        out = self.drop(out)
        out = self.act(self.fc2(out))
        out = self.drop(out)
        out = self.fc3(out)

        return out




## === cell 8
class TrainingModule:
    """
    Training Module to train the model
    """

    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True
            self.model = model.cuda()

    def train_epoch(self, epoch, train_iterator):
        total_loss = 0
        stats_string = "Epoch : {:3d} | Iteration : {:4d} | Loss : {:4.4f}"
        for i, data in enumerate(train_iterator):
            inputs = data["image"]
            labels = data["label"].long()

            if self.is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(stats_string.format(epoch + 1, i, total_loss / (i + 1)))
                total_loss = 0

    def train_model(self, train_iterator, dev_iterator, num_epocs=30):
        self.model.train()
        min_loss = 1000000
        stats_string = "Epoch : {:2d} | Train Loss : {:4.4f} | Dev Loss : {:4.4f}"
        for i in range(num_epocs):
            self.train_epoch(i, train_iterator)
            dev_loss = self.evaluate(dev_iterator)
            train_loss = self.evaluate(train_iterator)
            print(stats_string.format(i + 1, train_loss, dev_loss))
            if dev_loss <= min_loss:
                torch.save(self.model.state_dict(), "best_model")
                min_loss = dev_loss

        print("Finished Training")

    def evaluate(self, iterator):
        self.model.eval()
        loss_total = 0.0
        with torch.no_grad():
            for i, data in enumerate(iterator):
                inputs = data["image"]
                labels = data["label"].long()

                if self.is_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                preds = self.model(inputs)
                loss = self.loss_fn(preds, labels)
                loss_total += loss.item()

        return loss_total




## === cell 9
model = CactusIdentifier()
print(model)



## === cell 10
trainer = TrainingModule(model)

_root = os.path.normpath(getattr(dataset, "root_dir", ""))
if (
    os.path.basename(_root) == "train"
    and os.path.basename(os.path.dirname(_root)) == "train"
):
    dataset.root_dir = os.path.dirname(_root)  # drop the extra trailing "train/"

if not os.path.isdir(getattr(dataset, "root_dir", "")):
    raise IOError(
        "Image root_dir does not exist: '{}'".format(getattr(dataset, "root_dir", ""))
    )

if not os.path.exists("best_model"):
    trainer.train_model(train_loader, dev_loader, num_epocs=30)
else:
    print("Found existing 'best_model' checkpoint; skipping training to reuse it.")


## === cell 11
class CactusImageDataset(Dataset):
    "Aerial Cactus Classification Dataset"

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.cactus_annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.cactus_annotations.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.cactus_annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, filename)
        if not os.path.exists(img_name):
            alt_root = os.path.dirname(os.path.normpath(self.root_dir))
            alt_img_name = os.path.join(alt_root, filename)
            if os.path.exists(alt_img_name):
                img_name = alt_img_name
            else:
                raise IOError(
                    "No such file: '{}' (also tried '{}')".format(
                        img_name, alt_img_name
                    )
                )

        image = io.imread(img_name)
        has_cactus = self.cactus_annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label": has_cactus}
        return sample




## === cell 12
test_set = CactusImageDataset(
    "../input/aerial-cactus-identification/sample_submission.csv",
    "../input/aerial-cactus-identification/test/",
    image_transforms["test"],
)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)



## === cell 13
ckpt_name = "best_model"
ckpt_path = ckpt_name

if not os.path.exists(ckpt_path):
    candidate_paths = [
        os.path.join(os.getcwd(), ckpt_name),
        os.path.join("/kaggle/working", ckpt_name),
        os.path.join("/kaggle/data", ckpt_name),
        os.path.join("/kaggle/input", ckpt_name),
    ]
    for base in ["/kaggle/working", "/kaggle/data", "/kaggle/input"]:
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                if ckpt_name in files:
                    candidate_paths.append(os.path.join(root, ckpt_name))
                    break

    for p in candidate_paths:
        if os.path.exists(p):
            ckpt_path = p
            break

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print(
        "WARNING: Checkpoint '{}' not found. Proceeding with current in-memory model weights.".format(
            ckpt_name
        )
    )

model = model.to(device)
model.eval()

final_preds = []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data["image"].to(device)

        logits = model(inputs)
        probs = softmax(logits)[:, 1]  # P(has_cactus=1)
        final_preds.extend(probs.detach().cpu().numpy().tolist())

test_set.cactus_annotations["has_cactus"] = final_preds
test_set.cactus_annotations.to_csv("submission.csv", index=False)



## === cell 14
test_set.cactus_annotations.head()
