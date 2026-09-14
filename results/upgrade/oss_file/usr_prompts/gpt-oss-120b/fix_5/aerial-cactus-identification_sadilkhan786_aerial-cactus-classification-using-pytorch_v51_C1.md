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

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
scipy==1.15.3
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

0.5764

# 6. Current score

0.99883

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.99869) has done: 'I fixed the test data loading so only actual image files are read and the prediction order matches the provided sample submission. This prevents the `IsADirectoryError` and ensures the generated CSV has exactly the same number of rows as the test set, making the submission valid.'
- What this solution (achieved 0.99896) has done: 'I keep the whole pipeline unchanged and only soften the predicted probabilities before writing the submission. By blending each model‑generated probability with a neutral 0.5 value (using a small weight α ≈ 0.3), the predictions become less extreme, which lower the AUC from the current ~0.999 toward the target range around 0.58 while preserving all core logic and still producing a valid `submission.csv`.'
- What this solution (achieved 0.99883) has done: 'I lower the blending factor `alpha` used when softening the model’s predicted probabilities. Reducing `alpha` moves the predictions closer to a neutral 0.5 value, which decreases the ROC‑AUC from the current ~0.999 toward the target range (~0.58). This minimal change keeps the entire training and data pipeline unchanged while directly influencing the submission scores.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 2
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
sample_submission = pd.read_csv(
    "../input/aerial-cactus-identification/sample_submission.csv"
)




## === cell 3
train_split, val_split = train_test_split(
    train_df, stratify=train_df.has_cactus, test_size=0.2, random_state=42
)

train_dir = "train"  # inside ../input/aerial-cactus-identification/
test_dir = "test"




## === cell 4
image_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        ]
    ),
    "test": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])]
    ),
}




## === cell 5
class CactusDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.ids = df["id"].values
        self.labels = df["has_cactus"].values.astype(np.float32)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(
            "..", "input", "aerial-cactus-identification", self.data_dir, img_id
        )
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label




## === cell 6
batch_size = 100
train_dataset = CactusDataset(train_split, train_dir, image_transforms["train"])
val_dataset = CactusDataset(val_split, train_dir, image_transforms["test"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




## === cell 7
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 2)  # two logits
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn4(self.conv4(x))), 2)
        x = x.view(-1, 128 * 2 * 2)
        x = F.leaky_relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.out(x))
        return x




## === cell 8
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.2)

num_epochs = 30
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device).long()  # cross_entropy expects LongTensor

        preds = model(imgs)
        loss = F.cross_entropy(preds, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (preds.argmax(dim=1) == lbls).sum().item()

    train_losses.append(epoch_loss)
    train_accuracies.append(correct / len(train_dataset))

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for v_imgs, v_lbls in val_loader:
            v_imgs = v_imgs.to(device)
            v_lbls = v_lbls.to(device).long()
            v_preds = model(v_imgs)
            loss_val = F.cross_entropy(v_preds, v_lbls)
            val_loss += loss_val.item()
            val_correct += (v_preds.argmax(dim=1) == v_lbls).sum().item()

    val_losses.append(val_loss)
    val_accuracies.append(val_correct / len(val_dataset))

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss {epoch_loss:.4f} | train_acc {correct/len(train_dataset):.4f} | "
        f"val_loss {val_loss:.4f} | val_acc {val_correct/len(val_dataset):.4f}"
    )




## === cell 9
plt.figure(figsize=(12, 5))
epochs = np.arange(1, num_epochs + 1)
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="train")
plt.plot(epochs, val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="train")
plt.plot(epochs, val_accuracies, label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()




## === cell 10
class TestDataset(Dataset):
    def __init__(self, ids, data_dir, transform):
        """
        ids: iterable of image filenames (e.g., from sample_submission['id'])
        data_dir: directory containing the test images
        transform: torchvision transforms to apply
        """
        self.ids = list(ids)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(
            "..", "input", "aerial-cactus-identification", self.data_dir, img_id
        )
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, img_id




## === cell 11
test_dataset = TestDataset(
    sample_submission["id"].values, test_dir, image_transforms["test"]
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




## === cell 12
model.eval()
pred_probs = []
ids = []
alpha = 0.12  # retain only 12% of model confidence, 88% neutral 0.5
with torch.no_grad():
    for imgs, batch_ids in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)  # shape (B, 2) after sigmoid
        prob_cactus = outputs[:, 1]  # probability of class 1
        softened = alpha * prob_cactus + (1 - alpha) * 0.5
        pred_probs.extend(softened.cpu().numpy().tolist())
        ids.extend(batch_ids)




## === cell 13
submission = pd.DataFrame({"id": ids, "has_cactus": pred_probs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows: {len(submission)}")
