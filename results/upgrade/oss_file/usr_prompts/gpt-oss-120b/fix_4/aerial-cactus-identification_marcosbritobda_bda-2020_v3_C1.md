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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.9895

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I removed the stray non‑code text that caused a syntax error, deleted the stray markdown fence in the prediction cell, and eliminated the IPython magic commands that break when run as a script. I also increased the training epochs from 15 to 20 to give the model a bit more learning capacity, which should help move the AUC closer to the target while keeping the original architecture and training logic unchanged. The script now runs end‑to‑end, creates a “submission.csv” with the correct columns, and is ready for Kaggle submission.'
- What this solution (achieved 0.5) has done: 'I add a deterministic validation transform (no random flips/rotations) and compute the AU ROC on the validation set each epoch, saving the model that achieves the best AUC instead of only the lowest loss. This small change keeps the original CNN architecture and training loop while giving the model a clearer signal to improve the AUC, moving the score toward the target of 0.9895. I also import `roc_auc_score` for the metric calculation.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/"))

import matplotlib.pyplot as plt

plt.style.use("ggplot")

import cv2

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.optim as optim

from sklearn.metrics import roc_auc_score




## === cell 1
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
train_df.head()




## === cell 2
print(
    f"Train Size: {len(os.listdir('../input/aerial-cactus-identification/train/train'))}"
)
print(
    f"Test Size: {len(os.listdir('../input/aerial-cactus-identification/test/test'))}"
)




## === cell 3
value_counts = train_df.has_cactus.value_counts()
plt.pie(
    value_counts,
    labels=["Has Cactus", "No Cactus"],
    autopct="%1.1f",
    colors=["green", "red"],
    shadow=True,
)
plt.figure(figsize=(5, 5))
plt.show()




## === cell 4
train_path = "../input/aerial-cactus-identification/train/train/"
test_path = "../input/aerial-cactus-identification/test/test/"




## === cell 5
class CreateDataset(Dataset):
    def __init__(self, df_data, data_dir="./", transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            image = np.zeros((32, 32, 3), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image)

        label_tensor = torch.tensor(int(label), dtype=torch.long)

        return image, label_tensor




## === cell 6
transforms_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transforms_valid = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_data = CreateDataset(
    df_data=train_df, data_dir=train_path, transform=transforms_train
)
val_data = CreateDataset(
    df_data=train_df, data_dir=train_path, transform=transforms_valid
)

batch_size = 64
valid_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(val_data, batch_size=batch_size, sampler=valid_sampler)




## === cell 7
transforms_test = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

sample_sub = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
test_data = CreateDataset(
    df_data=sample_sub, data_dir=test_path, transform=transforms_test
)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)




## === cell 8
classes = ["No Cactus", "Cactus"]




## === cell 9
def imshow(img):
    """Helper function to un‑normalize and display an image."""
    img = img / 2 + 0.5  # reverse normalization
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(train_loader)
images, labels = next(dataiter)  # corrected iterator usage
imshow(torchvision.utils.make_grid(images[:20]))
plt.title("Sample Training Images")
plt.show()




## === cell 10
rgb_img = images[3].cpu().numpy()
channels = ["red channel", "green channel", "blue channel"]

fig = plt.figure(figsize=(12, 12))
for idx in range(rgb_img.shape[0]):
    ax = fig.add_subplot(3, 1, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap="gray")
    ax.set_title(channels[idx])
    width, height = img.shape[:2]
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(
                str(val),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                size=8,
                color="white" if img[x][y] < thresh else "black",
            )
plt.show()




## === cell 11
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




## === cell 12
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print("CUDA is not available. Training on CPU ...")
else:
    print("CUDA is available! Training on GPU ...")




## === cell 13
model = CNN()
print(model)
if train_on_gpu:
    model = model.cuda()




## === cell 14
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=0.001)




## === cell 15
n_epochs = 20  # increased epochs for better AUC
best_auc = 0.0

train_losses = []
valid_losses = []

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    val_targets = []
    val_probs = []
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

        probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        val_probs.extend(probs)
        val_targets.extend(target.cpu().numpy())

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    try:
        val_auc = roc_auc_score(val_targets, val_probs)
    except ValueError:
        val_auc = 0.0

    print(
        f"Epoch: {epoch} \\tTraining Loss: {train_loss:.6f} \\tValidation Loss: {valid_loss:.6f} \\tVal AUC: {val_auc:.4f}"
    )

    if val_auc > best_auc:
        print(
            f"Validation AUC improved ({best_auc:.4f} --> {val_auc:.4f}). Saving model ..."
        )
        torch.save(model.state_dict(), "best_model.pt")
        best_auc = val_auc




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1038879509.py in <cell line: 0>()
     31         valid_loss += loss.item() * data.size(0)
     32 
---> 33         probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
     34         val_probs.extend(probs)
     35         val_targets.extend(target.cpu().numpy())

RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

## === cell 16
plt.plot(train_losses, label="Training loss")
plt.plot(valid_losses, label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()




## === cell 17
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()
if train_on_gpu:
    model = model.cuda()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1955270837.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
      2 model.eval()
      3 if train_on_gpu:
      4     model = model.cuda()
      5 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pt'

## === cell 18
preds = []
model.eval()
with torch.no_grad():
    for data, _ in test_loader:  # underscore discards dummy label
        if train_on_gpu:
            data = data.cuda()
        output = model(data)
        pr = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        preds.extend(pr.tolist())

sample_sub["has_cactus"] = preds
sample_sub.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
