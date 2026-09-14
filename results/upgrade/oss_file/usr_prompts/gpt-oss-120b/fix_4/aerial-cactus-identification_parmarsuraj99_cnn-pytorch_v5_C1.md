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

0.9927

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))




## === cell 1
import cv2




## === cell 2
from PIL import Image
from matplotlib import pyplot as plt




## === cell 3
import torch




## === cell 4
BASE_DIR = "../input/aerial-cactus-identification/"

TRAIN_DIR = os.path.join(BASE_DIR, "train/")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_DIR = os.path.join(BASE_DIR, "test/")




## === cell 5
print("Test directory:", TEST_DIR)
print("Number of entries:", len(os.listdir(TEST_DIR)))




## === cell 6
train_names = pd.read_csv(TRAIN_CSV)
print(train_names.shape)




## === cell 7
print("Number of image files in train folder:", len(os.listdir(TRAIN_DIR)))




## === cell 8
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms




## === cell 9
def show_image(index):
    image_path = os.path.join(TRAIN_DIR, str(train_names["id"][index]))
    image = Image.open(image_path)
    plt.imshow(image)
    print("Label:", train_names["has_cactus"][index])


show_image(20)




## === cell 10
class HasCactusDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.img_names = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_names.iloc[idx, 0])
        image = cv2.imread(img_name)  # BGR uint8 ndarray
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
        image = torch.from_numpy(image).float() / 255.0  # C,H,W after permute
        image = image.permute(2, 0, 1)  # to (C, H, W)
        has_cactus = float(self.img_names.iloc[idx, 1])
        return image, has_cactus




## === cell 11
hasCactusDataset = HasCactusDataset(
    csv_file=TRAIN_CSV,
    root_dir=TRAIN_DIR,
)

plt.imshow(hasCactusDataset[19][0].permute(1, 2, 0).numpy())
plt.title(f"Label: {hasCactusDataset[19][1]}")
plt.show()




## === cell 12
dataloader = DataLoader(hasCactusDataset, batch_size=128, shuffle=True)




## === cell 13
import torch.nn as nn
import torch.nn.functional as F




## === cell 14
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


net = Net()




## === cell 15
import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)  # Adam often converges faster




## === cell 16
num_epochs = 20  # slightly more epochs to improve AUC
for epoch in range(num_epochs):
    print(f"epoch {epoch} started...")
    for i, data in enumerate(dataloader):
        images, labels = data
        optimizer.zero_grad()

        inputs = images  # already FloatTensor
        labels = labels.view(-1, 1)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} complete")
print("finished Training...")




## === cell 17
data = next(iter(dataloader))
print("Batch image shape:", data[0].shape, "Batch label shape:", data[1].shape)




## === cell 18
print("Test directory contents (filtered):", len(os.listdir(TEST_DIR)))




## === cell 19
class TestSet(Dataset):
    def __init__(self, root_dir):
        self.img_names = [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root_dir, self.img_names[idx])
        image = cv2.imread(img_loc)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_loc}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # C, H, W
        name = self.img_names[idx]
        return image, name




## === cell 20
testSet = TestSet(
    root_dir=TEST_DIR,
)




## === cell 21
testloader = DataLoader(
    testSet, batch_size=1, shuffle=False
)  # keep order for reproducible submission




## === cell 22
tstdata = next(iter(testloader))
print("First test image tensor shape:", tstdata[0].shape, "Name:", tstdata[1])




## === cell 23
plt.imshow(tstdata[0].permute(1, 2, 0).numpy())
plt.title(tstdata[1])
plt.show()




## === cell 24
tst_outputs = []
tst_names = []
net.eval()
with torch.no_grad():
    for i, (image, name) in enumerate(testloader):
        inputs = image  # already FloatTensor
        prob = net(inputs).item()  # probability after sigmoid
        tst_outputs.append(prob)
        tst_names.append(name[0])
print(f"Generated predictions for {len(tst_outputs)} test images.")




## === cell 25
print("First 5 predictions:", tst_outputs[:5])




## === cell 26
print("Number of prediction entries:", len(tst_names))




## === cell 27
my_submission = pd.DataFrame({"id": tst_names, "has_cactus": tst_outputs})
my_submission.to_csv("submission.csv", index=False)




## === cell 28
print("Submission saved to submission.csv with", len(my_submission), "rows.")
