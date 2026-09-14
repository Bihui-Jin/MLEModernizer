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

No external packages required in the script and installed.

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

0.505

# 6. Current score

0.97158

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.97158) has done: 'I fix the dataset handling so that only image files present in the label CSV are used, add a proper test‑dataset class that returns images and their filenames, and update the inference loop to build predictions for every test image. These changes remove the KeyError, define `train_loader`/`test_loader` correctly, and ensure the produced `submission.csv` has the exact number of rows required by Kaggle.'

# 9. Code solution

## === cell 0
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")




## === cell 1
train_csv_path = "../input/train.csv"
image_cat = pd.read_csv(train_csv_path, low_memory=False, index_col="id")[
    "has_cactus"
].to_dict()




## === cell 2
def make_dataset(dir_path, class_to_idx):
    """
    Build a list of (image_path, label) pairs, ignoring files that are not in `image_cat`.
    """
    images = []
    dir_path = os.path.expanduser(dir_path)

    for filename in os.listdir(dir_path):
        if filename not in image_cat:
            continue  # skip unexpected files such as sub‑folders
        path = os.path.join(dir_path, filename)
        images.append((path, image_cat[filename]))
    return images


class CactusImageFolder(datasets.DatasetFolder):
    """
    Custom dataset that uses the label dict `image_cat` instead of sub‑folder names.
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = make_dataset(self.root, self.class_to_idx)
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self, dir_path):
        def label_to_name(fname):
            return "cactus" if image_cat[fname] == 1 else "noncactus"

        class_names = set()
        for fname in os.listdir(dir_path):
            if fname not in image_cat:
                continue
            class_names.add(label_to_name(fname))
        class_names = sorted(class_names)  # deterministic order
        class_to_idx = {class_names[i]: i for i in range(len(class_names))}
        return class_names, class_to_idx


class TestImageDataset(torch.utils.data.Dataset):
    """
    Loads test images and returns (tensor, filename) pairs.
    """

    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.paths.sort()  # deterministic order

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, path




## === cell 3
data_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_root = "../input/train/train"
train_set = CactusImageFolder(root=train_root, transform=data_transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True, num_workers=0
)

test_root = "../input/test"
test_set = TestImageDataset(root=test_root, transform=data_transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=0
)

classes = train_set.classes  # e.g., ['cactus', 'noncactus']




## === cell 4
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


data_iter = iter(train_loader)
images, labels = next(data_iter)
imshow(torchvision.utils.make_grid(images))
print(
    "Labels:", " ".join("%5s" % classes[labels[j].item()] for j in range(len(labels)))
)




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # binary classification (2 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()




## === cell 6
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)




## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")




## === cell 8
net.eval()
preds = []

with torch.no_grad():
    for images, paths in test_loader:
        images = images.to(device)
        outputs = net(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of label 1
        prob_val = probs.item()
        file_name = os.path.basename(paths[0])
        preds.append([file_name, prob_val])




## === cell 9
submission_path = "submission.csv"
output = pd.DataFrame(preds, columns=["id", "has_cactus"])
output.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with {len(output)} rows.")




## === cell 10
rows, cols = 4, 5
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
fig.subplots_adjust(hspace=0.5, wspace=0.4)

for idx, (name, prob) in enumerate(output.iloc[: rows * cols].values):
    img_path = os.path.join(test_root, name)
    img = plt.imread(img_path)
    r = idx // cols
    c = idx % cols
    ax[r, c].imshow(img)
    ax[r, c].axis("off")
    ax[r, c].set_title(f"Cactus prob: {prob:.2f}")

plt.show()
