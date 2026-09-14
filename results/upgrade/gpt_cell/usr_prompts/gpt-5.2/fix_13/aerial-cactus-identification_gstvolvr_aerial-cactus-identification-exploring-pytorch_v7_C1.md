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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import warnings
import shutil

%matplotlib inline
warnings.filterwarnings("ignore")


## === cell 1
image_cat = pd.read_csv("../input/train.csv", low_memory=False, index_col="id").to_dict()["has_cactus"]


## === cell 2
??torchvision.datasets.ImageFolder


## === cell 3
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        item = (path, image_cat[filename])
        images.append(item)
    return images

class CactusImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):                                                                                              
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = make_dataset(self.root, self.class_to_idx)
        self.loader = loader
        self.targets = [s[1] for s in self.samples]
        
    def _find_classes(self, dir):
        def f(x):
            return "cactus" if image_cat[x] == 1 else "noncactus"
        classes = list(set([ f(filename) for filename in os.listdir(dir)]))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


## === cell 4
from torchvision import transforms, datasets

data_transorm = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def _resolve_image_dir(root_dir):
    root_dir = os.path.expanduser(root_dir)

    def _has_images(d):
        if not os.path.isdir(d):
            return False
        for e in os.listdir(d):
            p = os.path.join(d, e)
            if os.path.isfile(p) and e.lower().endswith(IMG_EXTENSIONS):
                return True
        return False

    cur = root_dir
    while os.path.isdir(cur) and not _has_images(cur):
        subdirs = [
            os.path.join(cur, e)
            for e in os.listdir(cur)
            if os.path.isdir(os.path.join(cur, e))
        ]
        if len(subdirs) == 1:
            cur = subdirs[0]
            continue
        found = None
        for p in subdirs:
            if _has_images(p):
                found = p
                break
        if found is not None:
            cur = found
        break

    return cur


def _resolve_leaf_image_dir(root_dir):
    d = _resolve_image_dir(root_dir)
    while os.path.isdir(d):
        entries = os.listdir(d)
        img_files = [
            e
            for e in entries
            if os.path.isfile(os.path.join(d, e)) and e.lower().endswith(IMG_EXTENSIONS)
        ]
        if len(img_files) > 0:
            return d
        subdirs = [e for e in entries if os.path.isdir(os.path.join(d, e))]
        if len(subdirs) == 1:
            d = os.path.join(d, subdirs[0])
            continue
        return d
    return d


train_root = _resolve_leaf_image_dir("../input/train")
test_root = _resolve_leaf_image_dir("../input/test")


def _filter_to_images(dir_path):
    return sorted(
        [
            fn
            for fn in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, fn))
            and fn.lower().endswith(IMG_EXTENSIONS)
        ]
    )


_orig_make_dataset = make_dataset


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)

    for filename in _filter_to_images(dir):
        path = os.path.join(dir, filename)
        item = (path, image_cat[filename])
        images.append(item)
    return images


_orig_find_classes = CactusImageFolder._find_classes


def _find_classes(self, dir):
    def f(x):
        return "cactus" if image_cat[x] == 1 else "noncactus"

    classes = list(set([f(filename) for filename in _filter_to_images(dir)]))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


CactusImageFolder._find_classes = _find_classes

train_set = CactusImageFolder(root=train_root, transform=data_transorm)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True, num_workers=4
)

test_class_dir = os.path.join(test_root, "unknown")
if os.path.isdir(test_root) and not os.path.isdir(test_class_dir):
    os.makedirs(test_class_dir, exist_ok=True)
    for fn in os.listdir(test_root):
        src = os.path.join(test_root, fn)
        if os.path.isfile(src) and fn.lower().endswith(IMG_EXTENSIONS):
            dst = os.path.join(test_class_dir, fn)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

test_set = datasets.ImageFolder(root=test_root, transform=data_transorm)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=True, num_workers=4
)

classes = train_set.classes


## === cell 5
len(test_set)


## === cell 6
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


data_iter = iter(train_loader)

images, labels = next(data_iter)

imshow(torchvision.utils.make_grid(images))
print(" ".join("%5s" % classes[labels[j]] for j in range(4)))


## === cell 7
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()


## === cell 8
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


## === cell 9
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


## === cell 10
test_iter = iter(test_loader)
images, _ = next(test_iter)

imshow(torchvision.utils.make_grid(images))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print(
    "Predicted: ",
    " ".join("%5s" % classes[predicted[j]] for j in range(predicted.shape[0])),
)


## === cell 11
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)


## === cell 12
preds = []

net.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        file_name = os.path.basename(test_set.imgs[i][0])
        images, labels = data
        images = images.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        cactus = predicted[0].item()
        preds.append([file_name, cactus])


## === cell 13
!head ../input/sample_submission.csv


## === cell 14
output = pd.DataFrame(preds, columns=["id", "has_cactus"])
output.to_csv("submission.csv", index=False)


## === cell 15
rows, cols = 4, 5
i = 0
fig, ax = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(10, 5))
fig.subplots_adjust(hspace=1.0, wspace=0.75)
for name, label in output.iloc[:20].values:    
    img = plt.imread(f"../input/test/test/{name}")
    row = int(i/cols)
    col = i%cols
    ax[row][col].imshow(img)
    ax[row][col].title.set_text(f"Cactus? {label == 1}")
    i += 1
