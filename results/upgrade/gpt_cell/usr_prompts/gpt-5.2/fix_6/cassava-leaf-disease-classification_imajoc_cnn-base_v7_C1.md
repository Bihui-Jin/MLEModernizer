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

0.2895

# 6. Current score

0.75822

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16704) has done: 'Cell 6 is failing with a `SyntaxError` because it contains plain English “Diagnosis / Patch summary …” text that is not commented out, so Python tries to parse it as code (and chokes on backticks and other characters). The minimal fix is to replace cell 6 with valid Python only, keeping the original intent: load a sample test image and display it. To avoid the earlier `FileNotFoundError` risk, the code resolve the dataset root by checking the known available locations and then construct the image path with `os.path.join`. This does not affect cell 7 since it does not depend on any outputs from cell 6.'
- What this solution (achieved 0.23655) has done: 'Diagnosis: Cell 6 assumes a Kaggle-style mount point and a specific test image filename (`2216849948.jpg`) that does not exist in this environment’s provided filesystem layout. While `dataset_root` is found, the constructed `img_path` points to a non-existent file, causing `PIL.Image.open` to raise `FileNotFoundError`. The robust fix is to keep the same intent (open and display a test image) but dynamically choose an existing `.jpg` from the detected `test_images` directory (falling back to the first file) and build the path from that.

Patch summary: In cell 6 only, replace the hardcoded filename with logic that lists the `test_images` folder, selects the requested file if present otherwise selects the first available image, and then opens that path. This preserves the visualization behavior while making it deterministic and compatible with the actual dataset location.

Updated cells:'
- What this solution (achieved 0.21338) has done: 'Diagnosis: Cell 13 crashes because it tries to load `./weights_fine_tuning.pth`, but that file is not present in the runtime filesystem, causing a `FileNotFoundError`. The notebook also redundantly loads the same path twice (once default, once with `map_location`), so even if the first load were skipped the second would still fail. The minimal safe fix is to conditionally load weights only if the file exists; otherwise keep the current `net` parameters as-is so execution can proceed to inference in cell 14.

Patch summary: In cell 13, add an `os.path.exists` guard around the weight loading logic and ensure a single deterministic load path is used. If the file is missing, skip loading without raising, leaving `net` unchanged.

Updated cells: Only cell 13 is modified.

Compatibility notes for cell k+1: Cell 14 expects `net` to exist and be moved to `device`; this remains true. Skipping weight loading preserves the same interface and tensor shapes (classifier already set to 5 outputs earlier), so downstream inference code still runs.

Assumptions: It is acceptable to proceed without fine-tuned weights when the checkpoint file is unavailable, since no other checkpoint path is provided in the notebook/environment.'
- What this solution (achieved 0.75822) has done: 'Your current score is below target (0.21338 vs 0.2895), so we should make a small, legitimate improvement without changing the overall approach (VGG16 + cross-entropy + the same training loop). The biggest issue hurting accuracy is that images are fed as raw 0–255 BGR floats with no ImageNet normalization, and with `pretrained=False` the model starts from scratch; switching to ImageNet pretrained weights and applying the standard VGG16 preprocessing is a minimal, metric-aligned fix that typically boosts accuracy. To keep evaluation semantics intact, we only adjust dataset preprocessing (BGR→RGB, scale to [0,1], normalize) and enable pretrained weights; inference use the same preprocessing so train/test match. We also remove verbose debug prints in the test dataset to avoid slowdowns/timeouts and call `train_model(...)` so training actually happens before inference.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm

import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import models, transforms
from sklearn.model_selection import train_test_split



## === cell 1
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)



## === cell 2
use_pretrained = False
net = models.vgg16(pretrained=use_pretrained)
net.eval()
print(net)



## === cell 3
PATH = "../input/cassava-leaf-disease-classification/train_images/"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CassavaDataset(data.Dataset):
    def __init__(self, path, image_ids, labels, image_size):
        self.image_ids = image_ids
        self.labels = labels
        self.path = path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = str(self.image_ids[item])
        label = self.labels[item]
        img_file = cv2.imread(os.path.join(self.path, image_id))
        if img_file is None:
            raise FileNotFoundError(
                f"Could not read image: {os.path.join(self.path, image_id)}"
            )

        img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img_file, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)

        return torch.tensor(img, dtype=torch.float), torch.tensor(
            label, dtype=torch.long
        )


dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
xtrain, xval, ytrain, yval = train_test_split(
    dfx["image_id"].values,
    dfx.label.values,
    test_size=0.1,
    random_state=42,
    stratify=dfx.label.values,
)
print(dfx.head())



## === cell 4
IMG_SIZE = 224

train_dataset = CassavaDataset(PATH, xtrain, ytrain, IMG_SIZE)
val_dataset = CassavaDataset(PATH, xval, yval, IMG_SIZE)

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}



## === cell 5
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=5)
net.train()
print("ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました")



## === cell 6
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

candidate_roots = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]

dataset_root = None
for r in candidate_roots:
    if os.path.isdir(r):
        dataset_root = r
        break

if dataset_root is None:
    raise FileNotFoundError(f"Could not find dataset root in any of: {candidate_roots}")

test_images_dir = os.path.join(dataset_root, "test_images")
if not os.path.isdir(test_images_dir):
    raise FileNotFoundError(f"test_images directory not found at: {test_images_dir}")

preferred_name = "2216849948.jpg"
jpg_files = sorted(
    [f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")]
)
if len(jpg_files) == 0:
    raise FileNotFoundError(f"No .jpg files found in: {test_images_dir}")

image_name = preferred_name if preferred_name in jpg_files else jpg_files[0]
img_path = os.path.join(test_images_dir, image_name)

im = Image.open(img_path)
im_list = np.asarray(im)
plt.imshow(im_list)
plt.show()



## === cell 7
criterion = nn.CrossEntropyLoss()



## === cell 8
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = ["features"]
update_param_names_2 = [
    "classifier.0.weight",
    "classifier.0.bias",
    "classifier.3.weight",
    "classifier.3.bias",
]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
    else:
        param.requires_grad = False



## === cell 9
optimizer = optim.SGD(
    [
        {"params": params_to_update_1, "lr": 1e-4},
        {"params": params_to_update_2, "lr": 5e-4},
        {"params": params_to_update_3, "lr": 1e-3},
    ],
    momentum=0.9,
)




## === cell 10
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    net.to(device)
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-------------")

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == "train"):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))




## === cell 11
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=2)



## === cell 12
import os

load_path = "./weights_fine_tuning.pth"
if os.path.exists(load_path):
    load_weights = torch.load(
        load_path, map_location=("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    net.load_state_dict(load_weights)
else:
    print(
        f"Warning: checkpoint not found at {load_path}. Skipping weight loading and using current model parameters."
    )



## === cell 13
from torch.utils.data import Dataset, DataLoader

TEST_FILE_PATH = "../input/cassava-leaf-disease-classification/test_images/"


class CassavaTestDataset(Dataset):
    def __init__(self, path, image_ids, image_size):
        self.image_ids = list(image_ids)
        self.path = path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = str(self.image_ids[item])
        img_file = cv2.imread(os.path.join(self.path, image_id))
        if img_file is None:
            raise FileNotFoundError(
                f"Could not read image: {os.path.join(self.path, image_id)}"
            )

        img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img_file, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)

        return torch.tensor(img, dtype=torch.float)


sample = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dataset = CassavaTestDataset(TEST_FILE_PATH, sample.image_id, IMG_SIZE)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

net.to(device)
net.eval()

fin_outputs = []

with torch.no_grad():
    for inputs in tqdm(test_loader):
        inputs = inputs.to(device, non_blocking=True)
        outputs = net(inputs)
        preds = torch.argmax(outputs, dim=1)
        fin_outputs.append(preds.cpu().numpy())

sample["label"] = np.concatenate(fin_outputs).reshape(-1).astype(int)
sample[["image_id", "label"]].to_csv("submission.csv", index=False)
print(sample.head())
