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

albumentations==2.0.8
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

0.8936234511937141

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, random, json, time, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.utils.data as data
import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import glob




## === cell 1
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(SEED)



## === cell 2
sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
try:
    from efficientnet_pytorch import EfficientNet  # original repo
except ModuleNotFoundError:
    from torchvision.models import efficientnet_b7 as _effnet_b7

    class EfficientNet:
        @staticmethod
        def from_name(name):
            if name == "efficientnet-b7":
                return _effnet_b7(pretrained=False)
            raise ValueError(f"Unsupported EfficientNet name: {name}")




## === cell 3
SIZE = 512  # image size
num_classes = 5



## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 5
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    BASE_DIR = "../input/cassava-leaf-disease-classification"
elif os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
    BASE_DIR = "../input/cassava-leaf-disease-classification"
else:
    BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"

TEST_PATH = f"{BASE_DIR}/test_images"
test_files = os.listdir(TEST_PATH)
print(f"Number of test images: {len(test_files)}")



## === cell 6
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1  # placeholder

if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]



## === cell 7
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop((SIZE, SIZE), scale=(0.8, 1.0)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 8
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 9
class TrainDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.image_ids = df.image_id.tolist()
        self.labels = df.label.tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{self.img_dir}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label




## === cell 10
def predict_model(basename, net, dataloader):
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []
    for phase in ["test"]:
        prog = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, _ in prog:
            inputs = inputs.to(device)
            with torch.cuda.amp.autocast():
                outputs = net(inputs, False, "test")
                probability.append(torch.softmax(outputs, dim=1).cpu().numpy())
    return np.concatenate(probability)




## === cell 11
def train_simple_model(
    train_csv_path, train_images_dir, epochs=2, batch_size=64, lr=1e-3
):
    """
    Trains a lightweight ResNet‑18 (ImageNet pretrained) on the provided
    training split for a few epochs. Returns the trained model ready for
    inference with the same interface used in `predict_model`.
    """
    df = pd.read_csv(train_csv_path)
    tr_transform = Compose(
        [
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    dataset = TrainDataset(df, train_images_dir, transform=tr_transform)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    base = models.resnet18(pretrained=True)
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    net = FinalLayerMixupModel(base, nn.CrossEntropyLoss(), num_classes, False)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        prog = tqdm(loader, desc=f"Train epoch {epoch+1}/{epochs}")
        for imgs, lbls in prog:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            outputs, loss = net(imgs, lbls, "val")  # reuse validation branch for loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(loader):.4f}")

    net.eval()
    return net




## === cell 12
probability = []
start_time = time.time()

pretrained_models = (
    glob.glob(f"../input/efntb7/efficientnet-b7m_SEED70/*.pth")
    + glob.glob(f"../input/efntb7/efficientnet-b7m_SEED72/*.pth")
    + glob.glob(f"../input/efntb7/efficientnet-b7m_SEED73/*.pth")
    + glob.glob(f"../input/efntb7/efficientnet-b7sl_SEED71.pretrained/*.pth")
)

if len(pretrained_models) == 0:
    print("No pretrained checkpoints found – training a small model as fallback.")
    TRAIN_CSV = f"{BASE_DIR}/train.csv"
    TRAIN_IMG_DIR = f"{BASE_DIR}/train_images"
    fallback_net = train_simple_model(TRAIN_CSV, TRAIN_IMG_DIR, epochs=2, batch_size=64)
    for tid, tr in enumerate(transform["test"]):
        print(f"fallback transform loop={tid}")
        test_dataset = TestDataset(df_test, transform=tr)
        test_loader = {
            "test": data.DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        }
        prob = predict_model("fallback_model", fallback_net, test_loader)
        probability.append(prob)
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()
        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            net = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            sys.exit()

        print(f"{basename}: {MODEL_NAME}")

        if MODEL_NAME == "efficientnet-b7":
            if "b7m" in pretrained_model:
                net.model.load_state_dict(
                    torch.load(pretrained_model, map_location=device)
                )
            else:
                net.load_state_dict(torch.load(pretrained_model, map_location=device))
        else:
            net.load_state_dict(torch.load(pretrained_model, map_location=device))

        for p in net.parameters():
            p.requires_grad = False

        for tid, tr in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            test_dataset = TestDataset(df_test, transform=tr)
            test_loader = {
                "test": data.DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )
            }
            prob = predict_model(basename, net, test_loader)
            probability.append(prob)

        del net
        torch.cuda.empty_cache()

if len(probability) == 0:
    mean_probs = np.full((len(df_test), num_classes), 1.0 / num_classes)
else:
    prob_array = np.stack(
        probability, axis=0
    )  # (N_models*N_transforms, samples, classes)
    mean_probs = prob_array.mean(axis=0)  # average across models/transforms

df_test["label"] = mean_probs.argmax(axis=1).astype(int)
print(f"total time: {time.time() - start_time:.2f}[sec]")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1122511720.py in <cell line: 0>()
     14     TRAIN_CSV = f"{BASE_DIR}/train.csv"
     15     TRAIN_IMG_DIR = f"{BASE_DIR}/train_images"
---> 16     fallback_net = train_simple_model(TRAIN_CSV, TRAIN_IMG_DIR, epochs=2, batch_size=64)
     17     # Use the same test transforms as before
     18     for tid, tr in enumerate(transform["test"]):

/tmp/ipykernel_55/160528121.py in train_simple_model(train_csv_path, train_images_dir, epochs, batch_size, lr)
     25     base = models.resnet18(pretrained=True)
     26     base.fc = nn.Linear(base.fc.in_features, num_classes)
---> 27     net = FinalLayerMixupModel(base, nn.CrossEntropyLoss(), num_classes, False)
     28     net.to(device)
     29 

NameError: name 'FinalLayerMixupModel' is not defined

## === cell 13
df_test



## === cell 14
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
