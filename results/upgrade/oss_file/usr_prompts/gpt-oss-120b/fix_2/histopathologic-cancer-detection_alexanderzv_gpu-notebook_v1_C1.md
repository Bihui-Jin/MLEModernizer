# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from typing import List, Tuple, Optional
import logging
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt



## === cell 1
DATA_FOLDER = "../input/histopathologic-cancer-detection"
LABELS = f"{DATA_FOLDER}/train_labels.csv"
TRAIN_IMAGES_FOLDER = f"{DATA_FOLDER}/train"
SAMPLE_SUBMISSION = f"{DATA_FOLDER}/sample_submission.csv"
USE_GPU = torch.cuda.is_available()



## === cell 2
logging.basicConfig(level="INFO")
logger = logging.getLogger()



## === cell 3
labels = pd.read_csv(LABELS)




## === cell 4
def format_labels_for_data_set(labels):
    return labels["label"].values.reshape(-1, 1)


def train_valid_split(df, split_percent, limit_df=10000):
    df = df.sample(frac=1, random_state=42)  # shuffle
    df = df.iloc[:limit_df]
    split = round(limit_df * split_percent / 100)
    train = df.iloc[:split]
    valid = df.iloc[split:]
    return train, valid


def format_path_to_images_for_dataset(labels, path):
    return [os.path.join(path, f"{f}.tif") for f in labels["id"].values]




## === cell 5
class MainDataset(Dataset):
    def __init__(self, x_dataset, y_dataset, x_tfms):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.x_tfms = x_tfms

    def __len__(self):
        return len(self.x_dataset)

    def __getitem__(self, index):
        x = self.x_dataset[index]
        y = self.y_dataset[index]
        if self.x_tfms is not None:
            x = self.x_tfms(x)
        return x, torch.tensor(y, dtype=torch.float32)


class ImageDataset(Dataset):
    def __init__(self, path_to_image):
        self.path_to_image = path_to_image

    def __len__(self):
        return len(self.path_to_image)

    def __getitem__(self, index):
        img = Image.open(self.path_to_image[index]).convert("RGB")
        augmentation_pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    p=0.5, brightness_limit=0.2, contrast_limit=0.2
                ),
                A.ShiftScaleRotate(p=0.5),
            ]
        )
        img_np = np.array(img)
        img_aug = augmentation_pipeline(image=img_np)["image"]
        return Image.fromarray(img_aug)


class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]




## === cell 6
train, valid = train_valid_split(labels, 70)

train_labels = format_labels_for_data_set(train)
valid_labels = format_labels_for_data_set(valid)

train_images = format_path_to_images_for_dataset(train, TRAIN_IMAGES_FOLDER)
valid_images = format_path_to_images_for_dataset(valid, TRAIN_IMAGES_FOLDER)

train_images_dataset = ImageDataset(train_images)
valid_images_dataset = ImageDataset(valid_images)
train_labels_dataset = LabelDataset(train_labels)
valid_labels_dataset = LabelDataset(valid_labels)



## === cell 7
x_tfms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = MainDataset(train_images_dataset, train_labels_dataset, x_tfms)
valid_dataset = MainDataset(valid_images_dataset, valid_labels_dataset, x_tfms)



## === cell 8
shuffle = True
batch_size = 512
num_workers = 0

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)




## === cell 9
def to_gpu(tensor):
    return tensor.cuda() if USE_GPU else tensor


def create_resnet9_model(output_dim: int = 1) -> nn.Module:
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    in_features = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(in_features, output_dim)
    return to_gpu(model)


resnet9 = create_resnet9_model()



## === cell 10
lr = 1e-3
optimizer = Adam(resnet9.parameters(), lr=lr)



## === cell 11
criterion = nn.BCEWithLogitsLoss()




## === cell 12
def auc_writer(y_true, y_predicted, iteration):
    try:
        score = roc_auc_score(np.vstack(y_true), np.vstack(y_predicted))
    except Exception:
        score = -1
    print(f"iteration: {iteration}, roc_auc: {score}")
    logger.info(f"iteration: {iteration}, roc_auc: {score}")


loss_writer_train = auc_writer
loss_writer_valid = auc_writer




## === cell 13
def T(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.FloatTensor(tensor)
    else:
        tensor = tensor.type(torch.FloatTensor)
    if USE_GPU:
        tensor = to_gpu(tensor)
    return tensor


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, Image.Image):
        return np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError("Unsupported type for to_numpy conversion")




## === cell 14
def iteration_trigger(iteration, every_x_iteration):
    return every_x_iteration == 1 or (
        iteration > 0 and iteration % every_x_iteration == 0
    )


def init_triggers(step=1, train=10, valid=10):
    do_step_trigger = lambda it: iteration_trigger(it, step)
    train_loss_trigger = lambda it: iteration_trigger(it, train)
    valid_loss_trigger = lambda it: iteration_trigger(it, valid)
    return do_step_trigger, train_loss_trigger, valid_loss_trigger


do_step_trigger, train_loss_trigger, valid_loss_trigger = init_triggers(1, 10, 20)




## === cell 15
def train_one_epoch(
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    loss_writer_train,
    loss_writer_valid,
    do_step_trigger,
    train_loss_trigger,
    valid_loss_trigger,
):
    model.train()
    y_true_train, y_hat_train = [], []
    for iteration, (x, y) in enumerate(train_loader):
        x_train = Variable(T(x), requires_grad=True)
        y_train = Variable(T(y), requires_grad=True)

        output = model(x_train)
        y_true_train.append(to_numpy(y_train))
        y_hat_train.append(to_numpy(output))

        loss_val = loss_fn(output, y_train)
        loss_val.backward()

        if do_step_trigger(iteration):
            optimizer.step()
            optimizer.zero_grad()

        if train_loss_trigger(iteration):
            loss_writer_train(y_true_train, y_hat_train, iteration)
            y_true_train, y_hat_train = [], []

        if valid_loss_trigger(iteration):
            y_true_valid, y_hat_valid = predict(model, valid_loader)
            loss_writer_valid(y_true_valid, y_hat_valid, iteration)

    return model




## === cell 16
def predict(model, dataloader):
    model.eval()
    y_true, y_hat = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = Variable(T(x))
            y = Variable(T(y))
            output = model(x)
            y_true.append(to_numpy(y))
            y_hat.append(to_numpy(output))
    return y_true, y_hat




## === cell 17
resnet9 = train_one_epoch(
    resnet9,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    loss_writer_train,
    loss_writer_valid,
    do_step_trigger,
    train_loss_trigger,
    valid_loss_trigger,
)



## === cell 18
TEST_IMAGES_FOLDER = f"{DATA_FOLDER}/test/"


def test_image_collection(directory: str) -> List[str]:
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tif")
    ]


test_image_paths = test_image_collection(TEST_IMAGES_FOLDER)
test_images_dataset = ImageDataset(test_image_paths)


class TestDataset(Dataset):
    def __init__(self, x_dataset: Dataset, x_tfms: Optional = None):
        self.x_dataset = x_dataset
        self.x_tfms = x_tfms

    def __len__(self):
        return len(self.x_dataset)

    def __getitem__(self, index):
        x = self.x_dataset[index]
        if self.x_tfms is not None:
            x = self.x_tfms(x)
        return x


test_dataset = TestDataset(test_images_dataset, x_tfms)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)




## === cell 19
def predict_test(model, dataloader):
    model.eval()
    y_hat = []
    with torch.no_grad():
        for x in dataloader:
            x = Variable(T(x))
            output = model(x)
            prob = torch.sigmoid(output)  # convert logits to probabilities
            y_hat.append(to_numpy(prob))
    return y_hat




## === cell 20
y_hat_test = predict_test(resnet9, test_dataloader)



## === cell 21
predictions = pd.DataFrame(
    {
        "id": [os.path.basename(p).split(".")[0] for p in test_image_paths],
        "label": np.vstack(y_hat_test).reshape(-1),
    }
)
predictions.to_csv("submission.csv", index=False)



## === cell 22
print("Submission file 'submission.csv' created with shape:", predictions.shape)
