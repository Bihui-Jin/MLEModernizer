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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

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
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
timm==1.0.19
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.05676

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import re
import sys
import time
import copy
import random
import glob

import zipfile
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import transformers

from tqdm import tqdm

from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 1
class CFG:
    debug_one_epoch = True
    debug_one_fold = False
    only_infer = False
    num_workers = 16
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    early_stopping_round = 5
    random_seed = 42
    n_splits = 5
    model_name = "resnet18"  # timm model name
    pretrained_path = None
    train_dir = None
    test_dir = None
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_linear_schedule_with_warmup
    input_imgsize = 224
    data_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.random_seed)

if CFG.debug_one_epoch:
    CFG.num_epochs = 1

print("KAGGLE_URL_BASE" in set(os.environ.keys()))



## === cell 2
submission = pd.read_csv(os.path.join(CFG.data_dir, "sample_submission.csv"))
submission.head(3)



## === cell 3
if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    kaggle_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
    if not os.path.exists(kaggle_train_dir):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "train.zip"), CFG.kaggle_working_dir
        )

    kaggle_test_dir = os.path.join(CFG.kaggle_working_dir, "test")
    if not os.path.exists(kaggle_test_dir):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "test.zip"), CFG.kaggle_working_dir
        )

    CFG.data_dir = CFG.kaggle_working_dir

CFG.train_dir = os.path.join(CFG.data_dir, "train")
CFG.test_dir = os.path.join(CFG.data_dir, "test")



## === cell 4
train_list = glob.glob(os.path.join(CFG.data_dir, "train", "*.jpg"))
test_list = glob.glob(os.path.join(CFG.data_dir, "test", "*.jpg"))

if len(train_list) == 0:
    train_list = glob.glob(os.path.join(CFG.train_dir, "**", "*.jpg"), recursive=True)
if len(test_list) == 0:
    test_list = glob.glob(os.path.join(CFG.test_dir, "**", "*.jpg"), recursive=True)

print(f"train data : {len(train_list)}")
print(f"test data : {len(test_list)}")



## === cell 5
print(
    "the number of dog : ", len([i for i in train_list if "dog" in os.path.basename(i)])
)
print(
    "the number of cat : ", len([i for i in train_list if "cat" in os.path.basename(i)])
)



## === cell 6
if len(train_list) == 0:
    raise FileNotFoundError(
        f"No training images found. Searched recursively under '{CFG.train_dir}'."
    )

random_img = random.choice(train_list)
img = Image.open(random_img).convert("RGB")
print(random_img)
print(img.size)
plt.imshow(img)



## === cell 7
img_array = np.array(img)
print(img_array.shape)



## === cell 8
print(img_array[:, :, 0])



## === cell 9
transform_tmp = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    ]
)
img_transformed_tmp = transform_tmp(image=np.array(img))
print(img_transformed_tmp.keys())
img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
print(img_transformed_tmp.size)
plt.imshow(img_transformed_tmp)



## === cell 10
transform_tmp = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.HorizontalFlip(p=1.0),
    ]
)
img_transformed_tmp = transform_tmp(image=np.array(img))
img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
plt.imshow(img_transformed_tmp)



## === cell 11
transform_tmp = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.Normalize(),
    ]
)
img_transformed_tmp = transform_tmp(image=np.array(img))["image"]
plt.imshow(img_transformed_tmp)



## === cell 12
transform_tmp = A.Compose(
    [A.Resize(CFG.input_imgsize, CFG.input_imgsize), ToTensorV2()]
)
img_transformed_tmp = transform_tmp(image=np.array(img))
print(img_transformed_tmp.keys())
print(type(img_transformed_tmp["image"]))
print(img_transformed_tmp["image"].shape)



## === cell 13
train_df = pd.DataFrame(train_list, columns=["path"])
train_df["class"] = train_df["path"].apply(lambda x: os.path.basename(x).split(".")[0])
train_df["class"] = train_df["class"].map({"dog": 1, "cat": 0})
if train_df["class"].isna().any():
    train_df["class"] = (
        train_df["path"]
        .apply(lambda x: os.path.basename(x).split(".")[0])
        .map({"dog": 1, "cat": 0})
    )

test_df = pd.DataFrame(test_list, columns=["path"])
test_df["class"] = -1
test_df["id"] = test_df["path"].apply(
    lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
test_df = test_df.sort_values("id").reset_index(drop=True)

train_df.head(3)



## === cell 14
test_df.head(3)



## === cell 15
train_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [A.Resize(CFG.input_imgsize, CFG.input_imgsize), A.Normalize(), ToTensorV2()]
)




## === cell 16
class DogsCatsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx, 0]).convert("RGB")
        img = self.transform(image=np.array(img))["image"]
        label = np.float32(self.df.iloc[idx, 1])
        return img, label




## === cell 17
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    for img, label in tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output.squeeze(-1), label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return np.mean(losses)




## === cell 18
def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    losses = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = criterion(output.squeeze(-1), label)
            losses.append(loss.item())
            all_labels.extend(label.cpu().numpy())
            pred = torch.sigmoid(output).cpu().numpy()
            all_outputs.extend(pred)

    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    return {
        "bce_loss": np.mean(losses),
        "log_loss": log_loss(all_labels, all_outputs),
        "labels": all_labels,
        "outputs": all_outputs,
    }




## === cell 19
def infer(model, dataloader, test=False):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            if test:
                if isinstance(label, torch.Tensor):
                    assert float(label[0].item()) == -1.0
                else:
                    assert float(label[0]) == -1.0
            img = img.to(device)
            output = model(img)
            all_outputs.extend(torch.sigmoid(output).cpu().numpy())

    all_outputs = np.array(all_outputs)
    return all_outputs




## === cell 20
def run_train_cv(train, test):
    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train), 1))
    predictions = []

    test_dataset = DogsCatsDataset(test, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"====================fold : {fold}====================")
        train_df_fold = train.iloc[train_idx].reset_index(drop=True)
        valid_df_fold = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = DogsCatsDataset(train_df_fold, transform=train_transform)
        valid_dataset = DogsCatsDataset(valid_df_fold, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.to(device)

        optimizer = CFG.optimizer(model.parameters(), lr=CFG.lr)
        training_steps = len(train_loader) * CFG.num_epochs
        warmup_steps = int(training_steps * 0.1)
        scheduler = CFG.scheduler(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

        best_loss = np.inf
        early_stopping_round = 0
        best_path = f"{CFG.model_name}_fold{fold}.pth"

        for epoch in range(CFG.num_epochs):
            start_time = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, CFG.criterion
            )
            valid_result = eval_one_epoch(model, valid_loader, CFG.criterion)
            print(
                f"epoch : {epoch} - train loss : {train_loss} - valid loss : {valid_result['bce_loss']} - valid log loss : {valid_result['log_loss']}"
            )

            if valid_result["bce_loss"] < best_loss:
                best_loss = valid_result["bce_loss"]
                early_stopping_round = 0
                torch.save(model.state_dict(), best_path)
            else:
                early_stopping_round += 1
                if early_stopping_round > CFG.early_stopping_round:
                    break

            print(f"spend time for epoch {epoch} : {time.time() - start_time}")

        model.load_state_dict(torch.load(best_path, map_location=device))
        oof[valid_idx] = infer(model, valid_loader)

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        if CFG.debug_one_fold:
            break

    for fold in range(CFG.n_splits):
        ckpt_path = f"{CFG.model_name}_fold{fold}.pth"
        if not os.path.exists(ckpt_path):
            continue
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        predictions.append(infer(model, test_loader, test=True))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if CFG.debug_one_fold:
            break

    if len(predictions) == 0:
        raise RuntimeError(
            "No fold checkpoints found; cannot generate test predictions."
        )

    predictions = np.mean(predictions, axis=0)

    return {"oof": oof, "predictions": predictions}




## === cell 21
def main():
    global test_df

    sub_ids = submission["id"].astype(int).to_numpy()

    if (test_df is None) or (len(test_df) == 0):
        test_list_fix = glob.glob(
            os.path.join(CFG.test_dir, "**", "*.jpg"), recursive=True
        )
        if len(test_list_fix) == 0:
            raise FileNotFoundError(
                f"No test images found. Searched recursively under '{CFG.test_dir}'."
            )
        test_df_fix = pd.DataFrame(test_list_fix, columns=["path"])
        test_df_fix["class"] = -1
        test_df_fix["id"] = test_df_fix["path"].apply(
            lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        test_df = test_df_fix

    test_df = test_df[test_df["id"].isin(sub_ids)].copy()
    test_df = test_df.sort_values("id").reset_index(drop=True)
    if len(test_df) != len(submission):
        raise ValueError(
            f"Test images after filtering do not match sample_submission rows: "
            f"found={len(test_df)} vs submission={len(submission)}. "
            f"Check CFG.test_dir='{CFG.test_dir}'."
        )

    if CFG.only_infer:
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.to(device)
        predictions = infer(model, test_loader, test=True)
        submission["label"] = np.asarray(predictions).reshape(-1)
        submission.to_csv("submission.csv", index=False)
        return

    result = run_train_cv(train_df, test_df)
    oof_preds = result["oof"]
    predictions = np.asarray(result["predictions"]).reshape(-1)

    if len(predictions) != len(submission):
        raise ValueError(
            f"Prediction length mismatch: predictions={len(predictions)} vs submission={len(submission)}. "
            f"Check test image discovery under CFG.test_dir='{CFG.test_dir}'."
        )

    submission["label"] = predictions
    submission.to_csv("submission.csv", index=False)

    train_df_out = train_df.copy()
    train_df_out["oof_preds"] = oof_preds
    train_df_out.to_csv("oof_preds.csv", index=False)

    if CFG.debug_one_fold is False:
        print(f"oof log loss : {log_loss(train_df_out['class'], oof_preds)}")


if __name__ == "__main__":
    main()
