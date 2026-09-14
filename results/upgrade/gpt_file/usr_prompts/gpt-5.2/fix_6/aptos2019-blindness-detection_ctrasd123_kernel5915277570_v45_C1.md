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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8637257072282227

# 6. Current score

0.61406

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `tensorboardX` (it isn’t installed) by falling back to PyTorch’s built-in `SummaryWriter` or a no-op stub so imports don’t fail. Then I fix the missing checkpoint error by switching to an inference-only pipeline that uses the same dataset/transforms and model forward semantics, but loads standard torchvision ImageNet weights for DenseNet-201 (available offline) instead of a non-existent `/kaggle/input/temp-file/...pkl`. I also eliminate a few runtime/logic pitfalls (wrong `from __future__` placement, `torch.tensor(x)` copies in `forward`, and the broken `content` building loop) while keeping the overall prediction approach identical (sigmoid → threshold → `get_preds`). Finally, I ensure a valid `/kaggle/working/submission.csv` is written with the exact required columns and row order matching `test.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the runtime import failure caused by a broken TensorBoard/TensorFlow dependency chain by replacing the `torch.utils.tensorboard` import with a safe no-op `SummaryWriter` stub (it isn’t used elsewhere). Then I ensure the script runs in the Kaggle Python 3.8 environment by removing the directory-walk cell that can be slow/noisy and by renumbering cells starting at 1 as required. Finally, to move the score off 0.0 toward the target while preserving the core “DenseNet201 + sigmoid + threshold + get_preds” semantics, I load offline-available ImageNet pretrained weights for DenseNet201 (features) deterministically and keep submission row order aligned to `test.csv`, writing `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.03991) has done: 'Your 0.0 score is mainly because the inference pipeline is not aligned with what this competition expects: the model head is randomly initialized (only DenseNet features are loaded) and `get_preds(sigmoid>0.5)` is effectively producing near-constant labels. To move the score upward toward the target with minimal semantic change, I keep your exact “DenseNet201 → sigmoid → threshold → get_preds” prediction logic, but (1) correctly use the model’s own `self.sigmoid` (so the forward semantics match the original code intent) and (2) calibrate the per-class threshold on a small validation split from `train.csv` by directly maximizing quadratic weighted kappa, then reuse that single scalar threshold for test inference. This does not change architecture, loss, or training loops (there are none), but it fixes the key metric-mismatch issue by choosing a better threshold than 0.5. I also ensure the submission rows are written in exactly the `test.csv` order and written efficiently in one pass.'
- What this solution (achieved -0.00793) has done: 'Your current score is far below the target, and the main reason is that the DenseNet base is ImageNet-pretrained but the classifier head is still random, so your logits/probabilities are essentially uncalibrated noise and the “sigmoid→threshold→get_preds” mapping collapses toward near-constant labels. To move the score upward toward the target while preserving your exact model architecture and inference semantics, I add a minimal training step that fits only the existing classification head (and keeps the DenseNet base frozen) using the provided `train.csv` images. Then I keep your existing validation-based single-threshold calibration for QWK, but it now calibrate meaningful probabilities. Finally, I apply ImageNet normalization (matching the pretrained backbone) to reduce distribution shift with a negligible, semantics-preserving transform change.'
- What this solution (achieved 0.61406) has done: 'Your score is far below the target, so we should increase performance with the smallest change that preserves your core model/inference semantics (“DenseNet201 → sigmoid → threshold → get_preds”). The biggest current issue is label encoding: you train with one-hot for 5 classes, but `get_preds` expects an ordinal multi-label (cumulative) encoding; this mismatch makes the learned head produce unusable probability patterns and hurts QWK. I change only the training target construction to the standard ordinal cumulative form (e.g., label 3 → [1,1,1,1,0]) while keeping the same model, loss (BCEWithLogitsLoss), optimizer, epochs, and the exact threshold calibration + inference mapping. This should move the score substantially upward toward the target without altering architecture or adding heavier training.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score


class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def close(self):
        pass




## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_df = pd.read_csv(name_file)
content = (test_df["id_code"].astype(str) + ".png").tolist()




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        densenet201 = torchvision.models.densenet201(weights=None)
        self.base = nn.Sequential(*list(densenet201.children())[:-1])  # features only
        self.feature_dim = 1920

        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            self.cal_score = nn.Linear(in_features=num_classes, out_features=1)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1):
        x = self.base(x1)

        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
        else:
            ys = x
        return ys




## === cell 3
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def get_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)


class eye_dataset_orl(Dataset):
    def __init__(self, txt_path, transform=None, root=None):
        self.imgs = list(txt_path)
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = cv2.imread(os.path.join(self.root, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_dataset_orl_with_label(Dataset):
    """Identical image loading/processing, but returns labels for training."""

    def __init__(self, df, transform=None, root=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        id_code = str(self.df.loc[index, "id_code"])
        y = int(self.df.loc[index, "diagnosis"])
        fn = id_code + ".png"

        img = cv2.imread(os.path.join(self.root, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.df)




## === cell 4
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform2 = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(imagenet_mean, imagenet_std)]
    )

    base_input = "/kaggle/input/aptos2019-blindness-detection"
    test_csv_path = os.path.join(base_input, "test.csv")
    train_csv_path = os.path.join(base_input, "train.csv")
    test_img_root = os.path.join(base_input, "test_images")
    train_img_root = os.path.join(base_input, "train_images")

    test_df = pd.read_csv(test_csv_path)
    test_files = (test_df["id_code"].astype(str) + ".png").tolist()

    net = Baseline_single(num_classes=5)

    try:
        weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
        pretrained = torchvision.models.densenet201(weights=weights)
        net.base.load_state_dict(pretrained.features.state_dict(), strict=True)
    except Exception as e:
        print(
            "Warning: could not load ImageNet weights for DenseNet201, running with random init. Error:",
            repr(e),
        )

    net.freeze_base()

    if use_gpu:
        net = net.cuda()

    train_df = pd.read_csv(train_csv_path)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    tr_idx, va_idx = next(splitter.split(train_df["id_code"], train_df["diagnosis"]))
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[va_idx].reset_index(drop=True)

    y_val = val_df["diagnosis"].values.astype(int)

    train_data = eye_dataset_orl_with_label(tr_df, transform2, root=train_img_root)
    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True, num_workers=2, pin_memory=use_gpu
    )

    val_files = (val_df["id_code"].astype(str) + ".png").tolist()
    val_data = eye_dataset_orl(val_files, transform2, root=train_img_root)
    val_loader = DataLoader(
        val_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(net.classifiers.parameters(), lr=1e-3)

    net.train()
    epochs = 1  # keep minimal compute; main fix is correct ordinal target encoding
    for ep in range(epochs):
        running = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"Train head ep {ep+1}/{epochs}"):
            if use_gpu:
                xb = xb.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True)

            y_ord = torch.zeros((yb.shape[0], 5), device=yb.device, dtype=torch.float32)
            if yb.numel() > 0:
                for i in range(yb.shape[0]):
                    k = int(yb[i].item())
                    if k > 0:
                        y_ord[i, :k] = 1.0

            optim.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = criterion(logits, y_ord)
            loss.backward()
            optim.step()

            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"Epoch {ep+1}: train loss {running/max(n,1):.5f}")

    net.eval()
    val_logits = []
    with torch.no_grad():
        for data, _name in val_loader:
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)
            val_logits.append(out.detach().cpu().numpy())
    val_logits = np.concatenate(val_logits, axis=0)

    val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_kappa = -1.0
    for thr in thresholds:
        preds = get_preds((val_probs > thr).astype(np.uint8))
        k = cohen_kappa_score(y_val, preds, weights="quadratic")
        if k > best_kappa:
            best_kappa = k
            best_thr = float(thr)

    print(f"Calibrated threshold on val: {best_thr:.3f} (val QWK={best_kappa:.5f})")

    test_data = eye_dataset_orl(test_files, transform2, root=test_img_root)
    dataloader_test = DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    preds_all = []
    ids_all = []
    with torch.no_grad():
        net.eval()
        for data, name in tqdm(
            dataloader_test, total=len(dataloader_test), desc="Infer test"
        ):
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)

            probs = net.sigmoid(out).detach().cpu().numpy()
            predicted = get_preds((probs > best_thr).astype(np.uint8))

            ids_all.extend(list(name))
            preds_all.extend(list(predicted.astype(int)))

    submission_path = "/kaggle/working/submission.csv"
    sub_df = pd.DataFrame({"id_code": ids_all, "diagnosis": preds_all})

    sub_df = test_df[["id_code"]].merge(sub_df, on="id_code", how="left")
    sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int)

    sub_df.to_csv(submission_path, index=False)
    print("Wrote:", submission_path, "shape:", sub_df.shape)
    print(sub_df.diagnosis.value_counts(dropna=False))
