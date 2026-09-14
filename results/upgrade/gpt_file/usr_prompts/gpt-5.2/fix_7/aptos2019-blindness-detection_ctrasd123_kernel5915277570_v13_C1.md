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

0.5402456897607164

# 6. Current score

0.2424

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00728) has done: 'I remove the hard dependency on `tensorboardX` (it isn’t installed) by falling back to PyTorch’s built-in TensorBoard writer (or a no-op) so imports don’t crash. Then I fix the model definition to correctly use DenseNet121 features (your current `children()[:-1]` + `feature_dim` mismatch break loading/inference), and I make weight loading robust: if the referenced `.pkl` file isn’t present in `/kaggle/input`, the script still run end-to-end and write a valid `submission.csv` (with deterministic, reasonable default predictions). Finally, I remove excessive per-line printing and ensure paths use `/kaggle/input/aptos2019-blindness-detection/...` consistently so the pipeline finishes within the time limit and produces a correctly formatted submission.'
- What this solution (achieved -0.00351) has done: 'I fix the import crash coming from TensorBoard by preventing any TensorBoard import/use entirely (it’s not needed for inference) and keeping a safe no-op `SummaryWriter`. Then I make the test-time preprocessing match DenseNet’s expected input distribution by resizing to 224 and applying ImageNet normalization; this is a minimal change that typically moves kappa from near-random toward a reasonable baseline without altering the model architecture or inference loop. I also make the test image path resolution robust to the two common dataset directory layouts so missing images don’t silently zero out the input. Finally, I keep the submission writing logic but ensure deterministic execution and correct row alignment/length checks.'
- What this solution (achieved -0.00853) has done: 'Your current score is far below the target (higher-is-better), so we should make a minimal change that legitimately boosts kappa without changing the model or training loop. The biggest issue is that you’re using a randomly initialized DenseNet because the external weight file isn’t present; that produces near-random predictions and negative kappa. I keep your exact architecture and inference logic, but load standard ImageNet-pretrained DenseNet121 weights (available inside torchvision) into the DenseNet feature extractor when the competition-specific `.pkl` isn’t found. This should move predictions from random toward a reasonable baseline and thus increase kappa substantially while preserving your pipeline and submission format.'
- What this solution (achieved 0.27916) has done: 'Your current kappa is far below the target, so we should make a minimal, metric-aligned change that improves predictions without altering the model/training logic. The biggest issue is that the model outputs 5 logits but you’re taking `argmax` directly, which is not consistent with your declared “single BCE” multi-label style head; a small, standard fix is to apply `sigmoid` at inference and convert the 5 binary outputs into a single ordinal class using a count-based rule. This preserves the architecture and loss semantics, but makes post-processing consistent and usually yields a large kappa jump from near-random toward a reasonable baseline. I also make the output layer initialization consistent with the original intent by using the already-defined `self.sigmoid` in forward (still same layers), and I keep submission alignment checks unchanged.'
- What this solution (achieved 0.23679) has done: 'Your current score (0.27916) is far below the target (0.5402), so we should make a small, metric-aligned inference change that tends to increase QWK without changing your model/training core. The main weakness is the fixed `0.5` threshold for converting sigmoid outputs into an ordinal class; for this competition, calibrating thresholds on the training set (using out-of-fold-style validation) usually improves kappa substantially while preserving the same architecture and “single BCE” semantics. I keep your network exactly the same, but add a minimal train/val split to learn 4 thresholds (0–4) that maximize QWK on the validation set, then apply them to test predictions. If the competition checkpoint is missing and we fall back to ImageNet features, threshold calibration still helps; if calibration fails for any reason, it falls back to your existing `>0.5` counting rule so you still get a valid submission.'
- What this solution (achieved 0.2424) has done: 'Your current score is far below the target (higher-is-better), so the smallest legitimate move toward the target is to improve inference calibration without changing the model or training loop. I keep your architecture, weights logic, and ordinal-from-sigmoid setup, but tune the 4 thresholds in a slightly more reliable way: use stratified K-fold out-of-fold predictions (instead of a single split) to fit thresholds, which reduces variance and usually increases QWK. To avoid altering “core logic”, the network is never trained; we only evaluate it on multiple validation folds and optimize thresholds on those out-of-fold scores, then apply the tuned thresholds to the test set. I also make the tuned thresholds the default only when tuning succeeds; otherwise your original 0.5-count rule remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
import time
import datetime
import argparse
import random
from PIL import Image
import cv2
import csv

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import cohen_kappa_score


class SummaryWriter:  # minimal no-op fallback
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def close(self):
        pass


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(0)



## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_df = pd.read_csv(name_file)
content = (test_df["id_code"].astype(str) + ".png").tolist()
print(f"Loaded test ids: {len(content)} images")




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        densenet = torchvision.models.densenet121(weights=None)
        self.base = densenet.features  # (B, 1024, H, W)
        self.feature_dim = 1024

        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1):
        x = self.base(x1)
        x = nn.functional.relu(x, inplace=True)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
        return ys




## === cell 3
def crop_image_from_gray(img, tol=7):
    if img is None:
        return img
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    return image


class eye_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform

        cand_dirs = [
            "/kaggle/input/aptos2019-blindness-detection/test_images",
            "/kaggle/input/test_images",
            "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/test_images",
        ]
        self.test_img_dir = next((d for d in cand_dirs if osp.isdir(d)), cand_dirs[0])

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = osp.join(self.test_img_dir, fn)
        img = cv2.imread(img_path)

        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = load_ben_yuan(img)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_train_dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        fn = str(row["id_code"]) + ".png"
        label = int(row["diagnosis"])
        img_path = osp.join(self.img_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = load_ben_yuan(img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)


def _predict_sigmoid_prob(net, loader, use_gpu):
    net.eval()
    all_prob = []
    all_y = []
    with torch.no_grad():
        for data, y in loader:
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)
            prob = torch.sigmoid(out).detach().cpu().numpy()
            all_prob.append(prob)
            all_y.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(all_prob, axis=0), np.concatenate(all_y, axis=0)


def _ordinal_from_thresholds(score, thr):
    thr = np.asarray(thr, dtype=np.float32)
    return (
        (score > thr[0]).astype(np.int64)
        + (score > thr[1]).astype(np.int64)
        + (score > thr[2]).astype(np.int64)
        + (score > thr[3]).astype(np.int64)
    )


def _tune_thresholds_by_greedy_search(y_true, score, init_thr=None):
    if init_thr is None:
        init_thr = [0.5, 1.5, 2.5, 3.5]
    thr = np.array(init_thr, dtype=np.float32)

    def kappa_for(thr_vec):
        pred = _ordinal_from_thresholds(score, thr_vec)
        return cohen_kappa_score(y_true, pred, weights="quadratic")

    best = kappa_for(thr)

    for _ in range(5):
        improved = False
        for i in range(4):
            lo = 0.0 if i == 0 else float(thr[i - 1] + 1e-3)
            hi = 4.0 if i == 3 else float(thr[i + 1] - 1e-3)
            if hi <= lo:
                continue

            candidates = np.linspace(lo, hi, 31, dtype=np.float32)
            local_best_thr = thr[i]
            local_best = best
            for v in candidates:
                thr_try = thr.copy()
                thr_try[i] = v
                if not (thr_try[0] < thr_try[1] < thr_try[2] < thr_try[3]):
                    continue
                k = kappa_for(thr_try)
                if k > local_best:
                    local_best = k
                    local_best_thr = v
            if local_best > best:
                thr[i] = local_best_thr
                best = local_best
                improved = True
        if not improved:
            break

    return thr.tolist(), float(best)


def _tune_thresholds_oof(net, train_df, train_img_dir, transform2, use_gpu, n_splits=5):
    y = train_df["diagnosis"].astype(int).values
    oof_score = np.empty(len(train_df), dtype=np.float32)
    oof_score[:] = np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(train_df)), y), start=1):
        val_df = train_df.iloc[va_idx].reset_index(drop=True)
        val_ds = eye_train_dataset(val_df, img_dir=train_img_dir, transform=transform2)
        val_loader = DataLoader(
            val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_gpu
        )
        val_prob, _ = _predict_sigmoid_prob(net, val_loader, use_gpu=use_gpu)
        val_score = val_prob.sum(axis=1).astype(np.float32)
        oof_score[va_idx] = val_score

    ok = np.isfinite(oof_score)
    if ok.sum() != len(train_df):
        raise RuntimeError(
            "OOF scoring failed: some folds did not produce predictions."
        )

    thr, best_k = _tune_thresholds_by_greedy_search(
        y_true=y[ok], score=oof_score[ok], init_thr=[0.5, 1.5, 2.5, 3.5]
    )
    return thr, best_k




## === cell 4
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
    test_df = pd.read_csv(name_file)
    content = (test_df["id_code"].astype(str) + ".png").tolist()

    test_data = eye_dataset(content, transform2)
    net = Baseline_single(num_classes=5)

    if use_gpu:
        net = net.cuda()

    candidate_weight_paths = [
        "/kaggle/input/temp-file/model_yuan492_dense_00001_adam_pre_z_ji.pkl",
        "/kaggle/input/aptos2019-blindness-detection/model_yuan492_dense_00001_adam_pre_z_ji.pkl",
        "/kaggle/input/model_yuan492_dense_00001_adam_pre_z_ji.pkl",
    ]
    weight_path = next((p for p in candidate_weight_paths if os.path.exists(p)), None)

    if weight_path is not None:
        state = torch.load(weight_path, map_location="cuda" if use_gpu else "cpu")
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"Loaded weights: {weight_path}")
        if missing:
            print(f"Missing keys (ignored): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys (ignored): {len(unexpected)}")
    else:
        try:
            imagenet_weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            d_im = torchvision.models.densenet121(weights=imagenet_weights)
            net.base.load_state_dict(d_im.features.state_dict(), strict=True)
            print(
                "WARNING: Competition checkpoint not found; loaded ImageNet-pretrained "
                "DenseNet121 features as a minimal performance-improving fallback."
            )
        except Exception as e:
            print(
                "WARNING: Competition checkpoint not found and ImageNet fallback failed "
                f"({type(e).__name__}: {e}). Proceeding with untrained model."
            )

    tuned_thr = None
    try:
        train_csv = "/kaggle/input/aptos2019-blindness-detection/train.csv"
        train_df = pd.read_csv(train_csv)
        train_img_cand = [
            "/kaggle/input/aptos2019-blindness-detection/train_images",
            "/kaggle/input/train_images",
            "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/train_images",
        ]
        train_img_dir = next(
            (d for d in train_img_cand if osp.isdir(d)), train_img_cand[0]
        )

        tuned_thr, best_k = _tune_thresholds_oof(
            net=net,
            train_df=train_df,
            train_img_dir=train_img_dir,
            transform2=transform2,
            use_gpu=use_gpu,
            n_splits=5,
        )
        print(
            f"OOF threshold tuning complete. OOF QWK={best_k:.5f}, thresholds={tuned_thr}"
        )
    except Exception as e:
        tuned_thr = None
        print(
            f"WARNING: Threshold tuning skipped due to {type(e).__name__}: {e}. Using default 0.5-count rule."
        )

    dataloader_test = DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    sub_path = "/kaggle/working/submission.csv"
    with open(sub_path, "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["id_code", "diagnosis"])

    net.eval()
    with torch.no_grad():
        for _, item in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            data, names = item
            if use_gpu:
                data = data.cuda(non_blocking=True)

            out = net(data)
            prob = torch.sigmoid(out).detach().cpu().numpy()

            if tuned_thr is None:
                predicted = (prob > 0.5).sum(axis=1)
                predicted = np.clip(predicted, 0, 4).astype(int)
            else:
                score = prob.sum(axis=1).astype(np.float32)
                predicted = _ordinal_from_thresholds(score, tuned_thr).astype(int)

            with open(sub_path, "a", newline="") as f:
                f_csv = csv.writer(f)
                for n, p in zip(list(names), list(predicted)):
                    f_csv.writerow([str(n), int(p)])

    sub_df = pd.read_csv(sub_path)
    assert list(sub_df.columns) == ["id_code", "diagnosis"]
    assert len(sub_df) == len(test_df)
    print(f"Wrote submission: {sub_path} with shape {sub_df.shape}")
