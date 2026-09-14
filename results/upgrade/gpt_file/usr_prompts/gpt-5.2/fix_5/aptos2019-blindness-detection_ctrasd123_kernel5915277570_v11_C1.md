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

0.7407251123209954

# 6. Current score

-0.00133

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00222) has done: 'I remove the unavailable `tensorboardX` dependency so imports succeed in Kaggle. Then I fix the model definition bugs (DenseNet feature dim and incorrect `torch.tensor(x)` copies) that would otherwise crash or silently break inference. Since the referenced pretrained weights file is missing, I add a minimal fallback: load ImageNet-pretrained DenseNet121 weights (same architecture) and run deterministic test-time inference, ensuring we still produce a valid `submission.csv`. Finally, I fix the test CSV parsing and transforms to avoid randomness at inference and guarantee the submission matches the required format.'
- What this solution (achieved -0.16311) has done: 'I fix the import failure coming from `torch.utils.tensorboard` by disabling TensorBoard import entirely, since it’s unused for inference and currently crashes in this environment. Then I keep the same DenseNet121 inference pipeline but restore the intended pretrained checkpoint behavior safely by (a) attempting to load the provided checkpoint if present and (b) otherwise using ImageNet weights plus a minimal, score-relevant calibration step: convert the 5-class logits into an ordinal regression style prediction via expected value and rounding/clipping (a common, small post-processing improvement for QWK without changing the model). Finally, I ensure the submission rows exactly match `test.csv` ordering and write a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved -0.03888) has done: 'Your current score is far below the target, so the smallest safe way to move it upward is to fix the biggest mismatch: you’re using an ImageNet backbone but a randomly initialized 5-class head, which makes predictions essentially noise and drives QWK negative. I keep the exact same model/loss semantics and inference flow, but I load a full DenseNet121 ImageNet state_dict into the model (backbone + classifier) with `strict=False` so it also initializes your 5-way head deterministically instead of random. Then I make inference slightly more stable by using a larger batch size on GPU (if available) and setting `num_workers=0` to avoid dataloader worker overhead/crashes in Kaggle, without changing predictions’ meaning. This should materially improve agreement vs. random and move the score toward the 0.74 target.'
- What this solution (achieved -0.00133) has done: 'Your current score is far below the target, so we should make the smallest changes that significantly increase agreement without changing the core model or training approach (this is still pure inference). The main fix is to load the ImageNet-pretrained DenseNet121 *features* into your model’s backbone while keeping your 5-class classifier head deterministic and non-random, then calibrate predictions with a tiny, QWK-relevant post-processing: optimize 4 cutpoints on the training set (using the same model outputs) to convert the expected value into ordinal classes (a standard minimal improvement for QWK). This keeps the architecture and inference pipeline intact, but replaces naive rounding with thresholding that better matches the ordinal metric. The script still writes `/kaggle/working/submission.csv` in the required format and keeps test ordering identical to `test.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os, sys, time, datetime, argparse, random
import os.path as osp
from PIL import Image
import cv2
import csv

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

SummaryWriter = None


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


seed_everything(0)



## === cell 1
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"

test_df = pd.read_csv(TEST_CSV_PATH)
content = (test_df["id_code"].astype(str) + ".png").tolist()

print("Loaded test ids:", len(content))
print("First 3:", content[:3])

train_df = pd.read_csv(TRAIN_CSV_PATH)
print("Loaded train rows:", len(train_df), "cols:", list(train_df.columns))
print(train_df.head())




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        backbone = torchvision.models.densenet121(weights=None)
        self.base = backbone.features  # DenseNet's convolutional features
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
        x = torch.relu(x)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
            return ys
        return x




## === cell 3
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


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
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    return image


class eye_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = "/kaggle/input/aptos2019-blindness-detection/test_images/" + fn
        img = cv_imread(img_path)
        img = load_ben_yuan(img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_dataset_train(Dataset):
    def __init__(self, df, transform=None):
        self.ids = df["id_code"].astype(str).tolist()
        self.y = df["diagnosis"].astype(int).tolist()
        self.transform = transform

    def __getitem__(self, idx):
        fn = self.ids[idx] + ".png"
        img_path = "/kaggle/input/aptos2019-blindness-detection/train_images/" + fn
        img = cv_imread(img_path)
        img = load_ben_yuan(img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[idx])

    def __len__(self):
        return len(self.ids)




## === cell 4
def apply_cutpoints(x, cutpoints):
    c0, c1, c2, c3 = cutpoints
    return np.where(
        x < c0,
        0,
        np.where(x < c1, 1, np.where(x < c2, 2, np.where(x < c3, 3, 4))),
    ).astype(np.int64)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    num = (W * O).sum()
    return 1.0 - (num / denom)


def optimize_cutpoints(x, y, n_classes=5, max_iter=3, grid_size=60):
    """
    Minimal, safe calibration for QWK: search for 4 thresholds on scalar predictions x
    to maximize QWK on train. Keeps model unchanged; only changes post-processing.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    qs = [20, 40, 60, 80]
    cut = np.percentile(x, qs).tolist()

    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return [xmin, xmin, xmin, xmin], 0.0

    best_cut = cut
    best_k = quadratic_weighted_kappa(
        y, apply_cutpoints(x, best_cut), n_classes=n_classes
    )

    for _ in range(max_iter):
        improved = False
        for ci in range(4):
            lo = xmin if ci == 0 else best_cut[ci - 1] + 1e-6
            hi = xmax if ci == 3 else best_cut[ci + 1] - 1e-6
            if not (lo < hi):
                continue

            grid = np.linspace(lo, hi, grid_size)
            local_best_k = best_k
            local_best_t = best_cut[ci]

            cand = best_cut.copy()
            for t in grid:
                cand[ci] = float(t)
                if not (cand[0] < cand[1] < cand[2] < cand[3]):
                    continue
                k = quadratic_weighted_kappa(
                    y, apply_cutpoints(x, cand), n_classes=n_classes
                )
                if k > local_best_k:
                    local_best_k = k
                    local_best_t = float(t)

            if local_best_k > best_k:
                best_k = local_best_k
                best_cut[ci] = local_best_t
                improved = True

        if not improved:
            break

    return best_cut, float(best_k)




## === cell 5
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        torch.cuda.manual_seed_all(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_data = eye_dataset(content, transform2)

    batch_size = 16 if use_gpu else 4
    dataloader_test = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_gpu,
    )

    net = Baseline_single(num_classes=5).to(device)

    missing_ckpt = (
        "/kaggle/input/temp-file/model_yuan512_dense121_00001_adam_avg_pre_ji_2.pkl"
    )
    if os.path.exists(missing_ckpt):
        state = torch.load(missing_ckpt, map_location=device)
        net.load_state_dict(state, strict=True)
        print("Loaded checkpoint:", missing_ckpt)
    else:
        print("Checkpoint not found:", missing_ckpt)

        print(
            "Falling back to ImageNet-pretrained DenseNet121 backbone initialization."
        )
        pretrained = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )
        msg = net.base.load_state_dict(pretrained.features.state_dict(), strict=True)
        print("Loaded pretrained.features into net.base. Status:", msg)

    net.eval()

    class_values = torch.arange(5, device=device, dtype=torch.float32).view(1, -1)

    calib_max = (
        1024  # small cap to keep runtime within 600s while improving above random
    )
    calib_df = train_df.sample(
        n=min(calib_max, len(train_df)), random_state=0
    ).reset_index(drop=True)
    calib_data = eye_dataset_train(calib_df, transform2)
    dataloader_calib = DataLoader(
        calib_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_gpu,
    )

    x_list = []
    y_list = []
    with torch.no_grad():
        for data, y in tqdm(
            dataloader_calib, total=len(dataloader_calib), desc="Calibrating"
        ):
            data = data.to(device, non_blocking=True)
            out = net(data)  # (B,5) logits
            probs = torch.softmax(out, dim=1)
            exp = (probs * class_values).sum(dim=1)  # (B,)
            x_list.append(exp.detach().cpu().numpy())
            y_list.append(np.asarray(y, dtype=np.int64))
    x_cal = np.concatenate(x_list, axis=0)
    y_cal = np.concatenate(y_list, axis=0)

    cutpoints, k_train = optimize_cutpoints(
        x_cal, y_cal, n_classes=5, max_iter=3, grid_size=60
    )
    print("Optimized cutpoints:", cutpoints)
    print("Approx train QWK on calib subset:", k_train)

    sub_path = "/kaggle/working/submission.csv"
    rows = []

    with torch.no_grad():
        for _, item in tqdm(
            enumerate(dataloader_test), total=len(dataloader_test), desc="Predicting"
        ):
            data, names = item
            data = data.to(device, non_blocking=True)

            out = net(data)  # (B,5) logits
            probs = torch.softmax(out, dim=1)
            exp = (probs * class_values).sum(dim=1)  # (B,)
            exp_np = exp.detach().cpu().numpy().astype(np.float64)

            predicted = apply_cutpoints(exp_np, cutpoints)

            names = list(names)
            for n, p in zip(names, predicted):
                rows.append((str(n), int(p)))

    pred_map = {k: v for k, v in rows}
    ordered_ids = test_df["id_code"].astype(str).tolist()
    ordered_preds = [int(pred_map[_id]) for _id in ordered_ids]

    sub_df = pd.DataFrame({"id_code": ordered_ids, "diagnosis": ordered_preds})
    sub_df.to_csv(sub_path, index=False)

    check_df = pd.read_csv(sub_path)
    print(
        "Wrote:", sub_path, "shape=", check_df.shape, "columns=", list(check_df.columns)
    )
    print(check_df.head())
