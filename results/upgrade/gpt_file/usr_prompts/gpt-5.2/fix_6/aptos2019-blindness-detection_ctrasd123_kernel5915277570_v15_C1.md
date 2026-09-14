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

0.7245341870411515

# 6. Current score

0.87096

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `tensorboardX` (not installed) by falling back to PyTorch’s built-in TensorBoard writer or a no-op writer. Then I fix the model definition so it matches DenseNet121’s actual output feature dimension and remove incorrect `torch.tensor(x)` conversions that would break GPU tensors and gradients. Finally, because the referenced pretrained weight file does not exist in the provided `/kaggle/input`, I make checkpoint loading optional and, if missing, run an end-to-end inference that still produces a valid `submission.csv` with the required columns (using a deterministic fallback prediction so the notebook always yields a submit-ready file).'
- What this solution (achieved 0.0) has done: 'I fix the import/runtime crash caused by TensorBoard/Protobuf incompatibility by avoiding `torch.utils.tensorboard` entirely and using a small no-op `SummaryWriter` (this is score-neutral and unblocks execution). Then I fix a logic bug in inference: the model outputs 5 logits but you were treating it like a 5-class softmax; with your “single BCE” head it should instead be converted to an ordinal class via `sigmoid` + threshold counting. Finally, I ensure the script always writes a valid `submission.csv` with correct row count/order and make checkpoint loading map to the correct device without crashing.'
- What this solution (achieved 0.11829) has done: 'Your current 0.0 score is coming from the “no checkpoint => always predict 0” fallback, which guarantees near-random agreement on quadratic kappa. To move the score upward toward the target while keeping core logic intact, I (1) enable ImageNet-pretrained DenseNet121 weights (same architecture; just better initialization) and (2) add a tiny, validation-based threshold calibration step (still using your ordinal “sigmoid + count > thr” decoding) so predictions better match the metric. I also keep the exact submission format and ensure id ordering matches `test.csv`, writing a single valid `submission.csv` at the end.'
- What this solution (achieved 0.87096) has done: 'The timeout is dominated by repeated, expensive image decoding + preprocessing (OpenCV read, gray-crop, resize) during both training and validation, and by underutilized GPU input pipelines. I keep the exact model, loss, training loops, and preprocessing logic, but eliminate redundant work by caching preprocessed images in RAM for train/val/test datasets (equivalent outputs, just reused). I also speed up dataloading safely by enabling persistent workers, prefetching, and setting workers based on CPU count, while keeping determinism via fixed seeds and a seeded DataLoader generator. Finally, I vectorize the quadratic weighted kappa computation (same formula) to reduce Python overhead during repeated validation/threshold scans.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import os.path as osp
import random
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


class SummaryWriter:  # no-op (avoid tensorboard/protobuf issues)
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
content = (
    pd.read_csv(name_file, usecols=["id_code"])["id_code"].astype(str) + ".png"
).tolist()




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        backbone = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.DEFAULT
        )
        self.base = backbone.features
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
    if image is None:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    if image is None or (hasattr(image, "size") and image.size == 0):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
    image = cv2.resize(image, (512, 512))
    return image


class _ImageCache:
    __slots__ = ("store",)

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key, None)

    def set(self, key, value):
        self.store[key] = value


class eye_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform
        self._cache = _ImageCache()

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = "/kaggle/input/aptos2019-blindness-detection/test_images/" + fn

        cached = self._cache.get(fn)
        if cached is None:
            img = cv2.imread(img_path)
            img = load_ben_yuan(img)
            self._cache.set(fn, img)
        else:
            img = cached

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
        self._cache = _ImageCache()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        fn = row["id_code"] + ".png"
        y = int(row["diagnosis"])

        cached = self._cache.get(fn)
        if cached is None:
            img_path = os.path.join(self.img_dir, fn)
            img = cv2.imread(img_path)
            img = load_ben_yuan(img)
            self._cache.set(fn, img)
        else:
            img = cached

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.df)




## === cell 4
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    k = n_classes
    mask = (y_true >= 0) & (y_true < k) & (y_pred >= 0) & (y_pred < k)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size == 0:
        return 0.0

    O = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k).astype(np.float64)

    act_hist = np.bincount(yt, minlength=k).astype(np.float64)
    pred_hist = np.bincount(yp, minlength=k).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    Es = E.sum()
    Os = O.sum()
    if Es > 0:
        E = E * (Os / Es)

    idx = np.arange(k, dtype=np.float64)
    W = ((idx[:, None] - idx[None, :]) ** 2) / ((k - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def predict_with_threshold(net, dataloader, device, thr):
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            out = net(x)  # [B,5] logits
            prob = torch.sigmoid(out)  # [B,5]
            pred = (prob > thr).sum(dim=1).clamp(0, 4).to(torch.int64).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(np.asarray(y, dtype=np.int64).tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def make_ordinal_targets(y, num_classes=5, device=None):
    y = y.to(torch.int64)
    ks = torch.arange(num_classes, device=y.device, dtype=torch.int64).view(1, -1)
    t = (y.view(-1, 1) > ks).to(torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def train_one_epoch(net, loader, optimizer, device):
    net.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        target = make_ordinal_targets(y, num_classes=5)

        optimizer.zero_grad(set_to_none=True)
        out = net(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


def evaluate_qwk(net, loader, device, thr):
    y_t, y_p = predict_with_threshold(net, loader, device, thr)
    return quadratic_weighted_kappa(y_t, y_p, n_classes=5)




## === cell 5
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_data = eye_dataset(content, transform2)
    net = Baseline_single(num_classes=5).to(device)

    ckpt_path = "/kaggle/input/temp-file/model_yuan512_dense_00001_adam_precrop_1.pkl"
    loaded = False
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(state)
            loaded = True
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Checkpoint found but failed to load ({e}). Proceeding without it.")
    else:
        print(
            f"Checkpoint not found at {ckpt_path}. Proceeding without it (will train a small finetune to improve score)."
        )

    num_workers = min(8, (os.cpu_count() or 2))
    g = torch.Generator()
    g.manual_seed(0)
    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=use_gpu,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    dataloader_test = DataLoader(
        test_data, batch_size=16, shuffle=False, **common_loader_kwargs
    )

    train_csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images/"
    df_train = pd.read_csv(train_csv_path)

    rng = np.random.RandomState(0)
    idx = np.arange(len(df_train))
    rng.shuffle(idx)
    val_size = min(512, max(256, len(df_train) // 5))  # keep runtime bounded, stable
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    df_val = df_train.iloc[val_idx].copy()
    df_tr = df_train.iloc[tr_idx].copy()

    tr_ds = eye_train_dataset(df_tr, train_img_dir, transform=transform2)
    val_ds = eye_train_dataset(df_val, train_img_dir, transform=transform2)

    tr_loader = DataLoader(
        tr_ds, batch_size=16, shuffle=True, generator=g, **common_loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, **common_loader_kwargs
    )

    net.freeze_base()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3
    )

    epochs_stage1 = 2
    for ep in range(epochs_stage1):
        tr_loss = train_one_epoch(net, tr_loader, optimizer, device)
        qwk_tmp = evaluate_qwk(net, val_loader, device, thr=0.5)
        print(
            f"[Stage1][Epoch {ep+1}/{epochs_stage1}] loss={tr_loss:.4f} val_QWK@0.50={qwk_tmp:.4f}"
        )

    for name, p in net.base.named_parameters():
        p.requires_grad = False
        if name.startswith("denseblock4") or name.startswith("norm5"):
            p.requires_grad = True
    for p in net.classifiers.parameters():
        p.requires_grad = True

    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4
    )
    epochs_stage2 = 1
    for ep in range(epochs_stage2):
        tr_loss = train_one_epoch(net, tr_loader, optimizer2, device)
        qwk_tmp = evaluate_qwk(net, val_loader, device, thr=0.5)
        print(
            f"[Stage2][Epoch {ep+1}/{epochs_stage2}] loss={tr_loss:.4f} val_QWK@0.50={qwk_tmp:.4f}"
        )

    thr_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_thr = 0.50
    best_kappa = -1.0
    for thr in thr_grid:
        y_t, y_p = predict_with_threshold(net, val_loader, device, thr)
        k = quadratic_weighted_kappa(y_t, y_p, n_classes=5)
        if k > best_kappa:
            best_kappa = k
            best_thr = thr
    print(f"Chosen threshold={best_thr:.2f} via val QWK={best_kappa:.4f}")

    net.eval()
    preds = []
    ids = []
    with torch.no_grad():
        for data, name in tqdm(dataloader_test, total=len(dataloader_test)):
            data = data.to(device, non_blocking=True)
            out = net(data)  # [B,5] logits
            prob = torch.sigmoid(out)
            pred_labels = (
                (prob > best_thr).sum(dim=1).clamp(0, 4).to(torch.int64).cpu().numpy()
            )
            preds.extend(pred_labels.tolist())
            ids.extend([str(n) for n in name])

    sub_path = "/kaggle/working/submission.csv"
    sub_df = pd.DataFrame({"id_code": ids, "diagnosis": preds})

    test_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")
    sub_df = test_df.merge(sub_df, on="id_code", how="left")
    sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int)

    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission to: {sub_path} (rows={len(sub_df)})")
