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

3.12

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

0.0780772011140781

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the dependency on a missing external `.pth` file and instead initialize the same model architecture locally so the pipeline can run end-to-end. I also fix inference-time shape/normalization mismatches by using the same 224×224 ImageNet normalization transform at prediction time as used elsewhere, and make AMP usage conditional on CUDA to avoid CPU autocast issues. Finally, I ensure the script always writes a valid `submission.csv` with exactly the required columns (`id_code`, `diagnosis`) and the correct row alignment with `test.csv`.'
- What this solution (achieved 0.76075) has done: 'I make the pipeline reliably yield a non-zero Kaggle score by (1) enabling pretrained ImageNet initialization for the same ResNet18 architecture (no architecture change) and (2) switching to an ordinal-regression-style post-processing (expected value + fixed rounding) that better matches quadratic weighted kappa than raw argmax, while keeping the same CrossEntropy training. I also make CPU runs stable by disabling `pin_memory` and workers when CUDA isn’t available (prevents common DataLoader stalls), ensuring the notebook completes and always writes a valid `submission.csv`. These are minimal changes directly tied to improving QWK and producing a valid submission under the 600s constraint. The rest of your training loop, transforms, loss, and dataset logic remain intact.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torchvision import transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import cohen_kappa_score

import copy
import timm




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        augmentations=None,
        max_count=None,
        test=False,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_count = max_count
        self.test = test

        if not test and "diagnosis" in self.annotations.columns:
            self.class_counts = (
                self.annotations["diagnosis"].value_counts().sort_index()
            )
        else:
            self.class_counts = None

        if max_count and (not test) and (self.class_counts is not None):
            self.oversample(max_count)

    def oversample(self, max_count):  # Over sampling classes to balance
        samples = []
        for diagnosis in self.class_counts.index:
            class_samples = self.annotations[self.annotations["diagnosis"] == diagnosis]
            oversampled_class = class_samples.sample(max_count, replace=True)
            samples.append(oversampled_class)
        self.annotations = pd.concat(samples).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image

        label = int(self.annotations.iloc[idx, 1])
        return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 3
def select_model(model_name, input_size):
    if model_name == "efficientnet_b5":
        return timm.create_model(
            "efficientnet_b5", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "inception_resnet_v2":
        return timm.create_model(
            "inception_resnet_v2", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "inception_v4":
        return timm.create_model(
            "inception_v4", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "seresnext50_32x4d":
        return timm.create_model(
            "seresnext50_32x4d", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "seresnext101_32x4d":
        return timm.create_model(
            "seresnext101_32x4d", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "resnet18(WD_1e-3)_aptos":
        return timm.create_model("resnet18", pretrained=True, num_classes=5, in_chans=3)
    else:
        raise ValueError(f"Unknown model name {model_name}")




## === cell 4
def load_model(model_name, input_size, model_path, device):
    """
    Keep same architecture selection and weight-loading behavior.
    """
    model = select_model(model_name, input_size).to(device)
    if model_path is not None and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Loaded weights from: {model_path}")
    else:
        print(
            f"[WARN] Weights not found at: {model_path}. Will train {model_name} locally."
        )
    model.eval()
    return model




## === cell 5
def load_all_models(model_paths, device):
    models_list = []
    for model_name, input_size, model_path in model_paths:
        model = load_model(model_name, input_size, model_path, device)
        models_list.append((model_name, model))
    return models_list




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1"
if not os.path.isdir(save_dir):
    save_dir = "/kaggle/input"  # safe fallback that exists



## === cell 7
"""
model_paths = [
    ('efficientnet_b5', 512, os.path.join(save_dir, 'efficientnet_b5.pth')),
    ('resnet18(WD_1e-3)_aptos', 512, os.path.join(save_dir, 'resnet18(WD_1e-3)_aptos.pth')),
    ('inception_resnet_v2', 512, os.path.join(save_dir, 'inception_resnet_v2.pth')),
    ('inception_v4', 512, os.path.join(save_dir, 'inception_v4.pth')),
    ('seresnext50_32x4d', 512, os.path.join(save_dir, 'seresnext50_32x4d.pth')),
    ('seresnext101_32x4d', 384, os.path.join(save_dir, 'seresnext101_32x4d.pth'))
]
"""

model_paths = [
    (
        "resnet18(WD_1e-3)_aptos",
        512,
        os.path.join(save_dir, "resnet18(WD_1e-3)_aptos.pth"),
    )
]



## === cell 8
loaded_models = load_all_models(model_paths, device)




## === cell 9
def _compute_class_weights(train_df, num_classes=5):
    counts = (
        train_df["diagnosis"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values
    )
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_single_model_minimal(
    model,
    train_csv_file,
    train_root_dir,
    device,
    epochs=2,
    batch_size=16,
    lr=1e-4,
    num_workers=2,
    seed=42,
):
    """
    Keep training core logic identical.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    full_df = pd.read_csv(train_csv_file)

    tr_df, va_df = train_test_split(
        full_df,
        test_size=0.15,
        random_state=seed,
        stratify=full_df["diagnosis"],
    )

    tr_path = "/kaggle/working/_train_split.csv"
    va_path = "/kaggle/working/_valid_split.csv"
    tr_df.to_csv(tr_path, index=False)
    va_df.to_csv(va_path, index=False)

    tr_ds = BlindnessDataset(tr_path, train_root_dir, transform=transform, test=False)
    va_ds = BlindnessDataset(va_path, train_root_dir, transform=transform, test=False)

    effective_workers = num_workers if device.type == "cuda" else 0
    effective_pin = torch.cuda.is_available()

    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=effective_pin,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=effective_pin,
    )

    class_weights = _compute_class_weights(tr_df, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for ep in range(epochs):
        running_loss = 0.0
        for images, labels in tr_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu())

        model.eval()
        va_preds, va_true = [], []
        with torch.no_grad():
            for images, labels in va_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)

                probs = torch.softmax(logits, dim=1)
                class_ids = torch.arange(5, device=probs.device, dtype=probs.dtype)[
                    None, :
                ]
                exp = (probs * class_ids).sum(dim=1)
                preds = (
                    torch.clamp(torch.round(exp), 0, 4).to(torch.int64).cpu().numpy()
                )

                va_preds.append(preds)
                va_true.append(labels.numpy())
        va_preds = np.concatenate(va_preds)
        va_true = np.concatenate(va_true)
        kappa = cohen_kappa_score(va_true, va_preds, weights="quadratic")
        print(
            f"[TRAIN] epoch={ep+1}/{epochs} loss={running_loss/len(tr_loader):.4f} val_qwk={kappa:.4f}"
        )
        model.train()

    model.eval()
    return model




## === cell 10
def _train_prior_mean(train_csv_file):
    df = pd.read_csv(train_csv_file)
    return float(df["diagnosis"].mean())


def _expected_from_model(model, loader, device):
    model.eval()
    use_amp = device.type == "cuda"
    exps = []
    ys = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
            class_ids = torch.arange(5, device=probs.device, dtype=probs.dtype)[None, :]
            exp = (probs * class_ids).sum(dim=1).float().cpu().numpy()
            exps.append(exp)
            ys.append(labels.numpy())
    return np.concatenate(exps), np.concatenate(ys)


def _apply_thresholds(exp, thresholds):
    t0, t1, t2, t3 = thresholds
    pred = np.zeros_like(exp, dtype=np.int64)
    pred[exp >= t0] = 1
    pred[exp >= t1] = 2
    pred[exp >= t2] = 3
    pred[exp >= t3] = 4
    return pred


def _tune_thresholds_for_qwk(exp, y_true):
    thresholds = [0.5, 1.5, 2.5, 3.5]
    best = cohen_kappa_score(
        y_true, _apply_thresholds(exp, thresholds), weights="quadratic"
    )

    grid = np.arange(-0.4, 0.41, 0.05)  # small local search around each threshold
    for _ in range(2):  # two passes is enough; keep runtime low
        for i in range(4):
            base = thresholds[i]
            candidates = []
            for d in grid:
                t = base + float(d)
                cand = thresholds.copy()
                cand[i] = t
                cand = np.clip(cand, 0.0, 4.0).tolist()
                cand.sort()
                score = cohen_kappa_score(
                    y_true, _apply_thresholds(exp, cand), weights="quadratic"
                )
                candidates.append((score, cand))
            score, cand = max(candidates, key=lambda x: x[0])
            if score > best:
                best = score
                thresholds = cand

    return thresholds, best


def predict_ensemble(
    models,
    test_csv_file,
    test_root_dir,
    submission_file,
    device,
    train_prior_mean=None,
    alpha=0.20,
    thresholds=None,
    sample_submission_file="/kaggle/input/aptos2019-blindness-detection/sample_submission.csv",
):
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    sample_df = pd.read_csv(sample_submission_file)
    test_df = pd.read_csv(test_csv_file)
    test_df = sample_df[["id_code"]].merge(test_df, on="id_code", how="left")[
        ["id_code"]
    ]

    tmp_test_path = "/kaggle/working/_test_ordered.csv"
    test_df.to_csv(tmp_test_path, index=False)

    test_dataset = BlindnessDataset(
        tmp_test_path, test_root_dir, transform=test_transform, test=True
    )

    effective_workers = 2 if device.type == "cuda" else 0
    effective_pin = torch.cuda.is_available()

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=effective_pin,
    )

    predictions = []
    use_amp = device.type == "cuda"

    for model_name, model in models:
        model_preds = []
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                model_preds.append(probs.float().cpu().numpy())
        model_preds = np.concatenate(model_preds, axis=0)
        predictions.append(model_preds)

    avg_predictions = np.mean(predictions, axis=0)

    class_ids = np.arange(5, dtype=np.float32)[None, :]
    exp = (avg_predictions * class_ids).sum(axis=1)

    if train_prior_mean is None:
        train_prior_mean = 0.0
    exp = (1.0 - float(alpha)) * exp + float(alpha) * float(train_prior_mean)

    if thresholds is None:
        final_predictions = np.clip(np.rint(exp), 0, 4).astype(int)
    else:
        final_predictions = _apply_thresholds(exp, thresholds).astype(int)

    assert len(final_predictions) == len(
        test_df
    ), "Prediction length must match test rows."

    submission_df = pd.DataFrame(
        {"id_code": test_df["id_code"].values, "diagnosis": final_predictions}
    )
    submission_df = submission_df[["id_code", "diagnosis"]]
    submission_df.to_csv(submission_file, index=False)
    return submission_df




## === cell 11
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

trained_models = []
for model_name, model in loaded_models:
    expected_path = dict([(mp[0], mp[2]) for mp in model_paths]).get(model_name, None)
    if expected_path is None or not os.path.exists(expected_path):
        model = train_single_model_minimal(
            model,
            train_csv_file=train_csv_file,
            train_root_dir=train_root_dir,
            device=device,
            epochs=2,  # keep fixed
            batch_size=16,
            lr=1e-4,
            num_workers=2,
            seed=42,
        )
    trained_models.append((model_name, model))

full_df = pd.read_csv(train_csv_file)
tr_df, va_df = train_test_split(
    full_df, test_size=0.15, random_state=42, stratify=full_df["diagnosis"]
)
va_path = "/kaggle/working/_valid_split.csv"
va_df.to_csv(va_path, index=False)

va_ds = BlindnessDataset(va_path, train_root_dir, transform=transform, test=False)
effective_workers = 2 if device.type == "cuda" else 0
effective_pin = torch.cuda.is_available()
va_loader = DataLoader(
    va_ds,
    batch_size=16,
    shuffle=False,
    num_workers=effective_workers,
    pin_memory=effective_pin,
)

model_for_tuning = trained_models[0][1]
va_exp, va_true = _expected_from_model(model_for_tuning, va_loader, device)
best_thresholds, best_qwk = _tune_thresholds_for_qwk(va_exp, va_true)
print(
    "[INFO] Tuned thresholds:", best_thresholds, "val_qwk:", round(float(best_qwk), 6)
)

test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
submission_file = "/kaggle/working/submission.csv"

prior_mean = _train_prior_mean(train_csv_file)
sub_df = predict_ensemble(
    trained_models,
    test_csv_file,
    test_root_dir,
    submission_file,
    device,
    train_prior_mean=prior_mean,
    alpha=0.20,
    thresholds=best_thresholds,
    sample_submission_file="/kaggle/input/aptos2019-blindness-detection/sample_submission.csv",
)
print(sub_df.head())
print("Wrote:", submission_file, "rows:", len(sub_df), "cols:", list(sub_df.columns))
print("Train prior mean (anchor):", prior_mean)
print("Diagnosis value counts:\n", sub_df["diagnosis"].value_counts().sort_index())
