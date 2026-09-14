# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_id + ".png")

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        label = int(self.annotations.iloc[idx, 1])
        return image, label




## === cell 2
FALLBACK_MODEL_KEY = "efficientnet_b2"
FALLBACK_MODEL_NAME = "efficientnet_b2"

fallback_data_cfg = timm.data.resolve_model_data_config(
    timm.create_model(FALLBACK_MODEL_NAME, pretrained=True, num_classes=5)
)
IMG_SIZE = int(fallback_data_cfg["input_size"][-1])  # 260 for efficientnet_b2
MEAN = fallback_data_cfg["mean"]
STD = fallback_data_cfg["std"]

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
model_paths = {
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b2.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = []
active_model_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        continue
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    models_list.append(model)
    active_model_keys.append(model_key)

if len(models_list) == 0:
    fallback_key = "efficientnet_b2"
    fallback_name = model_names[fallback_key]
    model = timm.create_model(fallback_name, pretrained=True, num_classes=5)
    model.to(device)
    model.eval()
    models_list = [model]
    active_model_keys = [fallback_key]



## === cell 5
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}



## === cell 6
active_scores = {
    k: validation_scores[k] for k in active_model_keys if k in validation_scores
}
if len(active_scores) == 0:
    active_scores = {active_model_keys[0]: 1.0}

total_score = float(sum(active_scores.values()))
weights = {k: (v / total_score) for k, v in active_scores.items()}




## === cell 7
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    denom = (num_classes - 1) ** 2
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den) if den > 0 else 0.0


def apply_thresholds(scores, thr):
    thr = np.asarray(thr, dtype=np.float64)
    return np.digitize(scores, thr, right=False).astype(int)


def fit_thresholds(scores, y_true, num_classes=5, iters=2):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    thr = np.quantile(scores, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)

    for _ in range(iters):
        for k in range(num_classes - 1):
            lo = scores.min() - 1e-6 if k == 0 else thr[k - 1] + 1e-6
            hi = scores.max() + 1e-6 if k == (num_classes - 2) else thr[k + 1] - 1e-6
            if not (lo < hi):
                continue

            cand = np.unique(np.clip(scores, lo, hi))
            if cand.size > 120:
                idx = np.linspace(0, cand.size - 1, 120).round().astype(int)
                cand = cand[idx]

            best_t = thr[k]
            best = -1e9
            for t in cand:
                thr_try = thr.copy()
                thr_try[k] = float(t)
                pred = apply_thresholds(scores, thr_try)
                q = quadratic_weighted_kappa(y_true, pred, num_classes=num_classes)
                if q > best:
                    best = q
                    best_t = float(t)
            thr[k] = best_t
    return thr


seed_everything(42)

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

_using_fallback = (
    (len(model_paths) > 0)
    and (len(active_model_keys) == 1)
    and (active_model_keys[0] == "efficientnet_b2")
    and (not os.path.exists(model_paths["efficientnet_b2"]))
)

if _using_fallback:
    df = pd.read_csv(train_csv_file)

    train_tmp = "/kaggle/working/_train_full.csv"
    df.to_csv(train_tmp, index=False)

    train_dataset = BlindnessDataset(
        train_tmp, train_root_dir, transform=train_transform, test=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = models_list[0]
    model.train()

    for p in model.parameters():
        p.requires_grad = False

    head_params = []
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        for p in model.classifier.parameters():
            p.requires_grad = True
        head_params = list(model.classifier.parameters())
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        for p in model.fc.parameters():
            p.requires_grad = True
        head_params = list(model.fc.parameters())
    elif hasattr(model, "head") and isinstance(model.head, nn.Module):
        for p in model.head.parameters():
            p.requires_grad = True
        head_params = list(model.head.parameters())
    else:
        head_params = [p for p in model.parameters() if p.requires_grad]

    label_counts = np.bincount(df["diagnosis"].astype(int).values, minlength=5).astype(
        np.float64
    )
    inv = 1.0 / np.maximum(label_counts, 1.0)
    class_weights = inv * (inv.size / inv.sum())
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(head_params, lr=3e-3, weight_decay=1e-4)

    epochs = 3
    for ep in range(epochs):
        model.train()
        running = 0.0
        for images, labels in tqdm(train_loader, desc=f"Train head ep {ep+1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
        print(f"Epoch {ep+1}: train_loss={running/ max(1,len(train_loader)):.4f}")

    model.eval()
    train_eval_ds = BlindnessDataset(
        train_tmp, train_root_dir, transform=transform, test=False
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    y_true, train_scores = [], []
    with torch.no_grad():
        for images, labels in tqdm(train_eval_loader, desc="Calibrate on train"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
            score = (probs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
            train_scores.extend(score.tolist())
            y_true.extend(labels.numpy().astype(int).tolist())

    thr = fit_thresholds(
        np.array(train_scores), np.array(y_true), num_classes=5, iters=2
    )
    train_pred_thr = apply_thresholds(np.array(train_scores), thr)
    qwk_train = quadratic_weighted_kappa(
        np.array(y_true), train_pred_thr, num_classes=5
    )
    print("Learned thresholds:", thr)
    print(f"Train QWK after thresholding (sanity): {qwk_train:.4f}")

    model.eval()
    models_list[0] = model
else:
    thr = None  # no calibration by default for external checkpoints



## === cell 8
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device, non_blocking=True)

        outputs = []
        for model_key, model in zip(active_model_keys, models_list):
            w = weights.get(model_key, 1.0 / len(models_list))
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)
            outputs.append(w * probs.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        weighted_outputs = torch.sum(outputs, dim=0)
        all_outputs.extend(weighted_outputs.cpu().numpy())

all_outputs = np.asarray(all_outputs)

if thr is not None:
    test_scores = (all_outputs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
    final_predictions = apply_thresholds(test_scores, thr).astype(int)
else:
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 9
test_ids = pd.read_csv(test_csv_file)["id_code"].values

assert len(final_predictions) == len(
    test_ids
), f"Preds ({len(final_predictions)}) != test ({len(test_ids)})"

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
