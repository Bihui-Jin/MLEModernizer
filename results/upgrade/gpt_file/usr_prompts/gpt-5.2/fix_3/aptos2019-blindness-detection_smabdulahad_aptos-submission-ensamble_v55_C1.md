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

# 5. Target score

0.7316677251078982

# 6. Current score

0.52064

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08041) has done: 'I fix the missing external model files issue by falling back to a locally available pretrained timm model when the ensemble checkpoint paths don’t exist, so the notebook runs end-to-end and still produces reasonable predictions. I also fix the empty-ensemble logic that caused `torch.cat()` to fail and ensure predictions are always generated. Finally, I make image loading robust (RGB conversion) and load state dicts safely on the right device. This keeps the same core “timm model → softmax → weighted ensemble → argmax” semantics, only changing the weight/model sources when the original files are unavailable.'
- What this solution (achieved 0.52064) has done: 'Your current negative kappa is consistent with a label mapping mismatch: the model outputs 5 ImageNet class indices that do not correspond to the DR grades 0–4, so `argmax` is essentially random with respect to the metric. To move the score toward the 0.73 target while preserving your “timm model → softmax → weighted ensemble → argmax” core logic, I keep the same inference pipeline but (1) use a timm backbone that is strong and widely available as pretrained without external checkpoints, and (2) replace the final `argmax` with a minimal calibration step that maps the continuous expected grade (dot(probs, [0..4])) to 0–4 via thresholds optimized on a small validation split using quadratic weighted kappa. This is a standard post-processing alignment for this competition’s metric and is the smallest legitimate change that directly targets the evaluation. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
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
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
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
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/efficientnet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_v4.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_model_paths = {k: p for k, p in model_paths.items() if os.path.exists(p)}
use_fallback_pretrained = len(available_model_paths) == 0

models_list = []
model_keys = []

if use_fallback_pretrained:
    fallback_key = "efficientnet_b0"
    fallback_name = "tf_efficientnet_b0"
    model = timm.create_model(fallback_name, pretrained=True, num_classes=5)
    model.to(device).eval()
    models_list = [model]
    model_keys = [fallback_key]
else:
    for model_key, path in available_model_paths.items():
        model_name = model_names[model_key]
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        models_list.append(model)
        model_keys.append(model_key)



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "inception_resnet_v2": 0.880,
    "inception_v4": 0.902,
    "seresnext101_32x4d": 0.9697,
}



## === cell 7
if use_fallback_pretrained:
    weights = {model_keys[0]: 1.0}
else:
    filtered_scores = {k: validation_scores.get(k, 1.0) for k in model_keys}
    total_score = sum(filtered_scores.values())
    weights = {k: v / total_score for k, v in filtered_scores.items()}




## === cell 8
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def apply_thresholds(x, thresholds):
    t0, t1, t2, t3 = thresholds
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > t0] = 1
    y[x > t1] = 2
    y[x > t2] = 3
    y[x > t3] = 4
    return y


def fit_thresholds_for_kappa(x, y_true, n_iter=3):
    thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

    steps = [0.25, 0.1, 0.05]
    for it in range(n_iter):
        step = steps[min(it, len(steps) - 1)]
        for k in range(4):
            low = -0.5 if k == 0 else thr[k - 1] + 1e-3
            high = 4.5 if k == 3 else thr[k + 1] - 1e-3
            candidates = np.arange(
                max(low, thr[k] - 1.0), min(high, thr[k] + 1.0) + 1e-9, step
            )
            best_thr_k = thr[k]
            best_score = -1e9
            for c in candidates:
                thr_try = thr.copy()
                thr_try[k] = float(c)
                y_pred = apply_thresholds(x, thr_try)
                score = quadratic_weighted_kappa(y_true, y_pred, n_classes=5)
                if score > best_score:
                    best_score = score
                    best_thr_k = float(c)
            thr[k] = best_thr_k
    return thr




## === cell 9
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)

rng = np.random.RandomState(42)
perm = rng.permutation(len(train_df))
val_size = 512 if len(train_df) >= 512 else max(128, len(train_df) // 5)
val_idx = perm[:val_size]
val_df = train_df.iloc[val_idx].reset_index(drop=True)

val_csv_tmp = "/kaggle/working/_val_split.csv"
val_df.to_csv(val_csv_tmp, index=False)

val_dataset = BlindnessDataset(
    val_csv_tmp, train_root_dir, transform=transform, test=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 10
grade_values = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)

val_x = []
val_y = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Val inference (threshold fit)"):
        images = images.to(device, non_blocking=True)
        labels = labels.numpy().astype(int)
        probs_ens = None

        for model_key, model in zip(model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            weighted = weights[model_key] * probs
            probs_ens = weighted if probs_ens is None else (probs_ens + weighted)

        exp_grade = (
            (probs_ens * grade_values.unsqueeze(0)).sum(dim=1).detach().cpu().numpy()
        )
        val_x.append(exp_grade)
        val_y.append(labels)

val_x = np.concatenate(val_x, axis=0)
val_y = np.concatenate(val_y, axis=0)

thresholds = fit_thresholds_for_kappa(val_x, val_y, n_iter=3)
val_pred = apply_thresholds(val_x, thresholds)
val_kappa = quadratic_weighted_kappa(val_y, val_pred, n_classes=5)
print("Fitted thresholds:", thresholds)
print("Val QWK (for calibration sanity-check):", float(val_kappa))



## === cell 11
all_exp_grades = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        probs_ens = None

        for model_key, model in zip(model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            weighted = weights[model_key] * probs
            probs_ens = weighted if probs_ens is None else (probs_ens + weighted)

        exp_grade = (
            (probs_ens * grade_values.unsqueeze(0)).sum(dim=1).detach().cpu().numpy()
        )
        all_exp_grades.append(exp_grade)

all_exp_grades = np.concatenate(all_exp_grades, axis=0)
final_predictions = apply_thresholds(all_exp_grades, thresholds).astype(int)



## === cell 12
test_ids = pd.read_csv(test_csv_file)["id_code"].values
submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})

assert submission_df.shape[0] == len(test_ids)
assert list(submission_df.columns) == ["id_code", "diagnosis"]

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
