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

3.8

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

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Image.MAX_IMAGE_PIXELS = None
try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass



## === cell 1
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform

        self._ids = self.eye_frame["id_code"].to_numpy()
        if self.filetype == "train":
            self._labels = self.eye_frame["diagnosis"].to_numpy(dtype=np.int64)
        else:
            self._labels = None

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/train_images",
                self._ids[idx] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(self._labels[idx])
        else:
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/test_images",
                self._ids[idx] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, self._ids[idx]




## === cell 2
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_df["id_code"] = train_df["id_code"].astype(str)

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"],
)

tr_csv = "train_split.csv"
va_csv = "valid_split.csv"
tr_df.to_csv(tr_csv, index=False)
va_df.to_csv(va_csv, index=False)

train_dataset = APTOSDataset(
    csv_file=tr_csv,
    filetype="train",
    transform=transform,
)

valid_dataset = APTOSDataset(
    csv_file=va_csv,
    filetype="train",
    transform=transform,
)

test_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_num_workers = 4
_common_loader_kwargs = dict(
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
)
if _num_workers > 0:
    _common_loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=True,
    **_common_loader_kwargs,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=24,
    shuffle=False,
    **_common_loader_kwargs,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    **_common_loader_kwargs,
)




## === cell 3
def _get_imagenet_weights(model_name: str):
    try:
        if model_name == "resnet152":
            return torchvision.models.ResNet152_Weights.DEFAULT
        if model_name == "resnet101":
            return torchvision.models.ResNet101_Weights.DEFAULT
    except Exception:
        return None
    return None


def build_resnet(model_name: str, weights_path: str, num_classes: int = 5):
    """
    Keep the same architectures and FC head, but fall back to ImageNet weights
    (or random init) so we can still fine-tune/infer end-to-end and write submission.csv.
    """
    imagenet_weights = _get_imagenet_weights(model_name)

    if model_name == "resnet152":
        model = torchvision.models.resnet152(weights=imagenet_weights)
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(weights=imagenet_weights)
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if weights_path is not None and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded competition weights: {weights_path}")
    else:
        print(
            f"WARNING: Weights not found at '{weights_path}'. "
            f"Falling back to torchvision ImageNet weights ({imagenet_weights is not None}) "
            "with a fresh 5-class head."
        )

    model = model.to(device)
    return model


model0 = build_resnet("resnet152", "../input/resnet/FinalResnet152_0.pt")
model1 = build_resnet("resnet101", "../input/resnet/FinalResnet02.pt")
model2 = build_resnet("resnet101", "../input/resnet/FinalResnet01.pt")
model3 = build_resnet("resnet101", "../input/resnet/FinalResnet00.pt")
model4 = build_resnet("resnet152", "../input/resnet0/pretrainedResnet151_0.pt")
model5 = build_resnet("resnet101", "../input/resnet0/pretrainedResnet1.pt")



## === cell 4
from tqdm.auto import tqdm
from sklearn.metrics import cohen_kappa_score


def compute_predictions(model, model_type, data_loader, device):
    use_amp = device.type == "cuda"
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        model.eval()
        with torch.inference_mode():
            for inputs, labels in tqdm(data_loader, desc="Predict(train)"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds)
                num_examples += labels.size(0)
                correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100
    else:
        predictions = []
        img_ids = []
        out = []
        model.eval()
        with torch.inference_mode():
            for inputs, img_id in tqdm(data_loader, desc="Predict(test)"):
                inputs = inputs.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.detach().cpu())
                img_ids.extend(list(img_id))
                out.extend(outputs.detach().cpu())
        predictions = [int(pred.item()) for pred in predictions]
        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        return final_predictions, out, img_ids


def predict_logits_and_labels(model, loader, device):
    model.eval()
    use_amp = device.type == "cuda"
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for x, y in tqdm(loader, desc="Predict(valid)"):
            x = x.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_labels.append(torch.as_tensor(y, dtype=torch.long).cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def logits_to_expected_value(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    classes = np.arange(probs.shape[1], dtype=np.float32)
    return (probs * classes[None, :]).sum(axis=1)


def apply_thresholds(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds).astype(np.int64)


def fit_thresholds(
    values: np.ndarray, y_true: np.ndarray, max_iter: int = 30
) -> np.ndarray:
    values = values.astype(np.float32)
    y_true = y_true.astype(np.int64)

    qs = [0.2, 0.4, 0.6, 0.8]
    t = np.quantile(values, qs).astype(np.float32)
    t = np.sort(t)

    best = cohen_kappa_score(y_true, apply_thresholds(values, t), weights="quadratic")

    for _ in range(max_iter):
        improved = False
        for k in range(4):
            lo = -np.inf if k == 0 else t[k - 1] + 1e-3
            hi = np.inf if k == 3 else t[k + 1] - 1e-3
            grid = np.linspace(
                max(np.min(values), lo) if np.isfinite(lo) else np.min(values),
                min(np.max(values), hi) if np.isfinite(hi) else np.max(values),
                30,
                dtype=np.float32,
            )
            cur_best_tk = t[k]
            cur_best = best
            for cand in grid:
                tc = t.copy()
                tc[k] = cand
                tc = np.sort(tc)
                score = cohen_kappa_score(
                    y_true, apply_thresholds(values, tc), weights="quadratic"
                )
                if score > cur_best + 1e-6:
                    cur_best = score
                    cur_best_tk = cand
            if cur_best > best + 1e-6:
                t[k] = cur_best_tk
                t = np.sort(t)
                best = cur_best
                improved = True
        if not improved:
            break

    print("Fitted thresholds:", t, "valid QWK:", best)
    return t




## === cell 5
def fine_tune_if_needed(
    model,
    local_ckpt_path: str,
    epochs: int = 1,
    lr: float = 3e-4,
    train_last_block: bool = True,
):
    if os.path.exists(local_ckpt_path):
        state = torch.load(local_ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded local fine-tuned weights: {local_ckpt_path}")
        return model

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    if train_last_block:
        for p in model.layer4.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Fine-tuning '{local_ckpt_path}' for {epochs} epoch(s). "
        f"Trainable params: {sum(p.numel() for p in trainable_params):,}"
    )

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(epochs):
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Fine-tune ep{ep+1}/{epochs}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {ep+1}: train loss={epoch_loss:.4f}")

    model.eval()
    torch.save(model.state_dict(), local_ckpt_path)
    print(f"Saved fine-tuned weights: {local_ckpt_path}")
    return model


model0 = fine_tune_if_needed(
    model0, "ft_resnet152_m0.pt", epochs=1, lr=3e-4, train_last_block=True
)
model1 = fine_tune_if_needed(
    model1, "ft_resnet101_m1.pt", epochs=1, lr=3e-4, train_last_block=True
)
model2 = fine_tune_if_needed(
    model2, "ft_resnet101_m2.pt", epochs=1, lr=3e-4, train_last_block=True
)
model3 = fine_tune_if_needed(
    model3, "ft_resnet101_m3.pt", epochs=1, lr=3e-4, train_last_block=True
)
model4 = fine_tune_if_needed(
    model4, "ft_resnet152_m4.pt", epochs=1, lr=3e-4, train_last_block=True
)
model5 = fine_tune_if_needed(
    model5, "ft_resnet101_m5.pt", epochs=1, lr=3e-4, train_last_block=True
)



## === cell 6
print("Computing Validation Predictions (for threshold fitting)")
va_logits0, va_y = predict_logits_and_labels(model0, valid_loader, device)
va_logits1, _ = predict_logits_and_labels(model1, valid_loader, device)
va_logits2, _ = predict_logits_and_labels(model2, valid_loader, device)
va_logits3, _ = predict_logits_and_labels(model3, valid_loader, device)
va_logits4, _ = predict_logits_and_labels(model4, valid_loader, device)
va_logits5, _ = predict_logits_and_labels(model5, valid_loader, device)

va_logits_ens = (
    va_logits0 + va_logits1 + va_logits2 + va_logits3 + va_logits4 + va_logits5
) / 6.0
va_vals = logits_to_expected_value(va_logits_ens)
thresholds = fit_thresholds(va_vals, va_y.numpy(), max_iter=25)



## === cell 7
print("Computing Test Predictions")
test_predictions0, out0, ids0 = compute_predictions(model0, "test", test_loader, device)
test_predictions1, out1, ids1 = compute_predictions(model1, "test", test_loader, device)
test_predictions2, out2, ids2 = compute_predictions(model2, "test", test_loader, device)
test_predictions3, out3, ids3 = compute_predictions(model3, "test", test_loader, device)
test_predictions4, out4, ids4 = compute_predictions(model4, "test", test_loader, device)
test_predictions5, out5, ids5 = compute_predictions(model5, "test", test_loader, device)

out = (
    torch.stack(out0)
    + torch.stack(out1)
    + torch.stack(out2)
    + torch.stack(out3)
    + torch.stack(out4)
    + torch.stack(out5)
) / 6.0

img_ids = np.array(ids0).astype(str)

test_vals = logits_to_expected_value(out)
predictions = apply_thresholds(test_vals, thresholds).astype(int)

final_predictions = pd.DataFrame(
    {"id_code": img_ids, "diagnosis": predictions.astype(int)}
)

test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")[
    "id_code"
].astype(str)
final_predictions["id_code"] = final_predictions["id_code"].astype(str)
final_predictions = test_ids.to_frame().merge(
    final_predictions, on="id_code", how="left"
)
final_predictions["diagnosis"] = final_predictions["diagnosis"].fillna(0).astype(int)

final_predictions.to_csv("submission.csv", index=False)

print(final_predictions.head())
print("Wrote submission.csv with shape:", final_predictions.shape)
print(
    "diagnosis value counts:\n",
    final_predictions["diagnosis"].value_counts().sort_index(),
)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
