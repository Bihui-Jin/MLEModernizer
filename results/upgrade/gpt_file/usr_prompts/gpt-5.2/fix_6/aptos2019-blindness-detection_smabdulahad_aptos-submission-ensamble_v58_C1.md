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

0.8237644199831609

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.06923) has done: 'I fix the runtime failure by making the code robust to missing external model weight files (the `/kaggle/input/aptos_ensamble-models/...` dataset isn’t present), so it can still run end-to-end and write `submission.csv`. To preserve the ensemble core logic, the script only load models whose weight paths exist and automatically re-normalize the ensemble weights to the actually-loaded models. I also fix the inference loop so `torch.cat()` never receives an empty list and ensure the submission length matches `test.csv`. Finally, I make image loading more robust (`RGB` conversion) and make `torch.load` compatible across devices via `map_location`.'
- What this solution (achieved 0.30211) has done: 'Your current negative kappa is consistent with a label-space mismatch: the fallback model uses ImageNet-pretrained weights but outputs 5 classes with random classifier head weights, so predictions are essentially noise. To move the score up toward the target with minimal change, I keep the same model/inference logic but switch the fallback to a true ImageNet head (1000 classes) and map its probabilities to 5 DR classes via a fixed binning of the expected severity (using the ImageNet class index expectation). This preserves your “softmax + weighted ensemble + argmax” semantics, but makes the fallback produce structured (non-random) predictions and typically improves kappa dramatically versus random outputs. I also add test-time augmentation with a horizontal flip averaged in-probability space (still the same core inference, just a deterministic augmentation) to nudge performance upward without changing training or architecture.'
- What this solution (achieved 0.2911) has done: 'Your current score (0.30211) is far below the target (0.82376), so we should improve the fallback behavior while keeping the same ensemble+softmax+argmax inference semantics. The biggest issue is the ImageNet-1000 fallback mapping: converting the expected class index into a hard 5-class one-hot is essentially arbitrary and discards useful uncertainty, so I replace it with a smooth, deterministic mapping from the scalar severity to a 5-class probability distribution (still derived only from the model’s softmax). I also apply this same smooth mapping under TTA (flip) to keep behavior consistent and stable, without changing model architecture, training, or the overall prediction pipeline. Everything else (paths, loader, weights, CSV writing) stays the same.'
- What this solution (achieved 0.11721) has done: 'Your current score is far below the target, and the biggest likely cause is that the ensemble weights dataset is missing so you’re effectively using the ImageNet-1000 fallback mapping, which is only weakly correlated with DR severity. To move the score toward the target with minimal change and without altering the model/inference “softmax → ensemble average → argmax” semantics, I keep the same pipeline but add a small on-the-fly fine-tuning step for the fallback model using the provided `train.csv` (same architecture, standard CE loss). This makes the fallback produce DR-relevant logits while preserving your overall approach and still finishing within the time budget by training only the classifier head for 1 short epoch. I also keep your flip TTA and submission-writing logic unchanged.'

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
import cv2




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
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext50_32x4d.pth",
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
missing = [k for k in model_paths.keys() if k not in available_model_paths]
if len(missing) > 0:
    print("Warning: missing model weights; these models will be skipped:", missing)

models_list = []
loaded_model_keys = []

for model_key, path in available_model_paths.items():
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

fallback_is_imagenet1000 = False

fallback_temperature = 1.0

if len(models_list) == 0:
    fallback_key = "resnet18"
    print(
        "Warning: no ensemble weights found. Falling back to pretrained resnet18 and lightly fine-tuning for 5-class DR."
    )

    model = timm.create_model(model_names[fallback_key], pretrained=True, num_classes=5)
    model.to(device)

    train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
    full_train_df = pd.read_csv(train_csv_file)

    rng = np.random.RandomState(42)
    idx = np.arange(len(full_train_df))
    rng.shuffle(idx)
    val_size = max(200, int(0.10 * len(full_train_df)))
    val_idx = idx[:val_size]
    trn_idx = idx[val_size:]

    trn_df = full_train_df.iloc[trn_idx].reset_index(drop=True)
    val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

    trn_csv_tmp = "train_split.csv"
    val_csv_tmp = "val_split.csv"
    trn_df.to_csv(trn_csv_tmp, index=False)
    val_df.to_csv(val_csv_tmp, index=False)

    train_dataset = BlindnessDataset(
        trn_csv_tmp, train_root_dir, transform=transform, test=False
    )
    val_dataset = BlindnessDataset(
        val_csv_tmp, train_root_dir, transform=transform, test=False
    )

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.get_classifier().parameters(), lr=3e-4, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    model.eval()
    model.get_classifier().train()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    for images, labels in tqdm(
        train_loader, desc="Finetuning fallback head", leave=False
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()

    def _qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        O = np.zeros((n_classes, n_classes), dtype=np.float64)
        for a, b in zip(y_true, y_pred):
            if 0 <= a < n_classes and 0 <= b < n_classes:
                O[a, b] += 1.0

        act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
        pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
        E = np.outer(act_hist, pred_hist)
        E = E / E.sum() * O.sum() if E.sum() > 0 else E

        W = np.zeros((n_classes, n_classes), dtype=np.float64)
        for i in range(n_classes):
            for j in range(n_classes):
                W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

        num = (W * O).sum()
        den = (W * E).sum()
        return 1.0 - num / den if den > 0 else 0.0

    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc="Calibrating temperature", leave=False
        ):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            val_logits_list.append(logits.detach().cpu())
            val_labels_list.append(labels.cpu())
    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0).numpy()

    temps = [0.7, 0.85, 1.0, 1.15, 1.3]
    best_t = 1.0
    best_k = -1e9
    for t in temps:
        probs = torch.softmax(val_logits / t, dim=1).numpy()
        preds = probs.argmax(axis=1).astype(int)
        k = _qwk(val_labels, preds, n_classes=5)
        if k > best_k:
            best_k = k
            best_t = float(t)

    fallback_temperature = best_t
    print(
        f"Fallback temperature selected: {fallback_temperature} (val QWK={best_k:.5f})"
    )

    models_list = [model]
    loaded_model_keys = [fallback_key]
    fallback_is_imagenet1000 = False  # now a true 5-class model



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "inception_resnet_v2": 0.880,  # 0.822,
    "inception_v4": 0.902,  # 0.888,
    "seresnext50_32x4d": 0.777,  # 0.709,
    "seresnext101_32x4d": 0.9697,  # 0.951
}

active_scores = {k: validation_scores.get(k, 1.0) for k in loaded_model_keys}
total_score = float(sum(active_scores.values()))
weights = {
    k: (v / total_score if total_score > 0 else 1.0 / len(active_scores))
    for k, v in active_scores.items()
}




## === cell 7
def _imagenet1000_to_dr5_probs(logits: torch.Tensor) -> torch.Tensor:
    probs1000 = nn.functional.softmax(logits, dim=1)  # [B,1000]
    idx = torch.arange(
        probs1000.shape[1], device=probs1000.device, dtype=probs1000.dtype
    )  # [1000]
    expected = (probs1000 * idx).sum(dim=1)  # [B]
    s = expected / (probs1000.shape[1] - 1)  # [B] in [0,1]

    sev = 4.0 * s  # [B]

    centers = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0], device=sev.device, dtype=sev.dtype
    )  # [5]

    sigma = torch.tensor(0.85, device=sev.device, dtype=sev.dtype)
    dist2 = (sev[:, None] - centers[None, :]) ** 2  # [B,5]
    unnorm = torch.exp(-dist2 / (2.0 * sigma * sigma))  # [B,5]
    probs5 = unnorm / unnorm.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs5


def _predict_probs(images: torch.Tensor) -> torch.Tensor:
    per_model = []
    for model_key, model in zip(loaded_model_keys, models_list):
        logits = model(images)

        if fallback_is_imagenet1000:
            probs = _imagenet1000_to_dr5_probs(logits)
        else:
            if len(models_list) == 1 and model_key == "resnet18":
                probs = nn.functional.softmax(
                    logits / float(fallback_temperature), dim=1
                )
            else:
                probs = nn.functional.softmax(logits, dim=1)

        per_model.append(weights[model_key] * probs)

    return torch.stack(per_model, dim=0).sum(dim=0)


all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)

        probs_a = _predict_probs(images)
        probs_b = _predict_probs(
            torch.flip(images, dims=[3])
        )  # horizontal flip (W dimension)
        weighted_outputs = 0.5 * (probs_a + probs_b)

        all_outputs.append(weighted_outputs.cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 8
test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(final_predictions) != len(test_ids):
    raise RuntimeError(
        f"Prediction length {len(final_predictions)} != test length {len(test_ids)}"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
