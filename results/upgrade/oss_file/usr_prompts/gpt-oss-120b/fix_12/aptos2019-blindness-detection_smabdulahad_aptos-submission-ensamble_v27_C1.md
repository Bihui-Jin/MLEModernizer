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

0.8483850813296373

# 6. Current score

0.75735

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.71738) has done: 'I fixed the script by handling missing pretrained checkpoints, adding a fallback that builds an ImageNet‑pretrained model and fine‑tunes it briefly on the available training data. The inference code is simplified to work with a single model, and the submission file is reliably written even when the original ensemble files are absent.'
- What this solution (achieved 0.74989) has done: 'I keep the overall pipeline unchanged but replace the hard arg‑max decision with a calibrated expected‑value prediction (rounded to the nearest integer). Using the soft probabilities from the ensemble and converting them to an expected class label usually yields higher quadratic weighted‑kappa scores, moving the result toward the target without altering the model architecture or training procedure.'
- What this solution (achieved 0.75569) has done: 'I add a lightweight validation split to compute a quadratic weighted kappa for each loaded model and use those scores as ensemble weights (instead of equal weights). This small change keeps the original architecture and training untouched, but gives the ensemble a better calibrated weighting, which should raise the overall quadratic weighted‑kappa toward the target score.'
- What this solution (achieved 0.73464) has done: 'I add a deterministic test‑time augmentation (horizontal flip) during inference and average the soft‑max probabilities from the original and flipped images before applying the ensemble weights. This low‑impact change usually boosts the quadratic weighted‑kappa without altering the core model or training pipeline, moving the score closer to the target.'
- What this solution (achieved 0.7489) has done: 'I add a light temperature‑scaling step that chooses, for each model, the temperature giving the highest quadratic weighted‑kappa on the small validation split. This calibration usually improves the expected‑value rounding used for the final predictions, nudging the score upward toward the target without changing the architecture or training loop.'
- What this solution (achieved 0.74427) has done: 'I add a lightweight post‑prediction calibration step: after selecting the best temperature for each model, I compute the weighted expected‑value predictions on a small validation split, fit a simple linear scaling (a·x + b) to the true labels, and then apply this scaling to the test‑time expected values before rounding. This modest calibration often nudges the quadratic weighted‑kappa upward toward the target without altering the model architecture or training process.'
- What this solution (achieved 0.74102) has done: 'I add a lightweight test‑time augmentation that averages model predictions over the original image, its horizontal flip, vertical flip, and both flips. This modest change keeps the core architecture and training untouched but usually yields a small boost in quadratic weighted‑kappa, moving the score upward toward the target. The same augmentation is applied during validation (used for weighting, temperature selection, and calibration) and during final test inference, ensuring the calibration remains consistent.'
- What this solution (achieved 0.74977) has done: 'I added a lightweight post‑processing step that searches for a small set of optimal decision thresholds on the validation expected values (using a greedy‑grid search). These thresholds replace the simple round‑after‑linear‑calibration step, which typically yields a higher quadratic weighted‑kappa. The rest of the pipeline—including model loading, TTA, ensemble weighting, temperature scaling, and calibration—remains unchanged, ensuring the core logic is preserved while nudging the score toward the target.'
- What this solution (achieved 0.74708) has done: 'I slightly improve the ensemble weighting by emphasizing stronger models (squaring their validation kappa‑based weights) and refine the decision thresholds with more passes and finer step sizes. These minor adjustments keep the core pipeline unchanged while nudging the quadratic weighted‑kappa closer to the target score.'
- What this solution (achieved 0.73919) has done: 'I replace the threshold‑based decision with a simple calibrated‑expected‑value rounding (clamped to 0‑4). This keeps all existing steps (ensemble weighting, temperature scaling, TTA, linear calibration) but uses a more direct mapping that usually yields a slightly higher quadratic weighted‑kappa, moving the score toward the target without altering the core architecture or training logic.'
- What this solution (achieved 0.75735) has done: 'I modify the inference step (cell 4) to actually use the optimized thresholds that were computed earlier, replacing the simple rounding of the calibrated expected value. By mapping the calibrated expectations to class labels with `np.digitize` and the learned thresholds, we obtain a slightly better calibrated decision rule, which should raise the quadratic weighted‑kappa toward the target while keeping the rest of the pipeline unchanged.'

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score




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
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

train_dataset = BlindnessDataset(
    train_csv_file, train_root_dir, transform=transform, test=False
)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth",
}
model_names = {
    "resnet18": "resnet18",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}

models_list = []
weights = {}
fallback_created = False
for key, path in model_paths.items():
    try:
        model = timm.create_model(model_names[key], pretrained=False, num_classes=5)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models_list.append((key, model))
        weights[key] = 1.0  # placeholder, will be re‑scaled later
    except FileNotFoundError:
        continue

if not models_list:
    fallback_key = "resnet18_fallback"
    model = timm.create_model("resnet18", pretrained=True, num_classes=5)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(2):
        epoch_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    model.eval()
    models_list.append((fallback_key, model))
    weights[fallback_key] = 1.0

if models_list:
    val_ratio = 0.1
    indices = np.arange(len(train_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=train_dataset.annotations["diagnosis"],
        random_state=42,
    )
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    val_loader = DataLoader(
        val_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    class_idxs = torch.arange(5, device=device, dtype=torch.float32)

    def tta_average_probs(model, imgs, temp):
        aug_imgs = [
            imgs,
            torch.flip(imgs, dims=[3]),  # horizontal flip
            torch.flip(imgs, dims=[2]),  # vertical flip
            torch.flip(torch.flip(imgs, dims=[3]), dims=[2]),  # both
        ]
        probs_sum = 0
        for a in aug_imgs:
            logits = model(a) / temp
            probs = nn.functional.softmax(logits, dim=1)
            probs_sum = probs_sum + probs
        return probs_sum / len(aug_imgs)

    for model_key, model in models_list:
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                probs = tta_average_probs(model, imgs, 1.0)
                expected = torch.sum(probs * class_idxs, dim=1)
                preds = torch.clamp(torch.round(expected), 0, 4).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(lbls.numpy())
        kappa = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        weights[model_key] = max(kappa, 0.01)

    weights = {k: (v**2) for k, v in weights.items()}
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    temps = {}
    temp_candidates = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    for model_key, model in models_list:
        best_temp = 1.0
        best_kappa = -np.inf
        model.eval()
        with torch.no_grad():
            for t in temp_candidates:
                all_preds = []
                all_true = []
                for imgs, lbls in val_loader:
                    imgs = imgs.to(device)
                    probs = tta_average_probs(model, imgs, t)
                    expected = torch.sum(probs * class_idxs, dim=1)
                    preds = torch.clamp(torch.round(expected), 0, 4).cpu().numpy()
                    all_preds.extend(preds)
                    all_true.extend(lbls.numpy())
                kappa = cohen_kappa_score(all_true, all_preds, weights="quadratic")
                if kappa > best_kappa:
                    best_kappa = kappa
                    best_temp = t
        temps[model_key] = best_temp

    val_expected = []
    val_true = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            weighted_sum = torch.zeros(imgs.size(0), 5, device=device)
            for model_key, model in models_list:
                temp = temps.get(model_key, 1.0)
                probs = tta_average_probs(model, imgs, temp)
                weighted_sum += probs * weights[model_key]
            expected = torch.sum(weighted_sum * class_idxs, dim=1)
            val_expected.extend(expected.cpu().numpy())
            val_true.extend(lbls.numpy())

    val_expected_np = np.array(val_expected)
    best_thresh = np.array([0.5, 1.5, 2.5, 3.5])

    def kappa_for_thresh(thresh):
        preds = np.digitize(val_expected_np, thresh)
        return cohen_kappa_score(val_true, preds, weights="quadratic")

    deltas = [-0.1, -0.05, 0.0, 0.05, 0.1]
    for _ in range(5):  # more iterations than the original 3
        improved = False
        for i in range(4):
            for delta in deltas:
                cand = best_thresh.copy()
                cand[i] = best_thresh[i] + delta
                if i > 0 and cand[i] <= cand[i - 1]:
                    continue
                if i < 3 and cand[i] >= cand[i + 1]:
                    continue
                if kappa_for_thresh(cand) > kappa_for_thresh(best_thresh):
                    best_thresh = cand
                    improved = True
        if not improved:
            break

    thresholds = best_thresh  # final thresholds to use

    if len(val_expected) > 0:
        a, b = np.polyfit(val_expected, val_true, 1)
    else:
        a, b = 1.0, 0.0
    calibration_a = float(a)
    calibration_b = float(b)
else:
    calibration_a = 1.0
    calibration_b = 0.0
    thresholds = np.array([0.5, 1.5, 2.5, 3.5])




## === cell 4
all_predictions = []
class_idxs = torch.arange(5, device=device, dtype=torch.float32)


def tta_average_probs_test(model, imgs, temp):
    aug_imgs = [
        imgs,
        torch.flip(imgs, dims=[3]),
        torch.flip(imgs, dims=[2]),
        torch.flip(torch.flip(imgs, dims=[3]), dims=[2]),
    ]
    probs_sum = 0
    for a in aug_imgs:
        logits = model(a) / temp
        probs = nn.functional.softmax(logits, dim=1)
        probs_sum = probs_sum + probs
    return probs_sum / len(aug_imgs)


with torch.no_grad():
    for images in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)

        weighted_preds = []
        for model_key, model in models_list:
            temp = temps.get(model_key, 1.0)

            probs = tta_average_probs_test(model, images, temp)

            weighted = probs * weights[model_key]
            weighted_preds.append(weighted)

        summed = torch.stack(weighted_preds).sum(dim=0)  # (B,5)
        expected = torch.sum(summed * class_idxs, dim=1)

        calibrated = expected * calibration_a + calibration_b

        calibrated_np = calibrated.cpu().numpy()
        preds_np = np.digitize(calibrated_np, thresholds)
        preds_np = np.clip(preds_np, 0, 4)  # safety clamp
        preds = torch.from_numpy(preds_np).to(device)

        all_predictions.append(preds.cpu().numpy())

final_predictions = np.concatenate(all_predictions, axis=0)




## === cell 5
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)
