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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.911371255572809

# 6. Current score

0.14399

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the broken file paths and checkpoint loading, add a safe fallback that predicts the most common training label when a model checkpoint is missing, and ensure a proper submission CSV is written. This resolves the runtime errors and guarantees a valid submission file.'
- What this solution (achieved 0.14259) has done: 'I add a lightweight image‑based heuristic that replaces the constant fallback prediction when a model checkpoint is unavailable. By computing each image’s average brightness and assigning the class whose training‑set average brightness is closest, we obtain a non‑trivial prediction that should raise the quadratic weighted kappa from 0.0 toward the target 0.911 while keeping the original architecture and inference flow unchanged.'
- What this solution (achieved 0.16532) has done: 'I enhance the simple brightness‑based fallback by also considering the per‑class brightness standard deviation, which gives a richer similarity measure and should raise the quadratic weighted kappa toward the target. The core model loading logic stays unchanged, and the submission CSV is still written exactly as before.'
- What this solution (achieved 0.21736) has done: 'The changes combine the three separate image reads into a single read per image, use faster pandas iteration (`itertuples`) for the training statistics loop, and adapt the feature‑based prediction to use the new combined function. This reduces I/O and Python overhead while keeping the exact same calculations and fallback logic, allowing the script to finish well within the 600‑second limit.'
- What this solution (achieved 0.53248) has done: 'I keep the overall pipeline unchanged but replace the simple centroid‑based fallback with a lightweight k‑nearest‑neighbors classifier that uses the same image‑level features (brightness, brightness‑std, and mean RGB). The training features are collected once while scanning the training images, then each test image finds its 3 closest training examples and votes for the most common label. This preserves the original model‑loading logic, adds only a small, deterministic computation, and is expected to raise the quadratic weighted kappa from 0.217 toward the target 0.911 while still writing a valid `submission.csv`.'
- What this solution (achieved 0.62185) has done: 'I improve the lightweight k‑NN fallback by (1) standardising each of the five image‑level features using the training‑set statistics (so the distance metric treats all dimensions fairly) and (2) increasing the neighbourhood size to 5 instances, which generally gives a more robust majority vote. These tweaks keep the original architecture and inference flow unchanged while modestly boosting the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.14399) has done: 'I replace the k‑NN fallback with a nearest‑centroid classifier that uses the same standardized image‑level features (brightness, brightness‑std, mean RGB). After computing the training‑set feature statistics I also compute a prototype vector for each diagnosis class. During inference the test image’s feature vector is standardized and the class whose prototype is closest (Euclidean distance) is chosen. This change keeps the overall pipeline and model‑loading logic intact while providing a more discriminative fallback, which should raise the quadratic weighted‑kappa toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import collections
import math
import re
from functools import partial
from sklearn.metrics import cohen_kappa_score




## === cell 1
base_dir = os.path.join("..", "input", "aptos2019-blindness-detection")
train_csv = os.path.join(base_dir, "train.csv")
test_csv = os.path.join(base_dir, "test.csv")
train_img_dir = os.path.join(base_dir, "train_images")
test_img_dir = os.path.join(base_dir, "test_images")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)


class RetinaTestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["id_code"]
        path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_id


tfms = T.Compose(
    [
        T.Resize((456, 456)),
        T.CenterCrop(456),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_dataset = RetinaTestDataset(test_df, test_img_dir, tfms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


def compute_image_features(path):
    """
    Reads the image once and returns:
    - average grayscale brightness
    - brightness standard deviation
    - mean RGB values (R,G,B)
    All values are in the 0‑1 range.
    """
    img = Image.open(path)
    img_rgb = img.convert("RGB")
    arr_rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0
    rgb_mean = arr_rgb.mean(axis=(0, 1))
    img_gray = img.convert("L")
    arr_gray = np.asarray(img_gray, dtype=np.float32) / 255.0
    bright = arr_gray.mean()
    bright_std = arr_gray.std()
    return bright, bright_std, rgb_mean


train_brightness = {}
class_means = {}
class_stds = {}
class_rgb_means = {}

train_features = []  # shape (N_train, 5)
train_labels = []  # shape (N_train,)

print("Computing training image brightness and color statistics...")
for row in train_df.itertuples(index=False):
    img_path = os.path.join(train_img_dir, f"{row.id_code}.png")
    try:
        bright, bright_std, rgb_mean = compute_image_features(img_path)
    except Exception:
        continue
    label = int(row.diagnosis)

    train_brightness.setdefault(label, []).append((bright, bright_std))
    class_rgb_means.setdefault(label, []).append(rgb_mean)

    train_features.append(np.concatenate(([bright, bright_std], rgb_mean)))
    train_labels.append(label)

for label, vals in train_brightness.items():
    mean_vals = [v[0] for v in vals]
    std_vals = [v[1] for v in vals]
    class_means[label] = np.mean(mean_vals)
    class_stds[label] = np.mean(std_vals)

for label, rgbs in class_rgb_means.items():
    class_rgb_means[label] = np.mean(rgbs, axis=0)  # shape (3,)

train_features = np.array(train_features)  # (N_train, 5)
train_labels = np.array(train_labels)  # (N_train,)

feat_means = train_features.mean(axis=0)
feat_stds = train_features.std(axis=0) + 1e-6  # avoid division by zero
train_features = (train_features - feat_means) / feat_stds

class_prototypes = {}
for label in range(5):
    mask = train_labels == label
    if mask.any():
        class_prototypes[label] = train_features[mask].mean(axis=0)
    else:
        class_prototypes[label] = np.zeros(train_features.shape[1])

fallback_pred = int(train_df["diagnosis"].mode().iloc[0])  # safety mode


def predict_by_features(img_path, k=5):
    """
    Nearest‑centroid fallback (k is kept for API compatibility but unused).
    Features are standardised with the training‑set statistics before distance
    computation. The class whose prototype is closest in Euclidean distance is
    returned.
    """
    try:
        b, s, rgb = compute_image_features(img_path)
        test_feat = np.concatenate(([b, s], rgb))
    except Exception:
        return fallback_pred

    test_feat = (test_feat - feat_means) / feat_stds

    distances = np.array(
        [np.linalg.norm(test_feat - class_prototypes[label]) for label in range(5)]
    )
    best_label = int(np.argmin(distances))
    return best_label




## === cell 2
model_path = os.path.join("models", "abcdef.pth")
model = None
if os.path.exists(model_path):
    try:
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=5)
        state = torch.load(model_path, map_location="cpu")
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        print("Model checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        model = None
else:
    print("Model checkpoint not found; using enhanced nearest‑centroid fallback.")

all_ids = []
all_preds = []

if model is not None:
    with torch.no_grad():
        for imgs, ids in test_loader:
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_ids.extend(ids)
            all_preds.extend(preds)
else:
    for idx in range(len(test_df)):
        img_id = test_df.iloc[idx]["id_code"]
        img_path = os.path.join(test_img_dir, f"{img_id}.png")
        pred = predict_by_features(img_path, k=5)
        all_ids.append(img_id)
        all_preds.append(pred)

submission = pd.DataFrame(
    {"id_code": all_ids, "diagnosis": np.array(all_preds, dtype=int)}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
