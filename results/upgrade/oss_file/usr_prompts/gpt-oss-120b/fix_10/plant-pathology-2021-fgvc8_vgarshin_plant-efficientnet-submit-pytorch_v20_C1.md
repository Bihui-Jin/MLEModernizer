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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7925577100646358

# 6. Current score

0.30565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24249) has done: 'I filter the test image list to include only actual image files (avoiding crashes on non‑image entries) and add a short safety check when loading each image so that a missing file won’t break the pipeline. These minimal adjustments guarantee that the notebook runs end‑to‑end and writes a valid `submission.csv`, moving the solution from “no score” to a usable submission.'
- What this solution (achieved 0.24533) has done: 'I lower the default per‑class thresholds (so more labels are emitted) and add a few simple test‑time augmentations (horizontal/vertical flips) that are already supported by the `flip` helper. The code now builds a loader for each TTA value, runs the model on all of them, averages the logits across both TTAs and folds, and then applies the lower thresholds. These minimal changes keep the original architecture and training untouched while giving a noticeably better F1‑score, moving the result closer to the target.'
- What this solution (achieved 0.24507) has done: 'I adjust the fallback EfficientNet loader to use ImageNet‑pretrained weights (so a reasonable model is always available even when the competition checkpoints are missing) and keep the original 0.5 class thresholds instead of forcing them to 0.3. These minimal changes preserve the overall pipeline while providing stronger predictions, moving the F1 score noticeably closer to the target.'
- What this solution (achieved 0.30565) has done: 'The fix ensures the data loader no longer returns tensors with negative strides by copying flipped images, sets more permissive class‑thresholds (0.3) to emit more labels, and improves the fallback heuristic: when the model cannot be loaded, non‑healthy images are assigned the two most frequent diseases instead of only the single most common one. These changes resolve the runtime error and give a better‑calibrated prediction, moving the F1 score toward the target while keeping the original architecture intact.'

# 9. Code solution

## === cell 0
import os, json, time, gc
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = "../input/plant-efficientnet-train-pytorch/models_v0"
else:
    DATA_PATH = "./data"
    MDLS_PATH = "./models_v0"

TEST = True
VER = "v0"
TTAS = [0, 1, 2, 3]
FOLDS = [0]  # single fold (model may be missing)
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()



## === cell 2
train_csv_path = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_csv_path)

all_labels = set()
for lst in train_df["labels"].astype(str):
    all_labels.update(lst.split())
all_labels = sorted(all_labels)
LABELS_ = {lbl: i for i, lbl in enumerate(all_labels)}  # label → index
LABELS = {i: lbl for lbl, i in LABELS_.items()}  # index → label

default_params = {
    "backbone": "efficientnet-b0",
    "dropout": 0.2,
    "img_size": 224,
    "batch_size": 32,
    "workers": 2,
}
if os.path.exists(os.path.join(MDLS_PATH, "params.json")):
    with open(os.path.join(MDLS_PATH, "params.json")) as f:
        params = json.load(f)
else:
    print("params.json not found – using defaults")
    params = default_params

ths_path = os.path.join(MDLS_PATH, "ths.json")
if os.path.exists(ths_path):
    with open(ths_path) as f:
        ths = json.load(f)
else:
    ths = {str(i): 0.5 for i in range(len(LABELS_))}

for k in ths:
    if ths[k] >= 0.5:
        ths[k] = 0.3
print("loaded params:", params)
print("thresholds loaded:", ths)

WORKERS = 2 if KAGGLE else params.get("workers", 2)




## === cell 3
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMGS_PATH, row["image"])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.labels is not None:
            lbl_vec = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row["labels"].split():
                lbl_vec[self.labels[lbl]] = 1.0
            return torch.tensor(img.transpose(2, 0, 1)), torch.tensor(lbl_vec)
        else:
            img = flip(img, axis=self.tta)
            img = np.ascontiguousarray(img)
            return torch.tensor(img.transpose(2, 0, 1))


try:
    from efficientnet_pytorch import model as enet
except Exception:

    class _EfficientNetWrapper:
        @staticmethod
        def EfficientNet_from_name(name):
            mapping = {
                "efficientnet-b0": torchvision.models.efficientnet_b0,
                "efficientnet-b1": torchvision.models.efficientnet_b1,
                "efficientnet-b2": torchvision.models.efficientnet_b2,
                "efficientnet-b3": torchvision.models.efficientnet_b3,
                "efficientnet-b4": torchvision.models.efficientnet_b4,
                "efficientnet-b5": torchvision.models.efficientnet_b5,
                "efficientnet-b6": torchvision.models.efficientnet_b6,
                "efficientnet-b7": torchvision.models.efficientnet_b7,
            }
            fn = mapping.get(name, torchvision.models.efficientnet_b0)
            model = fn(pretrained=True)
            return model

    enet = _EfficientNetWrapper


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super().__init__()
        backbone_name = params.get("backbone", "efficientnet-b0")
        self.enet = enet.EfficientNet_from_name(backbone_name)
        nc = (
            self.enet.classifier[1].in_features
            if hasattr(self.enet, "classifier")
            else self.enet.classifier.in_features
        )
        if hasattr(self.enet, "classifier"):
            self.enet.classifier = nn.Identity()
        else:
            self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(nc, nc // 4),
            nn.ReLU(),
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(nc // 4, out_dim),
        )

    def forward(self, x):
        x = self.enet(x)
        x = self.myfc(x)
        return x


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super().__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(pretrained=False)
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, nc // 4),
            nn.ReLU(),
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(nc // 4, out_dim),
        )
        self.rsnxt = nn.DataParallel(self.rsnxt)

    def forward(self, x):
        return self.rsnxt(x)




## === cell 4
models = []
for n_fold in FOLDS:
    try:
        if params.get("backbone") == "resnext":
            model = ResNext(params=params, out_dim=len(LABELS_))
        else:
            model = EffNet(params=params, out_dim=len(LABELS_))
        model_path = os.path.join(MDLS_PATH, f"model_best_{n_fold}.pth")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        models.append(model)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model for fold {n_fold}: {e}")
        if params.get("backbone") == "resnext":
            model = ResNext(params=params, out_dim=len(LABELS_))
        else:
            model = EffNet(params=params, out_dim=len(LABELS_))
        model.to(DEVICE).eval()
        models.append(model)
        print(f"Using fallback pretrained model for fold {n_fold}")

if not models:
    print("No models available – predictions will use label frequency fallback.")



## === cell 5
test_images = sorted([f for f in os.listdir(IMGS_PATH) if f.lower().endswith(".jpg")])
df_sub = pd.DataFrame({"image": test_images})
df_sub["labels"] = "healthy"  # default, will be overwritten if we have predictions

loaders_per_tta = []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    loader = data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
    )
    loaders_per_tta.append(loader)




## === cell 6
def get_labels(row_probs, label_map, thresholds):
    """
    Convert probability scores to a space‑delimited label string.
    Keep all labels whose probability exceeds the per‑class threshold.
    If no label passes the threshold, fall back to "healthy".
    """
    inds = [i for i, p in enumerate(row_probs) if p > thresholds.get(str(i), 0.3)]
    names = [label_map[i] for i in inds]
    if not names:
        return "healthy"
    return " ".join(names)


if models:
    all_logits = []  # list of (N, C) arrays
    with torch.no_grad():
        for model in models:
            tta_logits = []  # accumulate over TTAs for this model
            for loader in loaders_per_tta:
                batch_logits = []
                for imgs in loader:
                    imgs = imgs.to(DEVICE)
                    preds = torch.sigmoid(model(imgs)).cpu().numpy()
                    batch_logits.append(preds)
                batch_logits = np.vstack(batch_logits)  # (N, C)
                tta_logits.append(batch_logits)
            model_avg = np.mean(np.stack(tta_logits), axis=0)  # (N, C)
            all_logits.append(model_avg)
    avg_logits = np.mean(np.stack(all_logits), axis=0)  # (N, C)
    df_sub["labels"] = [get_labels(prob, LABELS, ths) for prob in avg_logits]
else:
    disease_counts = {}
    for lbls in train_df["labels"].astype(str):
        for lbl in lbls.split():
            if lbl != "healthy":
                disease_counts[lbl] = disease_counts.get(lbl, 0) + 1
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    top_two = [d for d, _ in sorted_diseases[:2]] if sorted_diseases else ["complex"]

    heuristic_labels = []
    for img_name in df_sub["image"]:
        img_path = os.path.join(IMGS_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            heuristic_labels.append("healthy")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        green_mean = img_rgb[:, :, 1].mean() / 255.0
        if green_mean > 0.4:
            heuristic_labels.append("healthy")
        else:
            heuristic_labels.append(" ".join(top_two))
    df_sub["labels"] = heuristic_labels

elapsed = time.time() - start_time
print(f"Finished in {int(elapsed // 60)}m {int(elapsed % 60)}s")



## === cell 7
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
