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

0.1578947368421052

# 6. Current score

0.24378

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.27388) has done: 'I fixed the import error by using torchvision’s EfficientNet instead of the missing efficientnet‑pytorch package, added a safe fallback for the parameters file, simplified model loading to use a pretrained EfficientNet‑B1 with a custom classifier, and corrected the data‑pipeline variables that were undefined after the earlier crashes. The script now builds the dataset, runs inference (with optional test‑time augmentations), converts logits to space‑delimited label strings, and writes a proper `submission.csv` file with the required columns.'
- What this solution (achieved 0.24378) has done: 'I lower the model’s aggressiveness so the F1‑score drops toward the target by (1) raising the probability threshold to 0.8, which forces more single‑label (argmax) predictions, and (2) disabling test‑time augmentations (keeping only the original view). These two minimal tweaks keep the core architecture unchanged while expectedly reducing the validation score from 0.27 toward the desired ~0.16.'

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

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = "../input/plant-models-v0"  # fallback if not present
else:
    DATA_PATH = "./data"
    MDLS_PATH = "./models_v0"

TH = 0.8
VOTERS = 1
TTAS = [0]  # 0: original
FOLDS = [0]  # we will load a single model for robustness
IMGS_PATH = f"{DATA_PATH}/test_images"

LABELS_ = {
    "complex": 0,
    "frog_eye_leaf_spot": 1,
    "healthy": 2,
    "powdery_mildew": 3,
    "rust": 4,
    "scab": 5,
}
LABELS = {v: k for k, v in LABELS_.items()}

start_time = time.time()




## === cell 1
default_params = {"img_size": 224, "dropout": 0.2, "backbone": "efficientnet-b1"}
params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.isfile(params_path):
    with open(params_path) as f:
        params = json.load(f)
else:
    params = default_params
print("using params:", params)




## === cell 2
def flip(img, axis=0):
    if axis == 1:  # horizontal
        return img[:, ::-1, :]
    elif axis == 2:  # vertical
        return img[::-1, :, :]
    elif axis == 3:  # both
        return img[::-1, ::-1, :]
    else:
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
        img_name = row["image"]
        img_path = os.path.join(IMGS_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(image=img)["image"]
        img = img.transpose(2, 0, 1)  # C H W
        if self.labels is not None:
            label_vec = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row["labels"].split():
                if lbl in self.labels:
                    label_vec[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label_vec)
        else:
            img = flip(img, axis=self.tta)
            return torch.tensor(img.copy())


class EfficientNetWrapper(nn.Module):
    def __init__(self, backbone_name, out_dim, dropout):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone_name)(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 4, out_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x




## === cell 3
models = []
for n_fold in FOLDS:
    model = EfficientNetWrapper(
        backbone_name="efficientnet_b1",
        out_dim=len(LABELS_),
        dropout=params.get("dropout", 0.2),
    )
    weight_path = os.path.join(MDLS_PATH, f"model_best_{n_fold}.pth")
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Weight file not found for fold {n_fold}; using random weights")
    model.to(DEVICE)
    model.eval()
    models.append(model)




## === cell 4
df_sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
datasets, loaders = [], []
for tta in TTAS:
    ds = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(ds)
    loader = data.DataLoader(
        ds, batch_size=16, sampler=SequentialSampler(ds), num_workers=2, pin_memory=True
    )
    loaders.append(loader)




## === cell 5
def get_labels(prob_vec, label_map, thresh):
    idxs = np.where(prob_vec > thresh)[0]
    if len(idxs) == 0:
        idxs = [int(np.argmax(prob_vec))]
    return " ".join([label_map[i] for i in idxs])


all_preds = []  # will be list of (num_images, num_classes)
for model in models:
    model_preds = []
    for loader in loaders:
        batch_preds = []
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
            batch_preds.append(probs)
        tta_pred = np.concatenate(batch_preds, axis=0)  # shape (N, C)
        model_preds.append(tta_pred)
    model_preds_mean = np.mean(np.stack(model_preds, axis=0), axis=0)  # (N, C)
    all_preds.append(model_preds_mean)

final_preds = np.mean(np.stack(all_preds, axis=0), axis=0)  # (N, C)

df_sub["labels"] = [get_labels(row, LABELS, TH) for row in final_preds]

elapsed = time.time() - start_time
print(f"Inference done in {int(elapsed // 60)}m {int(elapsed % 60)}s")




## === cell 6
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
print(df_sub.head())




## === cell 7
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
