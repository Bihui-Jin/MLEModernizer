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

0.8065189289012012

# 6. Current score

0.38173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The script failed because `torch` was used before being imported and the submission dataframe was built from a directory listing, causing a row‑count mismatch with the expected Kaggle submission format. I added an early `import torch`, switched to loading the provided `sample_submission.csv` as the submission skeleton, and corrected label lookup by using the ordered label list instead of the label‑to‑index dict. These fixes restore proper execution and generate a correctly‑sized CSV file.'
- What this solution (achieved 0.24507) has done: 'I lower the default class‐thresholds a bit (so more labels are emitted) and add a simple fallback that, when no pretrained model files are found, predicts labels according to their frequency in the training set. This keeps the original architecture intact but gives a richer baseline than always “healthy”, moving the F1 score toward the target.'
- What this solution (achieved 0.30565) has done: 'I adjust the label‑conversion logic so it no longer forces the “healthy” label whenever a healthy prediction appears. Instead the function return whatever classes exceed the threshold and fall back to the most frequent training label (or “healthy” if none exceed). This change keeps the overall pipeline intact while likely raising recall and therefore moving the mean F1 score closer to the target. I also lower the fallback threshold to 0.05 to emit more labels in the frequency‑based fallback case.'
- What this solution (achieved 0.30565) has done: 'I keep the overall pipeline unchanged but make two small adjustments that should raise the mean F1 score:  
1) Add a simple horizontal‑flip augmentation for the non‑zero TTA indices so the model (when available) sees more varied inputs.  
2) Lower the fallback‑threshold used when no pretrained model is found from 0.05 to 0.02, which makes the frequency‑based predictions emit more labels and improves recall. These tweaks stay within the existing code structure and move the score toward the target.'
- What this solution (achieved 0.38173) has done: 'I keep the overall pipeline unchanged but improve the fallback‑only predictions (used when no pretrained model is found). Instead of relying on a very low probability threshold, the fallback now assign the 3 most probable classes for each image, which usually yields a better balance of recall and precision and therefore raises the mean F1 score toward the target.'
- What this solution (achieved 0.38173) has done: 'I lower the class‑thresholds to emit more labels and, when the threshold‑based prediction yields only the default “healthy” label, replace it with the top‑k most probable classes (k=3). This keeps the overall pipeline unchanged while increasing recall, which should raise the mean F1 toward the target score.'
- What this solution (achieved 0.30565) has done: 'The patch lowers the per‑class thresholds to 0.05 so more labels are emitted, and refines the fallback‑only branch: it now first applies threshold‑based multi‑label selection (using the same low thresholds) and only resorts to the top‑3 most probable classes when nothing exceeds a threshold. These minimal changes keep the original architecture while improving recall and therefore moving the mean F1 score closer to the target.'
- What this solution (achieved 0.38173) has done: 'I lowered the per‑class thresholds from 0.05 to 0.02 to emit more labels and simplified the fallback‑only path: when no pretrained model is found we now assign the three most probable classes for every image (instead of first using a high‑threshold rule and only replacing “healthy”). These minimal tweaks keep the original architecture intact while increasing recall, which should move the mean F1 closer to the target score.'
- What this solution (achieved 0.3327) has done: 'I slightly lower the per‑class probability thresholds (from 0.02 to 0.01) to emit more labels and increase recall, and I also expand the fallback top‑k replacement for “healthy” predictions from 3 to 5 labels. These minimal adjustments keep the original model architecture and inference flow unchanged while moving the mean F1 score closer to the target.'
- What this solution (achieved 0.38173) has done: 'I keep the overall pipeline unchanged but stop overwriting the loaded class‑thresholds with an overly‑low value (0.01) and use a smaller top‑k fallback (k=3). Keeping the original thresholds (or the 0.5 default) improves precision, while a tighter top‑k reduces noisy extra labels, both of which should raise the mean F1 toward the target score.'

# 9. Code solution

## === cell 0
KAGGLE = True
if not KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
import os, gc, json, time
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.utils.data as data
import torchvision
from torchvision import models, transforms
from torch.utils.data.sampler import SequentialSampler

try:
    from efficientnet_pytorch import model as enet
except Exception:
    import subprocess, sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "efficientnet-pytorch"]
    )
    from efficientnet_pytorch import model as enet



## === cell 2
TEST = True
VER = "v101"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"
TTAS = [0, 1, 2, 3]  # TTA indices; non‑zero will now trigger a horizontal flip
FOLDS = [0]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()



## === cell 3
default_params = {
    "img_size": 224,
    "batch_size": 32,
    "dropout": 0.2,
    "backbone": "efficientnet-b0",
    "workers": 2,
    "labels_": [],  # will be filled later
    "labels": {},  # will be filled later
}
try:
    with open(f"{MDLS_PATH}/params.json") as f:
        params = json.load(f)
except Exception:
    params = default_params

if not params.get("labels_"):
    train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
    unique_labels = set()
    for lbls in train_df["labels"]:
        unique_labels.update(lbls.split())
    label_list = sorted(list(unique_labels))
    params["labels_"] = label_list
    params["labels"] = {lbl: idx for idx, lbl in enumerate(label_list)}

LABELS_ = params["labels_"]
LABELS = params["labels"]

WORKERS = 2 if KAGGLE else params.get("workers", 2)

try:
    with open(f"{MDLS_PATH}/ths.json") as f:
        ths = json.load(f)
except Exception:
    ths = {str(i): 0.5 for i in range(len(LABELS_))}

print("Parameters loaded:", params)
print("Thresholds loaded (kept as is):", ths)



## === cell 4
sample_sub_path = f"{DATA_PATH}/sample_submission.csv"
df_sub = pd.read_csv(sample_sub_path)
df_sub.columns = ["image", "labels"]
df_sub["labels"] = "healthy"
print("Submission skeleton:")
display(df_sub.head())



## === cell 5
models = []
model_loaded = False
for n_fold in FOLDS:
    model_path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.exists(model_path):

        class ResNext(nn.Module):
            def __init__(self, params, out_dim):
                super().__init__()
                self.rsnxt = torchvision.models.resnext50_32x4d(pretrained=False)
                nc = self.rsnxt.fc.in_features
                self.rsnxt.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(nc, int(nc / 4)),
                    nn.ReLU(),
                    nn.Dropout(params["dropout"]),
                    nn.Linear(int(nc / 4), out_dim),
                )
                self.rsnxt = nn.DataParallel(self.rsnxt)

            def forward(self, x):
                return self.rsnxt(x)

        model = ResNext(params, out_dim=len(LABELS_))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval().to(DEVICE)
        models.append(model)
        model_loaded = True
        print(f"Loaded model from {model_path}")
if not model_loaded:
    print(
        "No pretrained model files found – proceeding with fallback frequency‑based predictions."
    )




## === cell 6
def get_labels(row, label_list, ths, default_label="healthy"):
    """
    Convert raw sigmoid outputs (row) into a space‑delimited label string.
    Returns default_label when no class exceeds its threshold.
    """
    try:
        idxs = [i for i, x in enumerate(row) if x > ths.get(str(i), 0.5)]
        if not idxs:
            return default_label
        names = [label_list[i] for i in idxs]
        return " ".join(names)
    except Exception as e:
        print("Error in get_labels:", e)
        return default_label


def get_topk_labels(row, label_list, k=3):
    """
    Return the k most probable labels (by probability) as a space‑delimited string.
    """
    try:
        idxs = np.argsort(row)[-k:]  # indices of top‑k probabilities
        names = [label_list[i] for i in reversed(idxs)]  # keep descending order
        return " ".join(names)
    except Exception as e:
        print("Error in get_topk_labels:", e)
        return ""


if models:

    class PlantDataset(data.Dataset):
        def __init__(self, df, size, tta=0):
            self.df = df.reset_index(drop=True)
            self.size = size
            self.tta = tta  # tta index, non‑zero -> horizontal flip

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_name = self.df.iloc[idx].image
            img_path = os.path.join(IMGS_PATH, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.tta % 2 == 1:
                img = cv2.flip(img, 1)
            img = cv2.resize(img, (self.size, self.size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img.copy())
            return img

    all_logits = []
    for tta in TTAS:
        dataset = PlantDataset(df_sub, size=params["img_size"], tta=tta)
        loader = data.DataLoader(
            dataset,
            batch_size=params["batch_size"],
            sampler=SequentialSampler(dataset),
            num_workers=WORKERS,
        )
        logits_tta = []
        for batch in loader:
            batch = batch.to(DEVICE)
            preds = [torch.sigmoid(m(batch)).cpu().numpy() for m in models]
            preds = np.mean(preds, axis=0)  # shape (batch, n_classes)
            logits_tta.append(preds)
        logits_tta = np.concatenate(logits_tta, axis=0)  # (N, n_classes)
        all_logits.append(logits_tta)

    mean_logits = np.mean(np.stack(all_logits, axis=0), axis=0)  # (N, n_classes)

    df_sub["labels"] = [get_labels(row, LABELS_, ths) for row in mean_logits]

    default_mask = df_sub["labels"] == "healthy"
    if default_mask.any():
        topk_labels = [
            get_topk_labels(row, LABELS_, k=3)
            for row in mean_logits[default_mask.values]
        ]
        df_sub.loc[default_mask, "labels"] = topk_labels

else:
    train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
    class_counts = np.zeros(len(LABELS_))
    for lbls in train_df["labels"]:
        for lbl in lbls.split():
            if lbl in LABELS:
                class_counts[LABELS[lbl]] += 1
    class_probs = class_counts / class_counts.sum()

    most_common_label = LABELS_[int(np.argmax(class_counts))]

    dummy_logits = np.tile(class_probs, (len(df_sub), 1))

    df_sub["labels"] = [get_topk_labels(row, LABELS_, k=3) for row in dummy_logits]

elapsed = time.time() - start_time
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())



## === cell 7
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
