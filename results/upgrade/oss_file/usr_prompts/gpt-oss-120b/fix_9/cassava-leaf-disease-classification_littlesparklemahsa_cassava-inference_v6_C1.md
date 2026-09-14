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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.14

# 3. Installed packages

albumentations==2.0.8
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8909035962526443

# 6. Current score

0.76009

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The script crashed because `torch.cuda.amp.autocast` does not accept a `device_type` argument. We switch to the top‑level `torch.autocast` which supports `device_type`, and we adjust the averaging denominator to use the actual number of loaded folds (so missing model files don’t break the logic). The rest of the pipeline stays unchanged, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight fallback that, when none of the checkpoint files are found, predicts the most frequent label from the training set instead of a uniform distribution. This uses the existing training CSV, requires no model changes, and is expected to raise the accuracy from a random‑guess level (~5 %) toward a more reasonable value, moving the score closer to the target.'
- What this solution (achieved 0.68161) has done: 'I keep the existing inference flow but add a smarter fallback when no model checkpoints are found. Instead of always predicting the majority class, the fallback now use a lightweight K‑Nearest‑Neighbour classifier built on features extracted from a pretrained ConvNeXt‑tiny backbone (pretrained on ImageNet). This leverages visual similarity between test and a subset of training images, which should lift the accuracy from ~0.61 toward the target while preserving the original architecture and training logic.'
- What this solution (achieved 0.73655) has done: 'I keep the overall pipeline unchanged but improve the K‑Nearest‑Neighbour fallback used when no model checkpoints are found.  
Instead of a single‑nearest‑neighbor vote, the code now take the top‑5 most similar training embeddings and use majority voting, which usually gives a more robust prediction. I also increase the training subset size from 2 000 to 5 000 images to provide richer neighbours while staying memory‑friendly. These small, targeted tweaks should raise the validation accuracy toward the target without altering the core model or training logic.'
- What this solution (achieved 0.73655) has done: 'I add a richer test‑time augmentation in the inference loop – besides the existing horizontal flip, I also compute predictions on a vertical flip and on both flips combined, then average all four predictions. This keeps the original model and ensemble logic untouched while giving the model more views of each image, which should raise the validation accuracy and move the score closer to the target.'
- What this solution (achieved 0.75112) has done: 'I increase the K‑Nearest‑Neighbour fallback’s training pool and use a slightly larger k so the fallback becomes more accurate when no model checkpoints are available. This change keeps the core model‑inference logic untouched, only adjusts the KNN parameters, and is expected to raise the validation accuracy toward the target score.'
- What this solution (achieved 0.76009) has done: 'I increase the fallback K‑Nearest‑Neighbour pool to use more training samples and a larger k so that when model checkpoints are unavailable the predictions become more accurate, moving the validation score upward toward the target. This keeps the core model‑inference pipeline unchanged while only adjusting the KNN parameters.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import autocast  # top‑level autocast that accepts device_type
from contextlib import nullcontext
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
import timm


class CFG:
    img_size = 384
    batch_size = 64
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    model_paths = [
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold0.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold1.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold2.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold3.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold4.pth",
    ]
    knn_max_samples = 15000  # was 10000


test_tfms = A.Compose(
    [
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted([str(p) for p in Path(folder).glob("*.jpg")])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_tfms(image=img)["image"]
        return img, os.path.basename(self.paths[idx])


class TrainKNNDataset(Dataset):
    """A subset of the training data for the KNN fallback."""

    def __init__(self, csv_path, img_folder, max_samples=5000):
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.df = df.head(max_samples)
        self.img_folder = img_folder
        self.paths = [
            os.path.join(self.img_folder, img_id) for img_id in self.df["image_id"]
        ]
        self.labels = self.df["label"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_tfms(image=img)["image"]
        label = int(self.labels[idx])
        return img, label


class CassavaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=5
        )

    def forward(self, x):
        return self.backbone(x)


def compute_knn_predictions(test_loader, device, k_neighbors=7):
    """
    Build a KNN classifier from a larger subset of the training set
    using a pretrained ConvNeXt‑tiny feature extractor.
    Uses top‑k voting (k_neighbors) instead of a single nearest neighbour.
    """
    feature_extractor = timm.create_model(
        "convnext_tiny", pretrained=True, num_classes=0
    ).to(device)
    feature_extractor.eval()

    knn_dataset = TrainKNNDataset(
        CFG.train_csv, CFG.train_dir, max_samples=CFG.knn_max_samples
    )
    knn_loader = DataLoader(
        knn_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    train_embeds = []
    train_labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(knn_loader, desc="Train embeddings", leave=False):
            imgs = imgs.to(device)
            with autocast("cuda") if device.type == "cuda" else nullcontext():
                feats = feature_extractor(imgs)
                feats = F.normalize(feats, dim=1)  # L2‑norm for cosine similarity
            train_embeds.append(feats.cpu())
            train_labels.append(lbls)
    train_embeds = torch.cat(train_embeds)  # (N_train, D)
    train_labels = torch.cat(train_labels)  # (N_train,)

    pred_labels = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Test KNN", leave=False):
            imgs = imgs.to(device)
            with autocast("cuda") if device.type == "cuda" else nullcontext():
                feats = feature_extractor(imgs)
                feats = F.normalize(feats, dim=1)
            sim = torch.mm(feats.cpu(), train_embeds.T)  # (B, N_train)

            topk_vals, topk_idx = torch.topk(
                sim, k=min(k_neighbors, train_embeds.shape[0]), dim=1
            )
            topk_labels = train_labels[topk_idx]  # (B, k)
            mode_res = torch.mode(topk_labels, dim=1)
            pred_labels.append(mode_res.values.cpu())
    final_knn = torch.cat(pred_labels).numpy()
    return final_knn


@torch.no_grad()
def inference():
    dataset = TestDataset(CFG.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    ensemble_preds = None
    loaded_folds = 0
    am_context = autocast("cuda") if CFG.device.type == "cuda" else nullcontext()

    for fold, path in enumerate(CFG.model_paths):
        if not os.path.isfile(path):
            print(f"⚠️  Model file not found for fold {fold}: {path} – skipping.")
            continue

        print(f"Loading fold {fold} → {os.path.basename(path)}")
        model = CassavaModel().to(CFG.device)

        state = torch.load(path, map_location=CFG.device)
        model.load_state_dict(state)
        model.eval()

        fold_preds = []
        for imgs, _ in tqdm(loader, leave=False, desc=f"Fold {fold} TTA"):
            imgs = imgs.to(CFG.device)
            with am_context:
                p1 = torch.softmax(model(imgs), dim=1)  # original
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)  # h‑flip
                p3 = torch.softmax(model(torch.flip(imgs, dims=[2])), dim=1)  # v‑flip
                p4 = torch.softmax(
                    model(torch.flip(torch.flip(imgs, dims=[2]), dims=[3]))
                )  # hv‑flip
                avg_pred = (p1 + p2 + p3 + p4) / 4.0
            fold_preds.append(avg_pred.cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        ensemble_preds = (
            fold_preds if ensemble_preds is None else ensemble_preds + fold_preds
        )
        loaded_folds += 1

        del model, state
        torch.cuda.empty_cache()

    if ensemble_preds is None:
        print("⚠️  No model weights loaded; using KNN fallback.")
        knn_labels = compute_knn_predictions(loader, CFG.device, k_neighbors=11)
        final_labels = knn_labels
        loaded_folds = 1  # placeholder to keep downstream logic consistent
    else:
        final_probs = ensemble_preds / loaded_folds
        final_labels = np.argmax(final_probs, axis=1)

    sub = pd.DataFrame(
        {
            "image_id": [os.path.basename(p) for p in dataset.paths],
            "label": final_labels,
        }
    )
    sub = sub.sort_values("image_id").reset_index(drop=True)
    sub.to_csv("submission.csv", index=False)

    print(f"\nSUBMISSION READY → {len(sub)} predictions")
    print(sub.head())
    return sub


inference()
