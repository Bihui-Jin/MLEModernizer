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

3.9

# 3. Installed packages

albumentations==2.0.8
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

0.8612873980054397

# 6. Current score

0.68946

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the immediate runtime errors by replacing the removed `pytorch_lightning.metrics.functional.accuracy` import (it isn’t used here anyway) and ensuring `torchvision` is imported before the model is constructed. Since the external checkpoint path doesn’t exist in this environment, I switch to loading weights from `torchvision`’s built-in pretrained ResNet50 to keep the same architecture while producing reasonable predictions (and a score) instead of crashing. I also make the submission length match `sample_submission.csv` by iterating over that file exactly and writing `submission.csv` with the required columns. Finally, I add lightweight path fallback to `/kaggle/data/...` to match your provided filesystem.'
- What this solution (achieved 0.64985) has done: 'Your current score (0.10762) is far below the target (0.8613), and the main reason is that you’re using an ImageNet-pretrained ResNet50 with a *random* 5-class head (and heavy dropout), so predictions are essentially arbitrary for this task. To move the score toward the target while preserving your core architecture and inference flow, I add a minimal training step on `train.csv` to learn only the final `logits` layer (keeping your modified ResNet trunk frozen) and then use the trained head for test inference. This keeps the same model class, loss (CrossEntropy), and basic pipeline, but replaces random guessing with a task-adapted classifier. I also add standard deterministic seeding and a lightweight DataLoader to finish within the 600s budget, then write `submission.csv` in the exact same format as before.'
- What this solution (achieved 0.68797) has done: 'Main bottlenecks are CPU-side image loading/decoding (PIL per-sample), low DataLoader parallelism, and recomputing the frozen ResNet50 backbone every epoch even though only the final linear head is trained. To preserve core logic while cutting runtime, I cache the frozen 512-d embeddings for train/val once (same backbone, same transforms), then train the same `logits` layer on those cached features for the same number of epochs/optimizer/scheduler/loss. I also speed up I/O by using `torchvision.io.read_image` (libjpeg-turbo), increasing DataLoader workers with `persistent_workers/prefetch_factor`, and using `torch.inference_mode()` for feature extraction/inference while keeping determinism settings. Test-time still runs the same model end-to-end (no accuracy change), just with a faster image pipeline.'
- What this solution (achieved 0.68834) has done: 'Your current score (0.68797) is still well below the target (0.8613), so we should improve accuracy with the smallest changes that keep your exact model/backbone/head/loss and overall flow. The biggest issue is that your preprocessing resizes to 224 but does not center-crop (aspect distortion hurts ResNet), and you train only the linear head without any augmentation, which usually underfits this dataset. I keep the same frozen backbone + cached-embeddings + linear-head training, but (1) switch eval-time geometry to `Resize(256)->CenterCrop(224)` (standard ResNet) and (2) add lightweight train-time augmentation (RandomResizedCrop + horizontal flip) while keeping the same normalization. These are minimal, safe changes that typically move accuracy upward toward your target without changing the core approach.'
- What this solution (achieved 0.69357) has done: 'Your current score (0.688) is far below the target (0.861), so we should improve accuracy with minimal, low-risk changes that keep your frozen-ResNet50+linear-head approach intact. The biggest gain you can get without changing the core logic is to train the head on *deterministic* (non-augmented) cached embeddings, because right now you cache embeddings from a randomly augmented pipeline, which injects label-noise into the features and makes the linear head harder to fit. I keep your same model, loss, optimizer, scheduler, and epoch count, but (1) extract/caches train embeddings using the eval transform (Resize->CenterCrop) and (2) optionally use the train-time augmentation only as on-the-fly training would—since we’re caching, we should not randomize the cache. This should move accuracy upward toward your target while preserving evaluation semantics and runtime constraints, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.74925) has done: 'Your current score (0.69357) is far below the target (0.86129), so we should increase accuracy with the smallest changes that don’t alter your core “frozen ResNet50 backbone + cached embeddings + linear head trained with CrossEntropy” approach. The main low-risk issue is that you are training the head while the backbone is in `train()` mode during feature extraction, so BatchNorm uses batch statistics that differ from inference-time `eval()` behavior, harming generalization; we force the backbone to `eval()` for all embedding extraction and freeze its BN/Dropout behavior deterministically. Second, since you’re optimizing accuracy (not class-balanced accuracy), class-weighted CrossEntropy can reduce overall accuracy; we disable weights to better match the metric. Finally, we bump head-training epochs slightly (keeping the exact same optimizer/scheduler/loss family) to move the score upward without changing the model architecture or inference semantics.'
- What this solution (achieved 0.76196) has done: 'Your current gap to the target is large (0.74925 → 0.86129), so we should improve accuracy with minimal, low-risk changes that keep the same frozen-ResNet50 + cached-embeddings + linear head (CrossEntropy) core logic. The biggest issue is a feature mismatch: you train the linear head on embeddings produced by the backbone’s `fc` output **after Dropout+BN**, but at inference you call `lit_model(x)` which re-applies Dropout in `lit_model.model` unless the backbone is forced to `eval()`; we explicitly keep the backbone in `eval()` always for both caching and inference to eliminate stochasticity. Next, to better match the accuracy metric without changing the approach, we tune the head training slightly by (a) using a small validation-based model selection for the head weights (keep best epoch) and (b) using a cosine LR schedule (still SGD, same loss) to typically yield a modest uplift without touching architecture. Finally, we train the head on a larger split of data (reduce val to 5%) so the classifier sees more labeled examples, while still keeping a stratified holdout for selection.'
- What this solution (achieved 0.7074) has done: 'Your current score (0.76196) is still well below the target (0.86129), so we should make small, low-risk changes that improve generalization while preserving the same frozen-ResNet50 + cached-embeddings + linear-head (CrossEntropy) core approach. The biggest miss is that the backbone’s `fc` includes `BatchNorm1d(512)`, but you cache embeddings with a large batch size and then infer test embeddings with a different batch size; BN’s batch-dependent behavior shifts the feature distribution and hurts accuracy. We fix that by (1) extracting embeddings **one image at a time** (batch size 1) for train/val/test (BN becomes effectively deterministic and consistent) and (2) L2-normalizing the 512-d embeddings before the linear head during head training and inference (a minimal calibration step that often stabilizes linear classification on frozen features). These changes don’t alter your model architecture or loss and stay within the same pipeline, but should move accuracy upward toward your target.'
- What this solution (achieved 0.7074) has done: 'Your current score (0.7074) is still far below the target (0.8613), so we should make the smallest changes that legitimately improve accuracy without changing the core “frozen ResNet50 backbone + cached embeddings + linear head + CrossEntropy” approach. The biggest remaining issue is that the backbone’s embedding head contains `BatchNorm1d(512)`, so extracting embeddings in `train()` (even with batch size 1) can still introduce running-stat drift and mismatch; we explicitly freeze BN behavior by setting the backbone to `eval()` once and disabling BN running-stat updates. Next, we train the linear head on deterministic cached embeddings from the backbone but also include a small, low-risk validation-time logit temperature calibration (single scalar) to slightly improve argmax stability on the accuracy metric without changing labels or model structure. Finally, we keep submission ordering exactly aligned to `sample_submission.csv` and preserve all paths and the existing CSV writing.'
- What this solution (achieved 0.71001) has done: 'Your current score (0.7074) is far below the target (0.8613), so we should increase accuracy with the smallest safe changes that keep your exact frozen-ResNet50 + cached-embeddings + linear-head (CrossEntropy) approach intact. The most impactful low-risk issue is that you train the linear head on only 95% of the data (holding out 5%), but Kaggle test accuracy benefits from using all labeled data once you’ve selected hyperparameters—so we keep the 5% split only for model/temperature selection, then retrain the same logits layer on the full train set with the chosen temperature. Additionally, the BatchNorm1d inside `model.fc` can still cause subtle distribution mismatch; we make embeddings fully deterministic by forcing `model.fc[2]` (the BN) into pure “identity normalization” behavior (weight=1, bias=0, running stats fixed) and disabling its running-stat updates, while keeping the architecture unchanged. These are minimal changes focused on better generalization and consistency, and they still produce a valid `submission.csv` with the required format.'
- What this solution (achieved 0.71001) has done: 'Your current score (0.71001) is far below the target (0.86129), so we should improve accuracy with the smallest safe changes that keep your frozen-ResNet50 + cached-embeddings + linear-head (CrossEntropy) core approach intact. The biggest remaining mismatch is that you L2-normalize embeddings but train the head without any bias term, which often hurts linear separability; switching `logits` to `bias=True` is a minimal architectural tweak inside the same head that typically improves accuracy. Next, your “stabilize BN” routine currently forces the embedding BN to identity (running_mean=0, var=1), which can destroy useful ImageNet feature scaling; instead we should freeze BN (eval + no running stat updates) without overwriting its pretrained parameters/statistics. Finally, after selecting hyperparameters on the 5% split, we keep your full-data retrain but also apply the chosen temperature consistently at inference only (unchanged), ensuring submission ordering remains aligned to `sample_submission.csv`.'
- What this solution (achieved 0.71001) has done: 'Your current score (0.71001) is far below the target (0.86129), so we should increase accuracy with minimal, low-risk changes that keep the same frozen ResNet50 backbone + cached embeddings + linear head + CrossEntropy core approach intact. The most direct issue is that your embedding extractor includes Dropout and BatchNorm; currently Dropout is still active during embedding extraction because the model is never explicitly put into `eval()` before calling `model(x)`, which injects noise into cached features and harms the linear head’s fit. I force the backbone into deterministic inference mode for *all* embedding extraction by (1) putting the backbone in `eval()` and (2) temporarily disabling the Dropout layer inside `model.fc` during feature extraction (without changing architecture). I also stop disabling BN running stats via `track_running_stats=False` (that changes BatchNorm semantics); instead we keep BN in `eval()` to use its pretrained running stats consistently between train/val/test. These are small changes aimed at improving feature consistency and should move the score upward toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.68946) has done: 'Your current score (0.71001) is far below the target (0.86129), so we should improve accuracy with the smallest low-risk changes that keep the same frozen ResNet50 backbone + cached embeddings + linear head + CrossEntropy core intact. The biggest issue is that your cached embeddings come from the backbone’s modified `fc` block (Dropout/Linear/BN), which is a poorer “feature extractor” than the standard ResNet global-pooled 2048-d features; we cache the deterministic 2048-d pre-fc features and train a linear head on those instead (still the same architecture and training approach: frozen backbone, train only a linear layer). This also removes the need to fight BN/Dropout behavior in the embedding head, making train/test features match better. All paths, the overall flow (cache -> train head -> infer -> write submission.csv), and the final submission formatting remain unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from sklearn import model_selection




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 2
class LitModel(pl.LightningModule):
    def __init__(self, classify, n_cls=5, pretrained=False, t_data=None, v_data=None):
        super().__init__()
        self.classify = classify
        self.n_cls = n_cls
        self.pre_trained = pretrained
        self.model = self.modified_model()
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.1
        self.t_data = t_data
        self.v_data = v_data
        self.batch_size = 256

        self.logits = nn.Linear(2048, self.n_cls, bias=True)

    def forward(self, x):
        embeddings = self.model(x)
        if self.classify:
            logits = self.logits(embeddings)
            return logits
        else:
            return embeddings

    def modified_model(self):
        try:
            weights = (
                torchvision.models.ResNet50_Weights.DEFAULT
                if self.pre_trained
                else None
            )
            model = torchvision.models.resnet50(weights=weights)
        except Exception:
            model = torchvision.models.resnet50(pretrained=self.pre_trained)

        model.fc = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
        )
        return model




## === cell 3
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as TF


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def _read_rgb(self, img_path: str):
        img = read_image(img_path, mode=ImageReadMode.RGB)
        return TF.to_pil_image(img)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.img_dir, img_id)
        img = self._read_rgb(img_path)
        x = self.transform(img)
        if self.train:
            y = int(self.df.loc[idx, "label"])
            return x, y
        return x, img_id




## === cell 4
base_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
BASE = next((p for p in base_candidates if os.path.exists(p)), base_candidates[0])

train_csv_path = os.path.join(BASE, "train.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")
train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

train_df = pd.read_csv(train_csv_path)
sample_submission_df = pd.read_csv(sample_path)

preprocess_eval = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

preprocess_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 5
lit_model = LitModel(True, 5, pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model.to(device)

ckpt_path = "/kaggle/input/exp1-models/epoch9_best_val_acc.ckpt"
if os.path.exists(ckpt_path):
    dicts = torch.load(ckpt_path, map_location="cpu")
    state_dict = dicts.get("state_dict", dicts)
    lit_model.load_state_dict(state_dict, strict=False)



## === cell 6
lit_model.criterion = nn.CrossEntropyLoss()

for p in lit_model.model.parameters():
    p.requires_grad = False
for p in lit_model.logits.parameters():
    p.requires_grad = True

trn_idx, val_idx = model_selection.train_test_split(
    np.arange(len(train_df)),
    test_size=0.05,
    random_state=42,
    stratify=train_df["label"].values,
)
trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds_for_cache = CassavaDataset(trn_df, train_img_dir, preprocess_eval, train=True)
val_ds = CassavaDataset(val_df, train_img_dir, preprocess_eval, train=True)

cache_batch_size = 32 if torch.cuda.is_available() else 16
batch_size = 256 if torch.cuda.is_available() else 64
num_workers = min(8, os.cpu_count() or 2)

train_loader_for_cache = torch.utils.data.DataLoader(
    train_ds_for_cache,
    batch_size=cache_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=cache_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

optimizer = torch.optim.SGD(
    lit_model.logits.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
)

epochs = 14
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def _extract_embeddings_pre_fc(dataloader):
    backbone = lit_model.model
    backbone.eval()

    feats = []
    ys = []
    with torch.inference_mode():
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=True)

            x = backbone.conv1(xb)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            x = backbone.avgpool(x)
            x = torch.flatten(x, 1)  # (B, 2048)

            feats.append(x.detach().cpu())
            ys.append(yb.detach().cpu())

    feats = torch.cat(feats, dim=0).contiguous()
    ys = torch.cat(ys, dim=0).contiguous()
    return feats, ys


train_feats_cpu, train_y_cpu = _extract_embeddings_pre_fc(train_loader_for_cache)
val_feats_cpu, val_y_cpu = _extract_embeddings_pre_fc(val_loader)

train_feats_cpu = F.normalize(train_feats_cpu, p=2, dim=1)
val_feats_cpu = F.normalize(val_feats_cpu, p=2, dim=1)

feat_train_ds = torch.utils.data.TensorDataset(train_feats_cpu, train_y_cpu)
feat_val_ds = torch.utils.data.TensorDataset(val_feats_cpu, val_y_cpu)

feat_train_loader = torch.utils.data.DataLoader(
    feat_train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
feat_val_loader = torch.utils.data.DataLoader(
    feat_val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

best_val_acc = -1.0
best_state = None

lit_model.logits.train()
for epoch in range(epochs):
    for xb, yb in feat_train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = lit_model.logits(xb)
        loss = lit_model.criterion(logits, yb)
        loss.backward()
        optimizer.step()

    lit_model.logits.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for xb, yb in feat_val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = lit_model.logits(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    val_acc = correct / max(1, total)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in lit_model.logits.state_dict().items()
        }

    lit_model.logits.train()
    scheduler.step()

if best_state is not None:
    lit_model.logits.load_state_dict(best_state)


def _tune_temperature(logits_layer: nn.Module, val_loader_feats, device):
    logits_layer.eval()
    best_T = 1.0
    best_acc = -1.0
    Ts = [0.7, 0.85, 1.0, 1.2, 1.4]
    with torch.inference_mode():
        for T in Ts:
            correct = 0
            total = 0
            for xb, yb in val_loader_feats:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = logits_layer(xb) / T
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            acc = correct / max(1, total)
            if acc > best_acc:
                best_acc = acc
                best_T = T
    return float(best_T)


best_temperature = _tune_temperature(lit_model.logits, feat_val_loader, device)

full_train_ds_for_cache = CassavaDataset(
    train_df, train_img_dir, preprocess_eval, train=True
)
full_train_loader_for_cache = torch.utils.data.DataLoader(
    full_train_ds_for_cache,
    batch_size=cache_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)
full_feats_cpu, full_y_cpu = _extract_embeddings_pre_fc(full_train_loader_for_cache)
full_feats_cpu = F.normalize(full_feats_cpu, p=2, dim=1)

full_feat_ds = torch.utils.data.TensorDataset(full_feats_cpu, full_y_cpu)
full_feat_loader = torch.utils.data.DataLoader(
    full_feat_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

lit_model.logits.reset_parameters()
lit_model.logits.to(device)

optimizer_full = torch.optim.SGD(
    lit_model.logits.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
)
scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_full, T_max=epochs
)

lit_model.logits.train()
for epoch in range(epochs):
    for xb, yb in full_feat_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer_full.zero_grad(set_to_none=True)
        logits = lit_model.logits(xb)
        loss = lit_model.criterion(logits, yb)
        loss.backward()
        optimizer_full.step()
    scheduler_full.step()

lit_model.model.eval()
lit_model.logits.eval()
lit_model.eval()



## === cell 7
test_df = sample_submission_df.copy()

test_ds = CassavaDataset(test_df, test_img_dir, preprocess_eval, train=False)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32 if torch.cuda.is_available() else 16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

predictions = []
image_id = []

with torch.inference_mode():
    lit_model.model.eval()
    lit_model.logits.eval()
    backbone = lit_model.model
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)

        x = backbone.conv1(xb)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        x = backbone.avgpool(x)
        emb = torch.flatten(x, 1)

        emb = F.normalize(emb, p=2, dim=1)
        logits = lit_model.logits(emb) / best_temperature
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        predictions.extend(preds)
        image_id.extend(list(ids))



## === cell 8
assert len(predictions) == len(sample_submission_df), (
    len(predictions),
    len(sample_submission_df),
)
assert len(image_id) == len(sample_submission_df), (
    len(image_id),
    len(sample_submission_df),
)



## === cell 9
my_submission = pd.DataFrame({"image_id": image_id, "label": predictions})
my_submission = (
    my_submission.set_index("image_id").loc[sample_submission_df.image_id].reset_index()
)
my_submission.to_csv("submission.csv", index=False)

my_submission.head()
