# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

albumentations==2.0.8
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9996

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import time
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_INPUT = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing {TEST_DIR}"

print("Using device:", device)
print("Train CSV:", TRAIN_CSV)
print("Train dir:", TRAIN_DIR)
print("Test dir:", TEST_DIR)



## === cell 1
data_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

data_transforms_test = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)


def stratified_split_indices(y: np.ndarray, test_size=0.1, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0_val = int(round(len(idx0) * test_size))
    n1_val = int(round(len(idx1) * test_size))
    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    trn_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    rng.shuffle(val_idx)
    rng.shuffle(trn_idx)
    return trn_idx, val_idx


train_idx, valid_idx = stratified_split_indices(
    train_df["has_cactus"].values, test_size=0.1, seed=SEED
)
img_class_dict = dict(zip(train_df["id"].values, train_df["has_cactus"].values))

print(
    "Train rows:",
    len(train_df),
    "Train split:",
    len(train_idx),
    "Valid split:",
    len(valid_idx),
)




## === cell 3
class CactusDataset(Dataset):
    def __init__(
        self, datafolder, ids=None, datatype="train", transform=None, labels_dict=None
    ):
        self.datafolder = datafolder
        self.datatype = datatype
        self.transform = transform
        self.labels_dict = labels_dict if labels_dict is not None else {}

        if ids is None:
            self.image_files_list = sorted(
                [s for s in os.listdir(datafolder) if s.lower().endswith(".jpg")]
            )
        else:
            self.image_files_list = list(ids)

        if self.datatype == "train":
            self.labels = [
                np.float32(self.labels_dict[i]) for i in self.image_files_list
            ]
        else:
            self.labels = [np.float32(0.0) for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_id = self.image_files_list[idx]
        img_path = os.path.join(self.datafolder, img_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            out = self.transform(image=img)
            image = out["image"]
        else:
            image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, img_id




## === cell 4
train_ids = train_df.loc[train_idx, "id"].values
valid_ids = train_df.loc[valid_idx, "id"].values

train_set = CactusDataset(
    datafolder=TRAIN_DIR,
    ids=train_ids,
    datatype="train",
    transform=data_transforms,
    labels_dict=img_class_dict,
)
valid_set = CactusDataset(
    datafolder=TRAIN_DIR,
    ids=valid_ids,
    datatype="train",
    transform=data_transforms_test,
    labels_dict=img_class_dict,
)

sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["id"].values
test_set = CactusDataset(
    datafolder=TEST_DIR,
    ids=test_ids,
    datatype="test",
    transform=data_transforms_test,
    labels_dict=None,
)

batch_size = 512
num_workers = 0

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
valid_loader = DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

print("Loaders:", len(train_loader), len(valid_loader), len(test_loader))




## === cell 5
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Net(nn.Module):
    """
    Fix: replace missing 'pretrainedmodels' with torchvision DenseNet.
    Core logic preserved: DenseNet backbone -> Flatten -> BN -> Dropout -> Linear -> logits.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.2,
        arch: str = "densenet169",
        pretrained: str = "imagenet",
    ) -> None:
        super().__init__()

        if arch != "densenet169":
            raise ValueError(
                "This solution expects arch='densenet169' to match the original core logic."
            )

        if pretrained == "imagenet":
            weights = models.DenseNet169_Weights.IMAGENET1K_V1
        elif pretrained in (None, "none", ""):
            weights = None
        else:
            raise ValueError("Unsupported pretrained setting. Use 'imagenet' or None.")

        net = models.densenet169(weights=weights)

        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.pool(self.features(dummy)).view(1, -1)
            feat_dim = out.shape[1]

        self.head = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        logits = self.head(x)
        return torch.squeeze(logits, dim=-1)




## === cell 6
def roc_auc_score_numpy(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    scores_sorted = y_score[order]

    ranks = np.empty_like(scores_sorted, dtype=np.float64)
    i = 0
    r = 1.0
    n = len(scores_sorted)
    while i < n:
        j = i
        while j + 1 < n and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[i : j + 1] = avg_rank
        r += j - i + 1
        i = j + 1

    sum_ranks_pos = ranks[y_true_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)




## === cell 7
model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)


def run_one_epoch(model, loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_targets = []
    all_logits = []

    for x, y, _ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        losses.append(loss.detach().cpu().item())
        all_targets.append(y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_logits = np.concatenate(all_logits)
    auc = roc_auc_score_numpy(all_targets, all_logits)
    return float(np.mean(losses)), auc




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/280825812.py in <cell line: 0>()
----> 1 model = Net(num_classes=1).to(device)
      2 criterion = nn.BCEWithLogitsLoss()
      3 optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
      4 scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)
      5 

/tmp/ipykernel_11/3598675839.py in __init__(self, num_classes, p, arch, pretrained)
     43         with torch.no_grad():
     44             dummy = torch.zeros(1, 3, 32, 32)
---> 45             out = self.pool(self.features(dummy)).view(1, -1)
     46             feat_dim = out.shape[1]
     47 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/densenet.py in forward(self, init_features)
    120         features = [init_features]
    121         for name, layer in self.items():
--> 122             new_features = layer(features)
    123             features.append(new_features)
    124         return torch.cat(features, 1)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/densenet.py in forward(self, input)
     86             bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
     87         else:
---> 88             bottleneck_output = self.bn_function(prev_features)
     89 
     90         new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

/usr/local/lib/python3.11/dist-packages/torchvision/models/densenet.py in bn_function(self, inputs)
     47     def bn_function(self, inputs: List[Tensor]) -> Tensor:
     48         concated_features = torch.cat(inputs, 1)
---> 49         bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
     50         return bottleneck_output
     51 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/batchnorm.py in forward(self, input)
    191         used for normalization (i.e. in eval mode when buffers are not None).
    192         """
--> 193         return F.batch_norm(
    194             input,
    195             # If buffers are not to be tracked, ensure that they won't be updated

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
   2818         )
   2819     if training:
-> 2820         _verify_batch_size(input.size())
   2821 
   2822     return torch.batch_norm(

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in _verify_batch_size(size)
   2784         size_prods *= size[i + 2]
   2785     if size_prods == 1:
-> 2786         raise ValueError(
   2787             f"Expected more than 1 value per channel when training, got input size {size}"
   2788         )

ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 640, 1, 1])

## === cell 8
num_epochs = 10
best_state = None
best_auc = -1.0

start = time.time()
for epoch in range(1, num_epochs + 1):
    tr_loss, tr_auc = run_one_epoch(model, train_loader, train=True)
    va_loss, va_auc = run_one_epoch(model, valid_loader, train=False)

    scheduler.step(va_loss)

    if np.isfinite(va_auc) and va_auc > best_auc:
        best_auc = va_auc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    print(
        f"Epoch {epoch:02d}/{num_epochs} | train loss {tr_loss:.4f} auc {tr_auc:.5f} | valid loss {va_loss:.4f} auc {va_auc:.5f}"
    )

elapsed = time.time() - start
print(f"Training time: {elapsed:.1f}s, best valid AUC: {best_auc:.6f}")

if best_state is not None:
    model.load_state_dict(best_state)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/786144258.py in <cell line: 0>()
      5 start = time.time()
      6 for epoch in range(1, num_epochs + 1):
----> 7     tr_loss, tr_auc = run_one_epoch(model, train_loader, train=True)
      8     va_loss, va_auc = run_one_epoch(model, valid_loader, train=False)
      9 

NameError: name 'run_one_epoch' is not defined

## === cell 9
model.eval()
all_test_ids = []
all_test_probs = []

with torch.no_grad():
    for x, _y, ids in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_test_ids.extend(list(ids))
        all_test_probs.extend(list(probs))

sub = pd.DataFrame(
    {"id": all_test_ids, "has_cactus": np.array(all_test_probs, dtype=np.float64)}
)

sub = sample_sub[["id"]].merge(sub, on="id", how="left")
assert sub["has_cactus"].notna().all(), "Some test predictions are missing"

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Submission path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1042478979.py in <cell line: 0>()
----> 1 model.eval()
      2 all_test_ids = []
      3 all_test_probs = []
      4 
      5 with torch.no_grad():

NameError: name 'model' is not defined
