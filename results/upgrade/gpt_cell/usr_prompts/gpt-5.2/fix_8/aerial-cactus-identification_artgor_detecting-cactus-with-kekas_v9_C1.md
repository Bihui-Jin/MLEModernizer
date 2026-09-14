# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import albumentations

try:
    import pretrainedmodels  # type: ignore
except ModuleNotFoundError:
    import torchvision.models as pretrainedmodels  # type: ignore

try:
    import adabound  # type: ignore  # kept to preserve original imports (not necessarily used)
except ModuleNotFoundError:
    adabound = None  # type: ignore

try:
    from kekas import Keker, DataOwner, DataKek  # type: ignore
    from kekas.transformations import Transformer, to_torch, normalize  # type: ignore
    from kekas.modules import Flatten  # type: ignore
except ModuleNotFoundError:
    Keker = DataOwner = DataKek = Transformer = to_torch = normalize = Flatten = None  # type: ignore


## === cell 1
BASE_DIR = "../input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"

labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

try:
    fig = plt.figure(figsize=(25, 8))
    train_imgs = os.listdir(TRAIN_DIR)
    for idx, img in enumerate(np.random.choice(train_imgs, 20, replace=False)):
        ax = fig.add_subplot(4, 20 // 4, idx + 1, xticks=[], yticks=[])
        im = Image.open(os.path.join(TRAIN_DIR, img))
        plt.imshow(im)
        lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]
        ax.set_title(f"Label: {lab}")
    plt.close(fig)
except Exception:
    pass



## === cell 2
test_img = os.listdir(TEST_DIR)
test_df = pd.DataFrame(test_img, columns=["id"])
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

_ = labels.head()
_ = labels.loc[labels["data_type"] == "train", "has_cactus"].value_counts()



## === cell 3
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=42
)




## === cell 4
def reader_fn(i, row):
    if row["data_type"] == "train":
        img_path = os.path.join(TRAIN_DIR, row["id"])
    else:
        img_path = os.path.join(TEST_DIR, row["id"])

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {img_path}")
    image = image[:, :, ::-1]  # BGR -> RGB

    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}




## === cell 5
def augs(p=0.5):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.75
            ),
            albumentations.HueSaturationValue(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
        ],
        p=p,
    )


if Transformer is None:

    class Transformer:  # type: ignore
        def __init__(self, key, fn):
            self.key = key
            self.fn = fn

        def __call__(self, sample):
            sample = dict(sample)
            sample[self.key] = self.fn(sample[self.key])
            return sample


if to_torch is None:

    def to_torch():  # type: ignore
        def _to_torch(x):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            if x.ndim == 2:
                x = x[:, :, None]
            x = torch.from_numpy(x.transpose(2, 0, 1)).float().div(255.0)
            return x

        return _to_torch


if normalize is None:

    def normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):  # type: ignore
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)

        def _normalize(x):
            if not isinstance(x, torch.Tensor):
                x = to_torch()(x)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            return (x - mean_t.to(x.device)) / std_t.to(x.device)

        return _normalize




def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])
    NRM_TFMS = transforms.Compose(
        [
            Transformer(dataset_key, to_torch()),
            Transformer(dataset_key, normalize()),
        ]
    )

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])

    return train_tfms, val_tfms


train_tfms, val_tfms = get_transforms("image", 32, 0.5)


## === cell 6
if DataKek is None:

    class DataKek(torch.utils.data.Dataset):  # type: ignore
        def __init__(self, df, reader_fn, transforms=None):
            self.df = df.reset_index(drop=True)
            self.reader_fn = reader_fn
            self.transforms = transforms

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i].to_dict()
            sample = self.reader_fn(i, row)
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample


if Flatten is None:

    class Flatten(nn.Module):  # type: ignore
        def forward(self, x):
            return x.view(x.size(0), -1)


train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = torch.utils.data.DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = torch.utils.data.DataLoader(
    val_dk, batch_size=batch_size, num_workers=workers, shuffle=False
)

test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = torch.utils.data.DataLoader(
    test_dk, batch_size=batch_size, num_workers=workers, shuffle=False
)


## === cell 7
class Net(nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.2,
        pooling_size: int = 2,
        last_conv_size: int = 1664,
        arch: str = "densenet169",
        pretrained: str = "imagenet",
    ) -> None:
        super().__init__()
        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # delete last layer
        modules += [
            nn.Sequential(
                Flatten(),
                nn.BatchNorm1d(1664),
                nn.Dropout(p),
                nn.Linear(1664, num_classes),
            )
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits




## === cell 8
if DataOwner is None:

    class DataOwner:  # type: ignore
        def __init__(self, train_dl, val_dl, test_dl):
            self.train_dl = train_dl
            self.val_dl = val_dl
            self.test_dl = test_dl


if Keker is None:

    class Keker:  # type: ignore
        def __init__(
            self,
            model,
            dataowner,
            criterion,
            step_fn,
            target_key="label",
            metrics=None,
            opt=None,
            opt_params=None,
        ):
            self.model = model
            self.dataowner = dataowner
            self.criterion = criterion
            self.step_fn = step_fn
            self.target_key = target_key
            self.metrics = metrics or {}
            self.opt = opt
            self.opt_params = opt_params or {}

        def unfreeze(self, model_attr=None):
            module = getattr(self.model, model_attr) if model_attr else self.model
            for p in module.parameters():
                p.requires_grad = True

        def freeze_to(self, layer_num, model_attr=None):
            module = getattr(self.model, model_attr) if model_attr else self.model
            children = list(module.children())
            for idx, child in enumerate(children):
                if idx <= layer_num:
                    for p in child.parameters():
                        p.requires_grad = False
                else:
                    for p in child.parameters():
                        p.requires_grad = True


dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()


def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"]
    return model(inp)


def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target_np = target.cpu().detach().numpy()
    preds_np = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target_np, preds_np)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target_np = target.cpu().detach().numpy()
    preds_np = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target_np, preds_np)


keker = Keker(
    model=model,
    dataowner=dataowner,
    criterion=criterion,
    step_fn=step_fn,
    target_key="label",
    metrics={"acc": bce_accuracy, "auc": roc_auc},
    opt=torch.optim.SGD,
    opt_params={"momentum": 0.99},
)


## === cell 9
keker.unfreeze(model_attr="net")

layer_num = -1
keker.freeze_to(layer_num, model_attr="net")



## === cell 10
if DataOwner is None:

    class DataOwner:  # type: ignore
        def __init__(self, train_dl, val_dl, test_dl):
            self.train_dl = train_dl
            self.val_dl = val_dl
            self.test_dl = test_dl


if Keker is None:

    class Keker:  # type: ignore
        def __init__(
            self,
            model,
            dataowner,
            criterion,
            step_fn,
            target_key="label",
            metrics=None,
            opt=None,
            opt_params=None,
        ):
            self.model = model
            self.dataowner = dataowner
            self.criterion = criterion
            self.step_fn = step_fn
            self.target_key = target_key
            self.metrics = metrics or {}
            self.opt = opt
            self.opt_params = opt_params or {}

        def unfreeze(self, model_attr=None):
            module = getattr(self.model, model_attr) if model_attr else self.model
            for p in module.parameters():
                p.requires_grad = True

        def freeze_to(self, layer_num, model_attr=None):
            module = getattr(self.model, model_attr) if model_attr else self.model
            children = list(module.children())
            for idx, child in enumerate(children):
                if idx <= layer_num:
                    for p in child.parameters():
                        p.requires_grad = False
                else:
                    for p in child.parameters():
                        p.requires_grad = True

        def kek_one_cycle(
            self,
            max_lr,
            cycle_len,
            momentum_range=(0.95, 0.85),
            div_factor=25,
            increase_fraction=0.3,
            logdir=None,
        ):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            base_lr = float(max_lr) / float(div_factor) if div_factor else float(max_lr)
            opt_cls = self.opt if self.opt is not None else torch.optim.SGD
            self.optimizer = opt_cls(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=base_lr,
                **self.opt_params,
            )

            total_steps = max(1, int(cycle_len) * max(1, len(self.dataowner.train_dl)))
            self._kek_history = {"train_loss": [], "val_loss": [], "metrics": []}

            def _set_lr(lr_val: float):
                for pg in self.optimizer.param_groups:
                    pg["lr"] = float(lr_val)

            def _set_momentum(m_val: float):
                for pg in self.optimizer.param_groups:
                    if "momentum" in pg:
                        pg["momentum"] = float(m_val)
                    elif "betas" in pg:
                        b1, b2 = pg["betas"]
                        pg["betas"] = (float(m_val), b2)

            def _lr_for_step(t: int) -> float:
                inc_steps = int(total_steps * float(increase_fraction))
                inc_steps = max(1, min(total_steps, inc_steps))
                if t < inc_steps:
                    return base_lr + (float(max_lr) - base_lr) * (t / inc_steps)
                denom = max(1, total_steps - inc_steps)
                return float(max_lr) - (float(max_lr) - base_lr) * (
                    (t - inc_steps) / denom
                )

            def _mom_for_step(t: int) -> float:
                m_hi, m_lo = float(momentum_range[0]), float(momentum_range[1])
                inc_steps = int(total_steps * float(increase_fraction))
                inc_steps = max(1, min(total_steps, inc_steps))
                if t < inc_steps:
                    return m_hi - (m_hi - m_lo) * (t / inc_steps)
                denom = max(1, total_steps - inc_steps)
                return m_lo + (m_hi - m_lo) * ((t - inc_steps) / denom)

            global_step = 0
            for _epoch in range(int(cycle_len)):
                self.model.train()
                train_losses = []

                for batch in self.dataowner.train_dl:
                    batch = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }
                    target = batch[self.target_key].to(device)

                    lr_t = _lr_for_step(global_step)
                    mom_t = _mom_for_step(global_step)
                    _set_lr(lr_t)
                    _set_momentum(mom_t)

                    self.optimizer.zero_grad(set_to_none=True)
                    preds = self.step_fn(self.model, batch)
                    loss = self.criterion(preds, target)
                    loss.backward()
                    self.optimizer.step()

                    train_losses.append(float(loss.detach().cpu().item()))
                    global_step += 1

                self.model.eval()
                val_losses = []
                metric_vals = {name: [] for name in self.metrics.keys()}
                with torch.no_grad():
                    for batch in self.dataowner.val_dl:
                        batch = {
                            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                            for k, v in batch.items()
                        }
                        target = batch[self.target_key].to(device)
                        preds = self.step_fn(self.model, batch)
                        loss = self.criterion(preds, target)
                        val_losses.append(float(loss.detach().cpu().item()))
                        for name, fn in self.metrics.items():
                            try:
                                metric_vals[name].append(float(fn(target, preds)))
                            except Exception:
                                pass

                self._kek_history["train_loss"].append(
                    float(np.mean(train_losses)) if train_losses else float("nan")
                )
                self._kek_history["val_loss"].append(
                    float(np.mean(val_losses)) if val_losses else float("nan")
                )
                self._kek_history["metrics"].append(
                    {
                        k: (float(np.mean(v)) if v else float("nan"))
                        for k, v in metric_vals.items()
                    }
                )

            return self._kek_history

        def plot_kek(self, logdir=None):
            hist = getattr(self, "_kek_history", None)
            if not hist:
                return
            try:
                plt.figure(figsize=(10, 4))
                plt.plot(hist.get("train_loss", []), label="train_loss")
                plt.plot(hist.get("val_loss", []), label="val_loss")
                plt.legend()
                plt.tight_layout()
                plt.close()
            except Exception:
                pass


dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()


def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"]
    return model(inp)


def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target_np = target.cpu().detach().numpy()
    preds_np = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target_np, preds_np)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target_np = target.cpu().detach().numpy()
    preds_np = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target_np, preds_np)


keker = Keker(
    model=model,
    dataowner=dataowner,
    criterion=criterion,
    step_fn=step_fn,
    target_key="label",
    metrics={"acc": bce_accuracy, "auc": roc_auc},
    opt=torch.optim.SGD,
    opt_params={"momentum": 0.99},
)


## === cell 11
keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=4,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
    logdir="train_logs1",
)
try:
    keker.plot_kek("train_logs1")
except Exception:
    pass



## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2747633630.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m keker.kek_one_cycle(
[0m[1;32m      2[0m     [0mmax_lr[0m[0;34m=[0m[0;36m1e-3[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mcycle_len[0m[0;34m=[0m[0;36m4[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mmomentum_range[0m[0;34m=[0m[0;34m([0m[0;36m0.95[0m[0;34m,[0m [0;36m0.85[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mdiv_factor[0m[0;34m=[0m[0;36m25[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Keker' object has no attribute 'kek_one_cycle'

## === cell 12
preds = keker.predict_loader(loader=test_dl)

preds = np.asarray(preds).reshape(-1)
probs = 1.0 / (
    1.0 + np.exp(-preds)
)  # sigmoid in numpy (stable enough for typical logits here)

submission = pd.DataFrame(
    {"id": test_df["id"].values, "has_cactus": probs.astype(np.float32)}
)

sample = pd.read_csv(SAMPLE_SUB)
submission = sample[["id"]].merge(submission, on="id", how="left")

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
