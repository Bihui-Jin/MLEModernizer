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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.0316052125468551

# 6. Current score

0.02814

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.02814) has done: 'I eliminate the expensive ZIP extraction by using the already‑extracted dataset directories, update the configuration paths accordingly, and speed up training with mixed‑precision (AMP) while keeping the same model, loss, and training loop. These changes remove the 10‑minute unzip bottleneck and accelerate GPU computation without altering any core algorithmic logic.'

# 9. Code solution

## === cell 1
pass



## === cell 2
import os, glob, random, math, pprint
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2

try:
    import timm
except ImportError:
    timm = None  # fallback to torchvision models
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.model_selection import KFold
import tqdm




## === cell 3
class Config:
    dog = 1
    cat = 0
    train_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train"
    test_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test"
    n_fold = 5
    num_workers = 4
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    early_stopping = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (224, 224)


cfg = Config()




## === cell 4
def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


seed_everything()




## === cell 5
def _load_image(path):
    img = cv2.imread(path)
    if img is None:
        from PIL import Image

        img = np.array(Image.open(path).convert("RGB"))
    return img


def _default_transform(img):
    img = cv2.resize(img, cfg.size)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img


class DC_Dataset(Dataset):
    def __init__(self, paths, valid=False):
        self.paths = paths
        self.valid = valid
        if A is not None:
            if not self.valid:
                self.transform = A.Compose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.5
                        ),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Normalize(),
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = _load_image(self.paths[idx])
        img = cv2.resize(img, cfg.size)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img


class TrainDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.float32)
        if A is not None:
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.5
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = _load_image(self.paths[idx])
        img = cv2.resize(img, cfg.size)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, self.labels[idx]


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-2, -1)).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"


class DC_Model(pl.LightningModule):
    def __init__(
        self, model_name="convnext_small", pretrained=True, num_batch=0, fold=0
    ):
        super().__init__()
        if timm is not None:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, global_pool=""
            )
            num_features = self.model.num_features
        else:
            self.model = models.resnet18(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        self.model.head = nn.Sequential(GeM(), nn.Linear(num_features, 1))
        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        pred = torch.sigmoid(out) > 0.5
        acc = (pred == label).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr, weight_decay=0.1)
        return optimizer


def collate(x):
    return x




## === cell 6
test_paths = glob.glob(os.path.join(cfg.test_dir, "**", "*.jpg"), recursive=True)
image_ids = [os.path.basename(p).split(".")[0] for p in test_paths]



## === cell 7
model_paths = glob.glob(
    "/kaggle/input/dogs-vs-cats-lightning/lightning_logs/version_*/checkpoints/*.ckpt"
)
pprint.pprint(model_paths)



## === cell 8
if not model_paths:
    train_image_paths, train_labels = [], []
    for subdir, label in [("cat", cfg.cat), ("dog", cfg.dog)]:
        folder = os.path.join(cfg.train_dir, subdir)
        for p in glob.glob(os.path.join(folder, "*.jpg")):
            train_image_paths.append(p)
            train_labels.append(label)
    if len(train_image_paths) == 0:
        raise RuntimeError("No training images found – check train_dir path.")
    train_dataset = TrainDataset(train_image_paths, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=True,
    )
    trained_model = DC_Model()
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar()],
        deterministic=False,
        precision=16,  # mixed‑precision for speed
    )
    trainer.fit(trained_model, train_loader)
else:
    trained_model = None  # placeholder when checkpoints exist



## === cell 9
test_dataset = DC_Dataset(test_paths, valid=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    persistent_workers=True,
)



## === cell 10
outputs_list = []
if model_paths:
    for cp in model_paths:
        model = DC_Model.load_from_checkpoint(cp)
        model.eval()
        model = model.to(cfg.device)
        batch_outputs = []
        with torch.no_grad():
            for img in tqdm.tqdm(test_loader):
                out = model(img.to(cfg.device))
                batch_outputs += out.tolist()
        outputs_list.append(batch_outputs)
else:
    model = trained_model
    model.eval()
    model = model.to(cfg.device)
    batch_outputs = []
    with torch.no_grad():
        for img in tqdm.tqdm(test_loader):
            out = model(img.to(cfg.device))
            batch_outputs += out.tolist()
    outputs_list.append(batch_outputs)



## === cell 11
outputs = torch.tensor(outputs_list)  # (n_models, n_test)
outputs = outputs.mean(dim=0)  # ensemble average
outputs = torch.sigmoid(outputs)  # convert logits to probabilities



## === cell 12
sample_sub_path = (
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path, dtype={"id": str})
id_to_pred = dict(zip(image_ids, outputs.tolist()))
submission = pd.DataFrame(
    {
        "id": sample_sub["id"],
        "label": sample_sub["id"].map(lambda x: id_to_pred.get(x, 0.5)),
    }
)
submission["label"] = np.clip(submission["label"].astype(float), 0.001, 0.999)
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())



## === cell 13
pass
