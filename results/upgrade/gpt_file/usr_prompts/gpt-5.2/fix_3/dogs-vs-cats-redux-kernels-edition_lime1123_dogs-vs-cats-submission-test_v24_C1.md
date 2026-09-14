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

0.0311964116562683

# 6. Current score

0.70958

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'Your code currently can’t yield a Kaggle score because it won’t run end-to-end: the dataset returns only images (no labels) but the Lightning module expects `(img, label)`, `cfg.distill_ratio` is undefined, the teacher is never moved/loaded, and the test image glob points to the wrong directory level (so `id` extraction and ordering can break). I make the smallest changes needed to (1) correctly locate and sort test images, (2) make the dataset return ids for test-time, (3) ensure checkpoint inference runs on the right device, and (4) write a single valid `submission.csv` matching `sample_submission.csv` row count/order. I not change the model architecture or training; only inference/data plumbing and safe probability clipping for logloss stability. This should produce a valid submission and typically improves logloss versus misaligned ids / missing sigmoid / bad clipping.'
- What this solution (achieved 0.70958) has done: 'I fix the two blocking file/path issues: your `cfg.train_dir/test_dir` point to `/kaggle/working/...` but the provided data is under `/kaggle/input/dogs-vs-cats-redux-kernels-edition/...`, so I update the config to those real folders and broaden the test-image glob to include the `unknown/` nesting. Your checkpoint path is also missing in this environment, so instead of crashing I fall back to creating a `Distill_Model` with `pretrained=True` and run inference directly (same architecture/inference semantics, just no external weights). Finally, I ensure the submission ids match `sample_submission.csv` exactly (including row count/order) and keep safe probability clipping for logloss stability, which should move you off the 0.693 baseline toward a much better score.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import cv2
import timm
from torch.optim.lr_scheduler import _LRScheduler
import math
import pandas as pd
import tqdm
import pprint




## === cell 1
class Config:
    dog = 1
    cat = 0

    base_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
    train_dir = os.path.join(base_dir, "train")  # has cat/ and dog/ subfolders
    test_dir = os.path.join(base_dir, "test")  # has nested test/unknown/ in this dump

    n_fold = 5
    num_workers = 2  # keep modest to avoid dataloader spawn overhead/timeouts in some environments
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    early_stopping = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (224, 224)


cfg = Config()




## === cell 2
def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # deterministic inference


seed_everything()




## === cell 3
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return list(self.base_lrs)
            warmup_lr = [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            denom = self.total_epochs - self.warmup_epochs
            if denom <= 0:
                return list(self.base_lrs)
            cos_anneal_lr = [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / denom)
                )
                / 2
                for base_lr in self.base_lrs
            ]
            return cos_anneal_lr


class DC_Dataset(Dataset):
    """
    Minimal fix: return (image, id) for test-time so we can align predictions to ids.
    """

    def __init__(self, paths, valid=False, return_id=False):
        super().__init__()
        self.paths = paths
        self.valid = valid
        self.return_id = return_id

        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if not self.valid:
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.5
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RGBShift(p=0.5),
                    A.RandomSizedCrop(
                        min_max_height=(cfg.size[0], cfg.size[0] // 2),
                        height=cfg.size[0],
                        width=cfg.size[1],
                    ),
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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, cfg.size)
        img = self.transform(image=img)["image"]

        if self.return_id:
            image_id = os.path.basename(path).split(".")[0]
            return img, int(image_id)
        return img


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-1, -2)).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Distill_Model(pl.LightningModule):
    """
    Keep core model logic; distillation remains optional.
    """

    def __init__(
        self,
        model_name="convnext_small",
        teacher_path=None,
        pretrained=True,
        num_batch=0,
        fold=0,
        distill_ratio=0.0,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
        )
        if not ("vit" in model_name or "mobile" in model_name):
            self.model.head = nn.Sequential(
                GeM(), nn.Linear(self.model.head.in_features, 1)
            )

        self.teacher = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            num_classes=1,
        )
        self.teacher.head = nn.Sequential(
            GeM(), nn.Linear(self.teacher.head.in_features, 1)
        )

        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.model_name = model_name
        self.pretrained = pretrained
        self.distill_ratio = float(distill_ratio)

        self.save_hyperparameters(ignore=["teacher_path"])

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)

        if self.distill_ratio > 0:
            with torch.no_grad():
                t = torch.sigmoid(self.teacher(img).squeeze(-1))
            label = self.distill_ratio * t + label * (1 - self.distill_ratio)

        loss = self.criterion(output, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        if torch.isnan(loss):
            return None
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.model.parameters(), lr=cfg.lr)
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.warmup_epochs * max(self.num_batch, 1),
            total_epochs=cfg.epochs * max(self.num_batch, 1) + 1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }




## === cell 4
candidate_patterns = [
    os.path.join(cfg.test_dir, "*.jpg"),
    os.path.join(cfg.test_dir, "unknown", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "unknown", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "test", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "test", "unknown", "*.jpg"),
]
test_paths = []
for pat in candidate_patterns:
    test_paths = glob.glob(pat)
    if len(test_paths) > 0:
        break

if len(test_paths) == 0:
    raise FileNotFoundError(
        f"No test images found under {cfg.test_dir}. Tried: {candidate_patterns}"
    )

test_paths = sorted(test_paths, key=lambda p: int(os.path.basename(p).split(".")[0]))
print("Found test images:", len(test_paths), "example:", test_paths[0])



## === cell 5
model_paths = glob.glob(
    "/kaggle/input/dogs-vs-cats-distillation/lightning_logs/version_*/checkpoints/*.ckpt"
)
pprint.pprint(model_paths)

if len(model_paths) == 0:
    print(
        "WARNING: No checkpoints found at /kaggle/input/dogs-vs-cats-distillation/... . "
        "Falling back to a single pretrained convnext_small model."
    )



## === cell 6
outputs_per_model = []



## === cell 7
test_dataset = DC_Dataset(test_paths, valid=True, return_id=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    drop_last=False,
)



## === cell 8
id_order = []
models_to_run = model_paths if len(model_paths) > 0 else [None]

for model_path in models_to_run:
    if model_path is None:
        model = Distill_Model(pretrained=True)
        model_name_for_desc = "pretrained_convnext_small"
    else:
        model = Distill_Model.load_from_checkpoint(model_path)
        model_name_for_desc = os.path.basename(model_path)

    model.to(cfg.device)
    model.eval()

    logits = {}
    with torch.no_grad():
        for imgs, ids in tqdm.tqdm(
            test_loader, desc=f"Predicting {model_name_for_desc}"
        ):
            imgs = imgs.to(cfg.device, non_blocking=True)
            out = model(imgs).detach().float().cpu()
            ids = ids.detach().cpu().numpy().tolist()
            for i, image_id in enumerate(ids):
                logits[image_id] = out[i].item()

    if not id_order:
        id_order = sorted(logits.keys())
    outputs_per_model.append([logits[i] for i in id_order])



## === cell 9
logits = torch.tensor(outputs_per_model, dtype=torch.float32).mean(dim=0)
probs = torch.sigmoid(logits)

clip = 0.005
probs = torch.clamp(probs, min=clip, max=1 - clip)



## === cell 10
sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"id": id_order, "label": probs.cpu().numpy().tolist()})
pred_df["id"] = pd.to_numeric(pred_df["id"])
pred_df = pred_df.sort_values("id")

submission = sample[["id"]].merge(pred_df, on="id", how="left")
if submission["label"].isna().any():
    submission["label"] = submission["label"].fillna(0.5)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(
    "Wrote:",
    out_path,
    "rows:",
    len(submission),
    "na_filled:",
    int(submission["label"].isna().sum()),
)



## === cell 11
submission.head()



## === cell 12
print("Done.")
