# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
import pandas as pd
import tqdm
import pprint
import albumentations as A
from albumentations.pytorch import ToTensorV2




## === cell 1
class Config:
    dog = 1
    cat = 0

    train_dir = "/kaggle/working/train"
    test_dir = "/kaggle/working/test"

    n_fold = 5

    num_workers = min(8, os.cpu_count() or 4)
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




## === cell 2
def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()



## === cell 3
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def seed_worker(worker_id: int):
    worker_seed = (cfg.seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class DC_Dataset(Dataset):
    """
    Bugfixes:
    - Properly parse labels from path when return_label=True (was always returning 0.0).
    - Convert BGR->RGB for correct normalization/model input.
    - Keep core augmentation logic unchanged.
    """

    def __init__(self, paths, valid=False, return_label=False):
        super().__init__()
        self.paths = list(paths)
        self.valid = valid
        self.return_label = return_label

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
            self.transform = A.Compose([A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.paths)

    def _path_to_label(self, p: str) -> float:
        base = os.path.basename(p).lower()
        parent = os.path.basename(os.path.dirname(p)).lower()
        if parent in ("cat", "dog"):
            return float(cfg.dog if parent == "dog" else cfg.cat)
        if base.startswith("dog."):
            return float(cfg.dog)
        if base.startswith("cat."):
            return float(cfg.cat)
        return float(cfg.cat)

    def __getitem__(self, index):
        path = self.paths[index]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, cfg.size, interpolation=cv2.INTER_AREA)
        img = self.transform(image=img)["image"]

        if self.return_label:
            label = torch.tensor(self._path_to_label(path), dtype=torch.float32)
            return img, label
        return img


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
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class DC_Model(pl.LightningModule):
    def __init__(
        self, model_name="convnext_small", pretrained=True, num_batch=0, fold=0
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1, global_pool=""
        )

        num_features = self.model.num_features
        self.model.head = nn.Sequential(GeM(), nn.Linear(num_features, 1))
        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.model_name = model_name
        self.pretrained = pretrained

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = (torch.sigmoid(output) > 0.5).float()
        acc = (pred == label).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = (torch.sigmoid(output) > 0.5).float()
        acc = (pred == label).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr, weight_decay=0.1)
        return optimizer




## === cell 4
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition",
]
DATA_ROOT = None
for r in DATA_ROOT_CANDIDATES:
    if os.path.exists(r):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset root in candidates: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_CAT_DIR = os.path.join(DATA_ROOT, "train", "cat")
TRAIN_DOG_DIR = os.path.join(DATA_ROOT, "train", "dog")

TEST_DIR = os.path.join(DATA_ROOT, "test", "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

print("DATA_ROOT:", DATA_ROOT)
print(
    "TRAIN_CAT_DIR exists:",
    os.path.exists(TRAIN_CAT_DIR),
    "n=",
    len(glob.glob(os.path.join(TRAIN_CAT_DIR, "*.jpg"))),
)
print(
    "TRAIN_DOG_DIR exists:",
    os.path.exists(TRAIN_DOG_DIR),
    "n=",
    len(glob.glob(os.path.join(TRAIN_DOG_DIR, "*.jpg"))),
)
print(
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    "n=",
    len(glob.glob(os.path.join(TEST_DIR, "*.jpg"))),
)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 5
test_paths = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
if len(test_paths) == 0:
    raise FileNotFoundError(f"No test images found in: {TEST_DIR}")

test_paths = sorted(
    test_paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)
image_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_paths]

print("Found test images:", len(test_paths))
print("First/last id:", (image_ids[0], image_ids[-1]) if image_ids else None)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
expected_ids = sample_sub["id"].astype(int).tolist()
print(
    "Sample submission rows:",
    len(sample_sub),
    "id range:",
    (min(expected_ids), max(expected_ids)),
)

id_to_path = {int(os.path.splitext(os.path.basename(p))[0]): p for p in test_paths}
missing = [i for i in expected_ids if i not in id_to_path]
if missing:
    raise ValueError(
        f"Missing {len(missing)} test ids from filesystem. Example: {missing[:10]}"
    )

ordered_test_paths = [id_to_path[i] for i in expected_ids]
ordered_image_ids = expected_ids



## === cell 6
model_paths = glob.glob(
    "/kaggle/input/dogs-vs-cats-lightning/lightning_logs/version_*/checkpoints/*.ckpt"
)
pprint.pprint(model_paths)




## === cell 7
def train_fallback_checkpoint(out_ckpt_path="/kaggle/working/fallback.ckpt"):
    if os.path.exists(out_ckpt_path) and os.path.getsize(out_ckpt_path) > 0:
        return out_ckpt_path

    cat_paths = glob.glob(os.path.join(TRAIN_CAT_DIR, "*.jpg"))
    dog_paths = glob.glob(os.path.join(TRAIN_DOG_DIR, "*.jpg"))
    if len(cat_paths) == 0 or len(dog_paths) == 0:
        raise FileNotFoundError("Train cat/dog directories are empty or missing.")

    all_paths = cat_paths + dog_paths
    rng = np.random.RandomState(cfg.seed)
    idx = np.arange(len(all_paths))
    rng.shuffle(idx)

    val_size = int(0.05 * len(all_paths))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    tr_paths = [all_paths[i] for i in tr_idx]
    val_paths = [all_paths[i] for i in val_idx]

    train_ds = DC_Dataset(tr_paths, valid=False, return_label=True)
    val_ds = DC_Dataset(val_paths, valid=True, return_label=True)

    persistent = cfg.num_workers > 0
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=persistent,
        prefetch_factor=8 if persistent else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=persistent,
        prefetch_factor=8 if persistent else None,
        worker_init_fn=seed_worker,
    )

    pl.seed_everything(cfg.seed, workers=True)
    model = DC_Model(model_name="convnext_small", pretrained=True)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.dirname(out_ckpt_path),
        filename=os.path.splitext(os.path.basename(out_ckpt_path))[0],
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32-true",
        callbacks=[ckpt_cb],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        benchmark=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = ckpt_cb.best_model_path
    if not best_path or not os.path.exists(best_path):
        trainer.save_checkpoint(out_ckpt_path)
        best_path = out_ckpt_path

    if best_path != out_ckpt_path:
        try:
            import shutil

            shutil.copy2(best_path, out_ckpt_path)
            best_path = out_ckpt_path
        except Exception:
            pass

    return best_path


if len(model_paths) == 0:
    print(
        "No external checkpoints found; training fallback model to create a usable checkpoint."
    )
    best_ckpt = train_fallback_checkpoint("/kaggle/working/fallback.ckpt")
    model_paths = [best_ckpt]
    print("Using fallback checkpoint:", model_paths[0])



## === cell 8
outputs = []

test_dataset = DC_Dataset(ordered_test_paths, valid=True, return_label=False)

persistent = cfg.num_workers > 0
g = torch.Generator()
g.manual_seed(cfg.seed)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    drop_last=False,
    persistent_workers=persistent,
    prefetch_factor=8 if persistent else None,
    worker_init_fn=seed_worker,
    generator=g,
)

cached_test_batches = []
for batch in tqdm.tqdm(test_loader, desc="Caching test tensors"):
    cached_test_batches.append(batch)

for model_path in model_paths:
    model = DC_Model.load_from_checkpoint(model_path)
    model.to(cfg.device)
    model.eval()

    preds_chunks = []
    with torch.no_grad():
        for batch in tqdm.tqdm(
            cached_test_batches, desc=f"Infer {os.path.basename(model_path)}"
        ):
            img = batch.to(cfg.device, non_blocking=True)
            output = model(img)
            preds_chunks.append(output.detach().float().cpu())
    outputs.append(torch.cat(preds_chunks, dim=0))



## === cell 9
outputs = torch.stack(outputs, dim=0).to(dtype=torch.float32)
outputs = outputs.mean(dim=0)
outputs = torch.sigmoid(outputs)

print("Preds:", outputs.shape, "min/max:", float(outputs.min()), float(outputs.max()))



## === cell 10
clips = [0.0, 0.01, 0.005, 0.015, 0.0125, 0.0025, 0.0075, 0.004]
final_submission = None

for clip in clips:
    probs = torch.clamp(outputs, min=clip, max=1 - clip).cpu().numpy().tolist()
    submission = pd.DataFrame({"id": ordered_image_ids, "label": probs})
    submission = submission.sort_values(by="id").reset_index(drop=True)
    out_path = f"/kaggle/working/submission-clip={clip}.csv"
    submission.to_csv(out_path, index=False)
    final_submission = submission

final_path = "/kaggle/working/submission.csv"
final_submission.to_csv(final_path, index=False)
print("Wrote:", final_path, "rows:", len(final_submission))
print(final_submission.head())



## === cell 11
sub = pd.read_csv("/kaggle/working/submission.csv")
assert list(sub.columns) == ["id", "label"], f"Bad columns: {sub.columns.tolist()}"
assert len(sub) == len(
    sample_sub
), f"Row mismatch: sub={len(sub)} sample={len(sample_sub)}"
assert sub["id"].astype(int).tolist() == sorted(
    sample_sub["id"].astype(int).tolist()
), "ID set mismatch vs sample submission"
assert sub["label"].between(0, 1).all(), "Labels must be probabilities in [0,1]"
sub.head()
