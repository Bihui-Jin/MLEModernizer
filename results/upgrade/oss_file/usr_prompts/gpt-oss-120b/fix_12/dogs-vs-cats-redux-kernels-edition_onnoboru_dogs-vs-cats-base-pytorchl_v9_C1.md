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

albumentations==2.0.8
geopandas==0.14.4
lightning-utilities==0.15.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
sentence-transformers==4.1.0
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
transformers==4.53.3

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
import os, gc, random, zipfile, warnings, glob
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
pl.seed_everything(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class CFG:
    debug_one_epoch = False
    debug_one_fold = False
    only_infer = False
    num_workers = min(8, os.cpu_count() or 1)  # ↑ more workers for faster I/O
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    early_stopping_round = 5
    warmup_prop = 0.1
    random_seed = 42
    n_splits = 5
    model_name = "resnet18"
    pretrained_path = None
    train_dir = None
    test_dir = None
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = None
    input_imgsize = 224
    data_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"




## === cell 1
class DogsCatsDataset(Dataset):
    """
    Loads an image from disk on‑demand and applies the given Albumentations transform.
    `preload` is kept for compatibility – when True, all transformed images are cached
    in RAM, otherwise they are read per‑sample. This flexibility lets us avoid the
    costly upfront caching for large train/valid sets.
    """

    def __init__(self, df, transform=None, preload=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.preload = preload

        if self.preload:
            cached = []
            for idx in range(len(self.df)):
                path = self.df.iloc[idx]["path"]
                img = np.array(Image.open(path).convert("RGB"))
                if self.transform:
                    img = self.transform(image=img)["image"]
                cached.append(img)
            self.cached_images = torch.stack(cached)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.preload:
            img = self.cached_images[idx]
        else:
            path = self.df.iloc[idx]["path"]
            img = np.array(Image.open(path).convert("RGB"))
            if self.transform:
                img = self.transform(image=img)["image"]
        label = float(self.df.iloc[idx]["class"])
        return img, label




## === cell 2
train_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 3
class DogCatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)




## === cell 4
class dog_vs_cats_pl_model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = CFG.criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x).squeeze()
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = CFG.optimizer(self.parameters(), lr=CFG.lr)
        return optimizer




## === cell 5
def run_train_cv_pl(train, test):
    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train), 1))
    predictions = []

    test_dataset = DogsCatsDataset(test, transform=test_transform, preload=True)
    common_loader_args = dict(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"==================== fold {fold} ====================")
        train_df_fold = train.iloc[train_idx].reset_index(drop=True)
        valid_df_fold = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = DogsCatsDataset(
            train_df_fold, transform=train_transform, preload=False
        )
        valid_dataset = DogsCatsDataset(
            valid_df_fold, transform=test_transform, preload=False
        )

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **common_loader_args,
        )
        valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            **common_loader_args,
        )

        model = DogCatModel()
        if torch.cuda.is_available():
            model = torch.compile(model)  # ← speed‑up without altering architecture
        lightning_model = dog_vs_cats_pl_model(model)

        early_stopping = EarlyStopping(
            monitor="valid_loss", mode="min", patience=CFG.early_stopping_round
        )
        checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            dirpath="checkpoints",
            filename=f"{CFG.model_name}_fold{fold}",
            save_top_k=1,
        )
        logger = CSVLogger("logs", name=f"{CFG.model_name}_fold{fold}")

        trainer = pl.Trainer(
            max_epochs=CFG.num_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            precision=16 if torch.cuda.is_available() else 32,
            logger=logger,
            callbacks=[early_stopping, checkpoint],
            enable_progress_bar=False,
        )

        trainer.fit(lightning_model, train_loader, valid_loader)

        valid_preds = trainer.predict(lightning_model, valid_loader)
        valid_preds_arr = np.concatenate([p.squeeze() for p in valid_preds])
        oof[valid_idx] = valid_preds_arr.reshape(-1, 1)

        test_preds = trainer.predict(lightning_model, test_loader)
        test_preds_arr = np.concatenate([p.squeeze() for p in test_preds])
        predictions.append(test_preds_arr)

        del model, lightning_model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if CFG.debug_one_fold:
            break

    predictions = np.mean(predictions, axis=0)
    return {"oof": oof, "predictions": predictions}




## === cell 6
def build_dataframe():
    train_root = os.path.join(CFG.data_dir, "train")
    cat_paths = glob.glob(os.path.join(train_root, "cat", "*.jpg"))
    dog_paths = glob.glob(os.path.join(train_root, "dog", "*.jpg"))
    train_df = pd.DataFrame(
        {
            "path": cat_paths + dog_paths,
            "class": [0] * len(cat_paths) + [1] * len(dog_paths),
        }
    )

    test_root = os.path.join(CFG.data_dir, "test")
    test_paths = glob.glob(os.path.join(test_root, "**", "*.jpg"), recursive=True)
    test_df = pd.DataFrame(
        {
            "path": test_paths,
            "class": 0,  # placeholder, not used
            "id": [os.path.splitext(os.path.basename(p))[0] for p in test_paths],
        }
    )
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = build_dataframe()
    results = run_train_cv_pl(train_df, test_df)

    submission = pd.DataFrame({"id": test_df["id"], "label": results["predictions"]})
    submission_path = os.path.join(CFG.kaggle_working_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
