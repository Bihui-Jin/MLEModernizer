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

0.03167

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import tqdm
import shutil

try:
    import timm

    def create_model(name, pretrained):
        return timm.create_model(name, pretrained=pretrained)

except Exception:
    def create_model(name, pretrained):
        model = tv_models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model




## === cell 1
class Config:
    dog = 1
    cat = 0
    train_dir = "/kaggle/working/train"
    test_dir = "/kaggle/working/test"
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
    size = (384, 384)


cfg = Config()




## === cell 2
def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()




## === cell 3
def square_pad_and_resize(image, size):
    h, w, _ = image.shape
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    resized = cv2.resize(padded, size)
    return resized


class DC_Dataset(Dataset):
    def __init__(self, paths, valid=False):
        self.paths = paths
        self.valid = valid
        self.transform = (
            tv_models.resnet18(pretrained=True).transform if False else None
        )
        self.normalize = (
            tv_models.resnet18(pretrained=True).transform if False else None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = square_pad_and_resize(img, cfg.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)




## === cell 4
class DC_Model(pl.LightningModule):
    def __init__(self, model_name="efficientnetv2_rw_s", pretrained=True, num_batch=0):
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained)
        if hasattr(self.model, "classifier"):
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        elif hasattr(self.model, "fc"):
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        else:
            self.model = nn.Sequential(
                self.model, nn.Linear(self.model.out_features, 1)
            )
        self.criterion = cfg.criterion
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        pred = torch.sigmoid(out) > 0.5
        acc = (pred == label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs * len(self.trainer.datamodule.train_dataloader()),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }




## === cell 5
test_paths = glob.glob(os.path.join(cfg.test_dir, "**", "*.*"), recursive=True)
test_paths = [p for p in test_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]

image_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_paths]




## === cell 6
model_paths = glob.glob(
    "/kaggle/input/dogs-vs-cats-lightning/lightning_logs/version_*/checkpoints/*.ckpt"
)
print(f"Found {len(model_paths)} checkpoint(s).")




## === cell 7
outputs = []  # will hold per‑model predictions




## === cell 8
test_dataset = DC_Dataset(test_paths, valid=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
)




## === cell 9
if not model_paths:
    model = DC_Model().to(cfg.device)
    model.eval()
    outputs.append([])
    with torch.no_grad():
        for img in tqdm.tqdm(test_loader, desc="Predict (fallback)"):
            out = model(img.to(cfg.device))
            outputs[-1] += out.cpu().tolist()
else:
    for ckpt in model_paths:
        model = DC_Model.load_from_checkpoint(ckpt).to(cfg.device)
        model.eval()
        outputs.append([])
        with torch.no_grad():
            for img in tqdm.tqdm(
                test_loader, desc=f"Predict ({os.path.basename(ckpt)})"
            ):
                out = model(img.to(cfg.device))
                outputs[-1] += out.cpu().tolist()




## === cell 10
outputs = torch.tensor(outputs)  # shape: (n_models, n_images)
outputs = outputs.mean(dim=0)  # mean over models
probabilities = torch.sigmoid(outputs)




## === cell 11
for clip in [0.01, 0.005, 0.015, 0.0125, 0.0025, 0.0, 0.0075, 0.004]:
    sub = pd.DataFrame(
        {
            "id": image_ids,
            "label": torch.clamp(probabilities, min=clip, max=1 - clip).tolist(),
        }
    )
    sub = sub.sort_values("id")
    sub.to_csv(f"/kaggle/working/submission-clip={clip}.csv", index=False)

final_sub = pd.DataFrame({"id": image_ids, "label": probabilities.tolist()})
final_sub = final_sub.sort_values("id")
final_sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to /kaggle/working/submission.csv")




## === cell 12
for folder in ["/kaggle/working/train", "/kaggle/working/test"]:
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
