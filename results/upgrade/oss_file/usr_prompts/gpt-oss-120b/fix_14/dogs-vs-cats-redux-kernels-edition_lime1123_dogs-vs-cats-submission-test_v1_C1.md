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
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Target score

0.0324629077409772

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.73868) has done: 'Implemented a fix for the EfficientNet classifier initialization, which was causing an AttributeError, and added a safeguard to ensure the model’s output dimensions are correctly handled. This resolves the runtime crash and guarantees valid probability values for the submission CSV, keeping all other logic unchanged.'

# 9. Code solution

## === cell 0
import os, subprocess, sys, glob, random, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import models as tv_models
from torch.optim.lr_scheduler import _LRScheduler
import tqdm

try:
    import cv2

    _cv2_available = True
except Exception:
    _cv2_available = False
    from PIL import Image

    def _imread(path):
        return np.array(Image.open(path).convert("RGB"))

    def _resize(img, size):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size, Image.BILINEAR)
        return np.array(pil_img)

    class _CV2Mock:
        COLOR_BGR2RGB = None

        @staticmethod
        def imread(path):
            return _imread(path)

        @staticmethod
        def resize(img, size):
            return _resize(img, size)

        @staticmethod
        def cvtColor(img, flag):
            return img

    cv2 = _CV2Mock()

best_clip = 0.0

if not os.path.isdir("/kaggle/working/dogs-vs-cats-redux-kernels-edition/train"):
    subprocess.run(
        [
            "unzip",
            "-q",
            "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip",
            "-d",
            "/kaggle/working",
        ],
        check=True,
    )
if not os.path.isdir("/kaggle/working/dogs-vs-cats-redux-kernels-edition/test"):
    subprocess.run(
        [
            "unzip",
            "-q",
            "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip",
            "-d",
            "/kaggle/working",
        ],
        check=True,
    )




## === cell 1
class Config:
    dog = 1
    cat = 0
    base_dir = "/kaggle/working/dogs-vs-cats-redux-kernels-edition"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    n_fold = 5
    num_workers = 2
    pin_memory = True
    batch_size = 32  # reduced to fit GPU memory during training
    seed = 2025
    drop_last = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 8  # increased modestly to give the model more learning time
    early_stopping = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (320, 320)


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
    torch.backends.cudnn.benchmark = False


seed_everything()




## === cell 3
class DC_Dataset(Dataset):
    def __init__(self, img):
        super().__init__()
        self.img = torch.tensor(np.stack(img, axis=0)).permute(
            0, 3, 1, 2
        )  # (N, C, H, W)

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):
        img = self.img[index].to(torch.float32) / 255.0
        return img




## === cell 4
class DC_Model(pl.LightningModule):
    def __init__(self, model_name="efficientnet_v2_s", pretrained=True, num_batch=0):
        super().__init__()
        try:
            if model_name.startswith("efficientnet"):
                self.model = tv_models.efficientnet_v2_s(pretrained=pretrained)
            else:
                raise ValueError
        except Exception:
            self.model = tv_models.resnet18(pretrained=pretrained)

        if hasattr(self.model, "classifier"):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier = nn.Linear(in_features, 1)
        elif hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)
        else:
            raise RuntimeError("Unsupported model architecture for head replacement.")

        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.save_hyperparameters()

    def forward(self, x):
        out = self.model(x)
        return out.squeeze(-1)

    def training_step(self, batch, batch_idx):
        img, label = batch
        label = label.to(self.device)  # ensure same device as model output
        output = self(img)
        loss = self.criterion(output, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        label = label.to(self.device)
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs * self.num_batch + 1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }




## === cell 5
def collate(batch):
    """
    Flexible collate that works for both:
    - test batches containing only image tensors
    - training/validation batches containing (image, label) tuples
    """
    first = batch[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(batch)
    else:
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)
        return imgs, labels




## === cell 6
sample_sub_path = (
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path, dtype={"id": str})
image_ids = sample_sub["id"].tolist()

test_images = []
for img_id in image_ids:
    possible_paths = [
        os.path.join(cfg.test_dir, "unknown", f"{img_id}.jpg"),
        os.path.join(cfg.test_dir, f"{img_id}.jpg"),
    ]
    img_path = next((p for p in possible_paths if os.path.isfile(p)), None)
    if img_path is not None:
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((cfg.size[1], cfg.size[0], 3), dtype=np.uint8)
    else:
        img = np.zeros((cfg.size[1], cfg.size[0], 3), dtype=np.uint8)
    img = cv2.resize(img, cfg.size)
    test_images.append(img)



## === cell 7
model_paths = glob.glob(
    "/kaggle/input/dogs-vs-cats-lightning/lightning_logs/version_*/checkpoints/*.ckpt"
)



## === cell 8
outputs = []  # will collect per‑model predictions



## === cell 9
test_dataset = DC_Dataset(test_images)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=collate,
)




## === cell 10
class TrainDataset(Dataset):
    def __init__(self, img_paths, labels):
        imgs = []
        kept_labels = []
        for p, lbl in zip(img_paths, labels):
            img = cv2.imread(p)
            if img is None:
                continue
            if _cv2_available:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, cfg.size)
            imgs.append(img)
            kept_labels.append(lbl)
        self.labels = torch.tensor(kept_labels, dtype=torch.float32)
        self.imgs = (
            torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]




## === cell 11
if model_paths:
    for model_path in model_paths:
        model = DC_Model.load_from_checkpoint(model_path, map_location=cfg.device)
        model.eval()
        model.to(cfg.device)
        outputs.append([])
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                batch = batch.to(cfg.device)
                out = model(batch)
                out_flip = model(torch.flip(batch, dims=[3]))
                out_avg = (out + out_flip) / 2.0
                outputs[-1] += out_avg.tolist()
else:
    train_image_paths = []
    train_labels = []
    for fname in os.listdir(cfg.train_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        label = cfg.dog if fname.startswith("dog") else cfg.cat
        train_image_paths.append(os.path.join(cfg.train_dir, fname))
        train_labels.append(label)

    full_train_dataset = TrainDataset(train_image_paths, train_labels)
    val_len = int(0.1 * len(full_train_dataset))
    train_len = len(full_train_dataset) - val_len
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # data fits in RAM
        pin_memory=cfg.pin_memory,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.pin_memory,
        persistent_workers=False,
    )

    model = DC_Model()
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    model.eval()
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for img, lbl in val_loader:
            img = img.to(cfg.device)
            out = model(img)
            val_logits.append(out.cpu())
            val_labels.append(lbl.cpu())
    val_logits = torch.cat(val_logits)
    val_labels = torch.cat(val_labels)
    val_probs = torch.sigmoid(val_logits).numpy()
    val_labels_np = val_labels.numpy()
    clips = [0.01, 0.005, 0.015, 0.0125, 0.0025, 0.0, 0.0075]
    best_clip = 0.0
    best_loss = float("inf")
    eps = 1e-15
    for clip in clips:
        p = np.clip(val_probs, clip, 1 - clip)
        loss = -(
            val_labels_np * np.log(p + eps) + (1 - val_labels_np) * np.log(1 - p + eps)
        ).mean()
        if loss < best_loss:
            best_loss = loss
            best_clip = clip

    model.eval()
    model.to(cfg.device)
    outputs.append([])
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            batch = batch.to(cfg.device)
            out = model(batch)
            out_flip = model(torch.flip(batch, dims=[3]))
            out_avg = (out + out_flip) / 2.0
            outputs[-1] += out_avg.tolist()



## === cell 12
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
outputs_mean = outputs_tensor.mean(dim=0)
outputs_prob = torch.sigmoid(outputs_mean)
outputs_prob = torch.nan_to_num(outputs_prob, nan=0.5)

if best_clip == 0.0:
    best_clip = 0.001

submission = pd.DataFrame(
    {
        "id": image_ids,
        "label": torch.clamp(outputs_prob, min=best_clip, max=1 - best_clip).tolist(),
    }
)
submission["id"] = pd.to_numeric(submission["id"])
submission = submission.sort_values("id")
submission.to_csv("/kaggle/working/submission.csv", index=False)
