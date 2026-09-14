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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.38356

# 6. Current score

0.80172

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51421) has done: 'The timeout is dominated by 5-fold training plus slow data loading/augmentation and extra overhead (tqdm per-batch, PIL decoding in main process, unnecessary Python loops for label indexing, and an extra CSV read/softmax pass). I keep the same ResNet50 head-only training, same transforms, same loss/optimizer/scheduler, and same fold averaging, but make data input and training loops materially faster via multi-worker DataLoaders (with pinned memory, persistent workers, prefetching), vectorized label encoding, removing plotting, and eliminating redundant CPU/GPU syncs and the second CSV I/O pass by applying softmax once at the end in-memory. These are provably equivalent to the original semantics (same data, same model updates), with only negligible floating-point differences. I also add deterministic seeding and fix the undefined `i` in the earlier split cell to avoid accidental reruns/time waste.'
- What this solution (achieved 0.45652) has done: 'The timeout is dominated by doing 5-fold training with a ResNet50 backbone plus repeated JPEG decode/augmentation overhead and extra CPU↔GPU synchronization inside the loops. I keep the exact same model, loss, optimizer, folds, epochs, and early-stopping logic, but reduce constant-factor overhead: (1) cache decoded images globally so they are reused across folds/epochs (still applying the same transforms per access), (2) remove per-batch `.cpu()` syncs by accumulating metrics on-device and only converting once per epoch, and (3) enable cuDNN benchmarking only for the non-deterministic RandomResizedCrop training pipeline (safe here because determinism was already broken by stochastic aug; it does not change the algorithm). These changes are correctness-preserving with only negligible floating-point differences while substantially reducing wall time.'
- What this solution (achieved 0.80172) has done: 'The timeout is dominated by doing 5-fold training with heavy CPU image decoding/augmentation plus an expensive full test-set tensor precompute; both create large constant overhead. I keep the same model, loss, optimizer, scheduler, epochs, and folds, but remove the test-tensor precompute (it’s unnecessary and duplicates work) and replace the PNG-bytes cache with a faster, correctness-preserving “raw file bytes” cache that avoids repeated disk I/O without changing transform randomness. I also make the CUDA prefetcher re-iterable each epoch (the current code exhausts after epoch 0), which fixes training semantics while also preventing wasted epochs/overhead. Finally, I tune DataLoader settings (workers/prefetch/pinning) in a strictly equivalent way to reduce input bottlenecks.'

# 9. Code solution

## === cell 0
import os
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split, KFold
from PIL import Image

SEED = 2021
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))



## === cell 1
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(train_data["breed"].unique().tolist())

label_to_idx = {b: i for i, b in enumerate(labels)}
train_data["number"] = train_data["breed"].map(label_to_idx).astype(np.int64)
train_data.shape



## === cell 2
file_names = sorted(os.listdir("/kaggle/input/dog-breed-identification/test"))
file_names = [name[:-4] for name in file_names if name.lower().endswith(".jpg")]
test_data = pd.DataFrame({"id": file_names})
test_data.head()



## === cell 3
transforms_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transforms_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 4
_GLOBAL_JPEG_BYTES_CACHE: dict[str, bytes] = {}


class Dog_Breed(Dataset):
    def __init__(self, train_csv, transform=None, test=False):
        super().__init__()
        self.train_csv = train_csv.reset_index(drop=True)
        self.image_path = self.train_csv["id"].tolist()
        self.test = test
        if not self.test:
            self.label_nums = self.train_csv["number"].to_numpy(dtype=np.int64)
        self.transform = transform

    def _get_path(self, stem: str) -> str:
        if self.test:
            return os.path.join(
                "/kaggle/input/dog-breed-identification/test", stem + ".jpg"
            )
        return os.path.join(
            "/kaggle/input/dog-breed-identification/train", stem + ".jpg"
        )

    def _load_pil_rgb(self, stem: str) -> Image.Image:
        import io

        p = self._get_path(stem)
        b = _GLOBAL_JPEG_BYTES_CACHE.get(p)
        if b is None:
            with open(p, "rb") as f:
                b = f.read()
            _GLOBAL_JPEG_BYTES_CACHE[p] = b
        return Image.open(io.BytesIO(b)).convert("RGB")

    def __getitem__(self, idx):
        stem = self.image_path[idx]
        image = self._load_pil_rgb(stem)

        if self.transform is not None:
            image = self.transform(image)

        if not self.test:
            label = int(self.label_nums[idx])
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_path)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
device



## === cell 6
train, valid = train_test_split(train_data, test_size=0.2, random_state=SEED)
trainset = Dog_Breed(train, transform=transforms_train)
validset = Dog_Breed(valid, transform=transforms_test)




## === cell 7
def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

_cpu = os.cpu_count() or 2
_num_workers = min(8, max(2, _cpu - 1))  # keep within reason for Kaggle CPU limits
_pin = device == "cuda"

train_loader = DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=_num_workers,
    pin_memory=_pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)
valid_loader = DataLoader(
    validset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=_num_workers,
    pin_memory=_pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)


class CUDAPrefetcher:
    """Overlap CPU->GPU transfer with GPU compute; semantics identical to iterating the underlying loader."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device == "cuda" else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.device != "cuda":
            yield from self.loader
            return

        it = iter(self.loader)
        next_batch = None

        def _preload():
            nonlocal next_batch
            try:
                batch = next(it)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x, y = batch
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    next_batch = (x, y)
                else:
                    x = batch
                    x = x.to(self.device, non_blocking=True)
                    next_batch = x

        _preload()
        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            _preload()
            yield batch




## === cell 8
pass




## === cell 9
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 10
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetCustom, self).__init__()
        raise NotImplementedError("EfficientNetCustom is not used in this solution.")

    def forward(self, x):
        raise NotImplementedError




## === cell 11
def train_model(
    model,
    train_loader,
    valid_loader,
    loss,
    optimizer,
    epoch,
    device=torch.device("cuda:0"),
    test_loader=None,
):
    net = model.to(device)

    best_epoch = 0
    best_valid_loss = float("inf")
    best_model_state = None
    early_stopping_round = 3
    losses = []

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    for i in range(epoch):
        net.train()

        acc = torch.zeros((), device=device, dtype=torch.long)
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        train_seen = 0

        train_iterable = (
            CUDAPrefetcher(train_loader, device) if device == "cuda" else train_loader
        )

        for batch in train_iterable:
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                x, y = batch  # already on GPU
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += loss_temp.detach()
            loss_temp.backward()
            optimizer.step()

            acc += (y_hat.argmax(dim=1) == y).sum()
            train_seen += int(y.shape[0])

        scheduler.step()

        loss_sum_f = float(loss_sum.detach().cpu())
        acc_i = int(acc.detach().cpu())
        losses.append(loss_sum_f / len(train_loader))

        print(
            "epoch: ",
            i,
            "loss=",
            loss_sum_f / max(1, train_seen),
            "训练集准确度=",
            acc_i / max(1, train_seen),
            end="",
        )

        net.eval()
        valid_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        valid_correct = torch.zeros((), device=device, dtype=torch.long)
        valid_seen = 0

        valid_iterable = (
            CUDAPrefetcher(valid_loader, device) if device == "cuda" else valid_loader
        )

        with torch.inference_mode():
            for batch in valid_iterable:
                if device == "cuda":
                    x, y = batch
                else:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                y_hat = net(x)
                valid_loss_sum += loss(y_hat, y).detach()
                valid_correct += (y_hat.argmax(dim=1) == y).sum()
                valid_seen += int(y.shape[0])

        valid_loss_sum_f = float(valid_loss_sum.detach().cpu())
        valid_correct_i = int(valid_correct.detach().cpu())
        valid_loss_avg = valid_loss_sum_f / max(1, len(valid_loader))
        print(
            "验证集准确度",
            valid_correct_i / max(1, valid_seen),
            "验证集CE",
            valid_loss_avg,
        )

        if valid_loss_avg < best_valid_loss:
            best_model_state = copy.deepcopy(net.state_dict())
            best_valid_loss = valid_loss_avg
            best_epoch = i
            print("best epoch save! (by val CE)")
        if i - best_epoch >= early_stopping_round:
            break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    if test_loader is None:
        raise ValueError("test_loader must be provided")

    test_iterable = (
        CUDAPrefetcher(test_loader, device) if device == "cuda" else test_loader
    )

    predictions = []
    net.eval()
    with torch.inference_mode():
        for batch in test_iterable:
            if device == "cuda":
                x = batch
            else:
                x = batch.to(device)
            y_hat = net(x)
            predictions.append(y_hat.detach().cpu())

    prediction = torch.cat(predictions, dim=0).reshape(-1, 120)
    return prediction




## === cell 12
learn_rate = 0.001
momentum = 0.9
epoch = 15



## === cell 13
sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
submit_labels = [c for c in sample_sub.columns if c != "id"]

testset = Dog_Breed(test_data, transform=transforms_test, test=True)
test_loader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=_num_workers,
    pin_memory=_pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

all_prob_sum = torch.zeros((len(test_data), 120), dtype=torch.float32)

for train_index, val_index in kfold.split(train_data):
    model = MyResNet50()

    train, valid = train_data.iloc[train_index], train_data.iloc[val_index]
    trainset = Dog_Breed(train, transform=transforms_train)
    validset = Dog_Breed(valid, transform=transforms_test)

    train_loader = DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=_num_workers,
        pin_memory=_pin,
        persistent_workers=(_num_workers > 0),
        prefetch_factor=4 if _num_workers > 0 else None,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=_num_workers,
        pin_memory=_pin,
        persistent_workers=(_num_workers > 0),
        prefetch_factor=4 if _num_workers > 0 else None,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    logits = train_model(
        model,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        epoch,
        device,
        test_loader=test_loader,
    )

    fold_prob = F.softmax(logits.float(), dim=1)
    fold_prob = fold_prob / fold_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)

    all_prob_sum += fold_prob * 0.2

result = pd.DataFrame(all_prob_sum.numpy(), columns=labels)
result = pd.concat([test_data, result], axis=1)

result = result[["id"] + submit_labels]
result.to_csv("dog_breed.csv", index=False)
