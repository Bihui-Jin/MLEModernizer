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

0.48647

# 6. Current score

1.03468

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.0347) has done: 'The timeout is dominated by (1) expensive Python-side label indexing in a loop, (2) slow dataloading/augmentation due to too few workers and re-opening images repeatedly without tuning loader settings, and (3) extra CPU/GPU synchronization from frequent `.cpu()` conversions inside training. I keep the exact model/training logic intact, but make equivalent speedups: vectorize label encoding, use a deterministic seed setup, enable cuDNN benchmarking for fixed-size 224 crops, optimize DataLoader settings (more workers, persistent workers, prefetching), and reduce sync overhead by accumulating metrics on-device and converting once per epoch. These changes preserve accuracy/evaluation semantics (only negligible floating-point differences) while significantly reducing wall-clock time.'
- What this solution (achieved 1.03466) has done: 'You’re currently selecting the “best” checkpoint using validation accuracy, but the competition metric is multi-class log loss; that mismatch commonly leaves a lot of score on the table, so we instead select the checkpoint that minimizes validation cross-entropy (log loss proxy) while keeping the same model, optimizer, scheduler, and training loop structure. We also compute validation loss alongside accuracy (no change to training objective) and use it for early-stopping/best-model tracking, which should move your public score down toward the 0.48647 target. Finally, we add a tiny probability clamp when writing the submission (purely numerical safety for log loss) without changing prediction semantics.'
- What this solution (achieved 1.03468) has done: 'Your current score (1.03466, lower is better) is still far from the target (0.48647), so we need a modest but meaningful improvement without changing the model architecture or training objective. The biggest “legal” gain here is to make validation match training-time data distribution: right now you validate with CenterCrop while training uses RandomResizedCrop, which can bias checkpoint selection and hurt log loss; switching validation to the same resize/crop policy typically improves log loss while keeping the same overall pipeline. I also align scheduler stepping to be per-batch (as CosineAnnealingWarmRestarts is intended) while keeping the same scheduler and epochs, which often improves convergence at the same compute budget. Finally, I keep your numerical-safe clipping/renorm for submission unchanged.'

# 9. Code solution

## === cell 0
import os
import copy
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from PIL import Image

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False  # keep fast kernels; crop size is fixed so determinism is not required by solution
torch.backends.cudnn.benchmark = True



## === cell 1
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(train_data["breed"].unique().tolist())
label2idx = {b: i for i, b in enumerate(labels)}
train_data["number"] = train_data["breed"].map(label2idx).astype(np.int64)
train_data.shape



## === cell 2
test_dir = "/kaggle/input/dog-breed-identification/test"
file_names = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
file_names = [os.path.splitext(name)[0] for name in file_names]
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

transforms_valid = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
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
class Dog_Breed(Dataset):
    def __init__(self, train_csv, transform=None, test=False):
        super().__init__()
        self.train_csv = train_csv
        self.image_path = list(self.train_csv["id"])
        self.test = test
        if not self.test:
            self.label_nums = list(self.train_csv["number"])
        self.transform = transform

    def __getitem__(self, idx):
        if self.test:
            img_path = os.path.join(
                "/kaggle/input/dog-breed-identification/test",
                self.image_path[idx] + ".jpg",
            )
        else:
            img_path = os.path.join(
                "/kaggle/input/dog-breed-identification/train",
                self.image_path[idx] + ".jpg",
            )

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if not self.test:
            label = self.label_nums[idx]
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
split_seed = 42
train, valid = train_test_split(
    train_data, test_size=0.2, random_state=split_seed, stratify=train_data["breed"]
)
trainset = Dog_Breed(train, transform=transforms_train)
validset = Dog_Breed(valid, transform=transforms_valid)




## === cell 7
def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

num_workers = min(8, os.cpu_count() or 2)
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
valid_loader = torch.utils.data.DataLoader(
    validset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)




## === cell 8
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        try:
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.net = models.resnet50(pretrained=True)

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.net(x)
        return x




## === cell 9
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetCustom, self).__init__()
        raise NotImplementedError(
            "EfficientNetCustom is defined but not used in this solution."
        )

    def forward(self, x):
        raise NotImplementedError




## === cell 10
def train_model(
    model,
    train_loader,
    valid_loader,
    loss,
    optimizer,
    epoch,
    device=torch.device("cuda:0"),
):
    net = model.to(device)

    best_epoch = 0
    best_valid_loss = float("inf")
    best_model_state = None
    early_stopping_round = 10
    losses = []

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    for i in range(epoch):
        correct_train = 0
        loss_sum = 0.0
        net.train()
        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"train epoch {i}", leave=False)
        ):
            optimizer.zero_grad(
                set_to_none=True
            )  # faster; equivalent to zeroing gradients
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += float(loss_temp.detach())
            loss_temp.backward()
            optimizer.step()

            scheduler.step(i + step / max(1, len(train_loader)))

            correct_train += int((y_hat.argmax(dim=1) == y).sum().detach())

        losses.append(loss_sum / max(1, len(train_loader)))
        train_acc = correct_train / len(trainset)
        print(
            "epoch:",
            i,
            "loss=",
            loss_sum / max(1, len(train_loader)),
            "训练集准确度=",
            train_acc,
            end="",
        )

        correct_valid = 0
        valid_loss_sum = 0.0
        net.eval()
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"valid epoch {i}", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_hat = net(x)

                vloss = loss(y_hat, y)
                valid_loss_sum += float(vloss.detach())
                correct_valid += int((y_hat.argmax(dim=1) == y).sum().detach())

        valid_acc = correct_valid / len(validset)
        valid_loss_mean = valid_loss_sum / max(1, len(valid_loader))
        print("验证集准确度", valid_acc, "验证集loss", valid_loss_mean)

        if valid_loss_mean < best_valid_loss:
            best_model_state = copy.deepcopy(net.state_dict())
            best_valid_loss = valid_loss_mean
            best_epoch = i
            print("best epoch save! (by valid loss)")

        if i - best_epoch >= early_stopping_round:
            break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    testset = Dog_Breed(test_data, transform=transforms_test, test=True)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    predictions = []
    net.eval()
    with torch.no_grad():
        for x in tqdm(test_loader, desc="infer", leave=False):
            x = x.to(device, non_blocking=True)
            logits = net(x)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.detach().cpu().numpy())

    prediction = np.concatenate(predictions, axis=0)  # shape: (N,120)
    return prediction




## === cell 11
learn_rate = 0.0001
momentum = 0.9
epoch = 20



## === cell 12
model = MyResNet50()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

prediction = train_model(
    model, train_loader, valid_loader, loss, optimizer, epoch, device=device
)

sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
result = pd.DataFrame(prediction, columns=labels)

eps = 1e-7
prob_cols = [c for c in result.columns if c != "id"]
result[prob_cols] = result[prob_cols].clip(eps, 1.0 - eps)
result[prob_cols] = result[prob_cols].div(result[prob_cols].sum(axis=1), axis=0)

result = pd.concat(
    [test_data[["id"]].reset_index(drop=True), result.reset_index(drop=True)], axis=1
)
result = result[sample_sub.columns]  # id + 120 breed columns in required order

out_path = "/kaggle/working/dog_breed.csv"
result.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape=", result.shape)



## === cell 13
df = pd.read_csv("/kaggle/working/dog_breed.csv")
assert df.shape[0] == len(test_data), "Row count mismatch with test set."
assert list(df.columns) == list(
    pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv").columns
), "Submission columns mismatch."
df.head()
