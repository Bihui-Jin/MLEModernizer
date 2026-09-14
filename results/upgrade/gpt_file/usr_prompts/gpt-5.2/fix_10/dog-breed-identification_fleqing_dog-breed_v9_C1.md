# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
from sklearn.model_selection import KFold
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass


def seed_everything(seed: int = 2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


seed_everything(2021)

INPUT_DIR = "/kaggle/input/dog-breed-identification"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

try:
    import torchvision

    torchvision.set_image_backend("accimage")
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 8) // 2)))
    torch.set_num_interop_threads(1)
except Exception:
    pass



## === cell 1
train_data = pd.read_csv(LABELS_CSV)

labels = sorted(train_data["breed"].unique().tolist())
breed_to_idx = {b: i for i, b in enumerate(labels)}
train_data["number"] = train_data["breed"].map(breed_to_idx).astype(int)

train_data.shape



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB)
test_data = sample_sub[["id"]].copy()

test_data["id"] = test_data["id"].astype(str).str.strip()
test_data = test_data[test_data["id"].ne("")].reset_index(drop=True)

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
from collections import OrderedDict

_TEST_TENSOR_CACHE = OrderedDict()
_VALID_TENSOR_CACHE = OrderedDict()


def _lru_get(cache: OrderedDict, key):
    v = cache.get(key)
    if v is not None:
        cache.move_to_end(key)
    return v


def _lru_put(cache: OrderedDict, key, value, max_items: int):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > max_items:
        cache.popitem(last=False)


_HAS_TV_READ = False


class Dog_Breed(Dataset):
    def __init__(
        self,
        train_csv: pd.DataFrame,
        transform=None,
        test: bool = False,
        cache_deterministic: bool = False,
        cache_max_items: int = 4096,
    ):
        super().__init__()
        self.train_csv = train_csv.reset_index(drop=True)
        self.image_ids = self.train_csv["id"].astype(str).tolist()
        self.test = bool(test)
        self.transform = transform
        self.cache_deterministic = bool(cache_deterministic)
        self.cache_max_items = int(cache_max_items)

        base_dir = TEST_DIR if self.test else TRAIN_DIR
        self.image_fps = [
            os.path.join(base_dir, img_id + ".jpg") for img_id in self.image_ids
        ]

        if not self.test:
            self.label_nums = self.train_csv["number"].astype(int).to_numpy()

        self._do_cache = self.cache_deterministic and (
            self.transform is transforms_test
        )

    def _read_rgb_pil(self, img_fp: str):
        with Image.open(img_fp) as im:
            return im.convert("RGB")

    def __getitem__(self, idx):
        img_fp = self.image_fps[idx]

        if self._do_cache:
            if self.test:
                cached = _lru_get(_TEST_TENSOR_CACHE, img_fp)
                if cached is not None:
                    return cached
            else:
                cached = _lru_get(_VALID_TENSOR_CACHE, img_fp)
                if cached is not None:
                    return cached, int(self.label_nums[idx])

        image = self._read_rgb_pil(img_fp)

        if self.transform is not None:
            image = self.transform(image)

        if self._do_cache:
            if self.test:
                _lru_put(_TEST_TENSOR_CACHE, img_fp, image, self.cache_max_items)
                return image
            else:
                _lru_put(_VALID_TENSOR_CACHE, img_fp, image, self.cache_max_items)
                return image, int(self.label_nums[idx])

        if not self.test:
            return image, int(self.label_nums[idx])
        return image

    def __len__(self):
        return len(self.image_fps)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
device



## === cell 6
pass



## === cell 7
pass




## === cell 8
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        try:
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.net = models.resnet50(pretrained=True)

        for param in self.net.parameters():
            param.requires_grad = False

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 9
EfficientNetCustom = None




## === cell 10
def _maybe_compile(model: nn.Module):
    return model


def _num_loader_workers():
    cpu = os.cpu_count() or 2
    return max(2, min(8, cpu))


def _seed_worker(worker_id: int):
    base_seed = 2021
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _dataloader_kwargs(nw: int, shuffle: bool):
    kwargs = dict(
        shuffle=shuffle,
        drop_last=False,
        num_workers=nw,
        pin_memory=(device == "cuda"),
        worker_init_fn=_seed_worker,
    )
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def train_model(model, train_loader, valid_loader, loss, optimizer, epoch, device):
    net = model.to(device)
    net = _maybe_compile(net)

    best_epoch = 0
    best_score = 0.0
    best_model_state = None
    early_stopping_round = 3
    losses = []

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    for ep in range(epoch):
        net.train()
        acc = 0

        loss_sum_t = torch.zeros((), device=device)

        for x, y in tqdm(train_loader, desc=f"train ep{ep}", leave=False, disable=True):
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = net(x)
            loss_temp = loss(y_hat, y)

            loss_sum_t = loss_sum_t + loss_temp.detach()

            loss_temp.backward()
            optimizer.step()

            acc += (y_hat.argmax(dim=1) == y).sum().item()

        scheduler.step()

        loss_epoch = (loss_sum_t / len(train_loader)).item()
        losses.append(loss_epoch)

        train_acc = acc / len(train_loader.dataset)
        print(f"epoch: {ep} loss={loss_epoch:.6f} 训练集准确度={train_acc:.6f}", end="")

        test_acc = 0
        net.eval()
        with torch.inference_mode():
            for x, y in tqdm(
                valid_loader, desc=f"valid ep{ep}", leave=False, disable=True
            ):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_hat = net(x)
                test_acc += (y_hat.argmax(dim=1) == y).sum().item()

        val_acc = test_acc / len(valid_loader.dataset)
        print(f" 验证集准确度 {val_acc:.6f}")

        if test_acc > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = test_acc
            best_epoch = ep
            print("best epoch save!")

        if ep - best_epoch >= early_stopping_round:
            break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net


def predict_test(net: nn.Module, test_loader, device):
    predictions = []
    net.eval()
    with torch.inference_mode():
        for x in tqdm(test_loader, desc="infer", leave=False, disable=True):
            x = x.to(device, non_blocking=True)
            logits = net(x)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.cpu())
    return torch.cat(predictions, dim=0).float()




## === cell 11
learn_rate = 0.001
momentum = 0.9
epoch = 15



## === cell 12
kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

testset = Dog_Breed(
    test_data,
    transform=transforms_test,
    test=True,
    cache_deterministic=True,
    cache_max_items=2048,
)
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    **_dataloader_kwargs(_num_loader_workers(), shuffle=False),
)

all_predictions_sum = torch.zeros((len(test_data), 120), dtype=torch.float32)

for fold, (train_index, val_index) in enumerate(kfold.split(train_data), start=1):
    print(f"\n=== Fold {fold}/5 ===")
    model = MyResNet50(num_classes=len(labels))

    train_fold = train_data.iloc[train_index].reset_index(drop=True)
    valid_fold = train_data.iloc[val_index].reset_index(drop=True)

    trainset = Dog_Breed(
        train_fold, transform=transforms_train, cache_deterministic=False
    )
    validset = Dog_Breed(
        valid_fold,
        transform=transforms_test,
        cache_deterministic=True,
        cache_max_items=4096,
    )

    nw = _num_loader_workers()
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        **_dataloader_kwargs(nw, shuffle=True),
    )
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=128,
        **_dataloader_kwargs(nw, shuffle=False),
    )

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    net = train_model(model, train_loader, valid_loader, loss, optimizer, epoch, device)

    prediction = predict_test(net, test_loader, device)
    all_predictions_sum += prediction * 0.2

sub = sample_sub.copy()
proba_df = pd.DataFrame(all_predictions_sum.numpy(), columns=labels)

breed_cols = [c for c in sample_sub.columns if c != "id"]
proba_df = proba_df.reindex(columns=breed_cols)

sub.loc[:, breed_cols] = proba_df.values

sub.to_csv("dog_breed.csv", index=False)
print("Wrote submission:", os.path.abspath("dog_breed.csv"), "shape:", sub.shape)



## === cell 13
sub_check = pd.read_csv("dog_breed.csv")
assert list(sub_check.columns) == list(
    sample_sub.columns
), "Submission columns do not match sample_submission"
row_sums = sub_check.drop(columns=["id"]).sum(axis=1).values
assert np.all(np.isfinite(row_sums)), "Non-finite probabilities found"
print("Submission looks valid. Mean row sum:", float(np.mean(row_sums)))



## === cell 14
pass
