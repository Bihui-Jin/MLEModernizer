# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.11

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm




## === cell 1
def _pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


BASE_INPUT = _pick_existing_path(
    [
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
    ]
)

TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(BASE_INPUT, "train_images")
TEST_IMAGE_PATH = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")



## === cell 2
INPUT_PATH = "../input/mymodelparam"  # kept, but not relied on (may not exist)
SUBMISSION_PATH = "submission.csv"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
if len(DEVICES) == 0:
    DEVICES = [torch.device("cpu")]

OUT_FEATURES = 5
NUM_EPOCHS = 20
BATCH_SIZE = 16
IMAGE_SIZE = 224
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 3

BEST_MODEL_PATH = "best_model.pth"  # Fix: consistent filename (no stray space)




## === cell 3
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor) + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * y_hat + (1 - y_true) * (1 - y_hat)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




## === cell 4
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    lr_start = LR_START
    lr_max = LR_MAX
    lr_final = LR_FINAL
    lr_warmup_epoch = 4
    lr_sustain_epoch = 0
    lr_decay_epoch = num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1

    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (torch.cos(torch.tensor(decay_factor)).numpy() + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)
plt.title("LR schedule")
plt.show()



## === cell 5
train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333),
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0
        ),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(IMAGE_SIZE // 20, IMAGE_SIZE // 5),
            hole_width_range=(IMAGE_SIZE // 20, IMAGE_SIZE // 5),
            fill=0,
            p=0.5,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)



## === cell 6
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    scale=(0.85, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 7
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)




## === cell 8
class MyCassavaLeafDataset(Dataset):
    @staticmethod
    def generate_index(num_total, ratio):
        all_index = [i for i in range(num_total)]
        k = int(max(1, round(ratio * 10)))
        valid_index = np.arange(0, num_total, k)
        valid_set = set(valid_index.tolist())
        train_index = [i for i in all_index if i not in valid_set]
        return train_index, valid_index

    def __init__(
        self,
        csv_path=None,
        images_path=None,
        transform=None,
        mode="train",
        train_ratio=0.5,
    ):
        super().__init__()
        self.transform = transform
        self.mode = mode
        self.images_path = images_path
        self.data_info = pd.read_csv(csv_path)
        self.data_len = self.data_info.shape[0]
        if self.mode == "train":
            train_index, _ = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.image_arr = np.asarray(self.data_info.iloc[train_index, 0])
            self.label_arr = np.asarray(self.data_info.iloc[train_index, 1])
            self.real_len = len(self.image_arr)
        elif self.mode == "valid":
            _, valid_index = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.image_arr = np.asarray(self.data_info.iloc[valid_index, 0])
            self.label_arr = np.asarray(self.data_info.iloc[valid_index, 1])
            self.real_len = len(self.image_arr)
        else:
            raise ValueError("mode must be 'train' or 'valid' for MyCassavaLeafDataset")

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        image = Image.open(os.path.join(self.images_path, single_image_name)).convert(
            "RGB"
        )
        image = np.array(image)
        label = int(self.label_arr[index])
        return self.transform(image=image)["image"], label

    def __len__(self):
        return self.real_len


class MyCassavaLeafTestDataset(Dataset):
    def __init__(self, image_dir, transform, image_ids):
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(os.path.join(self.image_dir, image_id)).convert("RGB")
        image = np.array(image)
        x = self.transform(image=image)["image"]
        return x, image_id




## === cell 9
train_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=train_augs,
    mode="train",
)
valid_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=valid_augs,
    mode="valid",
)

_NUM_WORKERS = min(8, os.cpu_count() or 2)
_loader_kwargs = dict(
    num_workers=_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=4 if _NUM_WORKERS > 0 else None,
)

my_train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)
my_valid_dataloader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)



## === cell 10
my_model = torchvision.models.efficientnet_b4(weights=None)
my_model.classifier[-1] = nn.Linear(my_model.classifier[-1].in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model.classifier[-1].weight)




## === cell 11
class MyTrainer:
    @staticmethod
    def accurate_count(y_hat, y_true):
        y_hat_idx = y_hat.argmax(dim=1)
        y_true_idx = y_true.argmax(dim=1)
        return float((y_hat_idx == y_true_idx).sum().item())

    @staticmethod
    def _to_one_hot(y_int, num_classes, device):
        y_int = y_int.to(device=device, dtype=torch.long)
        y_onehot = torch.zeros(
            (y_int.numel(), num_classes), device=device, dtype=torch.float32
        )
        y_onehot.scatter_(1, y_int.view(-1, 1), 1.0)
        return y_onehot

    @staticmethod
    def calc_valid_acc(model, valid_dataloader):
        model.eval()
        device = next(iter(model.parameters())).device
        test_num = 0
        test_acc_num = 0.0
        with torch.no_grad():
            for x, y_true in valid_dataloader:
                if isinstance(x, list):
                    x = [x_1.to(device, non_blocking=True) for x_1 in x]
                else:
                    x = x.to(device, non_blocking=True)
                y_true_oh = MyTrainer._to_one_hot(y_true, OUT_FEATURES, device)
                test_num += y_true_oh.shape[0]
                test_acc_num += MyTrainer.accurate_count(model(x), y_true_oh)
        return test_acc_num / test_num

    def __init__(
        self,
        optimizer,
        model,
        criterion,
        train_dataloader,
        valid_dataloader,
        param_group=True,
        learning_rate=lr_tune,
        num_epochs=NUM_EPOCHS,
        devices=DEVICES,
    ):
        self.optimizer_class = optimizer
        self.model = model
        self.criterion = criterion
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.param_group = param_group
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = None

    def _build_optimizer_once(self):
        if isinstance(self.model, nn.DataParallel):
            module = self.model.module
            named_params = list(self.model.named_parameters())
            exclude = {"module.classifier.1.weight", "module.classifier.1.bias"}
        else:
            module = self.model
            named_params = list(self.model.named_parameters())
            exclude = {"classifier.1.weight", "classifier.1.bias"}

        param_1x = [p for n, p in named_params if n not in exclude]
        classifier_params = list(module.classifier.parameters())

        self.optimizer = self.optimizer_class(
            [
                {"params": param_1x},
                {"params": classifier_params, "lr": self.learning_rate(0) * 10},
            ],
            lr=self.learning_rate(0),
            weight_decay=0.001,
        )

    def _set_epoch_lrs(self, epoch):
        lr = float(self.learning_rate(epoch))
        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.param_groups[1]["lr"] = lr * 10.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        train_num = 0
        train_acc_num = 0.0
        batch_num = len(self.train_dataloader)

        if self.optimizer is None:
            self._build_optimizer_once()
        self._set_epoch_lrs(epoch)

        print(f"epoch{epoch + 1} begins:")
        tk0 = tqdm(enumerate(self.train_dataloader), total=batch_num)
        for batch_idx, (x, y_true) in tk0:
            x = x.to(self.devices[0], non_blocking=True)
            y_true_oh = MyTrainer._to_one_hot(y_true, OUT_FEATURES, self.devices[0])

            self.optimizer.zero_grad(set_to_none=True)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y_true_oh)
            loss_sum = loss.sum()
            loss_sum.backward()
            self.optimizer.step()

            total_loss += float(loss_sum.detach().item())
            train_num += y_true_oh.shape[0]
            train_acc_num += MyTrainer.accurate_count(
                y_hat.detach(), y_true_oh.detach()
            )

        return total_loss / train_num, train_acc_num / train_num

    def train(self):
        best_valid_acc = 0

        if torch.cuda.is_available() and len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices).to(
                self.devices[0]
            )
        else:
            self.model = self.model.to(self.devices[0])

        for epoch in range(self.num_epochs):
            train_loss, train_acc = MyTrainer.train_epoch(self, epoch)
            valid_acc = MyTrainer.calc_valid_acc(self.model, self.valid_dataloader)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
            print(
                f"epoch{epoch + 1}:train_loss:{train_loss}, train_acc:{train_acc}, valid_acc:{valid_acc}"
            )




## === cell 12
torch.cuda.empty_cache()

trainer = MyTrainer(
    optimizer=OPTIMIZER,
    model=my_model,
    criterion=sigmoid_focal_cross_entropy,
    train_dataloader=my_train_dataloader,
    valid_dataloader=my_valid_dataloader,
)
trainer.train()



## === cell 13
device = DEVICES[0]
my_model = my_model.to(device)
my_model.eval()

state_dict = None
if os.path.exists(BEST_MODEL_PATH):
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
elif os.path.exists(os.path.join(INPUT_PATH, "best_model.pth")):
    state_dict = torch.load(
        os.path.join(INPUT_PATH, "best_model.pth"), map_location=device
    )
elif os.path.exists(os.path.join(INPUT_PATH, "best_model .pth")):  # legacy typo path
    state_dict = torch.load(
        os.path.join(INPUT_PATH, "best_model .pth"), map_location=device
    )

if state_dict is not None:
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    my_model.load_state_dict(state_dict, strict=True)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()

test_ds = MyCassavaLeafTestDataset(TEST_IMAGE_PATH, test_augs, test_image_ids)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

pred_map = {}
with torch.no_grad():
    for x, image_ids in tqdm(test_dl, total=len(test_dl)):
        x = x.to(device, non_blocking=True)
        logits_sum = torch.zeros((x.size(0), OUT_FEATURES), device=device)
        for _ in range(TTA):
            logits_sum.add_(my_model(x))
        pred = logits_sum.argmax(dim=1).detach().cpu().numpy()
        for img_id, p in zip(image_ids, pred):
            pred_map[img_id] = int(p)

submission = sample_sub.copy()
submission["label"] = submission["image_id"].map(pred_map).astype(int)
submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Wrote {SUBMISSION_PATH} with shape {submission.shape} and columns {list(submission.columns)}"
)
