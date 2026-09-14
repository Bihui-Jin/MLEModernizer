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

3.12

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

# 5. Target score

0.8522212148685403

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the Albumentations v2 API breakage by updating `RandomResizedCrop` argument names so transforms can be constructed (this unblocks the dataloaders). Then I make inference robust by (1) using the correct output class count (5 for this competition) while keeping your model head size unchanged, (2) moving the model to an available device, and (3) loading weights from either the provided INPUT_PATH or the locally trained `best_model.pth` if the input file is missing. Finally, I ensure the submission file is built from `sample_submission.csv` (correct columns/order/row count) and that predictions align with the same `image_id` order, producing a valid `submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your low score is very likely coming from a label-space mismatch: the model is trained/loaded with `OUT_FEATURES=6` and then truncated to 5 classes at inference, which can seriously distort the argmax and wastes capacity on a nonexistent class. To move accuracy upward toward your target with minimal semantic changes, I set `OUT_FEATURES` to the competition’s true class count (5) and make the checkpoint loader tolerant by slicing away any extra 6th-class weights if your saved model used 6 outputs. I also fix the focal-loss implementation bug where `p_t` incorrectly uses logits instead of probabilities (this keeps the same “focal BCE” core idea but makes it mathematically correct and typically improves convergence/accuracy). All paths and the EfficientNet-B4 architecture/training loop remain the same, and the script still write a valid `submission.csv` in the required format.'

# 9. Code solution

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
def _resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("../input/"):
        alt = "/kaggle/input/" + p[len("../input/") :]
        if os.path.exists(alt):
            return alt
    return p




## === cell 2
INPUT_PATH = "../input/modelparam1003"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

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

INPUT_PATH = _resolve_path(INPUT_PATH)
TRAIN_CSV_PATH = _resolve_path(TRAIN_CSV_PATH)
TRAIN_IMAGE_PATH = _resolve_path(TRAIN_IMAGE_PATH)
TEST_IMAGE_PATH = _resolve_path(TEST_IMAGE_PATH)



## === cell 3
NUM_CLASSES = 5




## === cell 4
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

    bce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * bce, dim=-1)




## === cell 5
def to_onehot(labels, num_classes):
    y = torch.zeros((len(labels), num_classes), dtype=torch.float32)
    for i, lab in enumerate(labels):
        y[i, int(lab)] = 1.0
    return y




## === cell 6
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




## === cell 7
def _rrc(h, w, p=1.0):
    return A.RandomResizedCrop(
        size=(h, w), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333), p=p
    )




## === cell 8
train_augs = A.Compose(
    [
        _rrc(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
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



## === cell 9
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                _rrc(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
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



## === cell 10
if len(DEVICES) == 0:
    DEVICES = [torch.device("cpu")]




## === cell 11
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)




## === cell 12
def _load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)




## === cell 13
class MyCassavaLeafDataset(Dataset):
    @staticmethod
    def generate_index(num_total, ratio):
        all_index = [i for i in range(num_total)]
        k = int(ratio * 10)
        if k <= 0:
            k = 1
        valid_index = np.arange(0, num_total, k)
        train_index = [i for i in all_index if i not in set(valid_index.tolist())]
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
            self.image_arr = np.asarray(self.data_info.iloc[:, 0])
            self.label_arr = None
            self.real_len = len(self.image_arr)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        image = _load_image_rgb(os.path.join(self.images_path, single_image_name))
        if self.mode != "test":
            label = int(self.label_arr[index])
            return self.transform(image=image)["image"], label
        return self.transform(image=image)["image"], single_image_name

    def __len__(self):
        return self.real_len




## === cell 14
train_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=train_augs,
    mode="train",
)
my_train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
valid_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=valid_augs,
    mode="valid",
)
my_valid_dataloader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 15
my_model = torchvision.models.efficientnet_b4(weights=None)
my_model.classifier[-1] = nn.Linear(my_model.classifier[-1].in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model.classifier[-1].weight)



## === cell 16
my_model = my_model.to(DEVICES[0])



## === cell 17
criterion = sigmoid_focal_cross_entropy




## === cell 18
class MyTrainer:
    @staticmethod
    def accurate_count(y_hat, y_true):
        y_hat = y_hat.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        correct_count = 0
        for i in range(len(y_hat)):
            if y_hat[i].type(y_true.dtype) == y_true[i]:
                correct_count += 1
        return float(correct_count)

    @staticmethod
    def calc_valid_acc(model, valid_dataloader):
        model.eval()
        device = next(iter(model.parameters())).device
        test_num = 0
        test_acc_num = 0
        for x, y_true in valid_dataloader:
            if isinstance(x, list):
                x = [x_1.to(device) for x_1 in x]
            else:
                x = x.to(device)
            y_true = to_onehot(y_true, OUT_FEATURES).to(device)
            test_num += y_true.shape[0]
            test_acc_num += MyTrainer.accurate_count(model(x), y_true)
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

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_num = 0
        train_acc_num = 0
        batch_num = len(self.train_dataloader)

        excluded = {
            "module.classifier.1.weight",
            "module.classifier.1.bias",
            "classifier.1.weight",
            "classifier.1.bias",
        }
        param_1x = [
            param
            for name, param in self.model.named_parameters()
            if name not in excluded
        ]

        classifier_params = (
            self.model.module.classifier.parameters()
            if hasattr(self.model, "module")
            else self.model.classifier.parameters()
        )
        optimizer = self.optimizer_class(
            [
                {"params": param_1x},
                {"params": classifier_params, "lr": self.learning_rate(epoch) * 10},
            ],
            lr=self.learning_rate(epoch),
            weight_decay=0.001,
        )

        print(f"epoch{epoch + 1} begins:")
        tk0 = tqdm(enumerate(self.train_dataloader), total=batch_num)
        for batch_idx, (x, y_true) in tk0:
            y_true = to_onehot(y_true, OUT_FEATURES)
            x, y_true = x.to(self.devices[0]), y_true.to(self.devices[0])
            optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y_true)
            loss.sum().backward()
            optimizer.step()
            total_loss += loss.sum().detach()
            train_num += y_true.shape[0]
            train_acc_num += MyTrainer.accurate_count(y_hat.detach(), y_true)

        return total_loss / train_num, train_acc_num / train_num

    def train(self):
        best_valid_acc = 0
        if len(self.devices) > 1 and self.devices[0].type == "cuda":
            self.model = nn.DataParallel(
                self.model, device_ids=list(range(len(self.devices)))
            ).to(self.devices[0])
        else:
            self.model = self.model.to(self.devices[0])

        for epoch in range(self.num_epochs):
            train_loss, train_acc = MyTrainer.train_epoch(self, epoch)
            valid_acc = MyTrainer.calc_valid_acc(self.model, self.valid_dataloader)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), os.path.join("best_model.pth"))
            print(
                f"epoch{epoch + 1}:train_loss:{train_loss}, train_acc:{train_acc}, valid_acc:{valid_acc}"
            )




## === cell 19
torch.cuda.empty_cache()




## === cell 20
def _load_checkpoint_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICES[0])

    has_module = any(k.startswith("module.") for k in ckpt.keys())
    model_is_module = any(k.startswith("module.") for k in model.state_dict().keys())

    state = ckpt
    if has_module and not model_is_module:
        state = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    elif (not has_module) and model_is_module:
        state = {"module." + k: v for k, v in ckpt.items()}

    for w_key, b_key in [
        ("classifier.1.weight", "classifier.1.bias"),
        ("module.classifier.1.weight", "module.classifier.1.bias"),
    ]:
        if w_key in state and b_key in state:
            w = state[w_key]
            b = state[b_key]
            if w.ndim == 2 and w.shape[0] != OUT_FEATURES:
                if w.shape[0] > OUT_FEATURES:
                    state[w_key] = w[:OUT_FEATURES].contiguous()
                    state[b_key] = b[:OUT_FEATURES].contiguous()

    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError:
        try:
            model.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False




## === cell 21
candidate_ckpts = [
    os.path.join(INPUT_PATH, "best_model.pth"),
    os.path.join(".", "best_model.pth"),
]
ckpt_to_use = None
for p in candidate_ckpts:
    if os.path.exists(p):
        ckpt_to_use = p
        break



## === cell 22
my_model = my_model.to(DEVICES[0])
my_model.eval()
if ckpt_to_use is not None:
    ok = _load_checkpoint_into_model(my_model, ckpt_to_use)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_to_use}")
else:
    print(
        "WARNING: No best_model.pth found. Proceeding with randomly initialized weights."
    )

test_device = DEVICES[0]
sample_sub_path = _resolve_path(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.exists(sample_sub_path):
    sample_sub_path = _resolve_path(
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
sample_df = pd.read_csv(sample_sub_path)

preds = []
for single_image_name in tqdm(sample_df["image_id"].tolist(), total=len(sample_df)):
    with torch.no_grad():
        ans = torch.zeros(NUM_CLASSES, device=test_device)
        for _ in range(TTA):
            image = _load_image_rgb(os.path.join(TEST_IMAGE_PATH, single_image_name))
            aug_image = test_augs(image=image)["image"]  # CHW float tensor
            test_image = aug_image.to(test_device).unsqueeze(0)
            a = my_model(test_image)[0, :NUM_CLASSES]
            ans += a
        label = int(ans.argmax(dim=-1).item())
        preds.append(label)

submission = sample_df.copy()
submission["label"] = preds
submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Wrote submission to: {SUBMISSION_PATH} with shape {submission.shape} and columns {list(submission.columns)}"
)
