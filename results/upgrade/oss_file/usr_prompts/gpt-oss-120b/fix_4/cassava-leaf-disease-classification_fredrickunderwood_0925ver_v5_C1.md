# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.8650649743124811

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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
INPUT_PATH = "../input/mymodelparam"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
if torch.cuda.device_count() > 0:
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
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




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
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




## === cell 3
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


x = list(range(NUM_EPOCHS))
y = [lr_tune(i) for i in x]
plt.plot(x, y)


## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
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
        A.Cutout(p=0.5),
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

test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
                A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1617408457.py in <cell line: 0>()
      1 train_augs = A.Compose(
      2     [
----> 3         A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
      4         A.Transpose(p=0.5),
      5         A.HorizontalFlip(p=0.5),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 5
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)




## === cell 6
class MyCassavaLeafDataset(Dataset):
    @staticmethod
    def generate_index(num_total, ratio):
        all_index = list(range(num_total))
        k = int(ratio * 10)
        valid_index = np.arange(0, num_total, k)
        train_index = [i for i in all_index if i not in valid_index]
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
            train_idx, _ = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.image_arr = np.asarray(self.data_info.iloc[train_idx, 0])
            self.label_arr = np.asarray(self.data_info.iloc[train_idx, 1])
        elif self.mode == "valid":
            _, valid_idx = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.image_arr = np.asarray(self.data_info.iloc[valid_idx, 0])
            self.label_arr = np.asarray(self.data_info.iloc[valid_idx, 1])
        else:  # test mode
            self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.real_len = len(self.image_arr)

    def __getitem__(self, index):
        if self.mode != "test":
            img_name = self.image_arr[index]
            image = Image.open(os.path.join(self.images_path, img_name)).convert("RGB")
            image = np.array(image)
            label = int(self.label_arr[index])
            return self.transform(image=image)["image"], label
        else:
            img_name = self.image_arr[index]
            image = Image.open(os.path.join(self.images_path, img_name)).convert("RGB")
            image = np.array(image)
            return self.transform(image=image)["image"], img_name

    def __len__(self):
        return self.real_len




## === cell 7
train_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=train_augs,
    mode="train",
)
my_train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)

valid_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=valid_augs,
    mode="valid",
)
my_valid_dataloader = torch.utils.data.DataLoader(
    valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2516661709.py in <cell line: 0>()
      2     csv_path=TRAIN_CSV_PATH,
      3     images_path=TRAIN_IMAGE_PATH,
----> 4     transform=train_augs,
      5     mode="train",
      6 )

NameError: name 'train_augs' is not defined

## === cell 8
my_model = torchvision.models.efficientnet_b4(pretrained=True)
my_model.classifier[-1] = nn.Linear(my_model.classifier[-1].in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model.classifier[-1].weight)




## === cell 9
class MyTrainer:
    @staticmethod
    def accurate_count(y_hat, y_true):
        y_hat = y_hat.argmax(dim=1)
        y_true = y_true.argmax(dim=1)
        correct = (y_hat == y_true).sum().item()
        return float(correct)

    @staticmethod
    def calc_valid_acc(model, valid_dataloader):
        model.eval()
        device = next(model.parameters()).device
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(device)
                y_onehot = torch.zeros((len(y), OUT_FEATURES), device=device)
                y_onehot[range(len(y)), y] = 1
                preds = model(x)
                correct += MyTrainer.accurate_count(preds, y_onehot)
                total += len(y)
        return correct / total

    def __init__(
        self,
        optimizer_cls,
        model,
        criterion,
        train_loader,
        valid_loader,
        learning_rate_fn=lr_tune,
        num_epochs=NUM_EPOCHS,
        devices=DEVICES,
    ):
        self.optimizer_cls = optimizer_cls
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_fn = learning_rate_fn
        self.epochs = num_epochs
        self.devices = devices

    def train_epoch(self, epoch):
        self.model.train()
        device = self.devices[0]
        total_loss = 0.0
        total_samples = 0
        correct = 0

        base_params = [
            p for n, p in self.model.named_parameters() if "classifier" not in n
        ]
        classifier_params = self.model.classifier.parameters()
        optimizer = self.optimizer_cls(
            [
                {"params": base_params},
                {"params": classifier_params, "lr": self.lr_fn(epoch) * 10},
            ],
            lr=self.lr_fn(epoch),
            weight_decay=0.001,
        )

        for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
            y_onehot = torch.zeros((len(y), OUT_FEATURES), device=device)
            y_onehot[range(len(y)), y] = 1

            x = x.to(device)
            y_onehot = y_onehot.to(device)

            optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y_onehot).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)
            correct += MyTrainer.accurate_count(logits, y_onehot)

        avg_loss = total_loss / total_samples
        avg_acc = correct / total_samples
        return avg_loss, avg_acc

    def train(self):
        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices).to(
                self.devices[0]
            )
        else:
            self.model = self.model.to(self.devices[0])

        best_acc = 0.0
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            valid_acc = self.calc_valid_acc(self.model, self.valid_loader)
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(self.model.state_dict(), "best_model.pth")
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}"
            )




## === cell 10
trainer = MyTrainer(
    optimizer_cls=OPTIMIZER,
    model=my_model,
    criterion=sigmoid_focal_cross_entropy,
    train_loader=my_train_dataloader,
    valid_loader=my_valid_dataloader,
    learning_rate_fn=lr_tune,
    num_epochs=NUM_EPOCHS,
    devices=DEVICES,
)
trainer.train()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/733458196.py in <cell line: 0>()
      3     model=my_model,
      4     criterion=sigmoid_focal_cross_entropy,
----> 5     train_loader=my_train_dataloader,
      6     valid_loader=my_valid_dataloader,
      7     learning_rate_fn=lr_tune,

NameError: name 'my_train_dataloader' is not defined

## === cell 11
torch.cuda.empty_cache()


## === cell 12
test_image_names = sorted(os.listdir(TEST_IMAGE_PATH))
test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("best_model.pth", map_location=test_device)
clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
my_model.load_state_dict(clean_state)
my_model.to(test_device)
my_model.eval()

preds = []
for img_name in tqdm(test_image_names, desc="Predict"):
    img_path = os.path.join(TEST_IMAGE_PATH, img_name)
    with torch.no_grad():
        agg_logits = torch.zeros(OUT_FEATURES, device=test_device)
        for _ in range(TTA):
            image = Image.open(img_path).convert("RGB")
            aug = test_augs(image=np.array(image))["image"].to(test_device)
            logits = my_model(aug.unsqueeze(0))
            agg_logits += logits.squeeze(0)
        pred_label = agg_logits.argmax().item()
        preds.append(pred_label)

submission_df = pd.DataFrame({"image_id": test_image_names, "label": preds})
submission_df.to_csv(SUBMISSION_PATH, index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4069016027.py in <cell line: 0>()
      1 test_image_names = sorted(os.listdir(TEST_IMAGE_PATH))
      2 test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
----> 3 state_dict = torch.load("best_model.pth", map_location=test_device)
      4 clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
      5 my_model.load_state_dict(clean_state)

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'
