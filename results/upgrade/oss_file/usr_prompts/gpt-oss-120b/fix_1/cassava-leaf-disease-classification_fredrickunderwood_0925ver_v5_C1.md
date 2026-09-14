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

## === cell 1
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
from  PIL import Image
import os
import random
from tqdm import tqdm

## === cell 3
INPUT_PATH = '../input/mymodelparam'
TRAIN_CSV_PATH = '../input/cassava-leaf-disease-classification/train.csv'
TRAIN_IMAGE_PATH = '../input/cassava-leaf-disease-classification/train_images/'
TEST_IMAGE_PATH = '../input/cassava-leaf-disease-classification/test_images/'
SUBMISSION_PATH = 'submission.csv'
DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
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

## === cell 5
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert  len(y.shape) == 2
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction='none')
    p_t = y_true * y_hat + (1 - y_true) * (1 - y_hat)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)


## === cell 7
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

## === cell 9
train_augs = A.Compose(
    [
        A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        A.CoarseDropout(p=0.5),
        A.Cutout(p=0.5),
        ToTensorV2(p=1.0)
    ], p=1.0)
valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ]
)

## --- ERROR in cell 9, traceback:
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

ValidationError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4244043746.py in <cell line: 0>()
      1 train_augs = A.Compose(
      2     [
----> 3         A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE),
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

ValueError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 11
test_augs = A.Compose([
    A.OneOf([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
        A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0)
    ], p=1.0),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0)
    ], p=1.0
)

## --- ERROR in cell 11, traceback:
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

ValidationError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2055867883.py in <cell line: 0>()
      3         A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
      4         A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
----> 5         A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0)
      6     ], p=1.0),
      7     A.Transpose(p=0.5),

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

ValueError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 13
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

## === cell 15
class MyCassavaLeafDataset(Dataset):
    @staticmethod
    def generate_index(num_total, ratio):
        all_index = [i for i in range(num_total)]
        k = ratio * 10
        valid_index = np.arange(0, num_total, k)
        train_index = [i for i in all_index if i not in valid_index]
        return train_index, valid_index

    def __init__(self, csv_path=None, images_path=None, transform=None, mode='train', train_ratio=0.5):
        super().__init__()
        self.transform = transform
        self.mode = mode
        self.images_path = images_path
        self.data_info = pd.read_csv(csv_path)
        self.data_len = self.data_info.shape[0]
        if self.mode == 'train':
            train_index, _ = MyCassavaLeafDataset.generate_index(self.data_len, train_ratio)
            self.image_arr = np.asarray(self.data_info.iloc[train_index, 0])
            self.label_arr = np.asarray(self.data_info.iloc[train_index, 1])
            self.real_len = len(self.image_arr)
        elif self.mode == 'valid':
            _, valid_index = MyCassavaLeafDataset.generate_index(self.data_len, train_ratio)
            self.image_arr = np.asarray(self.data_info.iloc[valid_index, 0])
            self.label_arr = np.asarray(self.data_info.iloc[valid_index, 1])
            self.real_len = len(self.image_arr)

    def __getitem__(self, index):
        if self.mode != 'test':
            single_image_name = self.image_arr[index]
            image = Image.open(os.path.join(self.images_path, single_image_name))
            image = np.array(image)
            label = self.label_arr[index]
            return self.transform(image=image)["image"], label


    def __len__(self):
        return self.real_len

## === cell 16
train_set = MyCassavaLeafDataset(csv_path=TRAIN_CSV_PATH, images_path=TRAIN_IMAGE_PATH, transform=train_augs, mode='train')
my_train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_set = MyCassavaLeafDataset(csv_path=TRAIN_CSV_PATH, images_path=TRAIN_IMAGE_PATH, transform=valid_augs, mode='valid')
my_valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1988755474.py in <cell line: 0>()
----> 1 train_set = MyCassavaLeafDataset(csv_path=TRAIN_CSV_PATH, images_path=TRAIN_IMAGE_PATH, transform=train_augs, mode='train')
      2 my_train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
      3 valid_set = MyCassavaLeafDataset(csv_path=TRAIN_CSV_PATH, images_path=TRAIN_IMAGE_PATH, transform=valid_augs, mode='valid')
      4 my_valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

NameError: name 'train_augs' is not defined

## === cell 18
my_model = torchvision.models.efficientnet_b4(pretrained=False)
my_model.classifier[-1] = nn.Linear(my_model.classifier[-1].in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model.classifier[-1].weight)
my_model

## === cell 20
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
            y_true_tensor = torch.zeros(size=(len(y_true), OUT_FEATURES))
            for i in range(len(y_true)):
                label = y_true[i]
                y_true_tensor[i, label] = 1
            y_true = y_true_tensor
            y_true = y_true.to(device)
            test_num += y_true.shape[0]
            test_acc_num += MyTrainer.accurate_count(model(x), y_true)
        return test_acc_num / test_num

    def __init__(self, optimizer, model, criterion, train_dataloader, valid_dataloader, param_group=True, learning_rate=lr_tune, num_epochs=NUM_EPOCHS, devices=DEVICES):
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
        param_1x = [param for name, param in self.model.named_parameters() if
                    name not in ['module.classifier.1.weight', 'module.classifier.1.bias']]
        optimizer = self.optimizer_class([{'params': param_1x},
                            {'params': self.model.module.classifier.parameters(),
                            'lr': self.learning_rate(epoch) * 10}],
                            lr=self.learning_rate(epoch), weight_decay=0.001)
        print(f'epoch{epoch + 1} begins:')
        tk0 = tqdm(enumerate(self.train_dataloader), total=batch_num)
        for batch_idx, (x, y_true) in tk0:
            y_true_tensor = torch.zeros(size=(len(y_true), OUT_FEATURES))
            for i in range(len(y_true)):
                label = y_true[i]
                y_true_tensor[i, label] = 1
            y_true = y_true_tensor
            x, y_true = x.to(self.devices[0]), y_true.to(self.devices[0])
            optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y_true)
            loss.sum().backward()
            optimizer.step()
            total_loss += loss.sum()
            train_num += y_true.shape[0]
            train_acc_num += MyTrainer.accurate_count(y_hat, y_true)

        return total_loss / train_num, train_acc_num / train_num
    def train(self):
        best_valid_acc = 0
        self.model = nn.DataParallel(self.model ,device_ids=self.devices).to(self.devices[0])
        for epoch in range(self.num_epochs):
            train_loss, train_acc = MyTrainer.train_epoch(self, epoch)
            valid_acc = MyTrainer.calc_valid_acc(self.model, self.valid_dataloader)
            if valid_acc > best_valid_acc:
                torch.save(self.model.state_dict(), os.path.join('best_model.pth'))
            print(f'epoch{epoch + 1}:train_loss:{train_loss}, train_acc:{train_acc}, valid_acc:{valid_acc}')

## === cell 22
torch.cuda.empty_cache()

## === cell 25
test_device = next(iter(my_model.parameters())).device
preds = []
my_model.eval()
model_param = torch.load(os.path.join(INPUT_PATH, 'best_model .pth'))
new_model_param = {k[7:]: v for k, v in model_param.items() if 'module.' in k}
my_model.load_state_dict(new_model_param)
test_image_list = np.asarray([image_name for image_name in os.listdir(TEST_IMAGE_PATH)])
for single_image_name in test_image_list:
    with torch.no_grad():
        ans = torch.zeros(5, device=test_device)
        for _ in range(TTA):
            image = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name))
            aug_image = test_augs(image=np.array(image))['image']
            test_image = torch.tensor(aug_image, dtype=torch.float, device=test_device).unsqueeze(0).permute([0, 1, 2, 3])
            print(1)
            ans += my_model(test_image).view(ans.shape)
            label = ans.argmax(dim=-1).cpu().numpy()
        preds.append(label)

df_submission = pd.DataFrame(columns=pd.read_csv(TRAIN_CSV_PATH).columns)
df_submission['image_id'] = pd.DataFrame(test_image_list)
df_submission['label'] = pd.DataFrame(preds)
df_submission.to_csv(SUBMISSION_PATH, index=False)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/925187960.py in <cell line: 0>()
      2 preds = []
      3 my_model.eval()
----> 4 model_param = torch.load(os.path.join(INPUT_PATH, 'best_model .pth'))
      5 new_model_param = {k[7:]: v for k, v in model_param.items() if 'module.' in k}
      6 # fixed_model_param = {}

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/mymodelparam/best_model .pth'
