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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8912058023572076

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!mkdir -p /tmp/pip/cache
!cp -r ../input/efficientnet /tmp/pip/cache
!mv /tmp/pip/cache/efficientnet/efficientnet_pytorch-0.7.0.xyz /tmp/pip/cache/efficientnet/efficientnet_pytorch-0.7.0.tar.gz
!ls /tmp/pip/cache/efficientnet/
!pip install --no-index --find-links /tmp/pip/cache/efficientnet efficientnet_pytorch

## === cell 1
import numpy as np
import cv2
import gc
import random
import torch
import os
import pandas as pd
from torch import optim
from torch import nn
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1999129853.py in <cell line: 0>()
     13 import time
     14 from tqdm import tqdm
---> 15 from efficientnet_pytorch import EfficientNet

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 2
image_size = 380

## === cell 3
config = dict(
    seed=22,
    experiment_name='enb7',
    test_location='../input/cassava-leaf-disease-classification/test_images',
    checkpoint_path='../input/savedmodels2',
    checkpoint='enb7best.pt',
    model='efficientnet-b7',
    epochs=10,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(
            name='RandomResizedCrop',
            params=dict(
                height=image_size,
                width=image_size
            )
        ),
        dict(
            name='HorizontalFlip',
            params=dict(
                always_apply=False,
                p=0.5,
            )
        ),
        dict(
            name='Transpose',
            params=dict(
                p=0.5
            )
        ),
        dict(
            name='VerticalFlip',
            params=dict(
                always_apply=False,
                p=0.5
                            )
        ),
        dict(
            name='HueSaturationValue',
            params=dict(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            )
        ),
        dict(
            name='RandomBrightnessContrast',
            params=dict(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            )
        ),
        dict(
            name='Normalize',
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            )
        ),
        dict(
            name='CoarseDropout',
            params=dict(
                p=0.5
            )
        ),
        dict(
            name='Cutout',
            params=dict(
                p=0.5
            )
        )
    ]
)

## === cell 4
test = pd.DataFrame()
test['image_id'] = list(os.listdir(config['test_location']))
test.head()

## === cell 5
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

## === cell 6
seed(config['seed'])
assert torch.cuda.is_available(), 'cuda not available'
device = torch.device('cuda')

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3168117085.py in <cell line: 0>()
      1 seed(config['seed'])
----> 2 assert torch.cuda.is_available(), 'cuda not available'
      3 device = torch.device('cuda')

AssertionError: cuda not available

## === cell 7
def load_model():
    model = EfficientNet.from_name(config['model'], num_classes=5)
    checkpoint = torch.load(os.path.join(config['checkpoint_path'], config['checkpoint']))
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    if 'epoch' in checkpoint:
        epoch = int(checkpoint['epoch'])
    if 'train_loss' in checkpoint:
        train_loss = checkpoint['train_loss']
    if 'val_loss' in checkpoint:
        val_loss = checkpoint['val_loss']
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
    if 'lr' in checkpoint:
        lr = checkpoint['lr']

    print('Loading model from checkpoint...')
    print('Epoch', epoch)
    print('Train loss', train_loss)
    print('Validation loss', val_loss)
    print('Accuracy', metrics)
    print('Learning rate', lr)

    return model.to(device)

## === cell 8
def get_transforms():
    transforms = [getattr(A, item['name'])(**item['params']) for item in config['inference_augmentations']]
    comp = A.Compose(transforms)
    return comp

## === cell 9
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        image = cv2.imread(os.path.join(config['test_location'], self.images[n]))
        image = self.transforms(image=image)['image']
        image = np.moveaxis(image, -1, 0)
        image = torch.FloatTensor(image)
        return image

    def __len__(self):
        return len(self.images)

## === cell 10
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test['image_id'])

    data = CassavaDataset(test_data, transforms)
    dataloader = DataLoader(data, shuffle=False, batch_size=config['batch_size'], pin_memory=False, num_workers=config['workers'])

    return dataloader

## === cell 11
def infer(model, dataloader):
    print('Running inference...')
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.squeeze(1).to(device)
            batch_hat = model(batch)
            predictions.append(batch_hat)

    return torch.cat(predictions, dim=0)

## === cell 12
if __name__ == '__main__':
    torch.cuda.empty_cache()

    dataloader = get_dataloader()
    model = load_model()
    predictions = None
    print('Inferring experiment', config['experiment_name'])

    for epoch in range(config['epochs']):
        print('Epoch:', epoch)
        start_time = time.time()

        if epoch == 0:
            predictions = infer(model, dataloader)
        else:
            predictions += infer(model, dataloader)

        print('Time:', time.time() - start_time)
        torch.cuda.empty_cache()
        gc.collect()

    predictions /= config['epochs']
    results = predictions.cpu().numpy()

    test['label'] = np.argmax(results, axis=-1)

## --- ERROR in cell 12, traceback:
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
/tmp/ipykernel_11/3981224584.py in <cell line: 0>()
      2     torch.cuda.empty_cache()
      3 
----> 4     dataloader = get_dataloader()
      5     model = load_model()
      6     predictions = None

/tmp/ipykernel_11/259806148.py in get_dataloader()
      1 # Create a Torch DataLoader with our data
      2 def get_dataloader():
----> 3     transforms = get_transforms()
      4     test_data = np.array(test['image_id'])
      5 

/tmp/ipykernel_11/3656798183.py in get_transforms()
      1 # Get train and test augmentations from the albumentations module
      2 def get_transforms():
----> 3     transforms = [getattr(A, item['name'])(**item['params']) for item in config['inference_augmentations']]
      4     comp = A.Compose(transforms)
      5     return comp

/tmp/ipykernel_11/3656798183.py in <listcomp>(.0)
      1 # Get train and test augmentations from the albumentations module
      2 def get_transforms():
----> 3     transforms = [getattr(A, item['name'])(**item['params']) for item in config['inference_augmentations']]
      4     comp = A.Compose(transforms)
      5     return comp

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

## === cell 13
test.to_csv('submission.csv', index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
