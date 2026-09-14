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

0.8830462375339981

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06801) has done: 'I fix the Albumentations pipeline by removing the unavailable `Cutout` transform, which also restores the definition of `sub_aug`. Then I replace the torchvision `ToTensor` conversion with a direct NumPy‑to‑torch conversion that respects the already‑normalized values, ensuring the inference loop runs without errors and produces a correctly‑named `submission.csv` file.'

# 9. Code solution

## === cell 0
config = {
    "DATA": {
        "IMAGES": "train_images",
        "LABELS": "train.csv",
        "SUB_IMAGES": "test_images",
        "SUB_LABELS": "sample_submission.csv",
        "SUB_OUTPUT": "submission.csv",
    },
    "DEVICE": "cuda",
    "NUM_GPU": torch.cuda.device_count(),
    "TRAIN_BATCH_SIZE": 64,
    "VAL_BATCH_SIZE": 32,
    "CLASSES": 5,
    "CV_FOLDS": 5,
    "NUM_EPOCHS": 6,  # keep original epoch count
    "MODEL_PATH": "model.pth",
    "SGD": {"LR": 0.0005, "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.001},
    "COS_ANN_LR": {"ETA_MIN": 0.00001},
    "MODEL_TYPE": "RESNET_50",  # keep as RESNET_50 for this script
}




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2524643270.py in <cell line: 0>()
      8     },
      9     "DEVICE": "cuda",
---> 10     "NUM_GPU": torch.cuda.device_count(),
     11     "TRAIN_BATCH_SIZE": 64,
     12     "VAL_BATCH_SIZE": 32,

NameError: name 'torch' is not defined

## === cell 1
train_csv_path = os.path.join(
    "..", "input", "cassava-leaf-disease-classification", "train.csv"
)
train_images_path = os.path.join(
    "..", "input", "cassava-leaf-disease-classification", "train_images"
)

train_df = pd.read_csv(train_csv_path)

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
train_subset = train_df.iloc[:-val_size].reset_index(drop=True)
val_subset = train_df.iloc[-val_size:].reset_index(drop=True)


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.transform = transform
        self.img_paths = [
            os.path.join(img_dir, img_id)
            for img_id in df["image_id"].astype(str).tolist()
        ]
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_tensor = tv_io.read_image(img_path)  # C x H x W, uint8
        img_np = img_tensor.permute(1, 2, 0).numpy()  # H x W x C
        if self.transform:
            img_np = self.transform(image=img_np)["image"]
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        label = self.labels[idx]
        return img_tensor, label


train_aug = A.Compose(
    [
        A.Resize(512, 512),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.CLAHE(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

val_aug = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

train_dataset = CassavaDataset(train_subset, train_images_path, transform=train_aug)
val_dataset = CassavaDataset(val_subset, train_images_path, transform=val_aug)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

num_workers = max(1, min(8, os.cpu_count() or 1))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["TRAIN_BATCH_SIZE"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config["VAL_BATCH_SIZE"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3081431073.py in <cell line: 0>()
----> 1 train_csv_path = os.path.join(
      2     "..", "input", "cassava-leaf-disease-classification", "train.csv"
      3 )
      4 train_images_path = os.path.join(
      5     "..", "input", "cassava-leaf-disease-classification", "train_images"

NameError: name 'os' is not defined

## === cell 2
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 5
batch_size = 32


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir):
        self.img_ids = df["image_id"].astype(str).tolist()
        self.img_paths = [os.path.join(img_dir, img_id) for img_id in self.img_ids]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_tensor = tv_io.read_image(img_path)  # C x H x W, uint8
        img_np = img_tensor.permute(1, 2, 0).numpy()
        return self.img_ids[idx], img_np


test_dataset = TestDataset(sample_sub, test_images_path)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

predictions = []

model.to(device)
model.eval()
with torch.no_grad():
    for ids, images_np in test_loader:
        batch_size_actual = len(ids)

        aug_batches = []
        for _ in range(tta_count):
            aug_images = [sub_aug(image=img)["image"] for img in images_np]
            batch_tensor = torch.stack(
                [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in aug_images]
            )
            aug_batches.append(batch_tensor)

        all_tensor = torch.cat(aug_batches, dim=0).to(
            device, non_blocking=True
        )  # (tta*B, C, H, W)

        with torch.cuda.amp.autocast():
            all_outputs = model(all_tensor)  # (tta*B, C)

        all_outputs = all_outputs.view(tta_count, batch_size_actual, config["CLASSES"])
        batch_logits = all_outputs.mean(dim=0)  # (B, C)

        _, pred_labels = torch.max(batch_logits, 1)
        for img_id, pred in zip(ids, pred_labels):
            predictions.append([img_id, int(pred.item())])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv(config["DATA"]["SUB_OUTPUT"], index=False)
print(sub_df.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1164650103.py in <cell line: 0>()
----> 1 sample_sub = pd.read_csv(sample_sub_path)
      2 tta_count = 5
      3 batch_size = 32
      4 
      5 

NameError: name 'pd' is not defined
