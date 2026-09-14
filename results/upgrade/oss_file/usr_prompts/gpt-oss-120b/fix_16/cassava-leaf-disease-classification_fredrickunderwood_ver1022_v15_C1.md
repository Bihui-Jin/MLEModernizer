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
timm==1.0.19
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

0.8964944091870656

# 6. Current score

0.5654

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59454) has done: 'I fixed the Albumentations augmentation definitions which were causing validation errors (they required a tuple for the size argument). By supplying the size as a tuple in `RandomResizedCrop` for both training and test augmentations, the script now builds the augmentation pipelines correctly, allowing the inference loop to run and generate a proper `submission.csv` file.'
- What this solution (achieved 0.59604) has done: 'The changes set `NUM_EPOCHS` to 0 so the heavy fine‑tuning loop is skipped (the provided checkpoints already contain trained weights), and the learning‑rate plot is removed to avoid unnecessary work. This keeps the model architecture, loading, inference, and ensemble logic unchanged while ensuring the script finishes well within the 600 s limit.'
- What this solution (achieved 0.5654) has done: 'The script now skips expensive training and validation passes by setting `NUM_EPOCHS = 0` and uses a single test‑time augmentation (`TTA = 1`). In the validation‑skip branch we directly assign equal weights to the two models, avoiding a full pass over the validation set. These changes keep the model architecture and inference logic unchanged while dramatically reducing total runtime, ensuring the whole pipeline finishes well within the 600‑second limit.'

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
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm
import timm
from sklearn.model_selection import train_test_split




## === cell 1
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
if torch.cuda.is_available():
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
    DEVICES = [torch.device("cpu")]
OUT_FEATURES = 5
NUM_EPOCHS = 0
BATCH_SIZE = 64  # larger batch reduces iteration count
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 1




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
    lr_warmup_epoch = 2
    lr_sustain_epoch = 0
    lr_decay_epoch = max(num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1, 1)
    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (math.cos(decay_factor) + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr




## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), p=1.0),
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
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ],
    p=1.0,
)




## === cell 5
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
                A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), p=1.0),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
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




## === cell 6
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


seed_everything(SEED)




## === cell 7
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, aug):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        image = self.aug(image=np.array(image))["image"]
        label = int(row["label"])
        return image, label


df = pd.read_csv(TRAIN_CSV_PATH)
train_df, valid_df = train_test_split(
    df, test_size=0.1, random_state=SEED, stratify=df["label"]
)

train_dataset = CassavaDataset(train_df, TRAIN_IMAGE_PATH, train_augs)
valid_dataset = CassavaDataset(valid_df, TRAIN_IMAGE_PATH, valid_augs)

num_workers = min(8, os.cpu_count() or 1)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)




## === cell 8
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)


def load_ckpt(model, ckpt_path):
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICES[0])
        new_state = {k[7:]: v for k, v in state.items() if "module." in k}
        model.load_state_dict(new_state)
    return model


my_model_1 = load_ckpt(my_model_1, os.path.join(INPUT_PATH, RESNEXT_PATH))
my_model_2 = load_ckpt(my_model_2, os.path.join(INPUT_PATH, B4_PATH))

device = DEVICES[0]
if len(DEVICES) > 1:
    my_model_1 = nn.DataParallel(my_model_1)
    my_model_2 = nn.DataParallel(my_model_2)

my_model_1 = my_model_1.to(device)
my_model_2 = my_model_2.to(device)

if hasattr(torch, "compile"):
    my_model_1 = torch.compile(my_model_1)
    my_model_2 = torch.compile(my_model_2)

criterion = nn.CrossEntropyLoss()
optimizer1 = OPTIMIZER(my_model_1.parameters(), lr=LR_MAX)
optimizer2 = OPTIMIZER(my_model_2.parameters(), lr=LR_MAX)




## === cell 9
print("Starting fine‑tuning...")
acc1 = acc2 = 0.0  # will hold validation accuracies
if NUM_EPOCHS > 0:
    scaler1 = torch.cuda.amp.GradScaler()
    scaler2 = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        my_model_1.train()
        my_model_2.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            with torch.cuda.amp.autocast():
                outputs1 = my_model_1(images)
                loss1 = criterion(outputs1, labels)

                outputs2 = my_model_2(images)
                loss2 = criterion(outputs2, labels)

            scaler1.scale(loss1).backward()
            scaler2.scale(loss2).backward()

            scaler1.step(optimizer1)
            scaler2.step(optimizer2)

            scaler1.update()
            scaler2.update()

            running_loss1 += loss1.item() * images.size(0)
            running_loss2 += loss2.item() * images.size(0)

        lr = lr_tune(epoch)
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr

        epoch_loss1 = running_loss1 / len(train_loader.dataset)
        epoch_loss2 = running_loss2 / len(train_loader.dataset)

        my_model_1.eval()
        my_model_2.eval()
        correct1 = correct2 = total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.autocast():
                    outs1 = my_model_1(images)
                    outs2 = my_model_2(images)

                preds1 = outs1.argmax(dim=1)
                preds2 = outs2.argmax(dim=1)

                correct1 += (preds1 == labels).sum().item()
                correct2 += (preds2 == labels).sum().item()
                total += labels.size(0)

        acc1 = correct1 / total
        acc2 = correct2 / total

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Loss1 {epoch_loss1:.4f} Acc1 {acc1:.4f} | "
            f"Loss2 {epoch_loss2:.4f} Acc2 {acc2:.4f}"
        )
else:
    print("NUM_EPOCHS == 0 – skipping training and validation.")
    my_model_1.eval()
    my_model_2.eval()
    acc1 = acc2 = 0.5




## === cell 10
def inference_batch(models, img_paths, tta, batch_size=64):
    """
    models: list of models (already on the correct device)
    Returns list of predictions tensors, one per model.
    """
    for m in models:
        m.eval()
    all_preds = [[] for _ in models]
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            aug_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img_np = np.array(img)
                for _ in range(tta):
                    aug = test_augs(image=img_np)["image"]
                    aug_tensors.append(aug)
            batch_tensor = torch.stack(aug_tensors).to(device)  # (B*tta, C, H, W)

            for idx, model in enumerate(models):
                with torch.cuda.amp.autocast():
                    outputs = model(batch_tensor)  # (B*tta, OUT_FEATURES)
                outputs = outputs.view(len(batch_paths), tta, -1).mean(
                    dim=1
                )  # (B, OUT_FEATURES)
                all_preds[idx].append(outputs.cpu())
    return [torch.cat(p, dim=0) for p in all_preds]


torch.cuda.empty_cache()

test_image_list = np.asarray(
    [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
)
test_paths = [os.path.join(TEST_IMAGE_PATH, fn) for fn in test_image_list]

preds_list = inference_batch(
    [my_model_1, my_model_2], test_paths, tta=TTA, batch_size=64
)
predictions_1, predictions_2 = preds_list

normalize_pred_1 = F.normalize(predictions_1, p=2, dim=1)
normalize_pred_2 = F.normalize(predictions_2, p=2, dim=1)

weight_sum = acc1 + acc2
if weight_sum > 0:
    w1 = acc1 / weight_sum
    w2 = acc2 / weight_sum
else:
    w1 = w2 = 0.5

final_pred = (normalize_pred_1 * w1) + (normalize_pred_2 * w2)
label = final_pred.argmax(dim=1).numpy()
df_submission = pd.DataFrame({"image_id": test_image_list, "label": label.tolist()})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
