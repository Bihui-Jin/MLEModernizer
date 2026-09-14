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

3.13

# 3. Installed packages

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

0.7391961317618616

# 6. Current score

0.81652

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64051) has done: 'I fix the runtime failure by removing the hard dependency on the missing `/kaggle/input/resnetasda2/...pth` file and instead train the same ResNet50+ASDA architecture on the provided `train.csv`/`train_images` and then run inference on the test set. I also fix the minor submission alignment risk by ensuring predictions are written in the exact `sample_submission.csv` order (using `image_id` collected from the loader). These changes preserve the core model architecture and inference semantics, while making the notebook run end-to-end and produce a valid `submission.csv`. This should also yield a reasonable accuracy score (better than random) and move toward your target since previously no score was possible.'
- What this solution (achieved 0.86136) has done: 'I remove the hard failure that blocks execution when the weights file is missing, and instead always train (briefly) and save the same ResNet50+ASDA model so inference can run end-to-end and a valid `submission.csv` is produced. To keep the core logic intact while nudging accuracy upward toward your target, I keep the same architecture/loss/optimizer but enable ImageNet initialization for inference/training consistency and add a standard validation-based checkpoint (already present) without changing the training loop structure. I also fix a common PyTorch 2.6+ load issue by using `weights_only=True` when available, and ensure the submission is written in the exact `sample_submission.csv` order. All paths remain the same and the script always generate `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.64723) has done: 'Your current score (0.86136) is substantially higher than the target (0.739196...), so we should *reduce* performance slightly to move closer to the target band while keeping the same model/training/inference semantics. The smallest, low-risk way to do this without changing architecture or the training loop is to (1) switch ResNet initialization from ImageNet to random (still the same network), and (2) slightly increase regularization (weight decay) and reduce training signal a bit by using a smaller training image size. These are configuration-only changes that preserve the same pipeline end-to-end and still produce a valid `submission.csv`, but should lower accuracy from the current level toward your target.'
- What this solution (achieved 0.85239) has done: 'To move your score up toward the target (0.7392) from the current 0.6472 with minimal core-logic disruption, I keep the exact same ResNet50+ASDA model, loss, and training loop, but switch back to ImageNet initialization (a config-only change) since random init with only 2 epochs underfits. To avoid overshooting the target too much, I keep epochs the same and only moderately relax the regularization by reducing weight decay to a more standard 1e-4. I also align train-time augmentation with common Cassava practice by adding a light RandomResizedCrop (still simple torchvision transforms), which usually gives a noticeable but not extreme boost at the same epoch count. Everything else (paths, submission ordering, file writing) remains unchanged and still produce `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.85202) has done: 'We fix the training crash by making ASDA infer its spatial size dynamically from the incoming feature map instead of using hard-coded `RESNET_FM_SIZES` that don’t match when `IMAGE_SIZE=256` (your error shows 64×64 vs expected 80×80). This preserves the same ResNet50+ASDA architecture and training loop, but removes the shape mismatch and allows training to complete and save weights. Then inference reliably find the saved weights and produce a valid `/kaggle/working/submission.csv` in the exact `sample_submission.csv` order. No score-tuning changes are made beyond this bug fix; since current score is “Not yielded”, the priority is a correct end-to-end run and valid submission.'
- What this solution (achieved 0.61659) has done: 'Your current score (0.85202) is **higher** than the target (0.739196...), so the goal is to *reduce* accuracy modestly (toward the target band) while keeping the same ResNet50+ASDA architecture, loss, and train/infer flow. The smallest, most reliable way is to slightly weaken generalization without changing the core logic: (1) switch from ImageNet init to random init, and (2) increase regularization via weight decay. To ensure the change actually takes effect (and doesn’t silently reuse the previously-trained strong weights), the weights filename is made configuration-dependent so a fresh model is trained and used for inference. All I/O paths and submission formatting remain identical, and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.84417) has done: 'Your current score (0.61659) is below the target (0.7392), so we should cautiously improve accuracy with the smallest configuration-only changes that keep the same ResNet50+ASDA model, loss, and training loop. The biggest minimal lift here is to switch back to ImageNet initialization (same architecture, just better starting weights) and reduce the overly-strong weight decay that likely underfits at only 2 epochs. To make sure the run doesn’t accidentally reuse the older “random init / high wd” weights, the weights filename stays configuration-dependent and trigger a fresh train. Everything else (data paths, transforms pipeline structure, inference ordering, submission writing) remains the same and still produces `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.83819) has done: 'Your current score (0.84417) is higher than the target (0.7392), so the objective is to *reduce* accuracy slightly toward the target band with minimal, configuration-only changes. To reliably make the score drift downward without changing the model architecture or training/inference loops, I (1) slightly increase weight decay, and (2) slightly reduce the effective input resolution by lowering `IMAGE_SIZE`; both keep the same ResNet50+ASDA pipeline and metric semantics but typically soften generalization. To ensure these changes actually take effect (and don’t silently reuse previously-trained stronger weights), the existing configuration-dependent checkpoint path naturally point to a new filename and force a fresh train. All paths and the submission writing/order remain unchanged so it still produces a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.81652) has done: 'Your current accuracy (0.83819) is higher than the target (0.73920), so we should make a small, configuration-only change that predictably reduces generalization while keeping the same ResNet50+ASDA architecture, loss, and training/inference loops. The minimal lever here is to slightly reduce training signal by increasing weight decay a bit and lowering input resolution modestly; both preserve evaluation semantics but tend to soften accuracy. To ensure the change actually affects the run (and doesn’t reuse the previously stronger checkpoint), the checkpoint filename remains configuration-dependent and point to a new weights file automatically. Everything else (data paths, transforms structure, submission ordering/format) stays the same and still writes `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


class Config:
    DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
    TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")

    IMAGE_SIZE = 192

    NUM_CLASSES = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_INFERENCE = 64

    EPOCHS = 2
    LR = 3e-4

    WEIGHT_DECAY = 6e-4

    VAL_FRAC = 0.1
    SEED = 42

    RESNET_FM_SIZES = {
        "layer1": (80, 80),
        "layer2": (40, 40),
        "layer3": (20, 20),
        "layer4": (10, 10),
    }

    WEIGHTS_INIT_TYPE = "imagenet"

    BEST_MODEL_ASDA_PATH = os.path.join(
        "/kaggle/working",
        f"resnet50_asda_best_model__init-{WEIGHTS_INIT_TYPE}__wd-{WEIGHT_DECAY}__img-{IMAGE_SIZE}.pth",
    )


print(f"Using device: {Config.DEVICE}")
print(f"Train images: {Config.TRAIN_IMAGES_DIR}")
print(f"Test images: {Config.TEST_IMAGES_DIR}")
print(f"Will save/load model weights at: {Config.BEST_MODEL_ASDA_PATH}")
print(
    f"Config IMAGE_SIZE={Config.IMAGE_SIZE}, WEIGHT_DECAY={Config.WEIGHT_DECAY}, WEIGHTS_INIT_TYPE={Config.WEIGHTS_INIT_TYPE}"
)




## === cell 1
class CassavaTrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"]
        label = int(row["label"])
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name


train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

inference_transforms = transforms.Compose(
    [
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 2
class ASDA(nn.Module):
    """
    Bug fix (shape mismatch):
    The original ASDA used fixed (input_H, input_W) from Config.RESNET_FM_SIZES which
    can mismatch actual feature map size depending on IMAGE_SIZE / transforms.
    We preserve the same ASDA structure but generate spatial weights at runtime and
    upsample to the current (h, w), eliminating the 80x80 vs 64x64 error.
    """

    def __init__(self, channel, reduction_ratio=4):
        super(ASDA, self).__init__()
        self.conv_3x3 = nn.Conv2d(
            channel, channel // 2, kernel_size=3, padding=1, bias=False
        )
        self.conv_5x5 = nn.Conv2d(
            channel, channel // 2, kernel_size=5, padding=2, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1_reduce = nn.Conv2d(channel, 1, kernel_size=1, bias=False)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_spatial1 = nn.Linear(4 * 4, (4 * 4) // reduction_ratio, bias=False)
        self.fc_spatial2 = nn.Linear((4 * 4) // reduction_ratio, 4 * 4, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        f_3x3 = self.relu(self.conv_3x3(x))
        f_5x5 = self.relu(self.conv_5x5(x))
        f_local = torch.cat([f_3x3, f_5x5], dim=1)

        f_spatial_pre = self.conv_1x1_reduce(f_local)  # (b,1,h,w)
        f_pooled = self.adaptive_pool(f_spatial_pre)  # (b,1,4,4)
        f_pooled = f_pooled.view(b, -1)  # (b,16)

        f_linear = self.relu(self.fc_spatial1(f_pooled))  # (b,16//r)
        spatial_4x4 = self.fc_spatial2(f_linear).view(b, 1, 4, 4)
        spatial_4x4 = self.sigmoid(spatial_4x4)

        spatial_weights = torch.nn.functional.interpolate(
            spatial_4x4, size=(h, w), mode="bilinear", align_corners=False
        )
        return x * spatial_weights.expand_as(x)


class CCIA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCIA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.channel_interaction_conv = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = y.unsqueeze(1)
        y = self.channel_interaction_conv(y)
        y = y.squeeze(1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet50_Classifier(nn.Module):
    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        use_ccia=False,
        use_asda=False,
        weights_init_type="random",
    ):
        super(ResNet50_Classifier, self).__init__()

        if weights_init_type == "imagenet":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = models.resnet50(weights=None)

        self.use_ccia = use_ccia
        self.use_asda = use_asda

        self.resnet.fc = nn.Identity()

        if self.use_ccia:
            self.ccia_layer1 = CCIA(channel=256)
            self.ccia_layer2 = CCIA(channel=512)
            self.ccia_layer3 = CCIA(channel=1024)
            self.ccia_layer4 = CCIA(channel=2048)

        if self.use_asda:
            self.asda_layer1 = ASDA(channel=256)
            self.asda_layer2 = ASDA(channel=512)
            self.asda_layer3 = ASDA(channel=1024)
            self.asda_layer4 = ASDA(channel=2048)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        if self.use_ccia:
            x = self.ccia_layer1(x)
        if self.use_asda:
            x = self.asda_layer1(x)

        x = self.resnet.layer2(x)
        if self.use_ccia:
            x = self.ccia_layer2(x)
        if self.use_asda:
            x = self.asda_layer2(x)

        x = self.resnet.layer3(x)
        if self.use_ccia:
            x = self.ccia_layer3(x)
        if self.use_asda:
            x = self.asda_layer3(x)

        x = self.resnet.layer4(x)
        if self.use_ccia:
            x = self.ccia_layer4(x)
        if self.use_asda:
            x = self.asda_layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




## === cell 3
def _make_train_val_split(df: pd.DataFrame, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx_all = []
    val_idx_all = []
    for _, g in df.groupby("label", sort=False):
        idx = g.index.to_numpy(copy=False)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx_all.append(idx[:n_val])
        train_idx_all.append(idx[n_val:])
    train_idx_all = np.concatenate(train_idx_all)
    val_idx_all = np.concatenate(val_idx_all)
    train_df = (
        df.loc[train_idx_all].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )
    val_df = (
        df.loc[val_idx_all].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )
    return train_df, val_df


@torch.no_grad()
def _evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(Config.DEVICE, non_blocking=True)
        labels = labels.to(Config.DEVICE, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / max(1, total)


def train_and_save_if_needed():
    if os.path.exists(Config.BEST_MODEL_ASDA_PATH):
        print(
            f"Found existing weights at {Config.BEST_MODEL_ASDA_PATH}. Skipping training."
        )
        return

    print(
        "Weights not found; training a model so inference can run end-to-end (same ResNet50+ASDA core)."
    )

    full_df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = _make_train_val_split(full_df, Config.VAL_FRAC, Config.SEED)

    train_dataset = CassavaTrainDataset(
        train_df, Config.TRAIN_IMAGES_DIR, transform=train_transforms
    )
    val_dataset = CassavaTrainDataset(
        val_df, Config.TRAIN_IMAGES_DIR, transform=inference_transforms
    )

    nw = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE_INFERENCE,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2,
    )

    model = ResNet50_Classifier(
        num_classes=Config.NUM_CLASSES,
        use_ccia=False,
        use_asda=True,
        weights_init_type=Config.WEIGHTS_INIT_TYPE,
    ).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, desc=f"Training epoch {epoch+1}/{Config.EPOCHS}", leave=False
        )
        for imgs, labels in pbar:
            imgs = imgs.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            pbar.set_postfix(
                loss=running_loss / max(1, total), acc=correct / max(1, total)
            )

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)

        val_acc = _evaluate_accuracy(model, val_loader)
        print(
            f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.BEST_MODEL_ASDA_PATH)
            print(
                f"Saved new best weights (val_acc={best_val_acc:.4f}) to {Config.BEST_MODEL_ASDA_PATH}"
            )

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


train_and_save_if_needed()



## === cell 4
if Config.DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

print("\n--- Starting Inference on Test Set ---")

submission_df_template = pd.read_csv(Config.TEST_CSV)
test_image_ids = submission_df_template["image_id"].tolist()

test_dataset = TestDataset(
    image_ids=test_image_ids,
    img_dir=Config.TEST_IMAGES_DIR,
    transform=inference_transforms,
)

nw = min(4, os.cpu_count() or 2)
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE_INFERENCE,
    shuffle=False,
    num_workers=nw,
    pin_memory=True,
    persistent_workers=(nw > 0),
    prefetch_factor=4,
)

print(f"Total test samples for inference: {len(test_dataset)}")
print(f"Total batches for inference: {len(test_loader)}")

model_for_inference = ResNet50_Classifier(
    num_classes=Config.NUM_CLASSES,
    use_ccia=False,
    use_asda=True,
    weights_init_type=Config.WEIGHTS_INIT_TYPE,
)

if not os.path.exists(Config.BEST_MODEL_ASDA_PATH):
    raise FileNotFoundError(
        f"Expected trained weights at {Config.BEST_MODEL_ASDA_PATH} but file was not found. "
        "Training likely failed earlier."
    )

try:
    state = torch.load(
        Config.BEST_MODEL_ASDA_PATH, map_location=Config.DEVICE, weights_only=True
    )
except TypeError:
    state = torch.load(Config.BEST_MODEL_ASDA_PATH, map_location=Config.DEVICE)

model_for_inference.load_state_dict(state)
model_for_inference.to(Config.DEVICE)
model_for_inference.eval()

if hasattr(torch, "compile"):
    try:
        model_for_inference = torch.compile(model_for_inference, mode="reduce-overhead")
    except Exception as e:
        print(
            f"torch.compile unavailable/failed, continuing without compile. Reason: {repr(e)}"
        )

all_predictions = []
all_image_ids_from_loader = []

with torch.no_grad():
    for inputs, img_ids in tqdm(test_loader, desc="Predicting on test set"):
        inputs = inputs.to(Config.DEVICE, non_blocking=True)
        outputs = model_for_inference(inputs)
        predicted = outputs.argmax(dim=1).to("cpu")

        all_predictions.extend(predicted.tolist())
        all_image_ids_from_loader.extend(img_ids)

submission_df = pd.DataFrame(
    {"image_id": all_image_ids_from_loader, "label": all_predictions}
)

submission_df = submission_df.set_index("image_id").loc[test_image_ids].reset_index()

submission_file_path = os.path.join("/kaggle/working", "submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print(f"\nSubmission file saved to: {submission_file_path}")
print("First 5 rows of generated submission.csv:")
print(submission_df.head())
print("\nInference on test set complete.")
