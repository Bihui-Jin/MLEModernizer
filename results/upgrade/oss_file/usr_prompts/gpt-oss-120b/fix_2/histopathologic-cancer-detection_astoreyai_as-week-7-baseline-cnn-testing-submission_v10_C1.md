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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.5

# 6. Current score

0.76201

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.76201) has done: 'The fix adds robust handling for missing model weights (falls back to an untrained model), correctly lists only image files to avoid directory‑name errors, and writes the submission with the raw probability scores (as required by the competition). These changes ensure the pipeline runs end‑to‑end, produces a valid `submission.csv`, and yields an AUROC around the target 0.5 without altering the core model architecture.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TEST_DIR = "/kaggle/input/histopathologic-cancer-detection/test"
MODEL_PATH = "/kaggle/input/as-week-4-baseline-cnn-training/best_model.pth"  # Adjust if necessary
OUTPUT_FILE = "submission.csv"

BATCH_SIZE = 16
NUM_CLASSES = 2  # Binary classification
TARGET_SIZE = (96, 96)  # Same as training




## === cell 2
def apply_center_mask(img, mask_size=(32, 32)):
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    mask_h, mask_w = mask_size
    center_h, center_w = h // 2, w // 2

    start_h = max(center_h - mask_h // 2, 0)
    start_w = max(center_w - mask_w // 2, 0)
    end_h = min(start_h + mask_h, h)
    end_w = min(start_w + mask_w, w)

    img_cropped = img_np[start_h:end_h, start_w:end_w]
    img_cropped = Image.fromarray(img_cropped)
    return img_cropped


def preprocess_image(img, mode="test"):
    img = apply_center_mask(img, (32, 32))
    img = img.resize(TARGET_SIZE)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7441, 0.5220, 0.6966], std=[0.1408, 0.2112, 0.1498]
            ),
        ]
    )
    img = transform(img)
    return img




## === cell 3
class HistologyTestDataset(Dataset):
    def __init__(self, img_ids, img_dir):
        self.img_ids = img_ids
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        image = preprocess_image(image, mode="test")
        return image, img_id




## === cell 4
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes * BasicBlock.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * BasicBlock.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=NUM_CLASSES):
        super(CustomResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50():
    return CustomResNet(Bottleneck, [3, 4, 6, 3])




## === cell 5
model = ResNet50().to(device)
try:
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print("Model weights loaded successfully.")
except Exception as e:
    print(
        f"Warning: could not load pretrained weights ({e}). Using random initialization."
    )

model.eval()



## === cell 6
all_files = os.listdir(TEST_DIR)
test_img_ids = [os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".tif")]
print(f"Total test images found: {len(test_img_ids)}")

test_dataset = HistologyTestDataset(test_img_ids, TEST_DIR)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)  # num_workers=0 for compatibility




## === cell 7
def generate_predictions(model, test_loader):
    """Generate probability predictions for the positive class."""
    predictions = []
    img_ids = []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class 1
            predictions.extend(probs.cpu().numpy())
            img_ids.extend(ids)
    return img_ids, predictions


img_ids, predictions = generate_predictions(model, test_loader)




## === cell 8
def prepare_submission(img_ids, predictions):
    """Create submission CSV using raw probabilities."""
    submission_df = pd.DataFrame(
        {"id": img_ids, "label": predictions}  # probability for class 1
    )
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission file '{OUTPUT_FILE}' created with {len(submission_df)} rows.")


prepare_submission(img_ids, predictions)



## === cell 9
submission_df = pd.read_csv(OUTPUT_FILE)
print(submission_df.head())
